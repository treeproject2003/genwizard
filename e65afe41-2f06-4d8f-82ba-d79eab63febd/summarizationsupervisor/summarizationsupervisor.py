
import os, sys, json, asyncio, uvicorn
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Part, TextPart, Task, UnsupportedOperationError
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.utils.errors import ServerError
from langgraph.prebuilt import create_react_agent       # Deprecated method use the one below
# from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

env_path = os.path.join(
    os.path.dirname(__file__), 
    "..", 
    "..", 
    "..",
    "..",
    ".env" 
)
load_dotenv(env_path)

import uuid, random, string, httpx, nest_asyncio
from datetime import datetime
from typing import List, Dict
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import (
    JSONRPCErrorResponse,
    SendMessageRequest,
    SendMessageResponse,
    AgentCard,
    MessageSendParams,
    Task
)


class SessionMemory:
    """Lightweight in-memory session tracker (no database persistence)."""

    def __init__(self):
        self.context_map: Dict[str, str] = {}
        self.session_data: Dict[str, List[Dict]] = {}
        self.context_state: Dict[str, Dict[str, str]] = {}  # added initialization
        self.context_summary: Dict[str, str] = {}

        self._active_session_id: str = self.create_session()

    def _generate_id(self, prefix: str) -> str:
        """Generate a simple readable session/context ID."""
        return f"{prefix}-" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

    def create_session(self) -> str:
        """Create a new session and its context mapping."""
        session_id = self._generate_id("SESS")
        context_id = self._generate_id("CTX")
        self.context_map[session_id] = context_id
        self.session_data[session_id] = []
        self.context_state[context_id] = {}  # initialize empty context dict
        self.context_summary[context_id] = ""
        self._active_session_id = session_id
        return session_id

    def active_session_id(self) -> str:
        return self._active_session_id

    def get_context_id(self, session_id: str) -> str:
        return self.context_map.get(session_id)

    def get_session_history(self, session_id: str) -> List[Dict]:
        return self.session_data.get(session_id, [])

    def update_context(self, context_id: str, key: str, value: str):
        """Update or add a key-value pair in the shared context for this session."""
        if context_id not in self.context_state:
            self.context_state[context_id] = {}
        self.context_state[context_id][key] = value
        #print(f"[SessionMemory] Context updated ({context_id}): {key} = {value}")

    def get_context(self, context_id: str) -> Dict[str, str]:
        """Retrieve the full shared context for a given context ID."""
        return self.context_state.get(context_id, {})
    
    def update_summary(self, context_id: str, user_input: str, agent_name: str, agent_output: str):
        """Update the semantic context summary for a session."""
        prev_summary = self.context_summary.get(context_id, "")
        new_event = f"\n[{agent_name}] {user_input.strip()} → {agent_output.strip()}"
        
        # Append the new event and trim to avoid overflow
        combined = (prev_summary + new_event).strip()
        if len(combined) > 500000:  # limit to manageable size
            combined = combined[-50000:]  # keep last part only
        
        self.context_summary[context_id] = combined


    def record_agent_call(self, session_id: str, agent: str, input_text: str, output_text: str, metadata=None):
        """Record a single agent interaction in session memory."""
        metadata = metadata or {}
        record = {
            "agent": agent,
            "input": input_text,
            "output": output_text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": metadata
        }
        self.session_data.setdefault(session_id, []).append(record)
        # print(f"Called_record_agent_call{self.session_data}")

    def dump_session(self, session_id: str):
        """Pretty-print the full session history for debugging."""
        context_id = self.context_map.get(session_id)  #  fixed undefined variable
        session = {
            "session_id": session_id,
            "context_id": context_id,
            "history": self.session_data.get(session_id, []),
            "context": self.context_state.get(context_id, {})  #  now safe
        }
        print("\n========== SESSION DUMP ==========")
        print(json.dumps(session, indent=2, default=str))

    def save_to_file(self, filename="a2a_memory.json"):
        """Save all session and context data to a JSON file."""
        data = {
            "context_map": self.context_map,
            "context_state": self.context_state,
            "context_summary": self.context_summary,
            "session_data": self.session_data
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[SessionMemory] Saved context to {filename}")

    def load_from_file(self, filename="a2a_memory.json"):
        """Load context and session data from JSON file if available."""
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
            self.context_map = data.get("context_map", {})
            self.context_state = data.get("context_state", {})
            self.context_summary = data.get("context_summary", {})
            self.session_data = data.get("session_data", {})
            print(f"[SessionMemory]  Loaded context from {filename}")
        else:
            print(f"[SessionMemory] No saved memory file found — starting fresh.")

    def list_saved_sessions(self) -> list:
        """Return a list of all saved session IDs."""
        return list(self.session_data.keys())

    def set_active_session(self, session_id: str):
        """Switch active session manually."""
        if session_id not in self.session_data:
            print(f"[SessionMemory] Session {session_id} not found.")
            return
        self._active_session_id = session_id
        print(f"[SessionMemory] Switched to session {session_id} (context {self.get_context_id(session_id)})")

#  Singleton instance
memory = SessionMemory()

class RemoteAgentConnections:
    """Handles async communication with a single remote agent."""
 
    def __init__(self, agent_card: AgentCard, agent_url: str):
        self.agent_card = agent_card
        self.agent_url = agent_url
        self._client = httpx.AsyncClient(timeout=1000)
        self._a2a_client = A2AClient(self._client, agent_card, url=agent_url)
        print(f"[RemoteAgentConnections] Connected to {agent_card.name} at {agent_url}")
 
    def get_agent(self) -> AgentCard:
        """Return the agent's metadata card."""
        return self.agent_card
 
    async def send_message(self, request: SendMessageRequest) -> SendMessageResponse:
        """Send a message request to the remote agent."""
        return await self._a2a_client.send_message(request)

_remote_addresses = ""
_clients: dict[str, RemoteAgentConnections] = {}
_discovered = False

async def discover_agents():
    """Discover remote agents by fetching their A2A cards."""
    global _discovered, _clients, _remote_addresses
    if _discovered:
        return

    async with httpx.AsyncClient(timeout=2000) as client:
        for addr in _remote_addresses:
            try:
                resolver = A2ACardResolver(client, addr)
                card = await resolver.get_agent_card()
                _clients[card.name] = RemoteAgentConnections(agent_card=card, agent_url=addr)
                print(f"[HostAgent] Discovered agent: {card.name} at {addr}")
            except Exception as e:
                print(f"[HostAgent] Could not fetch card from {addr}: {e}")

    _discovered = True
	
def list_agents_info() -> list:
    """Return a summary of discovered agents."""
    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(discover_agents())

    return [
        {
            "name": c.get_agent().name,
            "description": c.get_agent().description,
            "url": c.get_agent().url,
            "streaming": c.get_agent().capabilities.streaming,
        }
        for c in _clients.values()
    ]
	
def send_task(agent_name: str, raw_user_query: str, context_id: str) -> dict:
    """Send a user query to a chosen remote agent and return its response."""
    global _clients
    connection = _clients.get(agent_name)
    if not connection:
        return {"error": f"No agent found with name '{agent_name}'."}

    formatted_history = [
        {
            "role": "user",
            "parts": [{"type": "text", "text": raw_user_query}],
        }
    ]

    payload = {
        "message": {
            "messageId": str(uuid.uuid4()),
            "history": formatted_history,
            "role": "user",
            "parts": [{"type": "text", "text": raw_user_query}],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
    }

    request = SendMessageRequest(
        id=str(uuid.uuid4()), params=MessageSendParams.model_validate(payload)
    )

    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(connection.send_message(request))

    # Handle agent errors gracefully
    if isinstance(response.root, JSONRPCErrorResponse):
        return {
            "agent": agent_name,
            "input": raw_user_query,
            "output": f"[ERROR] {response.root.error.message}",
            "timestamp": datetime.now().isoformat(),
        }

    output = ""
    if isinstance(response.root.result, Task):
        task = response.root.result
        for artifact in task.artifacts or []:
            for part in artifact.parts:
                if isinstance(part.root, TextPart):
                    output += part.root.text + "\n"

        if not output and task.status and task.status.message:
            for part in task.status.message.parts:
                if isinstance(part.root, TextPart):
                    output += part.root.text + "\n"


    # --- Context and session tracking ---
    session_id = memory.active_session_id()
    context_id = memory.get_context_id(session_id)

    # Store latest interaction snapshot
    _store_context_snapshot(context_id, agent_name, raw_user_query, output)

    memory.save_to_file()


    # Record full call history in memory
    memory.record_agent_call(session_id, agent_name, raw_user_query, output)


    # Record full semantic summary in memory
    memory.update_summary(context_id, raw_user_query, agent_name, output)

    return {
        "agent": agent_name,
        "input": raw_user_query,
        "output": output.strip() or "No textual response from agent.",
        "timestamp": datetime.now().isoformat(),
    }
	
def _store_context_snapshot(context_id: str, agent_name: str, input_text: str, output_text: str):
	"""Lightweight context tracking: store last agent, input, and output."""
	memory.update_context(context_id, "last_agent_used", agent_name)
	memory.update_context(context_id, "last_input", input_text)
	memory.update_context(context_id, "last_output", output_text)


instructions=""
def load_instructions(
    file_path: str = "instructions.txt"
) -> str:
    """Reads the flow instructions from an external file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"WARNING: Flow instructions file not found at {file_path}. Agent flow will be incomplete.")
        return ""

def dynamic_instructions():
    instructions = load_instructions()
    print("flow instructions retrieved")
    return instructions

instructions=dynamic_instructions()

def get_model() -> AzureChatOpenAI:
    ''' Returns the Chat model for the agent. '''
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("OPENAI_API_VERSION")
    deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
    model = AzureChatOpenAI(
        azure_deployment=deployment_name,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        api_key=api_key,
        temperature=0.7,
        max_retries=3,
        max_completion_tokens=8000
    )

    return model


def get_supervisor_agent(name: str, prompt: str, tools: list):
    ''' Creates and provides the supervisor agent. '''
    model = get_model()
    supervisor_agent = create_react_agent(
        name=name,
        model=model,
        prompt=prompt,
        tools=tools
    )

    return supervisor_agent


class SummarizationSupervisorAgentExecuter(AgentExecutor):

    def __init__(self):
        self.agent_config = {"name": "summarizationsupervisor", "prompt": f"You are an expert agent named SummarizationSupervisor. Your goal is: Efficiently manage the team of agents and ensure high-quality task completion. Your primary goal is to coordinate all agents effectively to maximize productivity, ensuring each agent is actively contributing to achieve the best possible output. It is *critical* that no agent remains idle; if a task involves multiple files, you must ensure that each file is processed one by one. For each file, read its content, perform the necessary action, and save the updated file in the specified location, ensuring that the file paths and content are preserved throughout the process. When delegating tasks, it is of *utmost priority* to provide the entire context, as agents lack prior knowledge of task steps. Additionally, you *must* pass any relevant content received from one agent to another agent if it is essential for their task, ensuring smooth, uninterrupted information flow across agents. IMPORTANT: Always provide the full absolute path to agents for any file or directory they need to read from or save to. Avoid providing just the file or directory name; instead, ensure you specify the entire path to eliminate any ambiguity. In case of saving content, you must provide the entire content to the agent that needs to be saved, clearly indicating where it should be written. This is essential to ensure tasks are performed correctly and the content is saved in the appropriate location without confusion. Always include the full file path provided by the user to avoid ambiguity or misplacement. Important: Use this specific format when delegating tasks to coworkers to maintain clarity: {{\"coworker\": \"[Agent Name]\", \"task\": \"[Task Description with specific file path provided]\", \"context\": \"[Full context of the task, including previous content if applicable and the exact file path]\"}}. Your expertise includes: You are an experienced project manager known for managing complex projects and orchestrating seamless teamwork among a diverse team of agents. Your expertise lies in maximizing resource efficiency, utilizing each agent's strengths, and upholding high standards for collaboration and task completion. It is your *top priority* to ensure no agent remains underutilized, and to facilitate smooth, uninterrupted information transfer between agents to maintain workflow coherence. Efficiently manage the team of agents and ensure high-quality task completion. Your primary goal is to coordinate all agents effectively to maximize productivity, ensuring each agent is actively contributing to achieve the best possible output. It is *critical* that no agent remains idle; if a task involves multiple files, you must ensure that each file is processed one at a time. For each file, read its content, perform the necessary action, and save the updated file in the specified location, preserving the file path and content. When delegating tasks, it is of *utmost priority* to provide the entire context, as agents lack prior knowledge of task steps. Additionally, you *must* pass any relevant content received from one agent to another agent if it is essential for their task, ensuring smooth, uninterrupted information flow across agents. IMPORTANT: Always provide the full absolute path to agents for any file or directory they need to read from or save to. Avoid providing just the file or directory name; instead, ensure you specify the entire path to eliminate any ambiguity. Always include the full file path provided by the user to avoid ambiguity or misplacement. Important: Use this specific format when delegating tasks to coworkers to maintain clarity: {{\"coworker\": \"[Agent Name]\", \"task\": \"[Task Description with specific file path provided]\", \"context\": \"[Full context of the task, including previous content if applicable and the exact file path]\"}}.You can use list_agents_info tool to discover the utility agents and send_task tool to delegate tasks to utility agents. Ensure to strictly follow the instructions mentioned below: INSTRUCTIONS: {instructions}.", "tools": [list_agents_info, send_task]}

        self.agent = None

    async def execute(
            self,
            context: RequestContext,
            event_queue: EventQueue
            ) -> None:

        if not context.task_id or not context.context_id or not context.message:
            raise ValueError("Invalid request context.")
        
        updater = TaskUpdater(event_queue=event_queue, task_id=context.task_id, context_id=context.context_id)

        if not context.current_task:
            await updater.submit()

        await updater.start_work()
        
        query = context.get_user_input()
        try:
            if not self.agent:
                self.agent = get_supervisor_agent(self.agent_config['name'], self.agent_config['prompt'], self.agent_config['tools'])
            result = await self.agent.ainvoke({"messages": [HumanMessage(content=query)]})
            messages = result.get("messages", [])
            content = next((m for m in reversed(messages) if isinstance(m, AIMessage)), AIMessage(content="Agent execution completed."))
            text = content.text

        except Exception as e:
            raise ValueError("Error executing the SummarizationSupervisor Agent.")
        
        parts = [Part(root=TextPart(text=text))]

        await updater.add_artifact(parts=parts)
        await updater.complete()

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())

    def _validate_request(self, context: RequestContext) -> bool:
        return False


def create_agent_card(host: str, port: int | str):
    ''' Creates and returns the agent card for the Agent. '''
    skills = [
        AgentSkill(
            id='assign_summarization_task',
            name='Task Assignment for Summarization',
            description='Assigns input text to the Summarizer Agent for processing. Ensures efficient routing and task management for summarization requests.',
            tags=['task assignment', 'workflow management', 'summarization'],
            examples=['Assign the following text to be summarized.', 'Route input to Summarizer Agent for processing.'],
            input_modes=["text"],
            output_modes=["text"]
        ),
        AgentSkill(
            id='manage_workflow',
            name='Summarization Workflow Management',
            description='Oversees the complete workflow of summarization tasks, tracking their progress and ensuring all steps are executed efficiently.',
            tags=['workflow management', 'summarization process'],
            examples=['Monitor progress of ongoing summarization tasks.', 'Ensure all summarization requests are handled promptly.'],
            input_modes=["text"],
            output_modes=["text"]
        ),
        AgentSkill(
            id='quality_assurance_summaries',
            name='Quality Assurance for Summaries',
            description='Performs quality checks on produced summaries to verify conciseness, coherence, and preservation of context. Ensures summaries meet required standards.',
            tags=['quality assurance', 'summary validation', 'conciseness', 'context preservation'],
            examples=['Check this summary for conciseness and context.', 'Validate the quality of the generated summary.'],
            input_modes=["text"],
            output_modes=["text"]
        ),
        AgentSkill(
            id='summarization_expertise',
            name='Summarization Expertise',
            description='Provides expert guidance and feedback for improving summarization results, ensuring best practices in text condensation and information retention.',
            tags=['summarization', 'NLP', 'expert review'],
            examples=['Advise on improving summary quality.', 'Suggest best summarization strategies for complex texts.'],
            input_modes=["text"],
            output_modes=["text"]
        ),
        AgentSkill(
            id='summarize_text',
            name='Concise Text Summarization',
            description='Processes arbitrary input text and generates a concise summary (≤100 words), using NLP techniques to identify main points, remove redundancy, and preserve context.',
            tags=['text summarization', 'conciseness', 'context preservation', 'NLP'],
            examples=['Summarize the following article in ≤100 words.', 'Generate a concise summary for this paragraph.'],
            input_modes=["text"],
            output_modes=["text"]
        )
    ]
    
    agent_card = AgentCard(
        name='SummarizationSupervisor',
        description='Supervises the Summarizer Agent, ensuring all summarization tasks are handled efficiently and outputs meet the requirements of conciseness and context preservation.',
        url=f'http://{{host}}:{{port}}/summarizationsupervisor/',
        version='1.0.0',
        defaultInputModes=['text', 'text/plain'],
        defaultOutputModes=['text', 'text/plain'],
        capabilities=AgentCapabilities(streaming=True, push_notifications=True),
        skills=skills
    )
    

    return agent_card

def build_component_identifier_app(configured_card: AgentCard):
    executor = SummarizationSupervisorAgentExecuter()
    handler = DefaultRequestHandler(executor, InMemoryTaskStore())
    return A2AStarletteApplication(configured_card, handler).build()

if __name__ == "__main__":
    try:
        port = int(sys.argv[1]) 
    except (IndexError, ValueError):
        port = 8007
        
    HOST = "127.0.0.1" 

    async def redirect_agent_card(request):
        '''Serves the Agent Card directly from the expected well-known path.'''
        return JSONResponse(final_agent_card.model_dump(), media_type="application/json")
    
    final_agent_card = create_agent_card(host=HOST, port=port)
    app_with_card = build_component_identifier_app(final_agent_card)
    reminderstandardcollectionagent_app = Starlette(
        routes = [
            Route("/.well-known/agent.json", endpoint=redirect_agent_card),
            Mount("/summarizationsupervisor/", app_with_card),                      
        ],
        middleware=[
            Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
        ]
    )

    uvicorn.run(reminderstandardcollectionagent_app, host="127.0.0.1", port=port)

