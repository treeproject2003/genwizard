
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
from langchain_mcp_adapters.client import MultiServerMCPClient

async def get_tools():
    ''' Returns the list of tools for the agents. '''
    tools = []
    ENV_DIR = os.path.dirname(env_path)
    mcp_path = os.path.join(ENV_DIR, "mcp_config.json")
    try:
        with open(mcp_path, "r") as f:
            config = json.load(f)
        client = MultiServerMCPClient(config)
        tools = await client.get_tools()
    except Exception as e:
        pass
    return tools

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


def get_agent(name: str, prompt: str, tools: list):
    ''' Creates and provides new agents. '''
    model = get_model()
    new_agent = create_react_agent(
        name=name,
        prompt=prompt,
        model=model,
        tools=tools
    )

    return new_agent

class PromptEngineeringAgentAgentExecuter(AgentExecutor):

    def __init__(self):
        self.agent_config = {"name": "promptengineeringagent", "prompt": "You are an expert agent named PromptEngineeringAgent. Your goal is: Construct precise, context-aware prompts for LLM summarization, tailored to document type, style requirements, and word limits.. Your expertise includes: You are an expert in prompt engineering for large language models, with deep knowledge of summarization objectives, stylistic adaptations, and compliance with word limits. You dynamically compose prompts that clarify summarization goals, preserve essential context, and adapt instructions based on document domain (legal, medical, technical, etc.). You ensure optimal LLM performance and output fidelity. To complete your tasks, you can use following tools: prompt_template_generator, document_type_classifier, style_guide_lookup.", "tools": []}

        self.agent = None

    async def execute(
            self,
            context: RequestContext,
            event_queue: EventQueue
            ) -> None:


        if not self.agent_config['tools']:
            try:
                tools = await get_tools()
                self.agent_config['tools'] = tools
            except Exception as e:
                tools = []

        if not context.task_id or not context.context_id or not context.message:
            raise ValueError("Invalid request context.")
        
        updater = TaskUpdater(event_queue=event_queue, task_id=context.task_id, context_id=context.context_id)

        if not context.current_task:
            await updater.submit()

        await updater.start_work()
        
        query = context.get_user_input()
        try:
            if not self.agent:
                self.agent = get_agent(self.agent_config['name'], self.agent_config['prompt'], self.agent_config['tools'])
            result = await self.agent.ainvoke({"messages": [HumanMessage(content=query)]})
            messages = result.get("messages", [])
            content = next((m for m in reversed(messages) if isinstance(m, AIMessage)), AIMessage(content="Agent execution completed."))
            text = content.text

        except Exception as e:
            raise ValueError("Error executing the PromptEngineeringAgent Agent.")
        
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
            id='summarize_document',
            name='Context-Aware Document Summarization',
            description='Generates precise, context-aware prompts for LLM summarization tailored to specified document type, style, and word limit.',
            tags=['summarization', 'prompt engineering', 'document analysis', 'context aware'],
            examples=[
                'Create a summarization prompt for a research paper, formal style, 150 words.',
                'Generate a prompt for summarizing a legal contract in plain language, 100 words.',
                'Formulate a summary prompt for a news article, concise style, 80 words.'
            ],
            input_modes=['text'],
            output_modes=['text']
        ),
        AgentSkill(
            id='adapt_summarization_style',
            name='Summarization Style Adaptation',
            description='Constructs summarization prompts that adapt output style (e.g., formal, technical, conversational) based on user or document requirements.',
            tags=['summarization', 'style adaptation', 'prompt design'],
            examples=[
                'Draft a prompt for a conversational summary of a blog post, 50 words.',
                'Generate a technical summary prompt for an engineering report, 200 words.'
            ],
            input_modes=['text'],
            output_modes=['text']
        ),
        AgentSkill(
            id='enforce_word_limit',
            name='Word Limit Enforcement in Summarization',
            description='Precisely defines word limit constraints within LLM summarization prompts for accurate and concise outputs.',
            tags=['summarization', 'word limit', 'prompt engineering'],
            examples=[
                'Create a prompt for summarizing a case study within 75 words.',
                'Generate a summary prompt for a whitepaper, maximum 120 words.'
            ],
            input_modes=['text'],
            output_modes=['text']
        )
    ]
    
    agent_card = AgentCard(
        name='LLM Summarization Prompt Constructor',
        description='Constructs precise, context-aware prompts for LLM summarization, tailored to document type, style requirements, and word limits.',
        url=f'http://{{host}}:{{port}}/llm_summarization_prompt_constructor/',
        version='1.0.0',
        defaultInputModes=['text', 'text/plain'],
        defaultOutputModes=['text', 'text/plain'],
        capabilities=AgentCapabilities(streaming=True, push_notifications=True),
        skills=skills
    )
    

    return agent_card

def build_component_identifier_app(configured_card: AgentCard):
    executor = PromptEngineeringAgentAgentExecuter()
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
            Mount("/llm_summarization_prompt_constructor/", app_with_card),                      
        ],
        middleware=[
            Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
        ]
    )

    uvicorn.run(reminderstandardcollectionagent_app, host="127.0.0.1", port=port)

