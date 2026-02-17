
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

class PreprocessingAgentAgentExecuter(AgentExecutor):

    def __init__(self):
        self.agent_config = {"name": "preprocessingagent", "prompt": "You are an expert agent named PreprocessingAgent. Your goal is: Clean, normalize, and prepare document text by removing metadata and non-content sections, and chunking text as needed for downstream processing.. Your expertise includes: You are a specialist in advanced text preprocessing for NLP tasks. You expertly extract main textual content, remove headers, footers, tables of contents, and irrelevant metadata. You normalize encoding, punctuation, and whitespace, and split large documents into context-preserving chunks suitable for LLM processing. You also detect and handle multilingual or mixed-format content. To complete your tasks, you can use following tools: content_cleaner, metadata_remover, text_chunker, language_detector.", "tools": []}

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
            raise ValueError("Error executing the PreprocessingAgent Agent.")
        
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
            id='remove_metadata',
            name='Metadata Removal',
            description='Removes metadata and non-content sections from document text to retain only the main content.',
            tags=['text cleaning', 'metadata removal', 'document processing'],
            examples=['Remove metadata from a PDF document.', 'Strip headers and footers from report text.'],
            input_modes=["text"],
            output_modes=["text"]
        ),
        AgentSkill(
            id='normalize_text',
            name='Text Normalization',
            description='Normalizes document text by standardizing whitespace, punctuation, and correcting common formatting issues.',
            tags=['text normalization', 'text cleaning'],
            examples=['Normalize spacing and punctuation in extracted document text.'],
            input_modes=["text"],
            output_modes=["text"]
        ),
        AgentSkill(
            id='chunk_text',
            name='Text Chunking',
            description='Chunks document text into manageable sections based on length or semantic breaks for downstream processing.',
            tags=['text chunking', 'document segmentation', 'preprocessing'],
            examples=['Divide document text into 500-word chunks.', 'Split text at paragraph boundaries.'],
            input_modes=["text"],
            output_modes=["list"]
        )
    ]
    
    agent_card = AgentCard(
        name='Document Preparation Agent',
        description='Cleans, normalizes, and prepares document text by removing metadata and non-content sections, and chunking text as needed for downstream processing.',
        url=f'http://{{host}}:{{port}}/document_preparation_agent/',
        version='1.0.0',
        defaultInputModes=['text', 'text/plain'],
        defaultOutputModes=['text', 'list', 'text/plain'],
        capabilities=AgentCapabilities(streaming=True, push_notifications=True),
        skills=skills
    )
    

    return agent_card

def build_component_identifier_app(configured_card: AgentCard):
    executor = PreprocessingAgentAgentExecuter()
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
            Mount("/document_preparation_agent/", app_with_card),                      
        ],
        middleware=[
            Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
        ]
    )

    uvicorn.run(reminderstandardcollectionagent_app, host="127.0.0.1", port=port)

