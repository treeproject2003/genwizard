
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

class LLMSummarizationAgentAgentExecuter(AgentExecutor):

    def __init__(self):
        self.agent_config = {"name": "llmsummarizationagent", "prompt": "You are an expert agent named LLMSummarizationAgent. Your goal is: Generate high-quality, contextually faithful summaries from preprocessed and chunked document sections using LLMs.. Your expertise includes: You are a seasoned operator of LLM-based summarization workflows, skilled at managing input chunking, summarizing each segment with attention to coherence and context preservation. You aggregate partial summaries, handle edge cases such as cross-chunk references, and ensure the overall summary remains within the user-specified word limit. You specialize in multi-pass summarization for large or complex inputs. To complete your tasks, you can use following tools: summarize_document, chunk_summary_aggregator, summary_length_checker.", "tools": []}

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
            raise ValueError("Error executing the LLMSummarizationAgent Agent.")
        
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
            id='summarize_document_chunk',
            name='Document Chunk Summarization',
            description='Generates a high-quality, contextually faithful summary for a given preprocessed and chunked section of a document using LLMs.',
            tags=['summarization', 'document processing', 'LLM', 'chunked input'],
            examples=['Summarize the following chunk: "In 2020, the company achieved record growth..."'],
            input_modes=["text"],
            output_modes=["text"]
        ),
        AgentSkill(
            id='combine_chunk_summaries',
            name='Combine Chunk Summaries',
            description='Aggregates multiple chunk-level summaries into a coherent, contextually accurate summary for the overall document.',
            tags=['summary aggregation', 'document synthesis', 'LLM'],
            examples=['Combine these summaries into a full document summary: ["Chunk 1 summary...", "Chunk 2 summary..."]'],
            input_modes=["text"],
            output_modes=["text"]
        )
    ]
    
    agent_card = AgentCard(
        name='Generate high-quality, contextually faithful summaries from preprocessed and chunked document sections using LLMs.',
        description='This agent generates high-quality, contextually faithful summaries from preprocessed and chunked document sections using LLMs. It handles both individual chunk summarization and aggregation of chunk summaries for accurate document-level synthesis.',
        url=f'http://{{host}}:{{port}}/summarization_agent/',
        version='1.0.0',
        defaultInputModes=['text', 'text/plain'],
        defaultOutputModes=['text', 'text/plain'],
        capabilities=AgentCapabilities(streaming=True, push_notifications=True),
        skills=skills
    )
    

    return agent_card

def build_component_identifier_app(configured_card: AgentCard):
    executor = LLMSummarizationAgentAgentExecuter()
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
            Mount("/summarization_agent/", app_with_card),                      
        ],
        middleware=[
            Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
        ]
    )

    uvicorn.run(reminderstandardcollectionagent_app, host="127.0.0.1", port=port)

