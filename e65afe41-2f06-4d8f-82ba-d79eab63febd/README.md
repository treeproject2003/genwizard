```markdown
# Python Multi-Agent Summarization System

## Project Overview

This project implements a modular, multi-agent system for automated text summarization. The system is designed to process arbitrary input text and generate a concise, coherent summary of up to 100 words. It leverages a hierarchical agent architecture with clear task delegation, context preservation, and built-in quality assurance.

---

## Architecture Overview

The system is structured in two main layers:

- **Supervisor Agent:** Orchestrates the summarization workflow, delegates tasks, maintains context, and performs quality assurance.
- **Utility Agents:** Specialized agents handling sub-tasks such as text analysis, redundancy removal, summarization, and quality checking.

Agents communicate via the A2A (Agent-to-Agent) protocol, and all agent endpoints are managed using Starlette/FastAPI with async support.

---

## Agents & Tools Description

### SummarizationSupervisor (Supervisor Agent)
- **Description:** Supervises the full summarization workflow, delegates input to utility agents, manages session context, and assures output quality.
- **Skills:**
  - Task Assignment for Summarization
  - Summarization Workflow Management
  - Quality Assurance for Summaries
  - Summarization Expertise
  - Concise Text Summarization

#### Utility Agents under SummarizationSupervisor

##### 1. Summarizer Agent (S1)
- **Type:** Utility Agent
- **Description:** Processes arbitrary input text to generate a concise, coherent summary not exceeding 100 words. Uses advanced NLP to identify main points, remove redundancy, and preserve context.
- **Assigned Tool:**
  - **summarize_text**
    - **Description:** Processes arbitrary input text and generates a concise summary limited to 100 words. Utilizes natural language processing techniques to:
      - Identify key points and main ideas
      - Remove redundant or less relevant information
      - Ensure the summary is coherent, concise, and preserves the original context
    - **When to use:** Whenever there is a request to summarize arbitrary text input, regardless of source or content domain.

##### 2. ContextAnalyzerAgent
- **Type:** Utility Agent
- **Description:** Analyzes input text to identify and preserve critical contextual elements, ensuring no loss of meaning or misrepresentation during summarization.
- **Assigned Tools:**
  - Contextual Summarization Tool
    - Summarizes input text while accurately preserving critical contextual elements.
  - Critical Context Extraction
    - Identifies and highlights critical contextual elements in input text.
  - Meaning Validation Assistant
    - Validates summarized output against the original to ensure no loss of meaning.

##### 3. RedundancyRemoverAgent
- **Type:** Utility Agent
- **Description:** Eliminates redundant or repetitive information from input text to streamline content and improve summary quality.
- **Assigned Tools:**
  - Redundant Information Eliminator
    - Identifies and removes redundant or repetitive information from input text to streamline content.
  - Pre-summarization Text Optimizer
    - Optimizes input text by eliminating repetitive content before summarization.

##### 4. SummaryQualityCheckerAgent
- **Type:** Utility Agent
- **Description:** Evaluates generated summaries for coherence, conciseness, and context preservation, ensuring alignment with output guidelines before final delivery.
- **Assigned Tools:**
  - Coherence Evaluation
    - Assesses the logical flow and clarity of generated summaries.
  - Conciseness Evaluation
    - Measures if the summary is brief yet complete, removing redundancy.
  - Context Preservation Assessment
    - Checks if the summary maintains the essential meaning and context of the source.
  - Output Guidelines Compliance
    - Ensures the summary aligns with specified formatting and output guidelines.

---

## Workflow Execution Flow

The summarization workflow is linear and managed by the SummarizationSupervisor:

1. **User submits text for summarization.**
   - SummarizationSupervisor receives the request.

2. **SummarizationSupervisor delegates the task to Summarizer Agent.**
   - Summarizer Agent processes the input (analyzes, extracts key points, removes redundancy, generates ≤100 word summary).

3. **Summary returned to SummarizationSupervisor.**
   - Supervisor checks summary for conciseness, coherence, and context preservation.

4. **If requirements are met, summary is returned to user.**
   - If not, feedback is provided and summarization is repeated.

- **Note:** All delegation is direct; there are no branching decision points.

---

## Folder Structure

```
project_root/
│
├── summarizationsupervisor.py               # Supervisor agent (entry point)
├── summarizeragent.py                       # Summarizer utility agent
├── contextanalyzeragent.py                  # Context analyzer utility agent
├── redundancyremoveragent.py                # Redundancy remover utility agent
├── summaryqualitycheckeragent.py            # Summary quality checker agent
├── instructions.txt                         # Workflow instructions
├── mcp_config.json                          # MCP config for tool discovery (if used)
├── .env                                     # Environment variables for model endpoints
├── a2a/                                     # A2A protocol support (library/dependency)
└── ...                                      # Other dependencies and assets
```

---

## How to Run

1. **Install dependencies:**
   - Ensure you have Python 3.8+ and all requirements installed (`pip install -r requirements.txt`).
   - Ensure Starlette, Uvicorn, LangChain, and required adapters are installed.

2. **Set up environment:**
   - Fill `.env` with OpenAI/Azure credentials as required by the agents.

3. **(Optional) Edit `mcp_config.json`:**
   - If your agents/tools are registered in an MCP, update configuration accordingly.

4. **Start the Supervisor Agent:**
   ```bash
   python summarizationsupervisor.py [PORT]
   ```
   - Default port is 8007.

5. **(Optional) Start utility agents individually:**
   ```bash
   python summarizeragent.py [PORT]
   python contextanalyzeragent.py [PORT]
   python redundancyremoveragent.py [PORT]
   python summaryqualitycheckeragent.py [PORT]
   ```

6. **Submit requests:**
   - Send HTTP requests to the SummarizationSupervisor endpoint (e.g., `http://127.0.0.1:8007/summarizationsupervisor/`).

---

## Notes / Constraints

- **All task delegation and workflow logic is managed by SummarizationSupervisor.**
- **No agent persists state across sessions except via the shared session/context memory in the supervisor.**
- **All agent-to-agent and agent-to-tool communication uses the A2A protocol and HTTP APIs.**
- **All file and directory paths must be specified as full absolute paths (per supervisor requirements).**
- **Summary output strictly does not exceed 100 words.**
- **The system expects English text input.**
- **No workflow branching or parallelism; process is strictly linear.**

---

**End of README**
```