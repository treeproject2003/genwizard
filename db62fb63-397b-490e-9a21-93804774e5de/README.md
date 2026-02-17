```markdown
# Multi-Agent Document Summarization System

## 1. Project Overview

This project delivers an advanced, orchestrated multi-agent Python system for automated document summarization. Users upload one or more documents and specify a word limit (default: 100 words). The system processes each document through a sequenced workflow of validation, preprocessing, prompt engineering, LLM summarization, iterative refinement, and post-processing, ensuring concise, accurate, and contextually faithful summaries strictly within the requested word limit.

## 2. Architecture Overview

The system is structured as a hierarchical agent network, coordinated by a `SummarizerAgentSupervisor`. Each agent specializes in a distinct stage of the summarization workflow, and the supervisor ensures sequential task execution, error handling, and compliance with user requirements. The workflow is modular, with clear delegation and context passing between agents, and leverages specialized tools for document processing and summarization.

## 3. Agents & Tools Description

### SummarizerAgentSupervisor (Supervisor Agent)

- **Description:** Orchestrates and supervises the entire summarization workflow, ensuring each step is performed in sequence and output meets quality, compliance, and user constraints.
- **Skills:** Workflow supervision, process orchestration, quality control, sequential task delegation, error correction, user satisfaction monitoring.
- **Tools:**
  - `list_agents_info`: Discovers available utility agents.
  - `send_task`: Delegates tasks to utility agents.

#### Utility Agents (Supervised by SummarizerAgentSupervisor)

1. **DocumentInputValidator**
   - *Description:* Validates uploaded documents for format, integrity, size, and compliance requirements before processing.
   - *Assigned Tools:*
     - `document_metadata_extractor`: Extracts metadata from documents.
     - `file_integrity_checker`: Checks for file corruption and readability.
     - `filetype_validator`: Validates document formats.
   - *Skills:* Document format validation, integrity checking, size enforcement, compliance checking.

2. **PreprocessingAgent**
   - *Description:* Cleans, normalizes, and chunks document text, removing metadata and non-content sections.
   - *Assigned Tools:*
     - `content_cleaner`: Cleans document content.
     - `metadata_remover`: Removes document metadata.
     - `text_chunker`: Chunks text into manageable sections.
     - `language_detector`: Detects document language.
   - *Skills:* Metadata removal, text normalization, text chunking.

3. **PromptEngineeringAgent**
   - *Description:* Constructs precise, context-aware prompts for LLM summarization, tailored to document type, style, and word limits.
   - *Assigned Tools:*
     - `prompt_template_generator`: Generates prompt templates.
     - `document_type_classifier`: Classifies document type.
     - `style_guide_lookup`: Provides style guide references.
   - *Skills:* Context-aware summarization, style adaptation, word limit enforcement.

4. **LLMSummarizationAgent**
   - *Description:* Generates high-quality, contextually faithful summaries from preprocessed and chunked document sections using LLMs.
   - *Assigned Tools:*
     - `summarize_document`: Summarizes document chunks.
     - `chunk_summary_aggregator`: Aggregates chunk summaries.
     - `summary_length_checker`: Checks summary length.
   - *Skills:* Chunk summarization, summary aggregation.

5. **SummaryOutputValidator**
   - *Description:* Validates final summary for word count, factual accuracy, contextual faithfulness, and industry-standard formatting.
   - *Assigned Tools:*
     - `summary_length_checker`: Validates word count.
     - `fact_checker`: Checks factual accuracy.
     - `formatting_enforcer`: Ensures proper formatting.
     - `compliance_validator`: Checks compliance.
   - *Skills:* Word count validation, accuracy verification, formatting validation.

6. **SummaryRefinementAgent**
   - *Description:* Iteratively improves summary drafts to resolve detected issues, enhance clarity, and optimize for conciseness and faithfulness.
   - *Assigned Tools:*
     - `summary_editor`: Edits summaries.
     - `feedback_analyzer`: Analyzes feedback.
     - `iterative_resummarizer`: Refines summaries iteratively.
   - *Skills:* Issue detection, clarity enhancement, conciseness optimization, faithfulness assurance.

7. **SummaryDeliveryAgent**
   - *Description:* Delivers the validated and refined summary to the user in the requested format, ensuring successful receipt and feedback collection.
   - *Assigned Tools:*
     - `summary_exporter`: Exports summary in required format.
     - `user_notifier`: Notifies user of delivery.
     - `delivery_tracker`: Tracks delivery status.
   - *Skills:* Summary validation, refinement, delivery in requested format.

---

**Tool: summarize_document**
- *Description (from builder):* Generates a concise summary (≤ word_limit words) of the content in one or more user-uploaded documents. Reads documents, removes metadata, validates against token limits, preprocesses, constructs prompts, invokes LLM, validates output, iteratively refines, applies post-processing, and returns the final summary.
- *Inputs:* `document_files: List[str]`, `word_limit: int`
- *Outputs:* `summary_text: str`
- *When to use:* Whenever a user uploads one or more documents and requests a summary within a specified word limit.

## 4. Workflow Execution Flow

1. **User uploads document(s) and specifies word limit**
2. **DocumentInputValidator**: Validates format, integrity, size, and compliance.
3. **PreprocessingAgent**: Cleans, normalizes, removes metadata, and chunks text as needed.
4. **PromptEngineeringAgent**: Constructs a tailored prompt for LLM summarization.
5. **LLMSummarizationAgent**: Performs LLM-based summarization on each chunk, aggregates chunk summaries.
6. **SummaryOutputValidator**: Validates the summary for word count, accuracy, faithfulness, and formatting.
7. **SummaryRefinementAgent**: Refines the summary iteratively if validation fails.
8. **SummaryDeliveryAgent**: Formats and delivers the summary to the user, confirming receipt and collecting feedback.

**Branching Logic:**
- If document size exceeds token limits, preprocessing and chunking are triggered.
- If summary does not meet word limit or quality, iterative refinement and re-summarization are performed.
- Supervisor ensures context and content are passed between agents and workflow order is maintained.

## 5. Folder Structure

```
summarizer_system/
├── summarizeragentsupervisor.py          # Supervisor agent script
├── documentinputvalidator.py             # Utility agent: document validation
├── preprocessingagent.py                 # Utility agent: preprocessing/cleaning
├── promptengineeringagent.py             # Utility agent: prompt engineering
├── llmsummarizationagent.py              # Utility agent: LLM summarization
├── summaryoutputvalidator.py             # Utility agent: summary validation
├── summaryrefinementagent.py             # Utility agent: summary refinement
├── summarydeliveryagent.py               # Utility agent: summary delivery
├── summarize_document_tool.py            # Tool logic script (summarize_document)
├── instructions.txt                      # Workflow orchestration instructions
├── mcp_config.json                       # Model/config file (if present)
├── a2a_memory.json                       # Session/context memory (generated)
├── requirements.txt                      # Python dependencies
└── README.md                             # This documentation
```

## 6. How to Run

1. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```
2. **Set Environment Variables:**
   - Configure `.env` file (Azure/OpenAI keys, endpoints).
3. **Start Agents:**
   - Run each agent script (e.g., supervisor and utility agents) as separate services:
     ```
     python summarizeragentsupervisor.py [port]
     python documentinputvalidator.py [port]
     python preprocessingagent.py [port]
     ...
     ```
   - Ensure all agents are discoverable (see `instructions.txt`).
4. **Interact:**
   - Upload documents and specify word limit via the supervisor agent API or UI.
   - Supervisor agent orchestrates the workflow and delivers the summary.

## 7. Notes / Constraints

- **Strict Word Limit:** Summaries are guaranteed not to exceed the specified word limit (default: 100 words).
- **Sequential Workflow:** Agents execute tasks in strict order, with context and content passed between stages.
- **No Agent or Tool Inference:** The system uses only the agents/tools described; no additional agents or tools are inferred or invented.
- **File Path Requirement:** All file operations use absolute paths as per supervisor instructions to prevent ambiguity.
- **Session Management:** The system tracks session/context to maintain workflow state and ensure repeatability.
- **LLM Integration:** Summarization logic is modular, allowing for integration with any compatible LLM (e.g., OpenAI, Azure).
- **Error Handling:** Supervisor agent monitors for workflow errors and initiates corrective actions.
- **Compliance:** Output is formatted to industry standards and validated for accuracy, clarity, and faithfulness.

---

**For any additional details, refer to the agent scripts and tool descriptions included in this repository.**
```
