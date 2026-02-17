```markdown
# Document Summarizer Agent Python Multi-Agent System

## 1. Project Overview

The Document Summarizer Agent system is an autonomous Python-based solution designed to efficiently process and summarize documents. It receives documents in text or PDF format, extracts key points, and generates concise summaries (under 100 words) to help users quickly grasp main ideas. The system operates as a single self-contained agent without delegation to utility agents.

---

## 2. Architecture Overview

- **Single Autonomous Agent:**  
  The system is architected around one agent—the Document Summarizer Agent—which independently handles all document processing, parsing, and summarization tasks.
- **Linear Workflow:**  
  The workflow follows a straightforward sequence: document input → parsing and summarization → summary output.
- **Tool Integration:**  
  Summarization logic is encapsulated within a dedicated tool function (`summarize_document_tool`), directly invoked by the agent.

---

## 3. Agents & Tools Description

### Document Summarizer Agent (Supervisor Agent)

- **Description:**  
  An autonomous agent responsible for receiving documents, parsing them, extracting key points, and generating concise summaries under 100 words. Operates independently without delegation to utility agents.

#### Utility Agents

- **None**  
  The Document Summarizer Agent does not supervise or delegate tasks to any utility agents. All tasks are handled internally.

#### Tools

- **summarize_document**
  - **Description:**  
    Summarizes the provided document content by extracting key points and condensing the information into a concise summary of less than 100 words. The function ensures essential information is retained and reduces the length of the original document for quick comprehension.
  - **Logic:**  
    - Accept the input document content as a string (plain text).
    - Parse the text to identify main sections, sentences, and key points.
    - Select the most salient sentences reflecting main ideas.
    - Condense these sentences into a cohesive summary (≤100 words).
    - Return the summary.
  - **Inputs:**  
    - `document_content` (str): The textual content of the document to be summarized.
  - **Outputs:**  
    - `summary_text` (str): The concise summary of the document, containing less than 100 words.
  - **When to Use:**  
    - Whenever a new document is received for summarization by the Document Summarizer Agent.

---

## 4. Workflow Execution Flow

1. **Input Document**
   - The user submits a document (text or PDF).
   - The agent receives the document.
2. **Summarization Step**
   - The agent parses the document.
   - Extracts main content and key points.
   - Calls the `summarize_document` tool to generate a summary (≤100 words).
3. **Output Summary**
   - The agent outputs the concise summary to the user.

> **Note:** If the document format is unsupported or content is insufficient, the agent returns an error or requests a different format.

---

## 5. Folder Structure

```
project_root/
│
├── document_summarizer_agent.py      # Main agent script (with agent logic and server)
├── summarize_document_tool.py        # Tool implementation script (summarize_document function)
├── instructions.txt                  # Workflow and agent instructions
├── .env                              # Environment variables (API keys, config)
├── mcp_config.json                   # Tool configuration file
└── a2a_memory.json                   # Session/context memory file (created at runtime)
```

---

## 6. How to Run

### Requirements

- Python 3.8+
- Install dependencies (see agent script for required packages)

### Steps

1. **Prepare Environment:**
   - Place `.env` and `mcp_config.json` in `project_root/`.
   - Ensure `instructions.txt` is present.

2. **Install Dependencies:**
   ```bash
   pip install starlette uvicorn langchain langgraph langchain-openai python-dotenv httpx nest_asyncio
   ```

3. **Start Agent Server:**
   ```bash
   python document_summarizer_agent.py [PORT]
   ```
   - Default port is `8007` if not specified.

4. **Send Document for Summarization:**
   - Use HTTP requests or the provided API endpoints to submit a document and receive a summary.
   - Example endpoint: `http://127.0.0.1:8007/document_summarizer_agent/`

---

## 7. Notes / Constraints

- **Autonomous Operation:**  
  The Document Summarizer Agent processes all tasks internally; no utility agents or external delegation.
- **Supported Formats:**  
  Only text and PDF documents are supported. Unsupported formats will trigger an error.
- **Summary Limit:**  
  Output summaries are strictly limited to under 100 words.
- **Error Handling:**  
  If the document content is too short, empty, or cannot be parsed, a clear error message is returned.
- **No Branching:**  
  The workflow is linear; there are no formal branching decision points.
- **Session Memory:**  
  Session and context tracking are handled via in-memory structures and saved to `a2a_memory.json`.
- **Environment Variables:**  
  Ensure all required API keys and configuration are present in `.env` for model and tool operation.

---

## References

- [Tool Logic](summarize_document_tool.py): For direct implementation details of `summarize_document`.
- [Instructions](instructions.txt): Detailed workflow and agent responsibilities.

---

```
