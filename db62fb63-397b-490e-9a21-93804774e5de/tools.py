
from typing import Dict, Any, List

def summarize_document_tool(document_files: List[str], word_limit: int = 100) -> Dict[str, Any]:
    """
    Description:
        Generates a concise summary (≤ word_limit words) of the content in one or more user-uploaded documents.
        It reads the documents, removes metadata and non-content, validates against model token limits,
        preprocesses (cleans, normalizes, chunks) if needed, constructs a summarization prompt, 
        invokes an LLM summarizer, validates the output, iteratively refines if necessary, 
        and finally applies post-processing (formatting, language polish, metadata tagging).

    Logic:
        1. Accept a list of document file paths and a word limit.
        2. Read full text from each document.
        3. Remove metadata and non-content sections.
        4. Concatenate all documents (if multiple).
        5. Validate total content size against a model token limit (e.g., 8000 tokens, ~6000 words).
           If exceeded, chunk the content into semantically coherent pieces.
        6. For each chunk, construct a prompt specifying summarization objective, style, and word limit.
        7. For each chunk, run the LLM summarizer (placeholder: _llm_summarize).
        8. Aggregate all chunk summaries.
        9. Validate the aggregated summary for word count and quality.
        10. If not compliant, iteratively refine or re-summarize to fit the word limit.
        11. Apply post-processing: formatting, language polishing, and metadata tagging.
        12. Return the final summary as a string under the key "summary_text".

    Inputs:
        - document_files: List[str]
            List of absolute file paths to user-uploaded documents.
        - word_limit: int (default=100)
            The maximum number of words allowed in the output summary.

    Outputs:
        - summary_text: str
            The final, post-processed, concise summary (≤ word_limit words).

    When to use:
        Call this tool whenever a user uploads one or more documents and requests a summary within a specified word limit.
    """
    import os
    import re

    # --- Internal Utility Functions ---
    def _read_document(file_path: str) -> str:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _remove_metadata(text: str) -> str:
        # Remove common metadata sections: headers, footers, TOC, etc.
        # Remove lines that look like metadata (e.g., author, date, page numbers, copyright).
        lines = text.split('\n')
        content_lines = []
        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                continue
            # Skip lines with copyright, author, page, etc.
            if re.search(r'(copyright|author|page|all rights reserved|table of contents|toc|version|\d{4})', line_strip, re.IGNORECASE):
                continue
            if len(line_strip) < 4:
                continue
            content_lines.append(line_strip)
        return '\n'.join(content_lines)

    def _normalize_text(text: str) -> str:
        # Normalize whitespace, punctuation, and encoding.
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'\s+', ' ', text)
        # Remove extraneous punctuation at start/end
        text = text.strip(' \t\n\r')
        return text

    def _word_count(text: str) -> int:
        return len(re.findall(r'\w+', text))

    def _split_into_chunks(text: str, max_words: int = 2000) -> List[str]:
        # Split text into chunks with at most max_words words, prefer paragraph boundaries.
        words = text.split()
        chunks = []
        current_chunk = []
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= max_words:
                # Try to break at last sentence end if possible
                joined = ' '.join(current_chunk)
                m = re.search(r'(.+[.!?]) ', joined[::-1])
                if m:
                    idx = len(joined) - m.end(1)
                    chunks.append(joined[:idx].strip())
                    remainder = joined[idx:].strip()
                    current_chunk = remainder.split() if remainder else []
                else:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def _construct_prompt(chunk: str, word_limit: int) -> str:
        return (f"Summarize the following document content in clear, concise, and accurate prose. "
                f"Do not include any metadata or extraneous information. The summary must not exceed {word_limit} words. "
                f"Preserve main ideas and context:\n\n{chunk}\n")

    def _llm_summarize(prompt: str) -> str:
        # Placeholder for LLM summarization, replace with actual model integration.
        # For now, return first word_limit words of prompt content (simulate summarization).
        # In production, integrate with an LLM (e.g., OpenAI, Azure, etc.).
        content = prompt.split('\n\n', 1)[-1]
        # Extract up to word_limit words
        words = content.split()
        return ' '.join(words[:word_limit])

    def _post_process_summary(text: str) -> str:
        # Formatting, language polish, and metadata tagging (here: just clean up whitespace and add tags).
        text = text.strip()
        # Capitalize first letter, ensure proper ending.
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        if text and text[-1] not in '.!?':
            text += '.'
        # Metadata tag (for traceability)
        tag = "[Summary generated by SummarizerAgent]"
        return f"{text}\n\n{tag}"

    # --- Main Logic ---
    if not document_files or not isinstance(document_files, list):
        raise ValueError("document_files must be a non-empty list of file paths.")

    all_text = ""
    for file_path in document_files:
        doc_text = _read_document(file_path)
        doc_text = _remove_metadata(doc_text)
        doc_text = _normalize_text(doc_text)
        all_text += doc_text + "\n"

    all_text = all_text.strip()
    if not all_text:
        raise ValueError("No valid content found in the provided documents.")

    # Assume a model token limit (e.g. 8000 tokens ≈ 6000 words)
    MODEL_TOKEN_LIMIT_WORDS = 6000

    if _word_count(all_text) > MODEL_TOKEN_LIMIT_WORDS:
        # Preprocessing: chunking
        chunks = _split_into_chunks(all_text, max_words=2000)
    else:
        chunks = [all_text]

    # Summarize each chunk
    chunk_summaries = []
    for chunk in chunks:
        prompt = _construct_prompt(chunk, min(word_limit, 1000))
        summary = _llm_summarize(prompt)
        chunk_summaries.append(summary.strip())

    # Aggregate summaries if multiple chunks
    if len(chunk_summaries) > 1:
        aggregated = ' '.join(chunk_summaries)
        # Re-summarize aggregated summary if necessary
        if _word_count(aggregated) > word_limit:
            prompt = _construct_prompt(aggregated, word_limit)
            final_summary = _llm_summarize(prompt)
        else:
            final_summary = aggregated
    else:
        final_summary = chunk_summaries[0]

    # Output Validation: enforce word count and basic quality
    def valid_summary(summary: str) -> bool:
        return _word_count(summary) <= word_limit and len(summary.split()) >= max(10, word_limit // 2)

    MAX_ITER = 3
    iteration = 0
    while not valid_summary(final_summary) and iteration < MAX_ITER:
        # Iterative refinement: re-summarize with stricter prompt or truncate
        prompt = _construct_prompt(final_summary, word_limit)
        final_summary = _llm_summarize(prompt)
        iteration += 1

    # Last resort: truncate if still not compliant
    words = final_summary.split()
    if len(words) > word_limit:
        final_summary = ' '.join(words[:word_limit])

    # Post-processing
    summary_text = _post_process_summary(final_summary)

    return {
        "summary_text": summary_text
    }
