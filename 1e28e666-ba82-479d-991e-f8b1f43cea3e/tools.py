
from typing import Dict, Any, List

def summarize_document_tool(document_content: str) -> Dict[str, Any]:
    """
    Description:
        Summarizes the provided document content by extracting key points and condensing the information
        into a concise summary of less than 100 words. The function ensures essential information is retained
        and reduces the length of the original document for quick comprehension.

    Logic:
        - Accept the input document content as a string (assumed to be plain text).
        - Parse the text to identify main sections, sentences, and key points.
        - Select the most salient sentences that reflect the main ideas.
        - Condense these sentences into a cohesive summary.
        - Ensure the summary does not exceed 100 words.
        - Return the summary.

    Inputs:
        - document_content (str): The textual content of the document to be summarized.

    Outputs:
        - summary_text (str): The concise summary of the document, containing less than 100 words.

    When to use:
        - Whenever a new document is received for summarization by the Document Summarizer Agent.

    """
    import re

    def clean_text(text: str) -> str:
        # Remove excessive whitespace, normalize
        return re.sub(r'\s+', ' ', text.strip())

    def split_sentences(text: str) -> List[str]:
        # Simple sentence splitter (does not handle complex cases)
        return re.split(r'(?<=[.!?])\s+', text)

    def extract_key_points(sentences: List[str]) -> List[str]:
        # Basic scoring: prioritize sentences containing keywords (heuristic)
        keywords = [
            "summary", "conclusion", "results", "finding", "key", "main", "important", "objective",
            "purpose", "goal", "we found", "recommend", "suggest", "demonstrate", "reveals", "shows"
        ]
        scored = []
        for idx, sent in enumerate(sentences):
            score = 0
            sent_lower = sent.lower()
            for kw in keywords:
                if kw in sent_lower:
                    score += 2
            if idx == 0:
                score += 1  # often the opening sentence contains context
            if len(sent.split()) > 5:
                score += 1  # prefer longer, informative sentences
            scored.append((score, sent))
        # Sort by score, descending
        scored.sort(reverse=True, key=lambda x: x[0])
        return [s for _, s in scored]

    def compose_summary(key_sentences: List[str], word_limit: int = 100) -> str:
        summary = []
        total_words = 0
        for sent in key_sentences:
            sent_clean = clean_text(sent)
            sent_word_count = len(sent_clean.split())
            if (total_words + sent_word_count) <= word_limit:
                summary.append(sent_clean)
                total_words += sent_word_count
            else:
                remaining = word_limit - total_words
                if remaining > 0:
                    # Add only as much as fits
                    partial = ' '.join(sent_clean.split()[:remaining])
                    summary.append(partial)
                    total_words += remaining
                break
        return ' '.join(summary).strip()

    try:
        text = clean_text(document_content)
        sentences = split_sentences(text)
        if not sentences or len(text) < 10:
            return {"summary_text": "Error: Provided document content is too short or empty to summarize."}
        key_sentences = extract_key_points(sentences)
        summary_text = compose_summary(key_sentences, word_limit=100)
        if not summary_text:
            summary_text = "Error: Unable to generate summary from the provided content."
        return {"summary_text": summary_text}
    except Exception as e:
        return {"summary_text": f"Error during summarization: {str(e)}"}
