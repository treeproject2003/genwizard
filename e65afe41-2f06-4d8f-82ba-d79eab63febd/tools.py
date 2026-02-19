
from typing import Dict, Any, List
import re

def summarize_text_tool(input_text: str) -> Dict[str, Any]:
    """
    Description:
        Processes arbitrary input text and generates a concise summary limited to 100 words.
        Utilizes natural language processing techniques to:
            - Identify key points and main ideas
            - Remove redundant or less relevant information
            - Ensure the summary is coherent, concise, and preserves the original context

    Logic:
        1. Tokenize the input text into sentences.
        2. Score sentences for importance using term frequency (TF) of non-stopword tokens.
        3. Select highest scoring sentences, avoiding redundancy, until the 100-word limit is reached.
        4. Concatenate selected sentences to form the summary.
        5. Ensure the summary does not exceed 100 words, trimming if necessary.

    Inputs:
        input_text (str): The text to be summarized.

    Outputs:
        summary (str): The concise summary of input_text, not exceeding 100 words.

    When to use:
        Whenever there is a request to summarize arbitrary text input, regardless of source or content domain.
    """
    import string

    # Define a simple list of English stopwords
    STOPWORDS = {
        'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him',
        'his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves',
        'what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being',
        'have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as',
        'until','while','of','at','by','for','with','about','against','between','into','through','during','before',
        'after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then',
        'once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some',
        'such','no','nor','not','only','own','same','so','than','too','very','can','will','just','don','should','now'
    }

    # 1. Sentence tokenization (split on punctuation followed by space or end of string)
    sentences = re.split(r'(?<=[.!?])\s+', input_text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

    # 2. Build frequency table (TF) for non-stopword tokens
    def tokenize(text):
        return [
            word.lower().strip(string.punctuation)
            for word in text.split()
            if word.lower().strip(string.punctuation) not in STOPWORDS and word.isalpha()
        ]

    word_freq = {}
    for sent in sentences:
        for word in tokenize(sent):
            word_freq[word] = word_freq.get(word, 0) + 1

    # 3. Score each sentence by sum of word TFs (importance)
    sent_scores = []
    for idx, sent in enumerate(sentences):
        score = sum(word_freq.get(word, 0) for word in tokenize(sent))
        sent_scores.append((idx, sent, score))

    # 4. Sort sentences by score (descending), preserve original order for equal scores
    sent_scores.sort(key=lambda tup: (-tup[2], tup[0]))

    # 5. Greedily select sentences, avoiding redundancy, up to 100 words
    summary_sents = []
    summary_word_count = 0
    used_texts = set()

    for idx, sent, score in sent_scores:
        # Check for redundancy (skip very similar sentences already in summary)
        sent_norm = ' '.join(tokenize(sent))
        if any(sent_norm in used or used in sent_norm for used in used_texts):
            continue
        sent_word_count = len(sent.split())
        if summary_word_count + sent_word_count <= 100:
            summary_sents.append((idx, sent))
            summary_word_count += sent_word_count
            used_texts.add(sent_norm)
        elif summary_word_count < 100:
            # Add as much as possible from this sentence
            words_remaining = 100 - summary_word_count
            partial_sent = ' '.join(sent.split()[:words_remaining])
            if partial_sent:
                summary_sents.append((idx, partial_sent))
                summary_word_count += len(partial_sent.split())
            break
        else:
            break

    # 6. Restore original order for coherence
    summary_sents.sort(key=lambda tup: tup[0])
    summary_text = ' '.join([s for _, s in summary_sents]).strip()

    # 7. Final trim to 100 words in case of overrun
    summary_words = summary_text.split()
    if len(summary_words) > 100:
        summary_text = ' '.join(summary_words[:100])

    return {
        "summary": summary_text
    }
