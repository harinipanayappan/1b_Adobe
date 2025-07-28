import re
from typing import List, Dict

def extract_summary_enhanced(refined_chunks: List[Dict], max_length: int = 300) -> str:
    """Enhanced summary generation that considers headings and structure"""
    try:
        from summarizer import Summarizer
        model_summarizer = Summarizer()
        
        # Combine text from refined chunks with heading structure
        combined_parts = []
        
        for chunk in refined_chunks:
            text = chunk.get('refined_text', '')
            if text:
                # Clean and format text
                clean_text = _clean_text_for_summary(text)
                if clean_text:
                    combined_parts.append(clean_text)
        
        if not combined_parts:
            return "No relevant content found for summarization."
        
        combined_text = ' '.join(combined_parts)
        
        # If text is short enough, return as is
        if len(combined_text.split()) < 60:
            return combined_text
        
        # Generate summary
        summary = model_summarizer(combined_text, min_length=40, max_length=max_length)
        return _post_process_summary(summary)
        
    except ImportError:
        return _fallback_summary(refined_chunks, max_length)
    except Exception as e:
        print(f"Error in summarization: {e}")
        return _fallback_summary(refined_chunks, max_length)

def extract_summary(text):
    """Backward compatibility function"""
    if isinstance(text, str):
        # Convert string to chunk format for consistency
        fake_chunks = [{'refined_text': text}]
        return extract_summary_enhanced(fake_chunks)
    elif isinstance(text, list):
        return extract_summary_enhanced(text)
    else:
        return "Invalid input for summary generation."

def _clean_text_for_summary(text: str) -> str:
    """Clean text specifically for summary generation"""
    if not text:
        return ""
    
    # Remove excessive newlines but preserve paragraph structure
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Convert paragraph breaks to periods for better sentence processing
    text = re.sub(r'\n\n+', '. ', text)
    
    # Clean up spacing
    text = ' '.join(text.split())
    
    # Remove very short fragments
    sentences = text.split('.')
    meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    return '. '.join(meaningful_sentences)

def _post_process_summary(summary: str) -> str:
    """Post-process generated summary"""
    if not summary:
        return "Summary could not be generated."
    
    # Ensure proper sentence endings
    summary = summary.strip()
    if summary and not summary.endswith(('.', '!', '?')):
        summary += '.'
    
    # Clean up any remaining formatting issues
    summary = ' '.join(summary.split())
    
    return summary

def _fallback_summary(refined_chunks: List[Dict], max_length: int = 300) -> str:
    """Fallback summary when summarizer is not available"""
    if not refined_chunks:
        return "No content available for summarization."
    
    # Extract key sentences from each chunk
    key_sentences = []
    
    for chunk in refined_chunks:
        text = chunk.get('refined_text', '')
        if text:
            sentences = _extract_key_sentences(text, max_sentences=2)
            key_sentences.extend(sentences)
    
    if not key_sentences:
        # Very basic fallback - first words from first chunk
        first_text = refined_chunks[0].get('refined_text', '')
        words = first_text.split()
        if len(words) > 50:
            return " ".join(words[:50]) + "..."
        return first_text
    
    # Combine key sentences
    summary = ' '.join(key_sentences)
    
    # Truncate if too long
    if len(summary) > max_length:
        words = summary.split()
        truncated_words = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_length - 3:  # Leave room for "..."
                truncated_words.append(word)
                current_length += len(word) + 1
            else:
                break
        
        summary = ' '.join(truncated_words) + "..."
    
    return summary

def _extract_key_sentences(text: str, max_sentences: int = 2) -> List[str]:
    """Extract key sentences from text using simple heuristics"""
    if not text:
        return []
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if not sentences:
        return []
    
    # Score sentences based on simple criteria
    scored_sentences = []
    
    for sentence in sentences:
        score = 0
        words = sentence.split()
        
        # Prefer sentences of moderate length
        if 8 <= len(words) <= 25:
            score += 2
        elif len(words) < 8:
            score -= 1
        elif len(words) > 35:
            score -= 2
        
        # Prefer sentences at the beginning (more likely to be important)
        if sentences.index(sentence) == 0:
            score += 1
        
        # Prefer sentences with certain keywords
        important_words = ['important', 'key', 'main', 'primary', 'essential', 'significant']
        for word in important_words:
            if word in sentence.lower():
                score += 1
        
        # Avoid sentences that are likely lists or fragments
        if sentence.count(',') > 3:
            score -= 1
        
        scored_sentences.append((sentence, score))
    
    # Sort by score and take top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    return [sentence for sentence, score in scored_sentences[:max_sentences]]