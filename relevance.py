# ========== relevance.py ==========
import re
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import Counter
from text_processor import TextProcessor # type: ignore

class StandaloneRelevanceClassifier:
    """
    Pure semantic search - no hardcoded patterns or keywords
    """
    
    def __init__(self):
        self.text_processor = TextProcessor()
        
    def search_and_rank(self, query: str, documents: List[Dict], persona: str = "", 
                       job_description: str = "", max_results: int = 10) -> Tuple[List[Dict], List[Dict]]:
        """
        Pure semantic search based on query similarity
        """
        print(f"Semantic search for: '{query}' in {len(documents)} documents")
        
        # Step 1: Calculate semantic similarity for each document
        scored_docs = self._calculate_semantic_scores(query, documents)
        print(f"- Calculated semantic scores for {len(scored_docs)} documents")
        
        # Step 2: Rank by similarity
        scored_docs.sort(key=lambda x: x['semantic_score'], reverse=True)
        
        # Step 3: Select top results with diversity
        diverse_docs = self._select_diverse_results(scored_docs, max_results)
        print(f"- Selected {len(diverse_docs)} diverse results")
        
        # Step 4: Create output
        extracted, refined = self._create_output(diverse_docs)
        
        return extracted, refined
    
    def _calculate_semantic_scores(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        Calculate pure semantic similarity between query and documents
        """
        query_words = self._extract_words(query)
        query_vector = self._create_text_vector(query)
        
        scored_docs = []
        
        for doc in documents:
            # Get document content
            content = doc.get('chunk', '')
            heading = doc.get('heading_text', '')
            
            if not content:
                continue
            
            # Calculate similarity scores
            content_score = self._calculate_text_similarity(query, content)
            heading_score = self._calculate_text_similarity(query, heading) if heading else 0.0
            
            # Vector-based similarity
            doc_vector = self._create_text_vector(content)
            vector_score = self._cosine_similarity(query_vector, doc_vector)
            
            # Word overlap similarity
            doc_words = self._extract_words(content)
            heading_words = self._extract_words(heading) if heading else set()
            
            content_overlap = self._jaccard_similarity(query_words, doc_words)
            heading_overlap = self._jaccard_similarity(query_words, heading_words) if heading_words else 0.0
            
            # Combined semantic score
            semantic_score = (
                content_score * 0.3 +      # Content text similarity
                heading_score * 0.4 +      # Heading similarity (higher weight)
                vector_score * 0.2 +       # Vector similarity
                content_overlap * 0.05 +   # Content word overlap
                heading_overlap * 0.05     # Heading word overlap
            )
            
            doc['semantic_score'] = min(semantic_score, 1.0)
            doc['content_score'] = content_score
            doc['heading_score'] = heading_score
            doc['vector_score'] = vector_score
            
            scored_docs.append(doc)
        
        return scored_docs
    
    def _extract_words(self, text: str) -> Set[str]:
        """Extract normalized words from text"""
        if not text:
            return set()
        
        # Extract words, normalize case
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        
        # Very minimal filtering - only remove extremely common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        return {word for word in words if word not in common_words and len(word) >= 2}
    
    def _create_text_vector(self, text: str) -> np.ndarray:
        """Create simple frequency-based vector representation"""
        words = self._extract_words(text)
        
        if not words:
            return np.zeros(100)  # Empty vector
        
        # Create word frequency vector
        word_counts = Counter(re.findall(r'\b[a-zA-Z]{2,}\b', text.lower()))
        
        # Use hash-based vector (simple but effective)
        vector = np.zeros(100)
        for word, count in word_counts.items():
            if word in words:  # Only meaningful words
                hash_val = hash(word) % 100
                vector[hash_val] += count
        
        # Normalize vector
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except:
            return 0.0
    
    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between word sets"""
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_text_similarity(self, query: str, text: str) -> float:
        """Calculate text-to-text similarity using multiple methods"""
        if not text or not query:
            return 0.0
        
        query_lower = query.lower()
        text_lower = text.lower()
        
        # Method 1: Substring matching with context
        substring_score = 0.0
        query_words = query_lower.split()
        
        for word in query_words:
            if len(word) >= 3:  # Skip very short words
                if word in text_lower:
                    # Boost score based on word length and frequency
                    word_count = text_lower.count(word)
                    word_boost = min(word_count * len(word) / 100, 0.3)
                    substring_score += word_boost
        
        # Method 2: Phrase proximity
        phrase_score = 0.0
        if len(query_words) >= 2:
            # Check for word pairs appearing near each other
            text_words = text_lower.split()
            for i in range(len(query_words) - 1):
                word1, word2 = query_words[i], query_words[i + 1]
                
                # Find positions of both words
                pos1 = [j for j, w in enumerate(text_words) if word1 in w]
                pos2 = [j for j, w in enumerate(text_words) if word2 in w]
                
                # Check if they appear close to each other
                for p1 in pos1:
                    for p2 in pos2:
                        if abs(p1 - p2) <= 5:  # Within 5 words
                            phrase_score += 0.1
                            break
        
        # Method 3: Semantic word overlap
        query_words_set = self._extract_words(query)
        text_words_set = self._extract_words(text)
        overlap_score = self._jaccard_similarity(query_words_set, text_words_set)
        
        # Method 4: Length-normalized similarity
        length_factor = min(len(text) / max(len(query), 1), 5.0) / 5.0  # Prefer reasonable length
        
        # Combine all methods
        total_score = (
            substring_score * 0.4 +
            phrase_score * 0.3 +
            overlap_score * 0.2 +
            length_factor * 0.1
        )
        
        return min(total_score, 1.0)
    
    def _select_diverse_results(self, scored_docs: List[Dict], max_results: int) -> List[Dict]:
        """Select diverse results while maintaining relevance"""
        if len(scored_docs) <= max_results:
            return scored_docs
        
        selected = []
        seen_pdfs = set()
        
        # First pass: Select best from each PDF
        for doc in scored_docs:
            if len(selected) >= max_results:
                break
            
            pdf_name = doc.get('filename', 'unknown')
            
            # Take best from each PDF first
            if pdf_name not in seen_pdfs:
                selected.append(doc)
                seen_pdfs.add(pdf_name)
        
        # Second pass: Fill remaining slots with highest scores
        for doc in scored_docs:
            if len(selected) >= max_results:
                break
                
            if doc not in selected:
                selected.append(doc)
        
        return selected[:max_results]
    
    def _create_output(self, documents: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Create output format"""
        extracted = []
        refined = []
        
        for rank, doc in enumerate(documents):
            text = doc.get('chunk', '')
            pdf_name = doc.get('filename', 'Unknown')
            page = doc.get('page', -1)
            
            # Extract title using heading or content
            title = self._extract_best_title(doc)
            
            # Create extracted entry
            extracted_entry = {
                "document": pdf_name,
                "section_title": title,
                "heading_text": doc.get('heading_text', ''),
                "display_heading": self._format_heading_for_display(doc),
                "importance_rank": rank + 1,
                "page_number": page,
                "relevance_score": round(doc.get('semantic_score', 0), 3)
            }
            
            # Add semantic breakdown for debugging
            if 'content_score' in doc:
                extracted_entry["score_breakdown"] = {
                    "content_similarity": round(doc.get('content_score', 0), 3),
                    "heading_similarity": round(doc.get('heading_score', 0), 3),
                    "vector_similarity": round(doc.get('vector_score', 0), 3)
                }
            
            # Add heading info if available
            if doc.get('has_heading'):
                extracted_entry.update({
                    "heading_confidence": round(doc.get('heading_confidence', 0), 2),
                    "is_heading_based": True
                })
            else:
                extracted_entry["is_heading_based"] = False
            
            extracted.append(extracted_entry)
            
            # Create refined entry
            refined_text = self._format_text(text)
            refined.append({
                "document": pdf_name,
                "refined_text": refined_text,
                "page_number": page,
                "paragraph_count": len(refined_text.split('\n\n')) if refined_text else 1
            })
        
        return extracted, refined
    
    def _extract_best_title(self, doc: Dict) -> str:
        """Extract the best title from document"""
        # Priority 1: Use heading if available and confident
        if (doc.get('has_heading') and 
            doc.get('heading_text') and 
            doc.get('heading_confidence', 0) >= 0.5):
            
            title = doc['heading_text'].strip()
            return self._format_heading_for_display(title)
        
        # Priority 2: Extract from content
        content = doc.get('chunk', '')
        if content:
            return self._extract_title_from_content(content)
        
        return "Content Section"
    
    def _extract_title_from_content(self, content: str) -> str:
        """Extract title from content using heuristics"""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        if not lines:
            return "Content Section"
        
        # Look for title-like first line
        first_line = lines[0]
        
        # Good title characteristics
        words = first_line.split()
        
        if (3 <= len(words) <= 15 and 
            len(first_line) <= 100 and
            not first_line.endswith('.') and
            not first_line.startswith('•')):
            return self._clean_title(first_line)
        
        # Look for patterns like "Topic: Description"
        for line in lines[:3]:
            if ':' in line and len(line) < 80:
                parts = line.split(':', 1)
                potential_title = parts[0].strip()
                if 2 <= len(potential_title.split()) <= 10:
                    return self._clean_title(potential_title)
        
        # Fallback: use first few words
        first_words = ' '.join(content.split()[:8])
        if len(first_words) > 60:
            first_words = first_words[:57] + "..."
        
        return self._clean_title(first_words) if first_words else "Content Section"
    
    def _clean_title(self, title: str) -> str:
        """Clean up title"""
        if not title:
            return "Content Section"
        
        # Remove common prefixes/suffixes
        title = re.sub(r'^[•\-\*]\s*', '', title)  # Remove bullets
        title = re.sub(r'[,;:\-]+$', '', title)    # Remove trailing punctuation
        
        # Ensure reasonable length
        if len(title) > 80:
            words = title.split()
            title = ' '.join(words[:10]) + "..."
        
        return title.strip() or "Content Section"
    
    def _format_text(self, text: str) -> str:
        """Format text for output"""
        if not text:
            return ""
        
        # Use text processor if available
        try:
            return self.text_processor.format_json_safe_text(text)
        except:
            # Fallback: basic cleaning
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            return text.strip()
    
    def _format_heading_for_display(self, doc_or_text) -> str:
        """Format heading specifically for JSON display"""
        if isinstance(doc_or_text, dict):
            heading_text = doc_or_text.get('heading_text', '')
        else:
            heading_text = str(doc_or_text)
        
        if not heading_text:
            return ""
        
        import unicodedata
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', heading_text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-.,():]', '', text)
        
        # Remove control characters
        text = re.sub(r'[\u0000-\u001F\u007F-\u009F]', '', text)
        
        # Clean up spacing
        text = ' '.join(text.split())
        
        # Ensure reasonable length
        if len(text) > 80:
            text = text[:77] + "..."
        
        return text.strip()
    
    def get_debug_info(self, documents: List[Dict]) -> Dict:
        """Get debug information for scored documents"""
        if not documents:
            return {}
        
        scores = [doc.get('semantic_score', 0) for doc in documents]
        has_headings = sum(1 for doc in documents if doc.get('has_heading'))
        
        return {
            'total_documents': len(documents),
            'documents_with_headings': has_headings,
            'heading_percentage': round((has_headings / len(documents)) * 100, 1),
            'average_semantic_score': round(np.mean(scores), 3),
            'max_semantic_score': round(max(scores), 3),
            'min_semantic_score': round(min(scores), 3),
            'score_std': round(np.std(scores), 3)
        }


# ===== INTEGRATION FUNCTION =====

def simple_persona_job_search(query: str, documents: List[Dict], persona: str, 
                             job_description: str, max_results: int = 10) -> Tuple[List[Dict], List[Dict]]:
    """
    Pure semantic search integration - no hardcoded patterns
    """
    classifier = StandaloneRelevanceClassifier()
    
    # Use query + context for better semantic understanding
    enhanced_query = f"{query} {persona} {job_description}".strip()
    
    extracted, refined = classifier.search_and_rank(
        enhanced_query, documents, persona, job_description, max_results
    )
    
    # Add debug info
    if extracted:
        debug_info = classifier.get_debug_info(documents)
        extracted[0]['debug_info'] = debug_info
    
    return extracted, refined

# ========== text_processor.py ==========
import re
import unicodedata
from typing import List, Dict, Optional

class TextProcessor:
    """Enhanced text processing with better character handling and structure preservation"""
    
    def __init__(self):
        # Characters to preserve (important punctuation and symbols)
        self.preserve_chars = {
            '"', '"', '"', "'", "'", '—', '–', '-', 
            '•', '·', '§', '©', '®', '™', '°', '%',
            '(', ')', '[', ']', '{', '}', ':', ';', ',', '.'
        }
        
        # Common OCR/PDF artifacts to clean
        self.artifacts = [
            r'Page\s*\d+\s*(?:of\s*\d+)?',  # Page numbers
            r'^\d+\s*$',                     # Standalone numbers
            r'\f',                           # Form feeds
            r'\r\n?',                        # Carriage returns
            r'[\x00-\x08\x0B\x0C\x0E-\x1F]' # Control characters except \t and \n
        ]
    
    def clean_text_enhanced(self, text: str) -> str:
        """Enhanced text cleaning that preserves structure and important characters"""
        if not text:
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove PDF artifacts
        for artifact_pattern in self.artifacts:
            text = re.sub(artifact_pattern, '', text, flags=re.MULTILINE)
        
        # Fix common OCR issues
        text = self._fix_ocr_issues(text)
        
        # Preserve paragraph breaks but clean up excessive whitespace
        text = self._normalize_whitespace(text)
        
        # Remove problematic characters but preserve important ones
        text = self._selective_character_cleaning(text)
        
        return text.strip()
    
    def _fix_ocr_issues(self, text: str) -> str:
        """Fix common OCR recognition errors"""
        fixes = [
            (r'\bfi\b', 'fi'),  # Common ligature issue
            (r'\bfl\b', 'fl'),  # Common ligature issue
            (r'(\w)- (\w)', r'\1\2'),  # Remove hyphenation at line breaks
            (r'([a-z])([A-Z])', r'\1 \2'),  # Add space between joined words
            (r'(\d+)([A-Z][a-z])', r'\1 \2'),  # Separate numbers from words
        ]
        
        for pattern, replacement in fixes:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph structure"""
        # Convert multiple spaces to single space (but preserve newlines)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Preserve double newlines (paragraph breaks) but clean up multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Clean up space around newlines
        text = re.sub(r' *\n *', '\n', text)
        
        return text
    
    def _selective_character_cleaning(self, text: str) -> str:
        """Remove problematic characters while preserving important ones"""
        cleaned_chars = []
        
        for char in text:
            # Keep ASCII printable characters
            if 32 <= ord(char) <= 126:
                cleaned_chars.append(char)
            # Keep important Unicode characters
            elif char in self.preserve_chars:
                cleaned_chars.append(char)
            # Keep newlines and tabs
            elif char in '\n\t':
                cleaned_chars.append(char)
            # Keep common accented characters
            elif self._is_acceptable_unicode(char):
                cleaned_chars.append(char)
            # Replace others with space
            else:
                cleaned_chars.append(' ')
        
        return ''.join(cleaned_chars)
    
    def _is_acceptable_unicode(self, char: str) -> bool:
        """Check if a Unicode character should be preserved"""
        # Allow common accented letters and symbols
        category = unicodedata.category(char)
        return category.startswith(('L', 'N', 'P', 'S')) and ord(char) < 0x1000
    
    def extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from text, preserving structure"""
        if not text:
            return []
        
        # Split on double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean each paragraph and filter out empty ones
        cleaned_paragraphs = []
        for para in paragraphs:
            cleaned = ' '.join(para.split())  # Normalize internal whitespace
            if len(cleaned.strip()) > 20:  # Only keep substantial paragraphs
                cleaned_paragraphs.append(cleaned)
        
        return cleaned_paragraphs
    
    def create_structured_chunk(self, heading: Optional[Dict], content: str, 
                              page_num: int, chunk_id: str) -> Dict:
        """Create a well-structured chunk with heading information"""
        # Clean the content
        clean_content = self.clean_text_enhanced(content)
        paragraphs = self.extract_paragraphs(clean_content)
        
        # Build the final chunk text
        chunk_parts = []
        
        # Add heading if available
        if heading and heading.get('text'):
            heading_text = heading['text'].strip()
            chunk_parts.append(heading_text)
            chunk_parts.append("")  # Empty line after heading
        
        # Add content paragraphs
        chunk_parts.extend(paragraphs)
        
        # Join with proper spacing
        final_text = '\n'.join(chunk_parts).strip()
        
        # Create chunk metadata
        chunk_data = {
            "filename": "",  # Will be set by caller
            "page": page_num,
            "chunk": final_text,
            "chunk_id": chunk_id,
            "chunk_length": len(final_text),
            "paragraph_count": len(paragraphs),
            "has_heading": heading is not None,
        }
        
        # Add heading metadata if available
        if heading:
            chunk_data.update({
                "heading_text": heading.get('text', '').strip(),
                "heading_level": heading.get('level', 'H2'),
                "heading_confidence": heading.get('confidence', 0.0),
                "heading_page": heading.get('page', page_num)
            })
        
        return chunk_data
    
    def format_json_safe_text(self, text: str) -> str:
        """Ensure text is safe for JSON serialization"""
        if not text:
            return ""
        
        # Ensure proper Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Replace problematic characters that might break JSON
        replacements = {
            '\x00': '',  # Null character
            '\x08': '',  # Backspace
            '\x0C': '',  # Form feed
            '\x0E': '',  # Shift out
            '\x0F': '',  # Shift in
        }
        
        for bad_char, replacement in replacements.items():
            text = text.replace(bad_char, replacement)
        
        return text
    
    def extract_title_from_chunk(self, chunk_text: str, heading_info: Optional[Dict] = None) -> str:
        """Extract the best title from a chunk, preferring heading information"""
        
        # First priority: use heading if available
        if heading_info and heading_info.get('text'):
            return heading_info['text'].strip()
        
        # Second priority: look for title patterns in the text
        lines = [line.strip() for line in chunk_text.split('\n') if line.strip()]
        
        if not lines:
            return "Content Section"
        
        # Look for title-like first line
        first_line = lines[0]
        if self._looks_like_title(first_line):
            return first_line
        
        # Look for lines with colons (Topic: Description format)
        for line in lines[:3]:
            if ':' in line and len(line) < 100:
                potential_title = line.split(':', 1)[0].strip()
                if self._is_good_title(potential_title):
                    return potential_title
        
        # Fallback: use first sentence or phrase
        sentences = re.split(r'[.!?]\s+', chunk_text)
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) <= 80 and self._is_good_title(first_sentence):
                return first_sentence
        
        # Last resort: first few words
        words = chunk_text.split()[:8]
        if words:
            title = ' '.join(words)
            return title[:60] + "..." if len(title) > 60 else title
        
        return "Content Section"
    
    def _looks_like_title(self, line: str) -> bool:
        """Check if a line looks like a title"""
        if len(line) < 3 or len(line) > 100:
            return False
        
        words = line.split()
        if len(words) > 15 or len(words) == 0:
            return False
        
        # Avoid lines that end with periods (usually sentences)
        if line.endswith('.') and not line.endswith('...'):
            return False
        
        # Check for title-like capitalization
        capitalized = sum(1 for word in words if word and word[0].isupper())
        return capitalized >= max(1, len(words) * 0.4)
    
    def _is_good_title(self, text: str) -> bool:
        """Check if text makes a good title"""
        if len(text) < 3 or len(text) > 120:
            return False
        
        words = text.split()
        if len(words) > 20 or len(words) == 0:
            return False
        
        # Avoid instruction-like text
        instruction_starters = [
            'add', 'mix', 'cook', 'heat', 'place', 'put', 'take', 'remove',
            'then', 'next', 'after', 'first', 'finally', 'now', 'step'
        ]
        
        first_word = words[0].lower()
        if first_word in instruction_starters:
            return False
        
        # Avoid very sentence-like patterns
        if text.count(',') > 3 or text.count('and') > 2:
            return False
        
        return True
    
    def _clean_title(self, title: str) -> str:
        """Clean up title"""
        if not title:
            return "Content Section"
        
        # Unicode normalization
        title = unicodedata.normalize('NFKC', title)
        
        # Remove special characters but preserve important ones
        title = re.sub(r'[^\w\s\-.,():]', '', title)
        
        # Remove common prefixes/suffixes
        title = re.sub(r'^[•\-\*]\s*', '', title)  # Remove bullets
        title = re.sub(r'[,;:\-]+$', '', title)    # Remove trailing punctuation
        
        # Ensure reasonable length
        if len(title) > 80:
            words = title.split()
            title = ' '.join(words[:10]) + "..."
        
        return title.strip() or "Content Section"

