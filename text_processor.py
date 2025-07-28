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
                "heading_text": heading.get('text', ''),
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