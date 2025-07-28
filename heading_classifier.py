import pdfplumber
import fitz
import numpy as np
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter

@dataclass
class ExtractedElement:
    text: str
    font_size: float
    is_bold: bool
    is_italic: bool
    page: int
    bbox: Tuple[float, float, float, float]
    font_family: str = ""
    spacing_before: float = 0.0
    spacing_after: float = 0.0

class HeadingClassifier:
    def __init__(self):
        self.document_stats = {}
        self.spacing_stats = {}
        
    def classify_pdf_headings(self, pdf_path: str):
        """Detect complete, clean headings only"""
        try:
            text_elements = self._extract_clean_text(pdf_path)
            if not text_elements:
                return []
                
            # Calculate document statistics
            self.document_stats = self._calculate_document_stats(text_elements)
            
            # Add spacing analysis
            text_elements = self._calculate_spacing(text_elements)
            self.spacing_stats = self._calculate_spacing_stats(text_elements)
            
            # Find complete headings only
            headings = self._detect_complete_headings(text_elements)
            
            return [
                {
                    'text': h['text'].strip(), 
                    'page': h['page'], 
                    'confidence': h['confidence']
                } 
                for h in headings
            ]
        except Exception as e:
            print(f"Error in heading classification: {e}")
            return []
    
    def _detect_complete_headings(self, elements: List[ExtractedElement]) -> List[Dict]:
        """Detect only complete, well-formed headings"""
        headings = []
        
        for elem in elements:
            try:
                text = elem.text.strip()
                
                # Must pass strict completeness check
                if not self._is_complete_heading_text(text):
                    continue
                
                # Must have heading-like formatting
                if not self._has_heading_formatting(elem):
                    continue
                
                # Calculate confidence based on multiple factors
                confidence = self._calculate_heading_confidence(elem, text)
                
                # Only keep high-confidence, complete headings
                if confidence >= 0.7:  # High threshold for clean results
                    headings.append({
                        'text': text,
                        'page': elem.page,
                        'confidence': round(confidence, 2),
                        'element': elem
                    })
                    
            except Exception as e:
                print(f"Error processing heading candidate: {e}")
                continue
        
        # Remove duplicates and overlapping headings
        headings = self._remove_duplicate_headings(headings)
        
        # Sort by page and confidence
        headings.sort(key=lambda x: (x['page'], -x['confidence']))
        
        return headings
    
    def _is_complete_heading_text(self, text: str) -> bool:
        """Strict check for complete, well-formed heading text"""
        if not text or len(text) < 5 or len(text) > 120:
            return False
        
        words = text.split()
        if len(words) < 1 or len(words) > 20:
            return False
        
        # Must not end with incomplete words or fragments
        incomplete_endings = {
            'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'
        }
        
        last_word = words[-1].lower().rstrip('.,;:')
        if last_word in incomplete_endings:
            return False
        
        # Must not end with comma, colon, or semicolon (indicates continuation)
        if text.rstrip().endswith((':',)):
            return False
        
        # Must not start with bullet points followed by long text
        if text.startswith('•') and len(words) > 8:
            return False
        
        # Must not be obviously a sentence (multiple clauses)
        if text.count(',') > 2 or ' and ' in text.lower() and len(words) > 8:
            return False
        
        # Must not contain obvious sentence patterns
        sentence_patterns = [
            r'\b(this|these|that|those)\s+(is|are|was|were)\b',
            r'\b(here|there)\s+(is|are|was|were)\b',
            r'\bthe\s+\w+\s+(is|are|was|were)\b'
        ]
        
        text_lower = text.lower()
        for pattern in sentence_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Check for good heading characteristics
        return self._has_heading_characteristics(text)
    
    def _has_heading_characteristics(self, text: str) -> bool:
        """Check for positive heading characteristics"""
        words = text.split()
        
        # Good capitalization pattern (title case or sentence case)
        capitalized_words = sum(1 for word in words if word and word[0].isupper())
        good_capitalization = capitalized_words >= max(1, len(words) * 0.4)
        
        if not good_capitalization:
            return False
        
        # Prefer certain heading patterns
        heading_patterns = [
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Title Case
            r'^\d+\.\s*[A-Z]',  # Numbered headings
            r'^Chapter\s+\d+',  # Chapter headings
            r'^Part\s+[IVX\d]+',  # Part headings
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text):
                return True
        
        # Check for topic-like words (without being too restrictive)
        if len(words) >= 2 and len(words) <= 8:
            # Short, descriptive phrases are often good headings
            return True
        
        return False
    
    def _has_heading_formatting(self, elem: ExtractedElement) -> bool:
        """Check if element has heading-like formatting"""
        # Must have at least one distinguishing characteristic
        distinguishing_features = 0
        
        # Bold text
        if elem.is_bold:
            distinguishing_features += 1
        
        # Larger font size
        font_ratio = elem.font_size / self.document_stats.get('font_size_mean', 12.0)
        if font_ratio >= 1.1:
            distinguishing_features += 1
        
        # Significant spacing before
        large_spacing_threshold = self.spacing_stats.get('large_spacing_threshold', 20.0)
        if elem.spacing_before > large_spacing_threshold:
            distinguishing_features += 1
        
        # Position characteristics (near top of page or after spacing)
        if elem.bbox[1] < 150 or elem.spacing_before > 15:
            distinguishing_features += 1
        
        # Need at least 2 distinguishing features
        return distinguishing_features >= 2
    
    def _calculate_heading_confidence(self, elem: ExtractedElement, text: str) -> float:
        """Calculate confidence score for heading"""
        confidence = 0.0
        
        # Text quality (most important)
        if self._is_complete_heading_text(text):
            confidence += 0.4
        
        # Formatting characteristics
        if elem.is_bold:
            confidence += 0.2
        
        font_ratio = elem.font_size / self.document_stats.get('font_size_mean', 12.0)
        if font_ratio >= 1.3:
            confidence += 0.3
        elif font_ratio >= 1.1:
            confidence += 0.2
        elif font_ratio >= 1.05:
            confidence += 0.1
        
        # Spacing characteristics
        large_spacing_threshold = self.spacing_stats.get('large_spacing_threshold', 20.0)
        if elem.spacing_before > large_spacing_threshold * 1.5:
            confidence += 0.2
        elif elem.spacing_before > large_spacing_threshold:
            confidence += 0.1
        
        # Length characteristics (prefer moderate length)
        words = text.split()
        if 3 <= len(words) <= 8:
            confidence += 0.1
        elif 2 <= len(words) <= 12:
            confidence += 0.05
        
        # Position characteristics
        if elem.bbox[1] < 150:  # Near top of page
            confidence += 0.05
        
        # Pattern matching for common heading types
        if re.match(r'^\d+\.\s*[A-Z]', text):  # Numbered
            confidence += 0.1
        elif re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', text):  # Title Case
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _remove_duplicate_headings(self, headings: List[Dict]) -> List[Dict]:
        """Remove duplicate and overlapping headings"""
        if not headings:
            return headings
        
        # Remove exact duplicates
        seen_texts = set()
        unique_headings = []
        
        for heading in headings:
            text_key = heading['text'].lower().strip()
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_headings.append(heading)
        
        # Remove headings that are substrings of others (keep the longer, more complete one)
        filtered_headings = []
        
        for i, heading1 in enumerate(unique_headings):
            text1 = heading1['text'].strip()
            is_substring = False
            
            for j, heading2 in enumerate(unique_headings):
                if i == j:
                    continue
                    
                text2 = heading2['text'].strip()
                
                # If heading1 is a substring of heading2, and heading2 is more confident
                if (text1.lower() in text2.lower() and 
                    len(text1) < len(text2) and 
                    heading2['confidence'] >= heading1['confidence']):
                    is_substring = True
                    break
            
            if not is_substring:
                filtered_headings.append(heading1)
        
        return filtered_headings
    
    def _create_heading_pattern(self, heading_text):
        """Create regex pattern for heading matching"""
        try:
            escaped_text = re.escape(heading_text)
            pattern = escaped_text.replace(r'\ ', r'\s*').replace(r'\.', r'\.?')
            return re.compile(f'^{pattern}', re.IGNORECASE | re.MULTILINE)
        except:
            return re.compile(re.escape(heading_text), re.IGNORECASE)
    
    def _extract_clean_text(self, pdf_path: str) -> List[ExtractedElement]:
        """Extract text elements with error handling"""
        elements = []
        
        try:
            # Try pdfplumber first
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_elements = self._extract_page_elements(page, page_num + 1)
                        elements.extend(page_elements)
                    except Exception as e:
                        print(f"Error extracting page {page_num + 1}: {e}")
                        continue
        except Exception as e:
            print(f"pdfplumber failed, trying pymupdf: {e}")
            elements = self._extract_with_pymupdf(pdf_path)
        
        return self._clean_elements(elements)
    
    def _extract_page_elements(self, page, page_num: int) -> List[ExtractedElement]:
        """Extract elements from a page"""
        elements = []
        
        try:
            chars = page.chars
            if not chars:
                return elements
            
            lines = self._group_chars_into_lines(chars)
            
            for line in lines:
                if line and line.get('text', '').strip():
                    try:
                        element = ExtractedElement(
                            text=line['text'].strip(),
                            font_size=line.get('font_size', 12.0),
                            is_bold=line.get('is_bold', False),
                            is_italic=line.get('is_italic', False),
                            page=page_num,
                            bbox=line.get('bbox', (0, 0, 0, 0)),
                            font_family=line.get('font_family', '')
                        )
                        elements.append(element)
                    except Exception as e:
                        continue
        except Exception as e:
            print(f"Error in page element extraction: {e}")
        
        return elements
    
    def _group_chars_into_lines(self, chars: List[Dict]) -> List[Dict]:
        """Group characters into complete lines"""
        if not chars:
            return []
        
        try:
            chars = sorted(chars, key=lambda c: (round(c.get('top', 0), 1), c.get('x0', 0)))
        except:
            return []
        
        lines = []
        current_line = []
        
        for char in chars:
            try:
                if not current_line:
                    current_line = [char]
                else:
                    last_char = current_line[-1]
                    # More strict line grouping for complete text
                    if abs(char.get('top', 0) - last_char.get('top', 0)) <= 2:
                        current_line.append(char)
                    else:
                        if current_line:
                            processed_line = self._process_line(current_line)
                            if processed_line and processed_line.get('text'):
                                lines.append(processed_line)
                        current_line = [char]
            except:
                continue
        
        if current_line:
            processed_line = self._process_line(current_line)
            if processed_line and processed_line.get('text'):
                lines.append(processed_line)
        
        return [line for line in lines if line and line.get('text', '').strip()]
    
    def _process_line(self, chars: List[Dict]) -> Dict:
        """Process a line of characters into complete text element"""
        if not chars:
            return {}
        
        try:
            chars = sorted(chars, key=lambda c: c.get('x0', 0))
            
            # Build complete text from characters
            text_parts = []
            for char in chars:
                char_text = char.get('text', '')
                if char_text:
                    text_parts.append(char_text)
            
            complete_text = ''.join(text_parts).strip()
            
            if not complete_text or len(complete_text) < 3:
                return {}
            
            # Calculate bounding box
            x0 = min(c.get('x0', 0) for c in chars)
            y0 = min(c.get('top', 0) for c in chars)
            x1 = max(c.get('x1', 0) for c in chars)
            y1 = max(c.get('bottom', 0) for c in chars)
            
            # Get font characteristics
            font_sizes = [c.get('size', 12) for c in chars if c.get('size', 0) > 0]
            font_names = [c.get('fontname', '') for c in chars if c.get('fontname')]
            
            most_common_size = Counter(font_sizes).most_common(1)[0][0] if font_sizes else 12.0
            most_common_font = Counter(font_names).most_common(1)[0][0] if font_names else ''
            
            is_bold = any('bold' in str(name).lower() for name in font_names if name)
            is_italic = any('italic' in str(name).lower() for name in font_names if name)
            
            return {
                'text': complete_text,
                'font_size': float(most_common_size),
                'is_bold': is_bold,
                'is_italic': is_italic,
                'bbox': (x0, y0, x1, y1),
                'font_family': most_common_font
            }
        except:
            return {}
    
    def _extract_with_pymupdf(self, pdf_path: str) -> List[ExtractedElement]:
        """Fallback extraction using pymupdf"""
        elements = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    blocks = page.get_text("dict")["blocks"] # pyright: ignore[reportAttributeAccessIssue]
                    
                    for block in blocks:
                        if "lines" in block:
                            for line in block["lines"]:
                                if "spans" in line:
                                    # Combine spans in the same line for complete text
                                    line_text = ""
                                    line_font_size = 12.0
                                    line_is_bold = False
                                    line_bbox = None
                                    
                                    for span in line["spans"]:
                                        span_text = span.get("text", "").strip()
                                        if span_text:
                                            line_text += span_text + " "
                                            line_font_size = max(line_font_size, span.get("size", 12.0))
                                            if "bold" in span.get("font", "").lower():
                                                line_is_bold = True
                                            if line_bbox is None:
                                                line_bbox = span.get("bbox", (0, 0, 0, 0))
                                    
                                    line_text = line_text.strip()
                                    if line_text and len(line_text) >= 3:
                                        element = ExtractedElement(
                                            text=line_text,
                                            font_size=line_font_size,
                                            is_bold=line_is_bold,
                                            is_italic=False,
                                            page=page_num + 1,
                                            bbox=line_bbox or (0, 0, 0, 0),
                                            font_family=""
                                        )
                                        elements.append(element)
                except:
                    continue
            
            doc.close()
            
        except:
            pass
        
        return elements
    
    def _calculate_spacing(self, elements: List[ExtractedElement]) -> List[ExtractedElement]:
        """Calculate spacing between elements"""
        pages = {}
        for elem in elements:
            if elem.page not in pages:
                pages[elem.page] = []
            pages[elem.page].append(elem)
        
        enhanced_elements = []
        
        for page_num, page_elements in pages.items():
            try:
                page_elements.sort(key=lambda x: x.bbox[1])
                
                for i, elem in enumerate(page_elements):
                    spacing_before = 0.0
                    spacing_after = 0.0
                    
                    if i > 0:
                        prev_elem = page_elements[i-1]
                        spacing_before = max(0, elem.bbox[1] - prev_elem.bbox[3])
                    
                    if i < len(page_elements) - 1:
                        next_elem = page_elements[i+1]
                        spacing_after = max(0, next_elem.bbox[1] - elem.bbox[3])
                    
                    enhanced_elem = ExtractedElement(
                        text=elem.text,
                        font_size=elem.font_size,
                        is_bold=elem.is_bold,
                        is_italic=elem.is_italic,
                        page=elem.page,
                        bbox=elem.bbox,
                        font_family=elem.font_family,
                        spacing_before=spacing_before,
                        spacing_after=spacing_after
                    )
                    enhanced_elements.append(enhanced_elem)
            except:
                enhanced_elements.extend(page_elements)
        
        return enhanced_elements
    
    def _calculate_spacing_stats(self, elements: List[ExtractedElement]) -> Dict:
        """Calculate spacing statistics"""
        try:
            spacings_before = [elem.spacing_before for elem in elements if elem.spacing_before > 0]
            
            if not spacings_before:
                return {'mean_spacing': 12.0, 'large_spacing_threshold': 20.0}
            
            stats = {
                'mean_spacing': np.mean(spacings_before),
                'std_spacing': np.std(spacings_before),
                'p90_spacing': np.percentile(spacings_before, 90)
            }
            
            stats['large_spacing_threshold'] = stats['mean_spacing'] + 1.5 * stats['std_spacing']
            
            return stats
        except:
            return {'mean_spacing': 12.0, 'large_spacing_threshold': 20.0}
    
    def _calculate_document_stats(self, elements: List[ExtractedElement]) -> Dict:
        """Calculate document statistics"""
        try:
            if not elements:
                return {'font_size_mean': 12.0}
            
            font_sizes = [elem.font_size for elem in elements if elem.font_size > 0]
            
            return {
                'total_elements': len(elements),
                'font_size_mean': np.mean(font_sizes) if font_sizes else 12.0,
                'font_size_max': max(font_sizes) if font_sizes else 12.0,
                'font_size_std': np.std(font_sizes) if font_sizes else 0.0
            }
        except:
            return {'font_size_mean': 12.0}
    
    def _clean_elements(self, elements: List[ExtractedElement]) -> List[ExtractedElement]:
        """Clean extracted elements while preserving completeness"""
        cleaned = []
        
        for elem in elements:
            try:
                text = elem.text.strip()
                
                if len(text) < 3:
                    continue
                
                # Clean text while preserving structure
                # Remove excessive whitespace but keep single spaces
                text = re.sub(r'\s+', ' ', text)
                
                # Remove isolated punctuation
                text = re.sub(r'\s+[.,;:]\s+', ' ', text)
                
                # Clean up hyphenation artifacts
                text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
                
                if text and len(text) >= 3:
                    cleaned_elem = ExtractedElement(
                        text=text,
                        font_size=elem.font_size,
                        is_bold=elem.is_bold,
                        is_italic=elem.is_italic,
                        page=elem.page,
                        bbox=elem.bbox,
                        font_family=elem.font_family,
                        spacing_before=elem.spacing_before,
                        spacing_after=elem.spacing_after
                    )
                    cleaned.append(cleaned_elem)
            except:
                continue
        
        return cleaned