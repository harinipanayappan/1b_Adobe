import os
import json
import pdfplumber
from typing import List, Dict, Optional, Tuple
from config import PDF_DIR, JSON_CHUNKS, CHUNK_SIZE, CHUNK_OVERLAP, MIN_HEADING_CONFIDENCE, MAX_CHUNK_WITHOUT_HEADING
from langchain.text_splitter import RecursiveCharacterTextSplitter
from heading_classifier import HeadingClassifier
from text_processor import TextProcessor

class HeadingAwareChunker:
    def __init__(self):
        self.heading_classifier = HeadingClassifier()
        self.text_processor = TextProcessor()
        
    def load_and_chunk_pdfs(self, pdf_folder: Optional[str] = None) -> List[Dict]:
        """Main method to load PDFs and create heading-aware chunks"""
        docs = []
        processed_files = []

        pdf_dir = pdf_folder or PDF_DIR

        if not os.path.exists(pdf_dir):
            raise ValueError(f"PDF directory {pdf_dir} does not exist")

        all_files = os.listdir(pdf_dir)
        pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]

        if not pdf_files:
            raise ValueError(f"No PDF files found in {pdf_dir}")

        for filename in pdf_files:
            filepath = os.path.join(pdf_dir, filename)
            print(f"Processing {filename}...")

            try:
                page_contents, all_headings = self._extract_pdf_content(filepath)

                if not page_contents:
                    print(f"No content extracted from {filename}")
                    continue

                file_chunks = self._create_heading_aware_chunks(
                    page_contents, all_headings, filename
                )

                docs.extend(file_chunks)
                processed_files.append(filename)
                print(f"Created {len(file_chunks)} chunks from {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

        os.makedirs(os.path.dirname(JSON_CHUNKS), exist_ok=True)
        with open(JSON_CHUNKS, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)

        print(f"Total chunks created: {len(docs)} from {len(processed_files)} files")
        return docs

    def _extract_pdf_content(self, pdf_path: str) -> Tuple[Dict[int, str], List[Dict]]:
        page_contents = {}
        all_headings = self.heading_classifier.classify_pdf_headings(pdf_path)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    try:
                        raw_text = page.extract_text()
                        if raw_text and len(raw_text.strip()) > 20:
                            cleaned_text = self.text_processor.clean_text_enhanced(raw_text)
                            if cleaned_text:
                                page_contents[page_num] = cleaned_text
                    except Exception as e:
                        print(f"Error extracting page {page_num}: {e}")
                        continue
        except Exception as e:
            print(f"Error opening PDF {pdf_path}: {e}")
            return {}, []

        return page_contents, all_headings

    def _create_heading_aware_chunks(self, page_contents: Dict[int, str],
                                     headings: List[Dict], filename: str) -> List[Dict]:
        chunks = []

        headings_by_page = {}
        for heading in headings:
            page = heading.get('page', 1)
            headings_by_page.setdefault(page, []).append(heading)

        for page in headings_by_page:
            headings_by_page[page].sort(key=lambda h: h.get('confidence', 0), reverse=True)

        chunk_counter = 0
        for page_num, page_text in page_contents.items():
            page_headings = headings_by_page.get(page_num, [])
            page_chunks = self._chunk_page_with_headings(
                page_text, page_headings, page_num, filename, chunk_counter
            )
            chunks.extend(page_chunks)
            chunk_counter += len(page_chunks)

        return chunks

    def _chunk_page_with_headings(self, page_text: str, headings: List[Dict],
                                  page_num: int, filename: str, start_counter: int) -> List[Dict]:
        if not headings:
            return self._create_traditional_chunks(page_text, page_num, filename, start_counter)

        good_headings = [h for h in headings if h.get('confidence', 0) >= MIN_HEADING_CONFIDENCE]

        if not good_headings:
            return self._create_traditional_chunks(page_text, page_num, filename, start_counter)

        text_sections = self._split_text_by_headings(page_text, good_headings)

        chunk_counter = start_counter
        chunks = []

        for section in text_sections:
            heading_info = section.get('heading')
            content = section.get('content', '')

            if len(content.strip()) < 50:
                continue

            section_chunks = self._create_chunks_from_section(
                content, heading_info, page_num, filename, chunk_counter
            )

            chunks.extend(section_chunks)
            chunk_counter += len(section_chunks)

        return chunks

    def _split_text_by_headings(self, text: str, headings: List[Dict]) -> List[Dict]:
        sections = []
        text_lines = text.split('\n')

        heading_positions = []
        for heading in headings:
            heading_text = heading.get('text', '').strip()
            if not heading_text:
                continue

            for i, line in enumerate(text_lines):
                if self._is_heading_match(line, heading_text):
                    heading_positions.append({
                        'line_num': i,
                        'heading': heading,
                        'text': heading_text
                    })
                    break

        heading_positions.sort(key=lambda x: x['line_num'])

        if not heading_positions:
            return [{'heading': None, 'content': text}]

        for i, pos in enumerate(heading_positions):
            start_line = pos['line_num']
            end_line = heading_positions[i + 1]['line_num'] if i + 1 < len(heading_positions) else len(text_lines)

            section_lines = text_lines[start_line:end_line]
            section_content = '\n'.join(section_lines).strip()

            if section_content:
                sections.append({
                    'heading': pos['heading'],
                    'content': section_content
                })

        if heading_positions and heading_positions[0]['line_num'] > 0:
            pre_heading_content = '\n'.join(text_lines[:heading_positions[0]['line_num']]).strip()
            if len(pre_heading_content) > 100:
                sections.insert(0, {'heading': None, 'content': pre_heading_content})

        return sections if sections else [{'heading': None, 'content': text}]

    def _is_heading_match(self, line: str, heading_text: str) -> bool:
        line_clean = line.strip().lower()
        heading_clean = heading_text.strip().lower()

        if line_clean == heading_clean:
            return True
        if line_clean.startswith(heading_clean):
            return True
        if heading_clean.startswith(tuple('0123456789')):
            heading_words = ' '.join(heading_clean.split()[1:])
            if heading_words and heading_words in line_clean:
                return True

        line_words = set(line_clean.split())
        heading_words = set(heading_clean.split())

        if len(heading_words) > 0:
            overlap = len(line_words & heading_words) / len(heading_words)
            return overlap > 0.7

        return False

    def _create_chunks_from_section(self, content: str, heading_info: Optional[Dict],
                                    page_num: int, filename: str, start_counter: int) -> List[Dict]:
        chunks = []

        if len(content) <= CHUNK_SIZE:
            chunk_id = f"{filename}_p{page_num}_c{start_counter}"
            chunk_data = self.text_processor.create_structured_chunk(
                heading_info, content, page_num, chunk_id
            )
            chunk_data['filename'] = filename
            chunks.append(chunk_data)
            return chunks

        paragraphs = self.text_processor.extract_paragraphs(content)

        if not paragraphs:
            return chunks

        current_chunk_parts = []
        current_size = 0
        chunk_counter = start_counter

        if heading_info:
            heading_text = heading_info.get('text', '')
            current_chunk_parts.append(heading_text)
            current_size = len(heading_text)

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size > CHUNK_SIZE and current_chunk_parts:
                chunk_content = '\n\n'.join(current_chunk_parts)
                chunk_id = f"{filename}_p{page_num}_c{chunk_counter}"

                chunk_data = self.text_processor.create_structured_chunk(
                    heading_info if chunk_counter == start_counter else None,
                    chunk_content, page_num, chunk_id
                )
                chunk_data['filename'] = filename
                chunks.append(chunk_data)

                current_chunk_parts = [para]
                current_size = para_size
                chunk_counter += 1
            else:
                current_chunk_parts.append(para)
                current_size += para_size

        if current_chunk_parts:
            chunk_content = '\n\n'.join(current_chunk_parts)
            chunk_id = f"{filename}_p{page_num}_c{chunk_counter}"

            chunk_data = self.text_processor.create_structured_chunk(
                None,
                chunk_content, page_num, chunk_id
            )
            chunk_data['filename'] = filename
            chunks.append(chunk_data)

        return chunks

    def _create_traditional_chunks(self, text: str, page_num: int,
                                   filename: str, start_counter: int) -> List[Dict]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_WITHOUT_HEADING,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]
        )

        raw_chunks = text_splitter.split_text(text)
        chunks = []

        for i, chunk_text in enumerate(raw_chunks):
            if len(chunk_text.strip()) < 50:
                continue

            chunk_id = f"{filename}_p{page_num}_c{start_counter + i}"
            chunk_data = self.text_processor.create_structured_chunk(
                None, chunk_text, page_num, chunk_id
            )
            chunk_data['filename'] = filename
            chunks.append(chunk_data)

        return chunks


# 🔁 ENTRY POINT FUNCTION for external import
def load_and_chunk_pdfs(pdf_folder: Optional[str] = None):
    chunker = HeadingAwareChunker()
    return chunker.load_and_chunk_pdfs(pdf_folder=pdf_folder)