"""Document ingestion and indexing module for RAG system."""

import os
import json
import re
from typing import List, Dict, Tuple, Optional
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class DocumentIngestor:
    """Handles PDF parsing and text chunking with semantic awareness."""
    
    # SEC Item extraction patterns
    SEC_ITEM_PATTERNS = [
        r"Item\s+(\d+(?:\.\d+)?(?:[A-Z])?)\s*[\.:]?\s*(.{1,100}?)(?:\n|\r|$)",
        r"ITEM\s+(\d+(?:\.\d+)?(?:[A-Z])?)\s*[\.:]?\s*(.{1,100}?)(?:\n|\r|$)",
        r"Part\s+([IV]+)\s*[,\-]\s*Item\s+(\d+(?:\.\d+)?(?:[A-Z])?)\s*[\.:]?\s*(.{1,100}?)(?:\n|\r|$)",
    ]
    
    def __init__(self, max_chunk_size: int = 1200, table_context_size: int = 300):
        """
        Initialize the document ingestor.
        
        Args:
            max_chunk_size: Maximum number of characters per chunk (targets smaller, more granular chunks)
            table_context_size: Number of characters of context to include before/after tables
        """
        self.max_chunk_size = max_chunk_size
        self.table_context_size = table_context_size
        self.converter = DocumentConverter()
        self.last_seen_item = None  # Track last seen SEC item for inheritance
    
    def parse_pdf(self, pdf_path: str, doc_name: str) -> List[Dict]:
        """
        Parse a PDF file into semantically coherent chunks with metadata.
        Uses Docling's native structure to preserve tables and context.
        
        Args:
            pdf_path: Path to the PDF file
            doc_name: Name of the document (e.g., "Apple 10-K")
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            # Convert document using Docling
            result = self.converter.convert(pdf_path)
            doc = result.document
            
            # Use Docling's native structure for intelligent chunking
            chunks = self._create_semantic_chunks(doc, doc_name)
            
            # Assign structured chunk IDs
            self._assign_structured_chunk_ids(chunks, doc_name)
            
            return chunks
        except Exception as e:
            raise RuntimeError(f"Error parsing PDF {pdf_path}: {str(e)}") from e
    
    def _extract_sec_items(self, text: str, page_num: int) -> List[Dict]:
        """
        Extract SEC item numbers and titles from text.
        
        Args:
            text: Text content to search
            page_num: Page number
            
        Returns:
            List of dictionaries with item_number, item_title, page, position
        """
        items = []
        
        for pattern in self.SEC_ITEM_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                groups = [g for g in match.groups() if g is not None]
                if not groups:
                    continue
                
                item_num, item_title = None, ""
                
                if len(groups) == 1:
                    potential = groups[0].strip()
                    if re.match(r'^[\dA-Z]+(\.\d+)?[A-Z]?$', potential):
                        item_num = potential
                elif len(groups) == 2:
                    a, b = groups
                    if re.match(r'^[\dA-Z]+(\.\d+)?[A-Z]?$', a.strip()):
                        item_num = a.strip()
                        item_title = b.strip()[:100]
                    elif re.match(r'^[\dA-Z]+(\.\d+)?[A-Z]?$', b.strip()):
                        if re.match(r'^(Part\s+)?[IVX]+$', a.strip(), re.IGNORECASE):
                            item_num = f"{a.strip()}-{b.strip()}"
                        else:
                            item_num = b.strip()
                elif len(groups) == 3:
                    part, potential, title = groups
                    if re.match(r'^[\dA-Z]+(\.\d+)?[A-Z]?$', potential.strip()):
                        item_num = f"{part.strip()}-{potential.strip()}"
                        item_title = (title or "").strip()[:100]
                
                if item_num:
                    items.append({
                        'item_number': item_num,
                        'item_title': item_title,
                        'page': page_num,
                        'position': match.start()
                    })
        
        # Remove duplicates while preserving order
        seen = set()
        unique_items = []
        for item in sorted(items, key=lambda x: x['position']):
            key = (item['item_number'], item['page'])
            if key not in seen:
                seen.add(key)
                unique_items.append(item)
        
        return unique_items
    
    def _extract_financial_patterns(self, text: str) -> Dict[str, List[str]]:
        """
        Extract financial patterns like dollar amounts, percentages, years.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of pattern types and matches
        """
        patterns = {
            'dollar_amounts': re.findall(r'\$\s*\d+(?:\.\d+)?(?:\s*(?:million|billion|trillion))?', text.lower()),
            'percentages': re.findall(r'\d+(?:\.\d+)?%', text),
            'years': re.findall(r'\b(?:19|20)\d{2}\b', text),
            'sec_items': re.findall(r'\bitem\s+\d+[a-z]?\b', text.lower())
        }
        return patterns
    
    def _assign_structured_chunk_ids(self, chunks: List[Dict], doc_name: str) -> None:
        """
        Assign structured chunk IDs in format: doc_name|pX|cY
        
        Args:
            chunks: List of chunk dictionaries to update
            doc_name: Document name
        """
        page_counters = {}
        
        for chunk in chunks:
            page = chunk.get('page', 1)
            counter = page_counters.get(page, 0)
            chunk['chunk_id'] = f"{doc_name}|p{page}|c{counter}"
            chunk['id'] = chunk['chunk_id']  # Keep backward compatibility
            page_counters[page] = counter + 1
    
    def _create_semantic_chunks(self, doc, doc_name: str) -> List[Dict]:
        """
        Create semantic chunks from Docling document structure.
        Keeps tables intact and preserves context.
        
        Args:
            doc: Docling document object
            doc_name: Name of the document
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        # Get document elements with their types and content
        elements = self._extract_elements(doc)
        
        if not elements:
            # Fallback to markdown export if structure parsing fails
            return self._fallback_chunking(doc, doc_name)
        
        # Group elements into semantic chunks
        current_chunk = []
        current_size = 0
        
        for i, elem in enumerate(elements):
            elem_text = elem['text']
            elem_type = elem['type']
            elem_size = len(elem_text)
            
            # Tables are always kept as complete units
            if elem_type == 'table':
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, doc_name, len(chunks)
                    ))
                    current_chunk = []
                    current_size = 0
                
                # Create table chunk with context
                table_chunk = self._create_table_chunk(
                    elements, i, doc_name, len(chunks)
                )
                chunks.append(table_chunk)
                
            # Titles start new chunks (unless current chunk is small)
            elif elem_type == 'title' and current_size > self.max_chunk_size // 2:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, doc_name, len(chunks)
                    ))
                    current_chunk = []
                    current_size = 0
                current_chunk.append(elem)
                current_size += elem_size
                
            # Regular elements: add to current chunk
            elif current_size + elem_size <= self.max_chunk_size:
                current_chunk.append(elem)
                current_size += elem_size
                
            # Chunk size exceeded: create new chunk
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, doc_name, len(chunks)
                    ))
                current_chunk = [elem]
                current_size = elem_size
        
        # Add remaining elements
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk, doc_name, len(chunks)
            ))
        
        return chunks
    
    def _extract_elements(self, doc) -> List[Dict]:
        """
        Extract structured elements from Docling document.
        
        Args:
            doc: Docling document object
            
        Returns:
            List of element dictionaries with type, text, and page info
        """
        elements = []
        
        try:
            # Iterate through document structure
            for item, level in doc.iterate_items():
                elem_type = 'text'  # default
                
                # Determine element type
                if hasattr(item, 'label'):
                    label = str(item.label).lower()
                    if 'title' in label or 'heading' in label:
                        elem_type = 'title'
                    elif 'table' in label:
                        elem_type = 'table'
                    elif 'list' in label:
                        elem_type = 'list'
                
                # Get text content
                text = item.export_to_markdown() if hasattr(item, 'export_to_markdown') else str(item)
                
                # Get page number from provenance (prov attribute)
                page_num = 1  # default
                if hasattr(item, 'prov') and item.prov is not None:
                    # Try multiple possible attributes for page number
                    for attr in ['page_no', 'page', 'page_number']:
                        if hasattr(item.prov, attr):
                            page_val = getattr(item.prov, attr)
                            if page_val is not None:
                                # Docling page numbers are 0-indexed, convert to 1-indexed
                                page_num = (page_val + 1) if isinstance(page_val, int) else int(page_val) + 1
                                break
                elif hasattr(item, 'page') and item.page is not None:
                    page_num = (item.page + 1) if isinstance(item.page, int) else int(item.page) + 1
                
                # Extract SEC items from this element
                sec_items = self._extract_sec_items(text, page_num)
                item_number = ""
                item_title = ""
                
                if sec_items:
                    # Use first detected item
                    item_number = sec_items[0]['item_number']
                    item_title = sec_items[0]['item_title']
                    # Update last seen item (with 5-page limit for Item 16)
                    self.last_seen_item = {
                        'item_number': item_number,
                        'item_title': item_title,
                        'page': page_num
                    }
                elif self.last_seen_item:
                    # Inherit from last seen item with 5-page limit for Item 16
                    if self.last_seen_item['item_number'] == '16' and (page_num - self.last_seen_item['page'] > 5):
                        # Don't inherit Item 16 beyond 5 pages
                        pass
                    else:
                        item_number = self.last_seen_item['item_number']
                        item_title = self.last_seen_item['item_title']
                
                # Extract financial patterns
                fin_patterns = self._extract_financial_patterns(text)
                
                if text.strip():
                    elements.append({
                        'type': elem_type,
                        'text': text.strip(),
                        'page': page_num,
                        'level': level,
                        'item_number': item_number,
                        'item_title': item_title,
                        'financial_patterns': fin_patterns
                    })
        except Exception:
            # If iteration fails, return empty to trigger fallback
            return []
        
        return elements
    
    def _create_table_chunk(self, elements: List[Dict], table_idx: int, 
                           doc_name: str, chunk_id: int) -> Dict:
        """
        Create a chunk for a table with surrounding context.
        
        Args:
            elements: All document elements
            table_idx: Index of the table element
            doc_name: Document name
            chunk_id: Chunk ID
            
        Returns:
            Chunk dictionary with table and context
        """
        table_elem = elements[table_idx]
        context_parts = []
        
        # Add preceding context (title, paragraphs)
        context_size = 0
        for i in range(table_idx - 1, -1, -1):
            elem = elements[i]
            if context_size + len(elem['text']) <= self.table_context_size:
                context_parts.insert(0, elem['text'])
                context_size += len(elem['text'])
            else:
                break
        
        # Add table
        chunk_text_parts = context_parts + [table_elem['text']]
        
        # Add following context
        context_size = 0
        for i in range(table_idx + 1, len(elements)):
            elem = elements[i]
            if elem['type'] == 'table':  # Stop at next table
                break
            if context_size + len(elem['text']) <= self.table_context_size:
                chunk_text_parts.append(elem['text'])
                context_size += len(elem['text'])
            else:
                break
        
        # Collect all financial patterns from context
        all_patterns = {'dollar_amounts': [], 'percentages': [], 'years': [], 'sec_items': []}
        for i in range(max(0, table_idx - 3), min(len(elements), table_idx + 4)):
            elem_patterns = elements[i].get('financial_patterns', {})
            for key in all_patterns:
                all_patterns[key].extend(elem_patterns.get(key, []))
        
        return {
            "id": f"{doc_name}_{chunk_id}",
            "text": "\n\n".join(chunk_text_parts),
            "document": doc_name,
            "page": table_elem['page'],
            "type": "table",
            "has_table": True,
            "item_number": table_elem.get('item_number', ''),
            "item_title": table_elem.get('item_title', ''),
            "financial_patterns": all_patterns
        }
    
    def _create_chunk(self, elements: List[Dict], doc_name: str, 
                     chunk_id: int) -> Dict:
        """
        Create a chunk from a list of elements.
        
        Args:
            elements: List of element dictionaries
            doc_name: Document name
            chunk_id: Chunk ID
            
        Returns:
            Chunk dictionary
        """
        text_parts = [elem['text'] for elem in elements]
        page_num = elements[0]['page'] if elements else 1
        
        # Determine chunk type
        chunk_type = 'text'
        has_table = any(elem['type'] == 'table' for elem in elements)
        if has_table:
            chunk_type = 'mixed'
        
        # Get SEC item metadata (prefer most recent/specific)
        item_number = ""
        item_title = ""
        for elem in reversed(elements):  # Check from most recent
            if elem.get('item_number'):
                item_number = elem['item_number']
                item_title = elem.get('item_title', '')
                break
        
        # Aggregate financial patterns from all elements
        all_patterns = {'dollar_amounts': [], 'percentages': [], 'years': [], 'sec_items': []}
        for elem in elements:
            elem_patterns = elem.get('financial_patterns', {})
            for key in all_patterns:
                all_patterns[key].extend(elem_patterns.get(key, []))
        
        return {
            "id": f"{doc_name}_{chunk_id}",
            "text": "\n\n".join(text_parts),
            "document": doc_name,
            "page": page_num,
            "type": chunk_type,
            "has_table": has_table,
            "item_number": item_number,
            "item_title": item_title,
            "financial_patterns": all_patterns
        }
    
    def _fallback_chunking(self, doc, doc_name: str) -> List[Dict]:
        """
        Fallback to simple markdown-based chunking if structure extraction fails.
        Uses document pages to estimate page numbers.
        
        Args:
            doc: Docling document object
            doc_name: Document name
            
        Returns:
            List of chunk dictionaries
        """
        full_text = doc.export_to_markdown()
        chunks = []
        
        # Try to build page mapping from document pages
        page_mapping = self._build_page_mapping(doc)
        
        # Split by double newlines (paragraphs/sections)
        sections = full_text.split('\n\n')
        
        current_chunk = []
        current_size = 0
        current_pos = 0
        
        for section in sections:
            section = section.strip()
            if not section:
                current_pos += 2  # account for \n\n
                continue
            
            section_size = len(section)
            
            if current_size + section_size <= self.max_chunk_size:
                current_chunk.append(section)
                current_size += section_size
            else:
                # Save current chunk
                if current_chunk:
                    page_num = self._estimate_page(current_pos, page_mapping)
                    chunks.append({
                        "id": f"{doc_name}_{len(chunks)}",
                        "text": "\n\n".join(current_chunk),
                        "document": doc_name,
                        "page": page_num,
                        "type": "text",
                        "has_table": False
                    })
                
                # Start new chunk
                current_chunk = [section]
                current_size = section_size
            
            current_pos += section_size + 2  # +2 for \n\n
        
        # Add remaining chunk
        if current_chunk:
            page_num = self._estimate_page(current_pos, page_mapping)
            chunks.append({
                "id": f"{doc_name}_{len(chunks)}",
                "text": "\n\n".join(current_chunk),
                "document": doc_name,
                "page": page_num,
                "type": "text",
                "has_table": False
            })
        
        return chunks
    
    def _build_page_mapping(self, doc) -> List[tuple]:
        """
        Build a character position to page number mapping.
        
        Args:
            doc: Docling document object
            
        Returns:
            List of (start_pos, end_pos, page_num) tuples
        """
        page_mapping = []
        
        try:
            if hasattr(doc, 'pages') and doc.pages:
                current_pos = 0
                for page_num, page in enumerate(doc.pages, start=1):
                    # Get page content length
                    page_text = page.export_to_markdown() if hasattr(page, 'export_to_markdown') else ""
                    page_len = len(page_text)
                    
                    page_mapping.append((current_pos, current_pos + page_len, page_num))
                    current_pos += page_len + 2  # +2 for page separator
        except Exception:
            pass  # Return empty mapping if this fails
        
        return page_mapping
    
    def _estimate_page(self, position: int, page_mapping: List[tuple]) -> int:
        """
        Estimate page number from character position.
        
        Args:
            position: Character position in document
            page_mapping: List of (start, end, page) tuples
            
        Returns:
            Estimated page number
        """
        if not page_mapping:
            return 1
        
        for start, end, page in page_mapping:
            if start <= position <= end:
                return page
        
        # If position is beyond all pages, return last page
        return page_mapping[-1][2] if page_mapping else 1


class VectorStore:
    """Manages embeddings and vector search."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
    
    def add_chunks(self, chunks: List[Dict]) -> None:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries
        """
        self.chunks.extend(chunks)
        
        # Generate embeddings
        texts = [chunk["text"] for chunk in self.chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Create FAISS index
        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(embeddings.astype(np.float32))
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (chunk, distance) tuples
        """
        if self.index is None or not self.chunks:
            raise RuntimeError("Vector store is empty. Add chunks before searching.")
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append((self.chunks[idx], float(distance)))
        
        return results
    
    def save(self, save_dir: str) -> None:
        """Save the vector store."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_dir, "index.faiss"))
        
        # Save chunks metadata
        with open(os.path.join(save_dir, "chunks.json"), "w") as f:
            json.dump(self.chunks, f)
    
    def load(self, save_dir: str) -> None:
        """Load the vector store."""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(save_dir, "index.faiss"))
        
        # Load chunks
        with open(os.path.join(save_dir, "chunks.json"), "r") as f:
            self.chunks = json.load(f)
