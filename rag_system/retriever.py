"""Retrieval and re-ranking module with 3-stage hybrid reranking."""

from typing import List, Tuple, Dict
from sentence_transformers import CrossEncoder
from rag_system.ingestion import VectorStore
from rag_system.keyword_reranker import StopwordKeywordReranker


class RetrieverWithReranker:
    """
    Retrieves and re-ranks relevant chunks using 3-stage approach:
    1. Keyword filtering (stopword removal + financial pattern matching)
    2. Semantic reranking (CrossEncoder)
    3. Hybrid scoring (combines keyword + semantic scores)
    """
    
    def __init__(
        self, 
        vector_store: VectorStore, 
        use_reranker: bool = True,
        use_hybrid: bool = True,
        hybrid_weight: float = 0.3
    ):
        """
        Initialize retriever with optional re-ranking.
        
        Args:
            vector_store: VectorStore instance
            use_reranker: Whether to use cross-encoder for re-ranking
            use_hybrid: Whether to use 3-stage hybrid reranking with keywords
            hybrid_weight: Weight for keyword scores in hybrid (0-1), default 0.3
        """
        self.vector_store = vector_store
        self.use_reranker = use_reranker
        self.use_hybrid = use_hybrid
        self.hybrid_weight = hybrid_weight
        
        if use_reranker:
            # Cross-encoder for semantic ranking
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
        
        if use_hybrid:
            # Keyword-based reranker with stopword removal
            self.keyword_reranker = StopwordKeywordReranker()
    
    def retrieve(self, query: str, top_k: int = 5, rerank: bool = True, prefetch_multiplier: int = 6) -> List[Dict]:
        """
        Retrieve relevant chunks using 3-stage hybrid reranking.
        
        Stage 1: Vector similarity search (prefetch more candidates)
        Stage 2: Keyword filtering with financial pattern matching
        Stage 3: Semantic reranking with CrossEncoder
        Stage 4: Hybrid scoring combining keyword + semantic scores
        
        Args:
            query: Search query
            top_k: Number of results to return
            rerank: Whether to apply re-ranking
            prefetch_multiplier: How many more candidates to fetch initially
            
        Returns:
            List of relevant chunks with hybrid scores
        """
        # Stage 1: Initial retrieval with many candidates
        initial_k = top_k * prefetch_multiplier if rerank and self.use_reranker else top_k
        results = self.vector_store.search(query, k=initial_k)
        chunks = [r[0] for r in results]
        
        if not chunks:
            return []
        
        # If hybrid reranking is enabled, use 3-stage approach
        if rerank and self.use_reranker and self.use_hybrid:
            # Stage 2: Keyword filtering and scoring
            keyword_scored = self.keyword_reranker.filter_and_score(
                query, 
                chunks, 
                min_similarity=0.05  # Low threshold to keep most candidates
            )
            
            if not keyword_scored:
                # Fallback: use all chunks if keyword filter too strict
                keyword_scored = [(chunk, {'combined_keyword_score': 0.0}) for chunk in chunks]
            
            # Stage 3: Semantic reranking with CrossEncoder
            candidate_chunks = [chunk for chunk, _ in keyword_scored]
            pairs = [[query, chunk["text"]] for chunk in candidate_chunks]
            semantic_scores = self.reranker.predict(pairs)
            
            # Stage 4: Hybrid scoring
            hybrid_results = self.keyword_reranker.compute_hybrid_scores(
                keyword_scored,
                semantic_scores.tolist() if hasattr(semantic_scores, 'tolist') else list(semantic_scores),
                hybrid_weight=self.hybrid_weight
            )
            
            # Format results with all scores
            return [
                {
                    **result['chunk'],
                    "score": float(result['final_score']),
                    "semantic_score": float(result['semantic_score']),
                    "keyword_score": float(result['keyword_score']),
                    "rerank_method": "hybrid_3stage"
                }
                for result in hybrid_results[:top_k]
            ]
        
        elif rerank and self.use_reranker:
            # Traditional 2-stage: vector search + semantic rerank
            pairs = [[query, chunk["text"]] for chunk in chunks]
            scores = self.reranker.predict(pairs)
            
            ranked = sorted(
                zip(chunks, scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            return [
                {
                    **chunk,
                    "score": float(score),
                    "rerank_method": "semantic_only"
                }
                for chunk, score in ranked[:top_k]
            ]
        else:
            # No reranking: just vector similarity
            return [
                {
                    **result[0],
                    "score": 1.0 / (1.0 + result[1]),  # Convert distance to similarity
                    "rerank_method": "vector_only"
                }
                for result in results[:top_k]
            ]
    
    def retrieve_diverse(self, query: str, top_k: int = 5, max_per_page: int = 2) -> List[Dict]:
        """
        Retrieve diverse chunks with page diversity constraint.
        Prevents over-reliance on a single page by limiting chunks per page.
        
        Args:
            query: Search query
            top_k: Number of results to return
            max_per_page: Maximum chunks allowed from same page
            
        Returns:
            List of diverse relevant chunks with scores
        """
        # Get more candidates than needed for diversity filtering
        candidates = self.retrieve(query, top_k=top_k * 2, rerank=True)
        
        selected = []
        page_count = {}
        
        for chunk in candidates:
            page = chunk.get('page', 0)
            current_count = page_count.get(page, 0)
            
            # Add chunk if under page limit
            if current_count < max_per_page:
                selected.append(chunk)
                page_count[page] = current_count + 1
            
            # Stop when we have enough diverse chunks
            if len(selected) >= top_k:
                break
        
        return selected
    
    @staticmethod
    def format_sources(chunks: List[Dict]) -> List[str]:
        """
        Format chunks into source citations with SEC item information.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of formatted source strings
        """
        sources = []
        for chunk in chunks:
            doc_name = chunk.get('document', 'Unknown')
            page = chunk.get('page', 0)
            item_number = chunk.get('item_number', '')
            
            # Format: "Document Name, Item X, p. Y" or "Document Name, p. Y"
            if item_number:
                source = f"{doc_name}, Item {item_number}, p. {page}"
            else:
                source = f"{doc_name}, p. {page}"
            
            sources.append(source)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sources = []
        for source in sources:
            if source not in seen:
                seen.add(source)
                unique_sources.append(source)
        
        return unique_sources
