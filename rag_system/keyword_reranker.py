"""Stopword removal and keyword-based reranking for improved retrieval accuracy."""

import re
import nltk
from typing import List, Dict, Set, Tuple
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class StopwordKeywordReranker:
    """
    Three-stage reranking with stopword removal and keyword matching.
    
    Stage 1: Remove stopwords and filter by keyword similarity
    Stage 2: Apply semantic reranking (CrossEncoder)
    Stage 3: Combine keyword + semantic scores with hybrid weighting
    """
    
    def __init__(self):
        """Initialize the reranker with financial-domain stopwords."""
        
        # Extended stopwords for financial documents
        self.english_stopwords = set(stopwords.words('english'))
        
        # Add SEC/Financial document common stopwords
        self.financial_stopwords = {
            'company', 'companies', 'business', 'operations', 'period', 
            'year', 'years', 'quarter', 'quarters', 'fiscal', 'ended',
            'including', 'related', 'certain', 'various', 'primarily',
            'approximately', 'substantially', 'significantly', 'generally',
            'may', 'could', 'would', 'should', 'might', 'will', 'shall',
            'also', 'however', 'therefore', 'furthermore', 'moreover',
            'item', 'part', 'section', 'table', 'note', 'see', 'refer'
        }
        
        self.all_stopwords = self.english_stopwords.union(self.financial_stopwords)
        
        # Important financial/SEC keywords that should NEVER be removed
        self.protected_keywords = {
            'revenue', 'earnings', 'profit', 'loss', 'cash', 'debt', 'assets',
            'liabilities', 'equity', 'shares', 'dividend', 'eps', 'ebitda',
            'operating', 'income', 'expenses', 'margin', 'growth', 'risk',
            'material', 'adverse', 'segment', 'goodwill', 'impairment',
            'depreciation', 'amortization', 'taxes', 'automotive', 'interest', 'cost',
            'sales', 'services', 'products', 'customers', 'market', 'competition',
            'regulatory', 'compliance', 'litigation', 'contingencies',
            'apple', 'tesla', 'automotive', 'technology', 'manufacturing', 
            'unresolved', 'leasing', 'filed', 'sec', 'common', 'stock',
            'outstanding', 'total', 'signed', 'comments'
        }
    
    def _clean_and_tokenize(self, text: str) -> List[str]:
        """
        Clean text and tokenize while preserving important terms.
        
        Args:
            text: Input text
            
        Returns:
            List of cleaned tokens
        """
        text = text.lower()
        
        # Remove punctuation but keep $ and %
        text = re.sub(r'[^\w\s\$%]', ' ', text)
        
        tokens = word_tokenize(text)
        
        cleaned_tokens = []
        for token in tokens:
            # Always keep protected keywords
            if token in self.protected_keywords:
                cleaned_tokens.append(token)
            # Keep numbers
            elif re.match(r'^\d+(?:\.\d+)?$', token):
                cleaned_tokens.append(token)
            # Keep financial symbols
            elif '$' in token or '%' in token:
                cleaned_tokens.append(token)
            # Keep other tokens if not stopwords and long enough
            elif token not in self.all_stopwords and len(token) > 2:
                cleaned_tokens.append(token)
        
        return cleaned_tokens
    
    def _extract_financial_patterns(self, text: str) -> List[str]:
        """
        Extract specific financial patterns like dollar amounts, percentages, years.
        
        Args:
            text: Input text
            
        Returns:
            List of matched patterns
        """
        patterns = []
        
        # Dollar amounts
        dollar_patterns = re.findall(
            r'\$\s*\d+(?:\.\d+)?(?:\s*(?:million|billion|trillion))?', 
            text.lower()
        )
        patterns.extend(dollar_patterns)
        
        # Percentages
        percent_patterns = re.findall(r'\d+(?:\.\d+)?%', text)
        patterns.extend(percent_patterns)
        
        # Years
        year_patterns = re.findall(r'\b(?:19|20)\d{2}\b', text)
        patterns.extend(year_patterns)
        
        # SEC items
        item_patterns = re.findall(r'\bitem\s+\d+[a-z]?\b', text.lower())
        patterns.extend(item_patterns)
        
        return patterns
    
    def _calculate_keyword_similarity(
        self, 
        query_tokens: List[str], 
        doc_tokens: List[str], 
        query_patterns: List[str], 
        doc_patterns: List[str]
    ) -> Dict[str, float]:
        """
        Calculate various keyword similarity metrics.
        
        Args:
            query_tokens: Cleaned query tokens
            doc_tokens: Cleaned document tokens
            query_patterns: Financial patterns in query
            doc_patterns: Financial patterns in document
            
        Returns:
            Dictionary of similarity scores
        """
        query_set = set(query_tokens)
        doc_set = set(doc_tokens)
        
        if not query_set:
            return {
                'token_overlap': 0.0, 
                'pattern_match': 0.0, 
                'jaccard': 0.0, 
                'weighted_score': 0.0
            }
        
        # 1. Token overlap score
        intersection = query_set.intersection(doc_set)
        token_overlap = len(intersection) / len(query_set)
        
        # 2. Financial pattern matching
        pattern_matches = sum(1 for pattern in query_patterns if pattern in doc_patterns)
        pattern_match = pattern_matches / max(1, len(query_patterns))
        
        # 3. Jaccard similarity
        union = query_set.union(doc_set)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # 4. Weighted score (protected keywords count double)
        weighted_intersection = 0
        for token in intersection:
            if token in self.protected_keywords:
                weighted_intersection += 2.0
            else:
                weighted_intersection += 1.0
        
        weighted_score = weighted_intersection / len(query_set)
        
        return {
            'token_overlap': token_overlap,
            'pattern_match': pattern_match,
            'jaccard': jaccard,
            'weighted_score': weighted_score
        }
    
    def filter_and_score(
        self, 
        query: str, 
        chunks: List[Dict], 
        min_similarity: float = 0.05
    ) -> List[Tuple[Dict, Dict[str, float]]]:
        """
        Filter and score chunks based on keyword similarity.
        
        Args:
            query: Search query
            chunks: List of chunk dictionaries
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (chunk, metrics) tuples sorted by keyword score
        """
        query_tokens = self._clean_and_tokenize(query)
        query_patterns = self._extract_financial_patterns(query)
        
        scored_chunks = []
        
        for chunk in chunks:
            doc_text = chunk.get('text', '')
            doc_tokens = self._clean_and_tokenize(doc_text)
            doc_patterns = self._extract_financial_patterns(doc_text)
            
            similarity_metrics = self._calculate_keyword_similarity(
                query_tokens, doc_tokens, query_patterns, doc_patterns
            )
            
            # Combined score with emphasis on protected keywords and patterns
            combined_score = (
                0.4 * similarity_metrics['weighted_score'] +
                0.3 * similarity_metrics['token_overlap'] +
                0.2 * similarity_metrics['pattern_match'] +
                0.1 * similarity_metrics['jaccard']
            )
            
            if combined_score >= min_similarity:
                scored_chunks.append((chunk, {
                    **similarity_metrics,
                    'combined_keyword_score': combined_score
                }))
        
        # Sort by keyword score
        scored_chunks.sort(key=lambda x: x[1]['combined_keyword_score'], reverse=True)
        
        return scored_chunks
    
    def compute_hybrid_scores(
        self,
        keyword_scored_chunks: List[Tuple[Dict, Dict]],
        semantic_scores: List[float],
        hybrid_weight: float = 0.3
    ) -> List[Dict]:
        """
        Combine keyword and semantic scores.
        
        Args:
            keyword_scored_chunks: Chunks with keyword scores
            semantic_scores: Semantic reranking scores
            hybrid_weight: Weight for keyword scores (0-1)
            
        Returns:
            List of chunks with combined scores
        """
        results = []
        
        for i, (chunk, keyword_metrics) in enumerate(keyword_scored_chunks):
            if i < len(semantic_scores):
                semantic_score = semantic_scores[i]
                # Normalize semantic score to 0-1 range (CrossEncoder scores can be negative)
                normalized_semantic = max(0, min(1, (semantic_score + 5) / 10))
            else:
                normalized_semantic = 0.0
            
            keyword_score = keyword_metrics['combined_keyword_score']
            
            # Hybrid scoring
            final_score = (1 - hybrid_weight) * normalized_semantic + hybrid_weight * keyword_score
            
            result = {
                'chunk': chunk,
                'final_score': final_score,
                'semantic_score': semantic_score if i < len(semantic_scores) else 0.0,
                'keyword_score': keyword_score,
                'keyword_metrics': keyword_metrics
            }
            
            results.append(result)
        
        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return results
