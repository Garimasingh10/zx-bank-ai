import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from src.logger import logger
from src.config import config

class HybridRetriever:
    def __init__(self):
        self.encoder = SentenceTransformer(config.EMBEDDING_MODEL)
        self.index = None
        self.bm25 = None
        self.chunks = []
        
    def build_index(self, chunks):
        if not chunks:
            logger.warning("No chunks to index.")
            return

        self.chunks = chunks
        
        # 1. Build FAISS Dense Index
        logger.info("Building FAISS vector index...")
        texts = [c.page_content for c in chunks]
        embeddings = self.encoder.encode(texts, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        
        # Using L2 distance Faiss index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # 2. Build BM25 Sparse Index
        logger.info("Building BM25 keyword index...")
        tokenized_corpus = [doc.split(" ") for doc in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info(f"Hybrid index built with {len(chunks)} documents.")

    def retrieve(self, query: str, top_k: int = config.MAX_RETRIEVED_DOCS):
        """
        Retrieve documents using a hybrid score (BM25 + Dense FAISS).
        """
        if not self.chunks:
            return [], "No documents indexed."

        # 1. BM25 Search
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25
        if np.max(bm25_scores) > 0:
            bm25_scores = bm25_scores / np.max(bm25_scores)
            
        # 2. FAISS Dense Search
        query_vector = self.encoder.encode([query], convert_to_numpy=True)
        distances, indices_faiss = self.index.search(query_vector, len(self.chunks))
        
        # Normalize FAISS distances (lower is better, so we invert)
        # Using a distance threshold to avoid hallucination
        dense_scores = np.zeros(len(self.chunks))
        for rank, (idx, dist) in enumerate(zip(indices_faiss[0], distances[0])):
            if dist < config.DISTANCE_THRESHOLD:
                # Closer to 0 distance means higher score (1.0)
                dense_scores[idx] = max(0, 1.0 - (dist / config.DISTANCE_THRESHOLD))
        
        # 3. Combine scores (weighted)
        final_scores = (0.7 * dense_scores) + (0.3 * bm25_scores)
        
        # 4. Get Top K
        top_indices = final_scores.argsort()[-top_k:][::-1]
        
        results = []
        confidences = []
        for idx in top_indices:
            score = final_scores[idx]
            if score > 0.1: # Only include if there is some valid match
                results.append(self.chunks[idx])
                confidences.append(score)
        
        if not results:
            log_event = "Retrieval Output - Insufficient Evidence"
            return [], "INSUFFICIENT_EVIDENCE"
            
        return results, f"Found {len(results)} relevant chunks."
