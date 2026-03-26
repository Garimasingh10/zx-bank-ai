import os
import glob
from langchain_text_splitters import MarkdownHeaderTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from src.logger import logger
from src.config import config

class DocumentProcessor:
    def __init__(self):
        self.chunks = []
        # split by standard markdown headers to preserve hierarchy
        self.headers_to_split_on = [
            ("#", "Header1"),
            ("##", "Header2"),
            ("###", "Header3"),
            ("####", "Header4")
        ]
        self.splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)

    def process_documents(self):
        docs_path = os.path.join(config.DOCS_DIR, "*.md")
        files = glob.glob(docs_path)
        
        if not files:
            logger.warning(f"No markdown documents found in {config.DOCS_DIR}. RAG will be disabled.")
            return []

        all_splits = []
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Split document and retain markdown structure in metadata
            splits = self.splitter.split_text(content)
            for split in splits:
                split.metadata["source"] = os.path.basename(file)
                all_splits.append(split)
                
        logger.info(f"Extracted {len(all_splits)} structured chunks from {len(files)} markdown files.")
        
        # Apply TF-IDF to extract top keywords per chunk as metadata (Lightweight NLP requirement)
        self._attach_tfidf_keywords(all_splits)
        self.chunks = all_splits
        return self.chunks

    def _attach_tfidf_keywords(self, chunks):
        if not chunks:
            return
            
        texts = [chunk.page_content for chunk in chunks]
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            for i, chunk in enumerate(chunks):
                row = tfidf_matrix.getrow(i).toarray()[0]
                top_indices = row.argsort()[-5:][::-1] # top 5 keywords
                keywords = [feature_names[idx] for idx in top_indices if row[idx] > 0]
                chunk.metadata["keywords"] = keywords
        except Exception as e:
            logger.error(f"Failed to generate TF-IDF keywords: {e}")
