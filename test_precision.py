import sys
import os

# Add src to path
sys.path.append(os.path.abspath("."))

from src.retriever import HybridRetriever
from src.agent import ConversationalAgent
from src.document_processor import DocumentProcessor

def test_final_precision():
    processor = DocumentProcessor()
    retriever = HybridRetriever()
    chunks = processor.process_documents()
    retriever.build_index(chunks, force=True)
    agent = ConversationalAgent(retriever)
    
    queries = [
        "Is ZX Bank's CEO a woman?",
        "Tell me about the branch at Alambagh Main Market in Lucknow."
    ]
    
    for q in queries:
        print(f"\nQUERY: {q}")
        res = agent.process_request("test_precision", q)
        print(f"RESPONSE:\n{res}")

if __name__ == "__main__":
    test_final_precision()
