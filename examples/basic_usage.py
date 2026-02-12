from ..rapidindex.core.indexer import RapidIndex
from loguru import logger
import os

# Configure logging
logger.add("rapidindex.log", rotation="1 MB")

def main():
    # Initialize RapidIndex
    index = RapidIndex(
        use_embeddings=True,
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY') or ""
    )
    
    # Index a document
    print("📄 Indexing document...")
    doc_id = index.index_document(
        file_path='examples/sample_documents/financial_report.pdf',
        document_type='pdf'
    )
    print(f"✓ Document indexed: {doc_id}")
    
    # Search
    print("\n🔍 Searching...")
    query = "What was the revenue growth in Q4?"
    result = index.search(query, top_k=3)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nReasoning:\n{result['reasoning']}")
    print(f"\nConfidence: {result['confidence']}")
    print(f"\nRelevant Sections:")
    for i, section in enumerate(result['sections'], 1):
        print(f"\n{i}. {section['title']} (Pages {section['pages']})")
        print(f"   Preview: {section['content'][:150]}...")
    
    # Stats
    print(f"\n{'='*60}")
    stats = index.get_stats()
    print(f"System Stats:")
    print(f"  Documents: {stats['total_documents']}")
    print(f"  Sections: {stats['total_sections']}")
    print(f"  Cache hits: {stats['cache_stats']['hits']}")
    print(f"  Cache hit rate: {stats['cache_stats']['hit_rate']:.2%}")

if __name__ == '__main__':
    main()