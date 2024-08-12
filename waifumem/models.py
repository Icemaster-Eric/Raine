from sentence_transformers import SentenceTransformer, CrossEncoder


embedding_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l")
reranking_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
