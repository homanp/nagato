from nagato.service import predict_with_embedding


def callback_method(item):
    print(item)


def main():
    result = predict_with_embedding(
        input="What was Teslas total revenue in Q2 2023?",
        provider="REPLICATE",
        model="homanp/test:bc8afbabceaec8abb9b15fade05ff42db371b01fa251541b49c8ba9a9d44bc1f",
        vector_db="PINECONE",
        embedding_model="jinaai/jina-embeddings-v2-base-en",
        embedding_filter_id="010",
        enable_streaming=True,
        callback=callback_method,
    )
    print(result)


main()
