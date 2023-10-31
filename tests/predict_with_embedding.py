from nagato.service import predict_with_embedding


def callback_method(item):
    print(item)


def main():
    result = predict_with_embedding(
        input="What was Teslas total revenue in Q2 2023?",
        provider="REPLICATE",
        model="meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d",
        vector_db="PINECONE",
        embedding_model="all-MiniLM-L6-v2",
        embedding_filter_id="011",
        enable_streaming=True,
        callback=callback_method,
    )
    print(result)


main()
