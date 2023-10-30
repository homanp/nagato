from nagato.service import query_embedding


def callback_method(item):
    print(item)


def main():
    result = query_embedding(
        query="What was Teslas total revenue in Q2 2023?",
        filter_id="010",
        model="jinaai/jina-embeddings-v2-base-en",
    )
    print(result)


main()
