from nagato.service import query_embedding


def callback_method(item):
    print(item)


def main():
    result = query_embedding(
        query="What was total revenues in Q2 2023?",
        filter_id="011",
        model="all-MiniLM-L6-v2",
    )
    print(result)


main()
