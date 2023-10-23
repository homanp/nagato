from nagato.service import query_embedding


def callback_method(item):
    print(item)


def main():
    result = query_embedding(
        query="How many cars did Tesla sell in Q2 2023?", filter_id="001"
    )
    print(result)


main()
