from nagato.service import query_embedding


def callback_method(item):
    print(item)


def main():
    result = query_embedding(
        query="How many cars were sold in total?",
        filter_id="007",
        model="thenlper/gte-large",
    )
    print(result)


main()
