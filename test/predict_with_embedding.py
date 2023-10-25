from nagato.service import predict_with_embedding


def callback_method(item):
    print(item)


def main():
    result = predict_with_embedding(
        input="How many cars did Tesla sell in Q2 2023?",
        provider="REPLICATE",
        model="homanp/test:383487e5c6ee28d189d6a2542a7de3a0ba28b9f26286de08cfda5f0390b92c09",
        enable_streaming=False,
        embedding_model="thenlper/gte-small",
        embedding_filter_id="004",
        embedding_provider="PINECONE",
    )
    print(result)


main()
