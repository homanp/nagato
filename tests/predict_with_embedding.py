from nagato.service import predict_with_embedding


def callback_method(item):
    print(item)


def main():
    result = predict_with_embedding(
        input="What was Teslas YoY revenue increase in Q2 2023?",
        provider="REPLICATE",
        model="homanp/test:bc8afbabceaec8abb9b15fade05ff42db371b01fa251541b49c8ba9a9d44bc1f",
        system_prompt="You are an helpful assistant",
        embedding_provider="",
        embedding_model="",
        embedding_filter_id="",
        enable_streaming=True,
        callback=callback_method,
    )
    print(result)


main()
