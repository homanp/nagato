from nagato.service import predict


def callback_method(item):
    print(item)


def main():
    result = predict(
        provider="REPLICATE",
        model="homanp/test:383487e5c6ee28d189d6a2542a7de3a0ba28b9f26286de08cfda5f0390b92c09",
        enable_streaming=False,
    )
    print(result)


main()
