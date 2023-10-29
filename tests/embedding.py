from nagato.service import create_vector_embeddings


def main():
    result = create_vector_embeddings(
        type="PDF",
        url="https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q2-2023-Update.pdf",
        filter_id="008",
        model="thenlper/gte-large",
    )
    print(result)


main()
