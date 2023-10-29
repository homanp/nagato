from nagato.service import create_vector_embeddings


def main():
    result = create_vector_embeddings(
        type="PDF",
        url="https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q2-2023-Update.pdf",
        filter_id="010",
        model="jinaai/jina-embeddings-v2-base-en",
    )
    print(result)


main()
