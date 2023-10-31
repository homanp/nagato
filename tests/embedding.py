from nagato.service import create_vector_embeddings


def main():
    result = create_vector_embeddings(
        type="PDF",
        url="https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q2-2023-Update.pdf",
        filter_id="011",
        model="all-MiniLM-L6-v2",
    )
    print(result)


main()
