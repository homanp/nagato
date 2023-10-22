from nagato.service import create_finetuned_model


# Define a function that calls the ingest function
def main():
    result = create_finetuned_model(
        url="https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q2-2023-Update.pdf",
        type="PDF",
        base_model="GPT_35_TURBO",
        provider="OPENAI",
        webhook_url="https://webhook.site/ebe803b9-1e34-4b20-a6ca-d06356961cd1",
    )
    print(f"🤖 MODEL: {result}")


# Run the function
main()
