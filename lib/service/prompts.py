GPT_DATA_FORMAT = (
    "{"
    '"messages": ['
    '{"role": "system", "content": "You are an AI agent that\'s an expert at answering questions."}, '
    '{"role": "user", "content": "What\'s the capital of France?"}, '
    '{"role": "assistant", "content": "Paris, as if everyone doesn\'t know that already."}'
    "]"
    "}"
)


def generate_qa_pair_prompt(format: str, context: str, num_of_qa_paris: int = 10):
    prompt = (
        "You are an AI assistant tasked with generating question and answer pairs"
        "for the given context using the given format. Only answer in the format with"
        f"no other text. You should create the following number of question/answer pairs: {num_of_qa_paris}"
        "Return the question/answer pairs as a JSONL."
        "Each dict in the list should have the full context provided,"
        "a relevant question to the context and an answer to the question.\n\n"
        f"Format:\n {format}\n\n"
        f"Context:\n {context}"
    )
    return prompt