# flake8: noqa

GPT_DATA_FORMAT = (
    "{"
    '"messages": ['
    '{"role": "system", "content": "You are an AI agent that\'s an expert at answering questions."}, '
    '{"role": "user", "content": "What\'s the capital of France?"}, '
    '{"role": "assistant", "content": "Paris, is the capital of France."}'
    "]"
    "}"
)

REPLICATE_FORMAT = (
    "{"
    '"prompt": "What\'s the capital of France?",'
    '"completion": "Paris, is the capital of France"'
    "}"
)

REPLICATE_SYSTEM_PROMPT = (
    "You are an AI Assistant that's an expert at answering questions.\n\n"
    "If a question does not make any sense, or is not factually coherent, "
    "explain why instead of answering something not correct.\n\n"
    "If you don't know the answer to a question, please don't share false information."
)


def generate_replicaste_system_prompt(context: str) -> str:
    prompt = (
        "You are an AI Assistant that's an expert at answering questions.\n"
        "Use the following context to answer any questions:\n\n"
        f"Context:\n {context}\n\n"
    )
    return prompt


def generate_qa_pair_prompt(
    format: str, context: str, num_of_qa_pairs: int = 10
) -> str:
    prompt = (
        "You are an AI assistant tasked with generating question and answer pairs"
        "for the given context using the given format. Only answer in the format with"
        f"no other text. You should create the following number of question/answer pairs: {num_of_qa_pairs}"
        "Return the question/answer pairs as a JSONL."
        "Each dict in the list should have the full context provided,"
        "a relevant question to the context and an answer to the question.\n\n"
        f"Format:\n {format}\n\n"
        f"Context:\n {context}"
    )
    return prompt
