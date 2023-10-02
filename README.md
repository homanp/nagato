<div align="center">

# 🌸 Nagato

### Nagato is a framework that enables any developer to streamline the creation of fine-tuned embedding and language models specifically tailored to a given corpus of data

<p>
<img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/homanp/Nagato" />
<img alt="GitHub Last Commit" src="https://img.shields.io/github/last-commit/homanp/Nagato" />
<img alt="" src="https://img.shields.io/github/repo-size/homanp/Nagato" />
<img alt="GitHub Issues" src="https://img.shields.io/github/issues/homanp/Nagato" />
<img alt="GitHub Pull Requests" src="https://img.shields.io/github/issues-pr/homanp/Nagato" />
<img alt="Github License" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
<img alt="Discord" src="https://img.shields.io/discord/1110910277110743103?label=Discord&logo=discord&logoColor=white&style=plastic&color=d7b023)](https://discord.gg/e8j7mgjDUK" />
</p>

</div>

-----
<p align="center">
  <a href="#quick-start-guide">Quick Start Guide</a> •
  <a href="#features">Features</a> •
  <a href="#key-benefits">Key benefits</a> •
  <a href="#how-it-works">How it works</a>
</p>

-----

## Features

- Data ingestion from various formats such as JSON, CSV, TXT, PDF, etc.
- Data embedding using pre-trained or finetuned models.
- Storage of embedded vectors
- Automatic generation of question/answer pairs for model finetuning
- Built in code interpreter
- API concurrency for scalalbility and performance
- Workflow management for ingestion pipelines

## Key benefits

- **Faster inference**: Generic models often bring overhead in terms of computational time due to their broad-based training. In contrast, our fine-tuned models are optimized for specific domains, enabling faster inference and more timely results.

- **Lower costs**: Utilizing fine-tuned models tailored for a specific corpus minimizes the number of tokens needed for accurate understanding and response generation. This reduction in token count translates to decreased computational costs and thus lower operational expenses.

- **Better results**: Fine-tuned models offer superior performance on specialized tasks when compared to generic, all-purpose models. Whether you're generating embeddings or answering complex queries, you can expect more accurate and contextually relevant outcomes.

## How it works

Nagato utilizes distinct strategies to process structured and unstructured data, aiming to produce fine-tuned models for both types. Below is a breakdown of how this is accomplished:

![Untitled-2023-10-01-2152](https://github.com/homanp/nagato/assets/2464556/d3db5fa8-28ed-4623-a54a-bb07e494d362)

### Unstructured data:

1. **Selection of Embedding Model**: The first step involves a careful analysis of the textual content to select an appropriate text-based embedding model. Based on various characteristics of the corpus such as vocabulary, context, and domain-specific jargon, Nagato picks the most suitable pre-trained text-based model for embedding.

2. **Fine-Tuning the Embedding Model**: Once the initial text-based model is selected, it is then fine-tuned to align more closely with the specific domain or subject matter of the corpus. This ensures that the embeddings generated are as accurate and relevant as possible.

3. **Fine-Tuning the Language Model**: After generating and storing embeddings, Nagato creates question-answer pairs for the purpose of fine-tuning a GPT-based language model. This yields a language model that is highly specialized in understanding and generating text within the domain of the corpus.

### Structured data:

1. **Sandboxed REPL**: Nagato features a secure, sandboxed Read-Eval-Print Loop (REPL) environment to execute code snippets against the structured text data. This facilitates flexible and dynamic processing of structured data formats like JSON, CSV or XML.

2. **Evaluation/Prediction Using a Code Interpreter**: Post-initial processing, a code interpreter evaluates various code snippets within the sandboxed environment to produce predictions or analyses based on the structured text data. This capability allows the extraction of highly specialized insights tailored to the domain or subject matter.
