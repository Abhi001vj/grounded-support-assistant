# Problem statement (verbatim copy of the assignment brief)

> Saved as received so the specs travel with the code. The wording is unchanged; the only
> edit is wrapping the JSON samples in ```json fences so they render. The sample data files
> are also copied byte-for-byte to [`sample-data/`](sample-data/).

---

Before you start coding: develop Specs or design notes. You can upload them with your final submission; Markdown is preferred.
Problem Statement
Build a small, runnable AI support assistant for an AI Engineer interview. The service should answer product-support questions using only the provided internal knowledge base, return grounded citations, and refuse to invent facts when the answer is not supported.

This challenge is intentionally focused on execution: clear requirements, limited ambiguity, and practical completion in 60 minutes. We want to see whether you can turn a simple retrieval + LLM workflow into a reliable, testable implementation rather than a demo that only works on one happy path.

Your task is to implement a local pipeline that:

loads a small support knowledge base from disk
retrieves the most relevant document chunks for a user question
builds a prompt for an LLM or a deterministic fallback answer generator
returns a structured answer with citations
handles unsupported questions safely
includes basic tests and a validation command
You may use any LLM provider if you want, but your solution must still run from a clean checkout without requiring private credentials. If no API key is present, the pipeline must still work in a deterministic fallback mode.

Input / Sample Data
Create the following input files in your repository.

kb/articles.json
Use this exact sample data for local testing:

```json
[
  {
    "id": "ART-101",
    "title": "Password reset flow",
    "product": "Account",
    "content": "Users can reset their password from the login page by selecting Forgot password. A reset link is valid for 15 minutes. If a user requests multiple reset emails, only the latest link remains valid. Support agents cannot view or manually set a user's password."
  },
  {
    "id": "ART-102",
    "title": "Two-factor authentication recovery",
    "product": "Account",
    "content": "If a user loses access to their authenticator app, they must complete identity verification before 2FA can be removed. Backup codes can be used if previously saved. Support cannot bypass identity verification."
  },
  {
    "id": "ART-201",
    "title": "Withdrawal processing times",
    "product": "Payments",
    "content": "E-wallet withdrawals are usually processed within 2 hours. Bank withdrawals may take 1 to 3 business days depending on the destination bank. Withdrawals can be delayed if account verification is incomplete or if there is a security review."
  },
  {
    "id": "ART-202",
    "title": "Card deposit reversals",
    "product": "Payments",
    "content": "If a card deposit is reversed by the issuer, the trading balance may be adjusted automatically. Users should contact their card issuer first for the reason. Support can confirm whether the platform received the reversal notification."
  },
  {
    "id": "ART-301",
    "title": "API rate limits",
    "product": "Developer",
    "content": "Public API requests are limited to 120 requests per minute per token. When the rate limit is exceeded, the API returns HTTP 429. Clients should implement exponential backoff and retry with jitter."
  },
  {
    "id": "ART-302",
    "title": "Webhook delivery behavior",
    "product": "Developer",
    "content": "Webhook events are delivered at least once. Duplicate deliveries are possible. Consumers should verify signatures, store event IDs, and process events idempotently."
  }
]
```

questions.json
Use this exact sample file for local testing:

```json
[
  {
    "id": "Q1",
    "question": "My reset email came twice. Which link should I use and how long does it work?"
  },
  {
    "id": "Q2",
    "question": "Can support disable my 2FA without verifying me first?"
  },
  {
    "id": "Q3",
    "question": "What should my client do after hitting the API limit?"
  },
  {
    "id": "Q4",
    "question": "Are webhooks exactly once?"
  },
  {
    "id": "Q5",
    "question": "Can you tell me the CEO's direct email address?"
  }
]
```

The evaluator may replace the questions with equivalent ones. Your implementation must not depend on exact wording or fixed IDs.

MUST COMPLETE
1. Knowledge Base Loading and Chunking
Build code that loads kb/articles.json and prepares documents for retrieval.

Requirements:

preserve article metadata: id, title, and product
create chunks in code, even if each sample article stays as one chunk
save processed retrieval records to artifacts/index_preview.json
each retrieval record must contain:

```json
{
  "chunk_id": "string",
  "article_id": "string",
  "title": "string",
  "product": "string",
  "text": "string"
}
```

2. Retrieval Pipeline
Implement a retrieval step that ranks relevant chunks for each question.

Requirements:

retrieval must be done in code, not by the LLM
use a simple and reliable approach such as TF-IDF, BM25, or embeddings
return top 3 chunks per question
save retrieval results to artifacts/retrieval_results.json
Each result must include:

```json
{
  "question_id": "string",
  "question": "string",
  "retrieved_chunks": [
    {
      "chunk_id": "string",
      "article_id": "string",
      "score": 0.0
    }
  ]
}
```

3. Grounded Answer Generation
Implement an answer stage that uses only retrieved context.

Requirements:

output must be structured JSON per question
answer must contain a short user-facing response
include cited article IDs used in the answer
if the retrieved context does not support the answer, return "insufficient_context": true and a safe refusal or escalation message
do not fabricate policies, people, contact details, or unsupported steps
if you use an LLM, the prompt must explicitly forbid using outside knowledge
if no LLM credentials are available, provide a deterministic fallback that still follows the same output schema
Save output to artifacts/answers.json using this shape:

```json
[
  {
    "question_id": "string",
    "answer": "string",
    "citations": ["ART-101"],
    "insufficient_context": false
  }
]
```

4. Basic Guardrails and Evaluation
Add simple checks that make the system safer and easier to verify.

Requirements:

unsupported questions should not receive invented answers
duplicate or conflicting citations should be cleaned up
add a small evaluation script that checks the sample questions
for supported questions, at least one expected source article should appear in citations
for unsupported questions like Q5, insufficient_context must be true
Save evaluation output to artifacts/eval_summary.json.

5. Runnable Entry Point and Validation
Provide a single command to run the full pipeline from a clean checkout.

Examples:

python main.py
or

make run
Also provide a validation command, for example:

python validate.py
Validation must check:

required input files exist
required artifact files are generated
output JSON files are valid
every answer has the required fields
every citation refers to an article in the knowledge base
unsupported sample question Q5 is marked with insufficient_context: true
SHOULD ATTEMPT
6. Simple Tests
Add a few focused tests for the most important behavior.

Suggested coverage:

retrieval returns the password article for a password reset question
unsupported question returns insufficient_context: true
citation list contains unique article IDs
7. Observability
Log one record per question to artifacts/run_log.jsonl.

Each line should include:

```json
{
  "question_id": "string",
  "retrieval_chunk_ids": ["string"],
  "used_fallback": true,
  "timestamp": "ISO-8601 string"
}
```

STRETCH
8. Minimal Service Interface
Expose the pipeline through a tiny local API, such as:

POST /answer with a question string
response in the same structured format as answers.json
A lightweight FastAPI or Flask wrapper is enough.

Required Artifacts or Expected Outcome
Your repository should produce:

kb/articles.json
questions.json
artifacts/index_preview.json
artifacts/retrieval_results.json
artifacts/answers.json
artifacts/eval_summary.json
validate.py
tests, if attempted
artifacts/run_log.jsonl, if attempted
The evaluator should be able to run your solution from a clean checkout and regenerate artifacts.

Tools
Python is required.

Recommended libraries:

scikit-learn for TF-IDF retrieval
pydantic or dataclasses for structured outputs
pytest for tests
FastAPI or Flask for stretch work
Technical Constraints
Do not require private data, internal systems, or paid services.
The solution must work without API keys.
Retrieval must happen in code.
Answers must be grounded in retrieved context.
Unsupported questions must be handled safely.
Keep the implementation simple and maintainable.
Prefer readable code over clever abstractions.
Deliverable Format
Include a short README.md with:

how to run the pipeline
how to run validation
whether you used an LLM or fallback mode
any assumptions or tradeoffs
Tools allowed:
Any AI tools, editors, browsers, terminals — work naturally.
