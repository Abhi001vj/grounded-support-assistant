"""Load test for the API.

    make serve                         # in one terminal (deterministic mode: no API cost)
    make loadtest                      # headless: 50 users, 60 s, CSVs in artifacts/loadtest/

Pointing this at an LLM-mode server makes every request a paid API call; keep users low.
"""

import random

from locust import HttpUser, between, task

QUESTIONS = [
    "My reset email came twice. Which link should I use and how long does it work?",
    "Can support disable my 2FA without verifying me first?",
    "What should my client do after hitting the API limit?",
    "Are webhooks exactly once?",
    "Can you tell me the CEO's direct email address?",
    "How long does a bank withdrawal take?",
    "My card deposit was reversed, what should I do?",
    "Who founded the company?",
]


class SupportUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(6)
    def answer(self):
        q = random.choice(QUESTIONS)
        with self.client.post("/answer", json={"question": q}, name="/answer", catch_response=True) as r:
            body = r.json() if r.ok else {}
            if not r.ok or not {"answer", "citations", "insufficient_context"} <= set(body):
                r.failure(f"bad response: {r.status_code}")

    @task(3)
    def query(self):
        self.client.post("/query", json={"question": random.choice(QUESTIONS)}, name="/query")

    @task(1)
    def batch(self):
        qs = random.sample(QUESTIONS, 4)
        self.client.post("/answer/batch", json={"questions": [{"question": q} for q in qs]}, name="/answer/batch")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
