
import json
import os
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from app.rag import (
    retrieve_phase2,
    aggregate_contexts,
    build_prompt,
    wants_workflow_output,
)

load_dotenv()

OUTPUT_DIR = Path("data/phase3_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

QUESTIONS = [
    "What are the best practices for preventing excavation damage to buried telecommunications facilities?"
]


def generate_answer(client, messages):
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )

    elapsed = time.perf_counter() - start

    answer = json.loads(
        response.choices[0].message.content
    )

    usage = response.usage

    return {
        "answer": answer,
        "latency_seconds": round(elapsed, 3),
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
    }


def main():
    client = OpenAI()

    results = []

    for question_number, question in enumerate(QUESTIONS, start=1):
        print(f"\nEvaluating question {question_number}")
        print(question)


        contexts = retrieve_phase2(
            query=question,
            top_k=8,
            min_score=0.55,
        )

        if not contexts:
            print("No contexts retrieved; skipping.")
            continue

        agg = aggregate_contexts(contexts)
        workflow = wants_workflow_output(question)

        context_path = OUTPUT_DIR / f"contexts_{question_number}.json"
        context_path.write_text(
            json.dumps(contexts, indent=2, default=str),
            encoding="utf-8",
        )

        record = {
            "question": question,
            "context_file": str(context_path),
            "retrieved_chunk_ids": [
                c.get("chunk_id") for c in contexts
            ],
        }

        for mode in ["baseline", "phase3"]:
            print(f"Running {mode}...")

            messages = build_prompt(
                query=question,
                contexts=contexts,
                agg=agg,
                workflow=workflow,
                prompt_mode=mode,
            )

            record[mode] = generate_answer(client, messages)

        results.append(record)


        (OUTPUT_DIR / "results.json").write_text(
            json.dumps(results, indent=2, default=str),
            encoding="utf-8",
        )

    print("\nEvaluation complete.")
    print(f"Results saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
