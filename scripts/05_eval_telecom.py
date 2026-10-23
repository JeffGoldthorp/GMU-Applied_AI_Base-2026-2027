# scripts/05_eval_telecom.py
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.rag import generate_grounded_response


SAMPLE_QUERIES = [
    "What causes 911 network outages and what actions can improve 911 reliability?",
    "How can power outages affect telecommunications networks, and what measures can improve network resilience during power failures?",
    "What causes underground telecommunications cable damage during construction, and what practices can prevent fiber cuts and service outages?",
    "What measures can telecommunications providers use to improve network resilience and prevent service outages?",
]


def _chunk_ids_from_items(items: Any) -> List[str]:
    chunk_ids = []

    if not isinstance(items, list):
        return chunk_ids

    for item in items:
        if isinstance(item, str):
            if item.startswith(("energy-", "telecom-")):
                chunk_ids.append(item)

        elif isinstance(item, dict):
            chunk_id = item.get("chunk_id")
            if chunk_id:
                chunk_ids.append(str(chunk_id))

            for key in (
                "evidence",
                "source_chunk_ids",
                "evidence_chunk_ids",
                "grounding_chunk_ids",
            ):
                values = item.get(key) or []

                if isinstance(values, list):
                    for value in values:
                        if isinstance(value, str) and value:
                            chunk_ids.append(value)

    return list(dict.fromkeys(chunk_ids))


def assert_schema_and_shape(out: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(out, dict):
        return ["Output is not a dict"]

    if "error" in out:
        return [f"Response returned error: {out.get('error')}"]

    if not out.get("answer_summary"):
        errors.append("Missing or empty answer_summary")

    has_general_answer = isinstance(out.get("key_points"), list) and isinstance(out.get("sources"), list)
    has_workflow_answer = (
        isinstance(out.get("key_requirements"), list)
        and isinstance(out.get("policy_recommendations"), list)
        and isinstance(out.get("draft_policy_language"), list)
    )

    if not has_general_answer and not has_workflow_answer:
        errors.append(
            "Output must include either key_points/sources or "
            "key_requirements/policy_recommendations/draft_policy_language"
        )

    return errors


def assert_grounding_ids(out: Dict[str, Any]) -> List[str]:
    errors = []
    referenced = set()

    for key in (
        "sources",
        "used_chunk_ids",
        "key_points",
        "key_requirements",
        "policy_recommendations",
        "draft_policy_language",
    ):
        referenced.update(_chunk_ids_from_items(out.get(key)))

    retrieved = {
        str(source.get("chunk_id"))
        for source in out.get("retrieval_sources", [])
        if isinstance(source, dict) and source.get("chunk_id")
    }

    if not referenced:
        errors.append("No grounding chunk IDs referenced")
        return errors

    if not retrieved:
        errors.append("No retrieved source IDs available")
        return errors

    invalid = sorted(referenced - retrieved)

    if invalid:
        errors.append(
            "Citations not found in retrieved sources: "
            + ", ".join(invalid)
        )

    return errors


def run_one(q: str, top_k: int, min_recurring_reviews: int) -> Tuple[Dict[str, Any], List[str], float]:
    start_time = time.perf_counter()

    out = generate_grounded_response(
        query=q,
        top_k=top_k,
        min_recurring_reviews=min_recurring_reviews,
        include_debug=True,
        min_score=0.40,
    )

    response_time = time.perf_counter() - start_time

    errs: List[str] = []
    errs += assert_schema_and_shape(out)

    if "error" not in out:
        errs += assert_grounding_ids(out)

    return out, errs, response_time


def main(top_k: int, min_recurring_reviews: int, out_path: str) -> None:
    all_fails = 0
    stamp = datetime.now().isoformat(timespec="seconds")
    output_file = Path(out_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for q in SAMPLE_QUERIES:
        out, errs, response_time = run_one(q, top_k, min_recurring_reviews)

        results.append({
            "timestamp": stamp,
            "query": q,
            "response_time_seconds": round(response_time, 3),
            "passed": not bool(errs),
            "errors": errs,
            "output": out,
        })

        if errs:
            all_fails += 1
            print("\n" + "=" * 80)
            print("QUERY:", q)
            print("FAILED CHECKS:")
            for e in errs:
                print("-", e)
        else:
            print(f"[PASS] {q}")

    with output_file.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote eval results -> {output_file}")

    if all_fails:
        raise SystemExit(f"\nEvaluation finished with {all_fails} failing queries.")
    print("\nAll checks passed.")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--min_recurring_reviews", type=int, default=2)

    ap.add_argument(
        "--namespace",
        required=True,
        choices=[
            "group38-telecom-test",
            "group38-telecom-structure-v1",
        ],
    )

    ap.add_argument("--out", default="data/eval_results.jsonl")

    args = ap.parse_args()

    os.environ["PINECONE_NAMESPACE"] = args.namespace
    os.environ["ENABLE_SERPAPI_SEARCH"] = "false"

    print(f"Evaluating namespace: {args.namespace}")

    main(args.top_k, args.min_recurring_reviews, args.out)