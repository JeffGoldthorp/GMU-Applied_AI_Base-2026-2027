
import re
from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def bm25_retrieve(query, corpus, top_k=20):
    if not corpus:
        return []

    documents = [
        item for item in corpus
        if not item.get("is_owner_response")
        and item.get("text")
        and item.get("chunk_id")
    ]

    if not documents:
        return []

    tokenized = [
        tokenize(
            " ".join([
                str(item.get("document_title") or ""),
                str(item.get("text") or "")
            ])
        )
        for item in documents
    ]

    model = BM25Okapi(tokenized)
    scores = model.get_scores(tokenize(query))

    ranked = sorted(
        zip(documents, scores),
        key=lambda pair: pair[1],
        reverse=True
    )

    results = []
    for item, score in ranked:
        if score <= 0:
            continue

        results.append({
            **item,
            "id": item["chunk_id"],
            "score": float(score),
            "retrieval_method": "bm25"
        })

        if len(results) >= top_k:
            break

    return results


def hybrid_fusion(semantic, keyword, top_k=8, k=60):
    combined = {}
    scores = {}

    for results in (semantic, keyword):
        for rank, item in enumerate(results, start=1):
            chunk_id = item.get("chunk_id") or item.get("id")
            if not chunk_id:
                continue

            if chunk_id not in combined:
                combined[chunk_id] = item.copy()

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + 1.0 / (k + rank)
            )

    ranked = sorted(
        combined,
        key=lambda cid: scores[cid],
        reverse=True
    )

    output = []
    for cid in ranked[:top_k]:
        item = combined[cid].copy()
        item["score"] = scores[cid]
        item["fusion_score"] = scores[cid]
        output.append(item)



    return output

def rerank_with_openai(
    client, query, candidates, top_k=8,
    model="gpt-4.1-mini"
):
    import json

    if not candidates:
        return []

    evidence = [
        {
            "index": i,
            "title": item.get("document_title", ""),
            "text": (item.get("text") or "")[:1200]
        }
        for i, item in enumerate(candidates)
    ]

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Rank the supplied document chunks by "
                    "relevance to the question. Return JSON "
                    'with {"ranked_indices": [0, 1, ...]}. '
                    "Use each index at most once. "
                    "Do not invent indices."
                )
            },
            {
                "role": "user",
                "content": json.dumps({
                    "question": query,
                    "candidates": evidence
                })
            }
        ]
    )

    data = json.loads(response.choices[0].message.content)
    indices = data.get("ranked_indices", [])

    ordered = []
    seen = set()

    for index in indices:
        if (
            type(index) is int
            and 0 <= index < len(candidates)
            and index not in seen
        ):
            ordered.append(candidates[index])
            seen.add(index)

    ordered.extend(
        item for i, item in enumerate(candidates)
        if i not in seen
    )

    return ordered[:top_k]
