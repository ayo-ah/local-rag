from ..domain.dto import EvalSample, RAGResponse, RetrievalMetrics


def evaluate_retrieval(
    samples: list[EvalSample],
    responses: list[RAGResponse],
) -> RetrievalMetrics:

    gt_map = {s.question_id: set(s.source) for s in samples}

    hits = 0
    recall_sum = 0
    mrr_sum = 0

    for resp in responses:
        gt = gt_map.get(resp.question_id, set())

        if not gt:
            continue

        intersection = gt.intersection(resp.sources)
        recall_sum += len(intersection) / len(gt)

        rank = 0
        for i, cid in enumerate(resp.sources):
            if cid in gt:
                rank = i + 1
                break

        if rank:
            hits += 1
            mrr_sum += 1 / rank

    n = len(responses)

    return RetrievalMetrics(
        recall_at_k=recall_sum / n,
        mrr=mrr_sum / n,
        hit_rate=hits / n,
    )
