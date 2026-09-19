import math
from src.recommender import recommend_top_k


def hit_rate_at_k(model, test_interactions, k=10):
    if len(test_interactions) == 0:
        return 0.0

    hits = 0
    total = 0

    for raw_user_id, true_item in test_interactions:
        recs = recommend_top_k(model, raw_user_id, k=k)
        if true_item in recs:
            hits += 1
        total += 1

    return hits / total


def ndcg_at_k(model, test_interactions, k=10):
    if len(test_interactions) == 0:
        return 0.0

    total_ndcg = 0.0
    total = 0

    for raw_user_id, true_item in test_interactions:
        recs = recommend_top_k(model, raw_user_id, k=k)

        if true_item in recs:
            rank = recs.index(true_item) + 1
            total_ndcg += 1.0 / math.log2(rank + 1)

        total += 1

    return total_ndcg / total


def online_protocol_metrics(model, test_interactions, k=10, w_new=4.0, online_iter=1):
    if len(test_interactions) == 0:
        return {"hr_at_k": 0.0, "ndcg_at_k": 0.0}

    hits = 0
    total_ndcg = 0.0
    total = 0

    for raw_user_id, true_item in test_interactions:
        recs = recommend_top_k(model, raw_user_id, k=k)

        if true_item in recs:
            hits += 1
            rank = recs.index(true_item) + 1
            total_ndcg += 1.0 / math.log2(rank + 1)

        model.update_model(raw_user_id, true_item, w_new=w_new, online_iter=online_iter)
        total += 1

    return {
        "hr_at_k": hits / total,
        "ndcg_at_k": total_ndcg / total,
    }