import numpy as np


def recommend_top_k(model, raw_user_id, k=10):
    """
    Recommend top-k items for one raw user_id.
    Excludes items already seen in training.
    Returns raw item IDs.
    """
    if raw_user_id not in model.user_to_index:
        return []

    u = model.user_to_index[raw_user_id]
    seen_items = set(model.user_items[u])

    scores = []
    for i in range(model.item_count):
        if i in seen_items:
            continue
        score = model.predict(u, i)
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_items = scores[:k]

    return [model.index_to_item[i] for i, _ in top_items]