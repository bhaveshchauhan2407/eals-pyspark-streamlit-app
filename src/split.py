from typing import List, Tuple, Dict


def leave_one_out_split(interactions: List[Tuple[int, int]]):
    """
    For each user, keep the last interaction encountered as test
    and the rest as train.
    This is a simple project-friendly split.
    """
    user_histories: Dict[int, List[int]] = {}

    for u, i in interactions:
        if u not in user_histories:
            user_histories[u] = []
        user_histories[u].append(i)

    train = []
    test = []

    for u, items in user_histories.items():
        if len(items) == 1:
            train.append((u, items[0]))
        else:
            for i in items[:-1]:
                train.append((u, i))
            test.append((u, items[-1]))

    return train, test