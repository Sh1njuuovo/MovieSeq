"""Convert MovieLens-1M ratings.dat into implicit-feedback JSONL."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def prepare_movielens(path, min_rating=4, negatives=1, max_history=50, seed=42):
    if negatives < 0 or max_history < 1:
        raise ValueError("negatives must be nonnegative and max_history positive")
    histories = defaultdict(list)
    all_items = set()
    with Path(path).open(encoding="latin-1") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.rstrip().split("::")
            if len(fields) != 4:
                raise ValueError(f"line {number} is not a MovieLens rating")
            user, item, rating, timestamp = map(int, fields)
            all_items.add(item)
            if rating >= min_rating:
                histories[user].append((timestamp, item))
    item_ids = {item: index + 1 for index, item in enumerate(sorted(all_items))}
    universe = set(item_ids.values())
    rng = random.Random(seed)
    rows = []
    for user in sorted(histories):
        events = [item_ids[item] for _, item in sorted(histories[user])]
        unseen = sorted(universe - set(events))
        for index in range(1, len(events)):
            history = events[max(0, index - max_history):index]
            rows.append({"history": history, "candidate": events[index], "label": 1})
            for item in rng.sample(unseen, k=min(negatives, len(unseen))):
                rows.append({"history": history, "candidate": item, "label": 0})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ratings", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-rating", type=int, default=4)
    parser.add_argument("--negatives", type=int, default=1)
    parser.add_argument("--max-history", type=int, default=50)
    args = parser.parse_args()
    rows = prepare_movielens(args.ratings, args.min_rating, args.negatives, args.max_history)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} examples to {args.output}")


if __name__ == "__main__":
    main()
