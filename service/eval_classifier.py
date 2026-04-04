"""
eval_classifier.py — Evaluate the classifier's ability to detect highlighted
(significant) sentences against ACLSum gold annotations.

A sentence is "highlighted" if the classifier assigns any label other than NONE.
Gold highlighted sentences are the union of challenge, approach, and outcome
sentences from ACLSum — i.e. any sentence the paper authors considered significant.

This mirrors SCIM's evaluation metric, allowing direct comparison:
    SCIM F1: 0.533  |  Annotator-annotator F1: 0.725

Usage:
    python service/eval_classifier.py
    python service/eval_classifier.py --split test --max-papers 50
"""
import sys
import argparse

sys.path.insert(0, ".")

from classifier import classify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation",
                        choices=["train", "validation", "test"])
    parser.add_argument("--max-papers", type=int, default=None)
    args = parser.parse_args()

    print("Loading ACLSum dataset (extractive config)...")
    from datasets import load_dataset
    ds = load_dataset("sobamchan/aclsum", "extractive", split=args.split)

    if args.max_papers:
        ds = ds.select(range(min(args.max_papers, len(ds))))

    print(f"Evaluating classifier on {len(ds)} papers from '{args.split}' split...\n")

    tp = fp = fn = tn = 0
    skipped = 0

    for i, paper in enumerate(ds, start=1):
        sentences = paper["source_sentences"]
        if not sentences or len(sentences) < 2:
            skipped += 1
            continue

        all_gold = set(
            s.strip().lower()
            for key in ("challenge_sentences", "approach_sentences", "outcome_sentences")
            for s in paper[key]
        )
        if not all_gold:
            skipped += 1
            continue

        classifications = classify(sentences)
        labels = [c[0] for c in classifications]

        for sent, label in zip(sentences, labels):
            is_gold = sent.strip().lower() in all_gold
            is_pred = label != "NONE"
            if is_gold and is_pred:
                tp += 1
            elif not is_gold and is_pred:
                fp += 1
            elif is_gold and not is_pred:
                fn += 1
            else:
                tn += 1

        if i % 10 == 0:
            print(f"  [{i}/{len(ds)}] running: "
                  f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    total = tp + fp + fn + tn
    print()
    print("=" * 55)
    print(f"Split            : {args.split}")
    print(f"Papers evaluated : {len(ds) - skipped}")
    print(f"Papers skipped   : {skipped}")
    print(f"Total sentences  : {total}")
    print(f"  Gold highlighted   : {tp + fn}  ({(tp + fn) / total:.1%})")
    print(f"  Pred highlighted   : {tp + fp}  ({(tp + fp) / total:.1%})")
    print()
    print("Highlighted Sentence Detection")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1        : {f1:.3f}")
    print()
    print("Comparison")
    print(f"  SCIM                : 0.533")
    print(f"  Annotator-annotator : 0.725")
    print(f"  Ours                : {f1:.3f}")


if __name__ == "__main__":
    main()
