#!/usr/bin/env python3
"""
validate.py  --  validation harness for the retraction-citation risk score.

QUESTION IT ANSWERS:
  Does FinalRisk actually predict anything? I.e. do papers that THEMSELVES got
  retracted tend to score higher (cite more/worse retracted work) than matched
  papers that did not?

DESIGN (case-control predictive validity):
  cases    = papers that were later retracted          (label 1)
  controls = matched papers not retracted              (label 0)
  We score each paper's reference list with score_paper(), then measure how well
  FinalRisk separates cases from controls:
     - ROC-AUC  : P(random case scores > random control). 0.5 = useless, 1.0 = perfect.
     - Mann-Whitney U + medians : distribution-level separation.
     - precision/recall at a chosen FinalRisk threshold.

CONFOUNDS TO CONTROL FOR (document, don't ignore):
  Retracted papers may differ from controls in year, field, and reference count
  for reasons unrelated to citing bad work. Controls MUST be matched on those, or
  the AUC is measuring "is this paper old/large", not "does it cite retractions".
  v1 takes user-supplied case/control DOI lists so YOU own the matching; the README
  describes how to build a matched set. Circularity guard: exclude cases whose own
  retraction reason is "cites retracted work".

INPUT (live):  a CSV with columns: doi,label   (label = 1 case / 0 control)
RUN OFFLINE:   --demo  synthesizes labeled scores and runs the full metric stack
               with no network, to prove the math.
"""

import argparse
import csv
import math
import random
import sys

try:
    from retraction_risk import (
        norm_doi, load_retraction_index, fetch_work, fetch_reference_meta, score_paper,
    )
except ImportError:
    # allow running the demo even if the core module path isn't set up yet
    norm_doi = load_retraction_index = fetch_work = fetch_reference_meta = score_paper = None


# ==========================================================================
# metrics  (no sklearn dependency — keep the repo light)
# ==========================================================================
def _ranks(values):
    """Average ranks (1-based), ties share the mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def mann_whitney_u(pos, neg):
    """U statistic for positives vs negatives (larger U => pos ranked higher)."""
    all_vals = pos + neg
    ranks = _ranks(all_vals)
    r_pos = sum(ranks[:len(pos)])
    n1, n2 = len(pos), len(neg)
    u_pos = r_pos - n1 * (n1 + 1) / 2.0
    return u_pos, n1, n2


def roc_auc(pos, neg):
    """AUC = P(random positive > random negative). Rank-based; handles ties."""
    if not pos or not neg:
        return float("nan")
    u_pos, n1, n2 = mann_whitney_u(pos, neg)
    return u_pos / (n1 * n2)


def precision_recall_at(pos, neg, threshold):
    tp = sum(1 for s in pos if s >= threshold)
    fn = len(pos) - tp
    fp = sum(1 for s in neg if s >= threshold)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    return prec, rec, tp, fp, fn


def median(xs):
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def histogram(scores, labels, bins=10, width=40):
    """Tiny text histogram of case vs control score distributions."""
    if not scores:
        return ""
    lo, hi = min(scores), max(scores)
    if hi == lo:
        hi = lo + 1e-9
    out = []
    for b in range(bins):
        b_lo = lo + (hi - lo) * b / bins
        b_hi = lo + (hi - lo) * (b + 1) / bins
        c = sum(1 for s, l in zip(scores, labels) if l == 1 and b_lo <= s < b_hi + (1e-9 if b == bins-1 else 0))
        k = sum(1 for s, l in zip(scores, labels) if l == 0 and b_lo <= s < b_hi + (1e-9 if b == bins-1 else 0))
        bar = "#" * min(c, width) + "." * min(k, width)
        out.append("  %6.2f-%6.2f | %s  (case %d / ctrl %d)" % (b_lo, b_hi, bar, c, k))
    return "\n".join(out)


# ==========================================================================
# report
# ==========================================================================
def report(scored):
    """scored: list of (doi, final_risk, label)."""
    pos = [s for _, s, l in scored if l == 1]
    neg = [s for _, s, l in scored if l == 0]
    auc = roc_auc(pos, neg)
    u, n1, n2 = mann_whitney_u(pos, neg) if pos and neg else (float("nan"), len(pos), len(neg))

    print("=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)
    print("cases (retracted):   %d   median FinalRisk %.3f" % (len(pos), median(pos)))
    print("controls:            %d   median FinalRisk %.3f" % (len(neg), median(neg)))
    print("-" * 70)
    print("ROC-AUC:             %.3f   (0.5 = no signal, 1.0 = perfect)" % auc)
    print("Mann-Whitney U:      %.0f" % u)
    print("-" * 70)
    print("precision/recall at FinalRisk thresholds:")
    for thr in (0.5, 1.2, 2.0):
        p, r, tp, fp, fn = precision_recall_at(pos, neg, thr)
        print("  thr %.2f :  precision %.3f  recall %.3f   (tp %d fp %d fn %d)"
              % (thr, p, r, tp, fp, fn))
    print("-" * 70)
    print("score distribution (# case, . control):")
    print(histogram([s for _, s, _ in scored], [l for _, _, l in scored]))
    print("-" * 70)
    if not math.isnan(auc):
        if auc >= 0.70:
            verdict = "real signal — the score discriminates. Worth tuning + publishing."
        elif auc >= 0.58:
            verdict = "weak signal — present but modest; investigate confounds / reliance."
        else:
            verdict = "little/no signal — likely needs reliance (v2) before it predicts."
        print("READ: AUC %.3f -> %s" % (auc, verdict))
    print("NOTE: AUC reflects your case/control MATCHING. If controls aren't matched")
    print("on year/field/ref-count, this measures confounds, not the hypothesis.")


# ==========================================================================
# live scoring
# ==========================================================================
def score_doi(doi, rw_index, email):
    work = fetch_work(doi, email)
    refs = fetch_reference_meta(work.get("referenced_works", []) or [], email)
    return score_paper(refs, rw_index)["final"]


def run_live(labels_csv, rw_csv, email):
    rw_index = load_retraction_index(rw_csv)
    print("Indexed %d retractions." % len(rw_index), file=sys.stderr)
    scored = []
    with open(labels_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows, 1):
        doi = norm_doi(row["doi"])
        label = int(row["label"])
        try:
            fr = score_doi(doi, rw_index, email)
            scored.append((doi, fr, label))
            print("  [%d/%d] %s  label=%d  FinalRisk=%.3f"
                  % (i, len(rows), doi, label, fr), file=sys.stderr)
        except Exception as e:                       # noqa: keep harness robust
            print("  [%d/%d] %s  FAILED: %s" % (i, len(rows), doi, e), file=sys.stderr)
    report(scored)


# ==========================================================================
# demo  (no network) — proves the metric stack
# ==========================================================================
def demo():
    random.seed(7)
    scored = []
    # cases: papers that got retracted -> skewed toward higher risk, lots of overlap
    for i in range(120):
        base = random.random()
        fr = max(0.0, random.gauss(0.9, 0.8)) if base < 0.55 else max(0.0, random.gauss(0.1, 0.2))
        scored.append(("10.case/%d" % i, fr, 1))
    # controls: mostly low risk, occasional incidental hit
    for i in range(120):
        fr = max(0.0, random.gauss(0.15, 0.3)) if random.random() < 0.85 else max(0.0, random.gauss(0.8, 0.6))
        scored.append(("10.ctrl/%d" % i, fr, 0))
    report(scored)


def main():
    ap = argparse.ArgumentParser(description="Validate the retraction-risk score")
    ap.add_argument("--labels", help="CSV with columns: doi,label (1=case,0=control)")
    ap.add_argument("--rw-csv", help="Retraction Watch CSV dump")
    ap.add_argument("--email", help="email for OpenAlex polite pool")
    ap.add_argument("--demo", action="store_true", help="synthetic data, no network")
    args = ap.parse_args()

    if args.demo:
        demo()
        return
    if not (args.labels and args.rw_csv):
        ap.error("live mode needs --labels and --rw-csv (or use --demo)")
    if score_paper is None:
        ap.error("could not import retraction_risk.py — keep both files together")
    run_live(args.labels, args.rw_csv, args.email)


if __name__ == "__main__":
    main()
