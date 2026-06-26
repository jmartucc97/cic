#!/usr/bin/env python3
"""
retraction_risk.py  --  v1

Scores how much a paper's foundations are at risk because it cites RETRACTED work.

Pipeline:
  1. Take a DOI.
  2. Fetch the work + its reference list from OpenAlex.
  3. Cross-check every reference DOI against the Retraction Watch database.
  4. For each retracted reference, compute:
        severity  (why it was retracted -> 4-tier ladder)
      x reliance  (how load-bearing the citation is; v1 = structural/default)
      x staleness (how long it's been retracted-and-unaddressed)
  5. Sum -> RawRisk, normalize by sqrt(#refs) -> AdjRisk.
  6. Print an ITEMIZED LEDGER (the scalar is for ranking; the ledger is the product).

DESIGN NOTES (read these before trusting the number):
  - severity/reliance/staleness MULTIPLY. A score only spikes when a SERIOUS
    retraction is RELIED UPON. Harmless retractions (authorship disputes, etc.)
    stay near zero even when load-bearing. That is intentional.
  - All coefficients (0.4, 0.8, the sqrt, the staleness slope) are PROVISIONAL
    defaults. Do NOT defend them. Tune them against a validation set later.
  - Output is RISK / ATTENTION, never a verdict. We flag for human review and
    quote the evidence; we do not pronounce a paper "unreliable".

NETWORK: live mode needs api.openalex.org and the Retraction Watch CSV.
Run `--demo` to exercise the full scoring + ledger on built-in mock data with
no network at all.
"""

import argparse
import csv
import math
import sys
from datetime import date, datetime

import requests

# --------------------------------------------------------------------------
# SEVERITY LADDER
# Maps Retraction Watch `Reason` substrings -> (tier, severity).
# Checked HIGHEST-tier-first; a retraction's severity = its WORST reason (max).
# Reason strings in RW are ';'-delimited and often '+'-prefixed, e.g.
#   "+Manipulation of Images;+Investigation by Journal;+Concerns About Data"
# --------------------------------------------------------------------------
SEVERITY_LADDER = [
    # (tier, severity, [lowercase substrings that imply this tier])
    (3, 1.00, [
        "fabrication of data",
        "fabrication of results",
        "fabrication of image",
    ]),
    (2, 0.80, [
        "falsification",
        "manipulation of data",
        "manipulation of results",
        "manipulation of images",
        "fake peer review",
        "paper mill",
        "plagiarism of data",
        "plagiarism of results",
        "fabrication",            # generic fabrication -> still severe
    ]),
    (1, 0.40, [
        "results not reproducible",
        "unreliable results",
        "unreliable data",
        "error in data",
        "error in analyses",
        "error in results",
        "error in methods",
        "error in materials",
        "concerns about data",
        "concerns/issues about data",
        "concerns about results",
        "concerns/issues about results",
        "unreliable",
        "plagiarism",             # generic plagiarism -> text issue, data may be fine
    ]),
    (0, 0.10, [
        "authorship",
        "duplication of article",
        "duplication of text",
        "duplication of image",   # self-duplication of own figure
        "copyright",
        "legal",
        "breach of policy",
        "conflict of interest",
        "lack of irb",
        "lack of iacuc",
        "ethical",                # ethics/consent: data not necessarily false
        "withdrawn",
        "publisher error",
        "no information",
        "removed",
    ]),
]

DEFAULT_TIER = 1          # unmapped reason -> cautious middle, never 0 or max
DEFAULT_SEVERITY = 0.40

# Tier at/above which a retraction counts as MISCONDUCT and is scored UNDILUTED.
# Tier 2 = manipulation/falsification/paper-mill, Tier 3 = fabrication.
# Below this (honest error, authorship, duplication) is density-normalized.
SEVERE_TIER = 2

# Display bands for FinalRisk. PROVISIONAL — re-derive from validation data.
# Calibrated so ONE load-bearing fabrication (Tier 3) lands in HIGH on its own,
# and ONE manipulation (Tier 2) lands at least MODERATE.
RISK_BANDS = [(1.2, "HIGH"), (0.5, "MODERATE"), (0.0, "LOW")]

OPENALEX = "https://api.openalex.org"

# Retraction Watch full dump (CSV), distributed free via Crossref.
# Download once, pass with --rw-csv. Get it at:
#   https://gitlab.com/crossref/retraction-watch-data  (or Crossref Labs)
RW_ORIGINAL_DOI_COL = "OriginalPaperDOI"
RW_DATE_COL = "RetractionDate"
RW_REASON_COL = "Reason"
RW_NATURE_COL = "RetractionNature"   # keep only "Retraction" (skip EoC/Correction)


# ==========================================================================
# helpers
# ==========================================================================
def norm_doi(doi):
    """Lowercase, strip URL/scheme so DOIs compare cleanly across sources."""
    if not doi:
        return None
    doi = doi.strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:"):
        if doi.startswith(p):
            doi = doi[len(p):]
    return doi or None


def classify_severity(reason_str):
    """Return (tier, severity, matched_reason). Worst reason wins."""
    if not reason_str:
        return DEFAULT_TIER, DEFAULT_SEVERITY, "(no reason given)"
    text = reason_str.lower()
    for tier, sev, needles in SEVERITY_LADDER:      # highest tier first
        for n in needles:
            if n in text:
                return tier, sev, n
    return DEFAULT_TIER, DEFAULT_SEVERITY, "(unmapped: %s)" % reason_str.strip()


def staleness_factor(retraction_dt, today=None):
    """1.0 (just retracted) -> 1.5 (long retracted, unaddressed). Capped."""
    if retraction_dt is None:
        return 1.0
    today = today or date.today()
    years = (today - retraction_dt).days / 365.25
    if years < 0:
        years = 0
    return min(1.0 + 0.10 * years, 1.5)


def parse_rw_date(s):
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s.split(" ")[0] if " " in s and ":" in s else s, fmt).date()
        except ValueError:
            continue
    return None


def band(adj):
    for thresh, label in RISK_BANDS:
        if adj >= thresh:
            return label
    return "LOW"


# ==========================================================================
# Retraction Watch index
# ==========================================================================
def load_retraction_index(csv_path):
    """Build {normalized_original_doi: {date, reason}} from the RW CSV."""
    index = {}
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nature = (row.get(RW_NATURE_COL) or "").strip().lower()
            if nature and "retraction" not in nature:
                continue  # skip Expressions of Concern / Corrections for v1
            doi = norm_doi(row.get(RW_ORIGINAL_DOI_COL) if False else row.get(RW_ORIGINAL_DOI_COL))
            if not doi:
                continue
            index[doi] = {
                "date": parse_rw_date(row.get(RW_DATE_COL)),
                "reason": (row.get(RW_REASON_COL) or "").strip(),
            }
    return index


# ==========================================================================
# OpenAlex
# ==========================================================================
def oa_get(path, params, email):
    params = dict(params or {})
    if email:
        params["mailto"] = email
    r = requests.get("%s/%s" % (OPENALEX, path), params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_work(doi, email):
    data = oa_get("works/https://doi.org/%s" % doi, {}, email)
    return data


def fetch_reference_meta(ref_ids, email):
    """ref_ids: list of OpenAlex IDs. Returns list of dicts {doi,title,year}."""
    out = []
    short = [rid.rsplit("/", 1)[-1] for rid in ref_ids]
    for i in range(0, len(short), 50):                  # OpenAlex filter cap
        chunk = short[i:i + 50]
        data = oa_get("works", {
            "filter": "ids.openalex:" + "|".join(chunk),
            "per-page": 50,
            "select": "id,doi,title,publication_year",
        }, email)
        for w in data.get("results", []):
            out.append({
                "doi": norm_doi(w.get("doi")),
                "title": w.get("title") or "(untitled)",
                "year": w.get("publication_year"),
            })
    return out


# ==========================================================================
# reliance  (v1 = default; structural/LLM upgrade is v2)
# ==========================================================================
def estimate_reliance(_ref, full_text_ctx=None):
    """
    v1: we do NOT have in-text citation context from OpenAlex, so we cannot
    tell load-bearing from incidental. Default to 1.0 (treat as fully relied
    upon) — conservative: it can only OVER-state risk, never hide it.

    v2 plan: fetch JATS XML (Europe PMC OA), locate <xref ref-type="bibr">
    for this reference, read its section + co-citation density, and grade:
      Methods / cited-alone  -> ~0.9
      Results-Discussion support -> ~0.6
      Intro background clump  -> ~0.2
    Then replace the heuristic with an LLM classifier (quote the sentence).
    """
    if full_text_ctx is not None:
        return full_text_ctx  # hook for v2
    return 1.0


# ==========================================================================
# scoring
# ==========================================================================
def score_paper(references, rw_index, today=None):
    hits = []
    for ref in references:
        if not ref["doi"]:
            continue
        rec = rw_index.get(ref["doi"])
        if not rec:
            continue
        tier, sev, matched = classify_severity(rec["reason"])
        rel = estimate_reliance(ref)
        stal = staleness_factor(rec["date"], today=today)
        risk = sev * rel * stal
        hits.append({
            "ref": ref, "tier": tier, "severity": sev, "matched": matched,
            "reliance": rel, "staleness": stal, "risk": risk,
            "reason": rec["reason"], "ret_date": rec["date"],
            "severe": tier >= SEVERE_TIER,
        })
    n_refs = max(len(references), 1)

    # Two-component score:
    #   severe (misconduct) hits  -> summed UNDILUTED. One fabrication has full,
    #       reference-count-independent bearing; two are twice as bad.
    #   minor hits (honest error, authorship, duplication) -> density-normalized,
    #       so a long bibliography of harmless retractions can't manufacture alarm.
    severe_sum = sum(h["risk"] for h in hits if h["severe"])
    minor_sum = sum(h["risk"] for h in hits if not h["severe"])
    minor_component = minor_sum / math.sqrt(n_refs)
    final = severe_sum + minor_component

    raw = severe_sum + minor_sum  # undiluted total, for reference
    hits.sort(key=lambda h: (h["severe"], h["risk"]), reverse=True)
    return {
        "hits": hits, "n_refs": n_refs, "raw": raw,
        "severe_sum": severe_sum, "minor_component": minor_component,
        "final": final,
        "n_severe": sum(1 for h in hits if h["severe"]),
    }


# ==========================================================================
# output
# ==========================================================================
def print_ledger(title, doi, result):
    n = result["n_refs"]
    final = result["final"]
    print("=" * 74)
    print("PAPER:  %s" % (title or "(unknown title)"))
    print("DOI:    %s" % doi)
    print("Refs:   %d   |   Retracted refs: %d   |   of which misconduct-tier: %d"
          % (n, len(result["hits"]), result["n_severe"]))
    print("-" * 74)
    print("  misconduct component (undiluted):  %.3f" % result["severe_sum"])
    print("  minor component (/sqrt(%d)):        %.3f" % (n, result["minor_component"]))
    print("  FinalRisk:  %.3f   [%s]" % (final, band(final)))
    print("=" * 74)
    if not result["hits"]:
        print("No retracted references found. (Clean on this check.)")
        return
    if result["n_severe"]:
        print("⚠ EGREGIOUS: cites %d retraction(s) for misconduct (fabrication/"
              "manipulation). Each counts at full weight." % result["n_severe"])
        print("-" * 74)
    for h in result["hits"]:
        ref = h["ref"]
        yr = ref.get("year") or "????"
        rd = h["ret_date"].isoformat() if h["ret_date"] else "date?"
        tag = "  «MISCONDUCT, undiluted»" if h["severe"] else "  (minor, normalized)"
        print("")
        print("  • [%s, %s]  %s%s" % (yr, ref["doi"], _trunc(ref["title"], 52), tag))
        print("      retracted %s  |  reason: %s" % (rd, h["reason"] or "(none)"))
        print("      tier %d (sev %.2f, matched '%s')  x  reliance %.2f  x  staleness %.2f"
              % (h["tier"], h["severity"], h["matched"], h["reliance"], h["staleness"]))
        print("      => risk contribution %.3f" % h["risk"])
    print("")
    print("-" * 74)
    print("NOTE: risk = attention flag, not a verdict. reliance is a v1 default")
    print("(1.0), so v1 cannot yet tell a load-bearing misconduct cite from a")
    print("passing mention — both score full. That errs toward flagging; v2")
    print("reliance (full-text context) is what relaxes the incidental case.")


def _trunc(s, n):
    s = s or ""
    return s if len(s) <= n else s[:n - 1] + "…"


# ==========================================================================
# demo data (no network) — proves scoring + ledger end to end
# ==========================================================================
def demo():
    today = date(2026, 6, 26)
    references = [
        {"doi": "10.1/fabricated", "title": "核 A miracle assay protocol", "year": 2016},
        {"doi": "10.2/authordispute", "title": "Useful background review", "year": 2018},
        {"doi": "10.3/notreproducible", "title": "Prior finding we build on", "year": 2017},
        {"doi": "10.4/clean", "title": "A perfectly fine paper", "year": 2020},
        {"doi": "10.5/imagemanip", "title": "Recently retracted blots", "year": 2022},
    ] + [{"doi": "10.9/ok%d" % i, "title": "filler ref %d" % i, "year": 2019} for i in range(45)]

    rw_index = {
        "10.1/fabricated": {"date": date(2021, 1, 1),
                            "reason": "+Fabrication of Data;+Investigation by Journal"},
        "10.2/authordispute": {"date": date(2024, 1, 1),
                               "reason": "+Authorship/Affiliation Disputes"},
        "10.3/notreproducible": {"date": date(2023, 6, 1),
                                 "reason": "+Results Not Reproducible;+Concerns About Data"},
        "10.5/imagemanip": {"date": date(2026, 3, 1),
                            "reason": "+Manipulation of Images"},
    }
    result = score_paper(references, rw_index, today=today)
    print_ledger("DEMO — synthetic paper citing 4 retracted works (of 50 refs)",
                 "10.0/demo", result)


# ==========================================================================
# main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="Retraction-citation risk scorer (v1)")
    ap.add_argument("doi", nargs="?", help="DOI of the paper to score")
    ap.add_argument("--rw-csv", help="Path to Retraction Watch CSV dump")
    ap.add_argument("--email", help="Your email (OpenAlex polite pool)")
    ap.add_argument("--demo", action="store_true", help="Run on built-in mock data, no network")
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    if not args.doi or not args.rw_csv:
        ap.error("live mode needs a DOI and --rw-csv (or use --demo)")

    doi = norm_doi(args.doi)
    print("Loading Retraction Watch index from %s ..." % args.rw_csv, file=sys.stderr)
    rw_index = load_retraction_index(args.rw_csv)
    print("  %d retractions indexed." % len(rw_index), file=sys.stderr)

    print("Fetching %s from OpenAlex ..." % doi, file=sys.stderr)
    work = fetch_work(doi, args.email)
    ref_ids = work.get("referenced_works", []) or []
    title = work.get("title")
    print("  %d references found. Resolving DOIs ..." % len(ref_ids), file=sys.stderr)

    references = fetch_reference_meta(ref_ids, args.email) if ref_ids else []
    result = score_paper(references, rw_index)
    print_ledger(title, doi, result)


if __name__ == "__main__":
    main()
