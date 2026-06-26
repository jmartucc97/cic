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

# Display bands for FinalRisk. PROVISIONAL â€” re-derive from validation data.
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


def fetch_works_meta(ref_ids, email, include_refs=False):
    """ref_ids: OpenAlex IDs. Returns dicts {id,doi,title,year[,referenced_works]}.
    include_refs=True also pulls each work's OWN reference list (needed to walk
    to depth 2)."""
    out = []
    short = [rid.rsplit("/", 1)[-1] for rid in ref_ids]
    select = "id,doi,title,publication_year"
    if include_refs:
        select += ",referenced_works"
    for i in range(0, len(short), 50):                  # OpenAlex filter cap
        chunk = short[i:i + 50]
        data = oa_get("works", {
            "filter": "ids.openalex:" + "|".join(chunk),
            "per-page": 50,
            "select": select,
        }, email)
        for w in data.get("results", []):
            rec = {
                "id": w.get("id"),
                "doi": norm_doi(w.get("doi")),
                "title": w.get("title") or "(untitled)",
                "year": w.get("publication_year"),
            }
            if include_refs:
                rec["referenced_works"] = w.get("referenced_works", []) or []
            out.append(rec)
    return out


# Back-compat alias (validate.py imports this name).
def fetch_reference_meta(ref_ids, email):
    return fetch_works_meta(ref_ids, email, include_refs=False)


# ==========================================================================
# depth-2 propagation
# ==========================================================================
DEPTH_DECAY = 0.35   # per-hop risk multiplier. PROVISIONAL â€” tune on validation.


def collect_depth2_targets(depth1_refs):
    """No network. Map each depth-2 OpenAlex id -> the depth-1 ref(s) that cite
    it (the contamination PATH). Returns (parents_map, ordered_id_list)."""
    parents = {}
    for r in depth1_refs:
        for d2 in r.get("referenced_works", []) or []:
            parents.setdefault(d2, [])
            if r not in parents[d2]:
                parents[d2].append(r)
    return parents, list(parents.keys())


def score_depth2(d2_meta, parents, root_doi, depth1_dois, rw_index, today=None):
    """No network. Given fetched depth-2 metadata, find retracted ones, attribute
    the path, decay the risk. Excludes anything already seen at depth 0/1 so
    each retraction is counted at its SHALLOWEST entry point only."""
    hits = []
    for w in d2_meta:
        doi = w["doi"]
        if not doi or doi == root_doi or doi in depth1_dois:
            continue                       # min-depth: already counted shallower
        rec = rw_index.get(doi)
        if not rec:
            continue
        tier, sev, matched = classify_severity(rec["reason"])
        rel = estimate_reliance(w)
        stal = staleness_factor(rec["date"], today=today)
        risk_raw = sev * rel * stal
        hits.append({
            "ref": w, "tier": tier, "severity": sev, "matched": matched,
            "reliance": rel, "staleness": stal,
            "risk_raw": risk_raw, "risk": risk_raw * DEPTH_DECAY,
            "reason": rec["reason"], "ret_date": rec["date"],
            "severe": tier >= SEVERE_TIER, "depth": 2,
            "via": parents.get(w["id"], []),
        })
    hits.sort(key=lambda h: (h["severe"], h["risk"]), reverse=True)
    parents_touched = {p["doi"] for h in hits for p in h["via"] if p.get("doi")}
    return {
        "hits": hits,
        "component": sum(h["risk"] for h in hits),
        "n_severe": sum(1 for h in hits if h["severe"]),
        "n_parents_touched": len(parents_touched),
    }


def traverse_depth2(depth1_refs, root_doi, rw_index, email, max_nodes, today=None):
    """Orchestrates: collect depth-2 ids -> fetch (capped) -> score."""
    parents, d2_ids = collect_depth2_targets(depth1_refs)
    truncated = len(d2_ids) > max_nodes
    d2_ids = d2_ids[:max_nodes]
    print("  depth-2: %d unique nodes to examine%s ..."
          % (len(d2_ids), " (capped)" if truncated else ""), file=sys.stderr)
    d2_meta = fetch_works_meta(d2_ids, email, include_refs=False) if d2_ids else []
    depth1_dois = {r["doi"] for r in depth1_refs if r["doi"]}
    res = score_depth2(d2_meta, parents, root_doi, depth1_dois, rw_index, today=today)
    res["n_nodes"] = len(d2_ids)
    res["truncated"] = truncated
    return res


# ==========================================================================
# reliance  (v1 = default; structural/LLM upgrade is v2)
# ==========================================================================
def estimate_reliance(_ref, full_text_ctx=None):
    """
    v1: we do NOT have in-text citation context from OpenAlex, so we cannot
    tell load-bearing from incidental. Default to 1.0 (treat as fully relied
    upon) â€” conservative: it can only OVER-state risk, never hide it.

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
def print_ledger(title, doi, result, depth2=None):
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
    print("  depth-1 FinalRisk:  %.3f   [%s]" % (final, band(final)))

    # ---- entry profile (where does contamination enter the citation tree?) ----
    if depth2 is not None:
        d2 = depth2
        combined = final + d2["component"]
        print("-" * 74)
        print("CONTAMINATION ENTRY PROFILE (where retracted work enters):")
        print("  depth 1 (cited directly):     %d retracted  (%d misconduct)"
              % (len(result["hits"]), result["n_severe"]))
        print("  depth 2 (via your sources):   %d retracted  (%d misconduct), "
              "reached through %d of your direct refs%s"
              % (len(d2["hits"]), d2["n_severe"], d2["n_parents_touched"],
                 "  [search capped]" if d2.get("truncated") else ""))
        print("  depth-2 risk (decayed x%.2f):  %.3f" % (DEPTH_DECAY, d2["component"]))
        print("  COMBINED RISK:  %.3f   [%s]" % (combined, band(combined)))
    print("=" * 74)

    # ---- depth-1 ledger ----
    if not result["hits"] and (depth2 is None or not depth2["hits"]):
        print("No retracted references found at any examined depth. (Clean.)")
        return
    if result["hits"]:
        if result["n_severe"]:
            print("âš  EGREGIOUS (depth 1): cites %d retraction(s) for misconduct."
                  % result["n_severe"])
            print("-" * 74)
        print("DEPTH 1 â€” directly cited retractions:")
        for h in result["hits"]:
            _print_hit(h)
    else:
        print("DEPTH 1 â€” no directly cited retractions.")

    # ---- depth-2 ledger with PATH attribution ----
    if depth2 is not None and depth2["hits"]:
        print("")
        print("-" * 74)
        print("DEPTH 2 â€” retractions reached THROUGH your references (decayed):")
        for h in depth2["hits"]:
            _print_hit(h, decayed=True)
            for p in h["via"][:3]:
                print("        â†³ enters via your ref: [%s] %s"
                      % (p.get("doi") or "?", _trunc(p.get("title"), 50)))
            if len(h["via"]) > 3:
                print("        â†³ ...and %d more of your refs cite it"
                      % (len(h["via"]) - 3))

    print("")
    print("-" * 74)
    print("NOTE: risk = attention flag, not a verdict. reliance is a v1 default")
    print("(1.0), so neither depth can yet tell load-bearing from incidental â€”")
    print("both score full. Depth-2 findings are EXPLORATORY: the entry attribution")
    print("(which ref a retraction enters through) is reliable, but the decayed")
    print("risk weight is provisional until the reliance module lands.")


def _print_hit(h, decayed=False):
    ref = h["ref"]
    yr = ref.get("year") or "????"
    rd = h["ret_date"].isoformat() if h["ret_date"] else "date?"
    tag = "  Â«MISCONDUCTÂ»" if h["severe"] else "  (minor)"
    print("")
    print("  â€¢ [%s, %s]  %s%s" % (yr, ref["doi"], _trunc(ref["title"], 50), tag))
    print("      retracted %s  |  reason: %s" % (rd, h["reason"] or "(none)"))
    print("      tier %d (sev %.2f, '%s')  x  reliance %.2f  x  staleness %.2f"
          % (h["tier"], h["severity"], h["matched"], h["reliance"], h["staleness"]))
    if decayed:
        print("      => raw %.3f  x decay %.2f  => contribution %.3f"
              % (h["risk_raw"], DEPTH_DECAY, h["risk"]))
    else:
        print("      => risk contribution %.3f" % h["risk"])


def _trunc(s, n):
    s = s or ""
    return s if len(s) <= n else s[:n - 1] + "â€¦"


# ==========================================================================
# demo data (no network) â€” proves scoring + ledger end to end
# ==========================================================================
def demo(depth=1):
    today = date(2026, 6, 26)
    # depth-1 refs; two of them (R_a, R_b) themselves cite a depth-2 retraction
    references = [
        {"id": "https://openalex.org/Wa", "doi": "10.1/fabricated",
         "title": "æ ¸ A miracle assay protocol", "year": 2016, "referenced_works": []},
        {"id": "https://openalex.org/Wb", "doi": "10.2/authordispute",
         "title": "Useful background review", "year": 2018,
         "referenced_works": ["https://openalex.org/Wdeep1"]},
        {"id": "https://openalex.org/Wc", "doi": "10.3/notreproducible",
         "title": "Prior finding we build on", "year": 2017, "referenced_works": []},
        {"id": "https://openalex.org/Wd", "doi": "10.4/clean",
         "title": "A perfectly fine paper", "year": 2020,
         "referenced_works": ["https://openalex.org/Wdeep1", "https://openalex.org/Wok"]},
        {"id": "https://openalex.org/We", "doi": "10.5/imagemanip",
         "title": "Recently retracted blots", "year": 2022, "referenced_works": []},
    ] + [{"id": "https://openalex.org/Wf%d" % i, "doi": "10.9/ok%d" % i,
          "title": "filler ref %d" % i, "year": 2019, "referenced_works": []}
         for i in range(45)]

    rw_index = {
        "10.1/fabricated": {"date": date(2021, 1, 1),
                            "reason": "+Fabrication of Data;+Investigation by Journal"},
        "10.2/authordispute": {"date": date(2024, 1, 1),
                               "reason": "+Authorship/Affiliation Disputes"},
        "10.3/notreproducible": {"date": date(2023, 6, 1),
                                 "reason": "+Results Not Reproducible;+Concerns About Data"},
        "10.5/imagemanip": {"date": date(2026, 3, 1),
                            "reason": "+Manipulation of Images"},
        # depth-2: a fabricated paper reached only THROUGH two clean-looking refs
        "10.7/deepfraud": {"date": date(2020, 1, 1),
                           "reason": "+Falsification/Fabrication of Data"},
    }
    result = score_paper(references, rw_index, today=today)

    depth2 = None
    if depth >= 2:
        # simulate the fetch of depth-2 nodes (no network in demo)
        d2_meta = [
            {"id": "https://openalex.org/Wdeep1", "doi": "10.7/deepfraud",
             "title": "The fabricated foundation everyone trusted", "year": 2015},
            {"id": "https://openalex.org/Wok", "doi": "10.8/fine",
             "title": "A clean deep reference", "year": 2014},
        ]
        parents, _ = collect_depth2_targets(references)
        depth1_dois = {r["doi"] for r in references if r["doi"]}
        depth2 = score_depth2(d2_meta, parents, "10.0/demo", depth1_dois,
                              rw_index, today=today)
        depth2["n_nodes"] = 2
        depth2["truncated"] = False

    print_ledger("DEMO â€” paper citing 4 retractions directly + 1 two hops down",
                 "10.0/demo", result, depth2=depth2)


# ==========================================================================
# main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="Retraction-citation risk scorer")
    ap.add_argument("doi", nargs="?", help="DOI of the paper to score")
    ap.add_argument("--rw-csv", help="Path to Retraction Watch CSV dump")
    ap.add_argument("--email", help="Your email (OpenAlex polite pool)")
    ap.add_argument("--depth", type=int, default=1, choices=[1, 2],
                    help="1 = direct citations only (fast). 2 = also walk one hop "
                         "deeper to find retractions reached THROUGH your refs.")
    ap.add_argument("--max-nodes", type=int, default=5000,
                    help="Cap on depth-2 nodes examined (controls API load).")
    ap.add_argument("--demo", action="store_true", help="Built-in mock data, no network")
    args = ap.parse_args()

    if args.demo:
        demo(depth=args.depth)
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

    need_refs = args.depth >= 2
    references = (fetch_works_meta(ref_ids, args.email, include_refs=need_refs)
                 if ref_ids else [])
    result = score_paper(references, rw_index)

    depth2 = None
    if args.depth >= 2 and references:
        depth2 = traverse_depth2(references, doi, rw_index, args.email,
                                 args.max_nodes)

    print_ledger(title, doi, result, depth2=depth2)


if __name__ == "__main__":
    main()

