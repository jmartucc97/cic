#!/usr/bin/env python3
"""
citation_context.py  --  citation STANCE + RELIANCE analysis (rules-based v1)

Answers, for a paper that cites a retracted work: does the citing paper
BUILD ON it, merely MENTION it, or DISPUTE / REFUTE it? A paper that refutes a
retraction is science's immune system working — it should NOT be penalised the
way a paper that relies on the fraud is.

PIPELINE (all four steps live here; only step 4 is "the rules"):
  1. find_fulltext()            fetch JATS XML from Europe PMC (open-access only)
  2. extract_citation_contexts() locate the in-text citation(s) to the target,
                                 pull the surrounding sentence + section
  3. classify_context()         rules-based stance + reliance from cue phrases
  4. analyze_citation()         orchestrate + aggregate across all citations

This is a BASELINE. Keyword rules are wrong a lot (negation, distance, hedging),
and exist to (a) prove the plumbing and (b) give the eventual LLM classifier a
number to beat. Swapping in an LLM is a one-function change at classify_context.

COVERAGE LIMIT: full text exists only for open-access papers (Europe PMC / PMC /
bioRxiv). Paywalled paper -> no context -> stance "unknown" -> caller falls back
to its conservative default. That's most of the real-world coverage gap; no
classifier fixes it.

CLI:
  python citation_context.py --citing 10.x/paper --target 10.1038/nature04533 --email you@x.com
  python citation_context.py --demo      # offline: parsing + rules on sample JATS
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET

import requests

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"


# ==========================================================================
# small helpers
# ==========================================================================
def norm_doi(doi):
    if not doi:
        return None
    doi = doi.strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:"):
        if doi.startswith(p):
            doi = doi[len(p):]
    return doi or None


def _local(tag):
    """Strip XML namespace: '{ns}ref' -> 'ref'."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _localize_tree(root):
    for el in root.iter():
        el.tag = _local(el.tag)
    return root


def _norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_sentences(text):
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


# ==========================================================================
# 1. fetch full text (Europe PMC, open access only)  [network]
# ==========================================================================
def find_fulltext(doi, email=None):
    """Return (jats_xml_str, pmcid) or (None, reason)."""
    try:
        r = requests.get("%s/search" % EPMC, params={
            "query": 'DOI:"%s"' % doi, "format": "json", "resultType": "core",
        }, timeout=30)
        r.raise_for_status()
        results = r.json().get("resultList", {}).get("result", [])
    except Exception as e:                                   # noqa
        return None, "search failed: %s" % e
    if not results:
        return None, "not indexed in Europe PMC"
    rec = results[0]
    pmcid = rec.get("pmcid")
    is_oa = rec.get("isOpenAccess")
    if not pmcid:
        return None, "no PMC full text (likely paywalled)"
    if is_oa != "Y":
        return None, "indexed but not open-access"
    try:
        r = requests.get("%s/PMC/%s/fullTextXML" % (EPMC, pmcid), timeout=30)
        r.raise_for_status()
        return r.text, pmcid
    except Exception as e:                                   # noqa
        return None, "fulltext fetch failed: %s" % e


# ==========================================================================
# 2. locate the citation(s) to the target & pull context  [offline-testable]
# ==========================================================================
def _build_parent_map(root):
    return {c: p for p in root.iter() for c in p}


def _match_target_ref_ids(root, target_doi, target_title):
    """Find <ref> elements whose citation is the target; return their id attrs."""
    ids = []
    tnorm = _norm_title(target_title)
    for ref in root.iter("ref"):
        ref_id = ref.get("id")
        if not ref_id:
            continue
        hit = False
        for pid in ref.iter("pub-id"):
            if pid.get("pub-id-type") == "doi" and norm_doi(pid.text) == target_doi:
                hit = True
                break
        if not hit and tnorm:
            for at in ref.iter("article-title"):
                if at.text and _norm_title("".join(at.itertext())).startswith(tnorm[:40]):
                    hit = True
                    break
        if hit:
            ids.append(ref_id)
    return set(ids)


def _block_text_with_marker(block, target_xref):
    """Serialize a block's text, inserting [[CITE]] where target_xref sits."""
    parts = []

    def walk(el):
        if el is target_xref:
            parts.append(" [[CITE]] ")
        if el.text:
            parts.append(el.text)
        for child in el:
            walk(child)
        if el.tail:
            parts.append(el.tail)

    walk(block)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _section_of(el, parent_map):
    """Walk up to the nearest <sec>; return its <title> text (e.g. 'Methods')."""
    cur = parent_map.get(el)
    while cur is not None:
        if _local(cur.tag) == "sec":
            for ch in cur:
                if _local(ch.tag) == "title":
                    return "".join(ch.itertext()).strip()
            return "(untitled section)"
        cur = parent_map.get(cur)
    return None


def extract_citation_contexts(jats_xml, target_doi, target_title=""):
    """Return list of {sentence, section, cocitations} for each in-text citation
    of the target. Pure parsing — no network."""
    target_doi = norm_doi(target_doi)
    root = _localize_tree(ET.fromstring(jats_xml))
    ref_ids = _match_target_ref_ids(root, target_doi, target_title)
    if not ref_ids:
        return []   # target not found in this paper's reference list
    parent_map = _build_parent_map(root)

    contexts = []
    for xref in root.iter("xref"):
        if xref.get("ref-type") != "bibr":
            continue
        rids = set((xref.get("rid") or "").split())
        if not (rids & ref_ids):
            continue
        # containing block = nearest <p>
        block = parent_map.get(xref)
        while block is not None and _local(block.tag) != "p":
            block = parent_map.get(block)
        if block is None:
            continue
        text = _block_text_with_marker(block, xref)
        sentence = next((s for s in split_sentences(text) if "[[CITE]]" in s), text)
        sentence = sentence.replace("[[CITE]]", "").strip()
        cocites = sum(1 for x in block.iter("xref") if x.get("ref-type") == "bibr")
        contexts.append({
            "sentence": sentence,
            "section": _section_of(xref, parent_map) or "(unknown)",
            "cocitations": cocites,
        })
    return contexts


# ==========================================================================
# 3. classify — RULES.  Swap this one function for an LLM later. [offline]
# ==========================================================================
DISPUTE_CUES = [
    "contrary to", "in contrast to", "contrast with", "unlike",
    "failed to replicate", "could not replicate", "did not replicate",
    "unable to replicate", "failed to reproduce", "could not reproduce",
    "did not reproduce", "inconsistent with", "we found no", "found no evidence",
    "no evidence", "challenges", "disputes", "contradicts", "contradict",
    "refute", "refutes", "questioned", "calls into question", "call into question",
    "at odds with", "does not support", "do not support", "however",
    "in disagreement", "fails to", "cannot account",
]
ENDORSE_CUES = [
    "based on", "building on", "build on", "builds on", "following the",
    "as described by", "as shown by", "as demonstrated by", "as reported by",
    "consistent with", "in agreement with", "in line with", "supports",
    "confirms", "corroborat", "extends", "extending", "we adopted",
    "we used the method", "using the method", "according to",
    "in accordance with", "similar to", "as established by", "drawing on",
]
METHODS_SECTIONS = ("method", "material", "procedure", "experimental")


def _scan(text, cues):
    t = text.lower()
    return [c for c in cues if c in t]


def classify_context(ctx):
    """Rules-based stance + reliance for one citation context."""
    sent = ctx["sentence"]
    disp = _scan(sent, DISPUTE_CUES)
    endo = _scan(sent, ENDORSE_CUES)

    # stance: dispute wins if present and not clearly outweighed by endorse
    if disp and not endo:
        stance = "dispute"
    elif endo and not disp:
        stance = "endorse"
    elif disp and endo:
        stance = "mixed"          # hedged; treat cautiously (lean neutral)
    else:
        stance = "neutral"

    # reliance heuristic (0..1)
    rel = 0.5
    sec = (ctx.get("section") or "").lower()
    if any(m in sec for m in METHODS_SECTIONS):
        rel = 0.9                 # cited in Methods => likely load-bearing
    if ctx.get("cocitations", 1) <= 1:
        rel = min(1.0, rel + 0.2) # cited alone => more load-bearing
    elif ctx.get("cocitations", 1) >= 5:
        rel = max(0.2, rel - 0.3) # big co-citation clump => background
    if endo:
        rel = min(1.0, rel + 0.1)

    return {
        "stance": stance,
        "reliance": round(rel, 2),
        "dispute_cues": disp,
        "endorse_cues": endo,
        "evidence": sent,
        "section": ctx.get("section"),
    }


# stance -> how much the citation should still count toward contamination risk
STANCE_RISK_MULT = {
    "dispute": 0.1,   # refuting a retraction is the immune response, not reliance
    "mixed": 0.6,
    "neutral": 1.0,
    "endorse": 1.0,
    "unknown": 1.0,   # no full text -> conservative default
}


# ==========================================================================
# 4. orchestrate + aggregate  [network]
# ==========================================================================
def analyze_citation(citing_doi, target_doi, target_title="", email=None):
    """Full pipeline for one (citing -> target) pair. Returns a summary dict."""
    citing_doi, target_doi = norm_doi(citing_doi), norm_doi(target_doi)
    xml, info = find_fulltext(citing_doi, email)
    if xml is None:
        return {"available": False, "reason": info, "stance": "unknown",
                "reliance": 1.0, "risk_multiplier": 1.0, "contexts": []}
    try:
        contexts = extract_citation_contexts(xml, target_doi, target_title)
    except ET.ParseError as e:
        return {"available": False, "reason": "XML parse error: %s" % e,
                "stance": "unknown", "reliance": 1.0,
                "risk_multiplier": 1.0, "contexts": []}
    if not contexts:
        return {"available": True, "reason": "citation not located in full text",
                "stance": "unknown", "reliance": 1.0,
                "risk_multiplier": 1.0, "contexts": []}

    classed = [classify_context(c) for c in contexts]
    stances = [c["stance"] for c in classed]
    # aggregate: dispute (and not endorsed elsewhere) dominates; else endorse; else neutral
    if "endorse" in stances:
        agg = "endorse"
    elif "dispute" in stances:
        agg = "dispute"
    elif "mixed" in stances:
        agg = "mixed"
    else:
        agg = "neutral"
    reliance = max(c["reliance"] for c in classed)
    return {
        "available": True, "reason": None,
        "stance": agg, "reliance": reliance,
        "risk_multiplier": STANCE_RISK_MULT[agg],
        "contexts": classed,
        "pmcid": info,
    }


# ==========================================================================
# pretty-print + CLI
# ==========================================================================
def print_analysis(citing, target, res):
    print("=" * 70)
    print("CITATION CONTEXT ANALYSIS")
    print("  citing: %s" % citing)
    print("  target (retracted): %s" % target)
    print("-" * 70)
    if not res["available"]:
        print("  No analysis: %s" % res["reason"])
        print("  -> stance UNKNOWN; caller uses conservative default (full risk).")
        return
    if not res["contexts"]:
        print("  Full text found, but %s." % res["reason"])
        return
    print("  AGGREGATE STANCE:  %s   (risk x%.2f)"
          % (res["stance"].upper(), res["risk_multiplier"]))
    print("  reliance: %.2f   |   %d citation instance(s) found"
          % (res["reliance"], len(res["contexts"])))
    print("-" * 70)
    for i, c in enumerate(res["contexts"], 1):
        print("  [%d] %s  (section: %s)" % (i, c["stance"].upper(), c["section"]))
        if c["dispute_cues"]:
            print("      dispute cues: %s" % ", ".join(c["dispute_cues"]))
        if c["endorse_cues"]:
            print("      endorse cues: %s" % ", ".join(c["endorse_cues"]))
        print("      \"%s\"" % _trunc(c["evidence"], 200))
        print("")
    print("  NOTE: rules-based baseline. Cues miss negation/distance/hedging;")
    print("  treat as a flag to verify, not a verdict. Stance only available for")
    print("  open-access full text.")


def _trunc(s, n):
    s = s or ""
    return s if len(s) <= n else s[:n - 1] + "…"


# --------------------------------------------------------------------------
SAMPLE_JATS = """<article>
<body>
<sec><title>Introduction</title>
<p>Soluble amyloid oligomers have been proposed as drivers of memory loss
(see <xref ref-type="bibr" rid="r1">1</xref>, <xref ref-type="bibr" rid="r2">2</xref>,
<xref ref-type="bibr" rid="r3">3</xref>).</p></sec>
<sec><title>Results</title>
<p>Contrary to <xref ref-type="bibr" rid="r1">1</xref>, we found no evidence that
this specific assembly impairs memory in our model.</p>
<p>We adopted the protocol established by <xref ref-type="bibr" rid="r4">4</xref>
to prepare the samples.</p></sec>
</body>
<back><ref-list>
<ref id="r1"><element-citation><article-title>A specific amyloid-beta protein
assembly in the brain impairs memory</article-title>
<pub-id pub-id-type="doi">10.1038/nature04533</pub-id></element-citation></ref>
<ref id="r4"><element-citation><article-title>Some methods paper</article-title>
<pub-id pub-id-type="doi">10.1/methods</pub-id></element-citation></ref>
</ref-list></back>
</article>"""


def demo():
    print("OFFLINE DEMO — parsing + rules on a built-in JATS sample")
    print("(target = the retracted Lesné 2006 amyloid paper, 10.1038/nature04533)\n")
    ctxs = extract_citation_contexts(SAMPLE_JATS, "10.1038/nature04533",
                                     "A specific amyloid-beta protein assembly")
    print("Located %d citation instance(s) of the target:\n" % len(ctxs))
    for c in ctxs:
        cls = classify_context(c)
        print("  section: %s | cocitations: %d" % (c["section"], c["cocitations"]))
        print("  -> stance %s, reliance %.2f" % (cls["stance"].upper(), cls["reliance"]))
        print("     \"%s\"\n" % c["sentence"])
    # show the aggregation the way analyze_citation would
    classed = [classify_context(c) for c in ctxs]
    stances = [c["stance"] for c in classed]
    agg = ("endorse" if "endorse" in stances else
           "dispute" if "dispute" in stances else
           "mixed" if "mixed" in stances else "neutral")
    print("AGGREGATE: %s  (risk x%.2f)" % (agg.upper(), STANCE_RISK_MULT[agg]))
    print("\nNote how the Introduction clump (3 co-cites) reads as low-reliance")
    print("background, while 'Contrary to [1]' in Results is caught as DISPUTE —")
    print("that paper should NOT be penalised for citing the retraction.")


def main():
    ap = argparse.ArgumentParser(description="Citation stance/reliance analysis")
    ap.add_argument("--citing", help="DOI of the citing paper")
    ap.add_argument("--target", help="DOI of the (retracted) cited paper")
    ap.add_argument("--target-title", default="", help="title of the cited paper (aids matching)")
    ap.add_argument("--email", help="contact email")
    ap.add_argument("--demo", action="store_true", help="offline parsing+rules demo")
    args = ap.parse_args()

    if args.demo:
        demo()
        return
    if not (args.citing and args.target):
        ap.error("need --citing and --target (or --demo)")
    res = analyze_citation(args.citing, args.target, args.target_title, args.email)
    print_analysis(args.citing, args.target, res)


if __name__ == "__main__":
    main()
