# CIC — Citation Integrity Checker

Score how much a scientific paper's foundations are at risk because it **cites retracted work** — and flag, loudly, when it relies on retractions for **misconduct** (fabrication, falsification, image/data manipulation).

This is a research prototype. It produces an **attention flag with an itemized evidence ledger**, not a verdict. A paper citing a retracted study is not automatically wrong, and the tool is built to say so.

**▶ [Interactive demo](https://jmartucc97.github.io/cic/)** — walk the scoring model on bundled sample papers, no install required.

---

## What it does

Given a paper's DOI, it:

1. Pulls the paper and its reference list from **OpenAlex** (free, no key).
2. Cross-checks every reference against the **Retraction Watch** database (free via Crossref).
3. For each retracted reference, computes a risk contribution from three factors:

   | factor | meaning | source |
   |---|---|---|
   | **severity** | *why* it was retracted, on a 4-tier ladder | Retraction Watch `Reason` |
   | **reliance** | how load-bearing the citation is | full text (v2); default `1.0` in v1 |
   | **staleness** | how long it's been retracted-and-unaddressed | Retraction Watch `RetractionDate` |

4. Combines them into a `FinalRisk` score and prints an itemized ledger.

### The scoring model

```
risk_i = severity_i × reliance_i × staleness_i

FinalRisk = Σ risk_i over MISCONDUCT-tier refs        (undiluted)
          + ( Σ risk_i over minor refs ) / sqrt(#refs) (density-normalized)
```

The split is deliberate. **Misconduct-tier retractions (fabrication, falsification, manipulation, paper mills) are summed at full weight** — a single load-bearing fabrication should move the score regardless of how long the bibliography is, and two are twice as bad. **Minor retractions (authorship disputes, duplication, honest error, non-reproducibility) are density-normalized**, so a long reference list of harmless retractions can't manufacture a false alarm.

The three factors **multiply**, so the score only spikes when a *serious* retraction is *relied upon*. A Tier-0 retraction (e.g. an authorship dispute) stays near zero even when load-bearing.

### Severity ladder

| tier | severity | examples (Retraction Watch reasons) |
|---|---|---|
| 3 | 1.00 | fabrication of data / results / images |
| 2 | 0.80 | falsification, data/image manipulation, fake peer review, paper mill |
| 1 | 0.40 | results not reproducible, unreliable results, error in data/analysis |
| 0 | 0.10 | authorship disputes, copyright/legal, duplicate publication, ethics/IRB |

A retraction's severity is its **worst** reason (retractions often carry several).

---

## Install

```bash
git clone https://github.com/<you>/cic.git
cd cic
pip install -r requirements.txt
```

Download the Retraction Watch database (CSV) once — it's distributed free via Crossref:
<https://gitlab.com/crossref/retraction-watch-data> (see Crossref's Retraction Watch announcement for the current link). Save it as `retractions.csv`.

---

## Usage

**Score a paper:**

```bash
python retraction_risk.py 10.1234/example.doi --rw-csv retractions.csv --email you@example.com
```

**Walk one hop deeper** — also find retractions reached *through* your references, with attribution of which reference each one enters through:

```bash
python retraction_risk.py 10.1234/example.doi --rw-csv retractions.csv --email you@example.com --depth 2
```

`--depth 2` is slower (many more API calls) and bounded by `--max-nodes` (default 2000). The default `--depth 1` checks direct citations only.

**Try it with no network (synthetic data):**

```bash
python retraction_risk.py --demo
```

**Validate the score against a labeled set:**

```bash
python validate.py --labels labels.csv --rw-csv retractions.csv --email you@example.com
# labels.csv has columns:  doi,label   (label 1 = retracted/case, 0 = control)
```

```bash
python validate.py --demo     # metric stack on synthetic data, no network
```

---

## Validation methodology

`validate.py` tests **predictive validity**: do papers that *themselves* got retracted (cases) score higher than matched papers that didn't (controls)? It reports **ROC-AUC**, Mann-Whitney U, a precision/recall threshold sweep, and a distribution histogram.

**Building a defensible labeled set:**
- **Cases:** papers in Retraction Watch that were themselves retracted. Exclude any whose retraction reason is "cites retracted work" (circularity).
- **Controls:** papers **matched** to cases on publication year, field, journal tier, and reference count. *This matters:* unmatched controls make the AUC measure confounds (paper age, size) instead of the hypothesis.
- **Stronger ground truth, where available:** the *Reproducibility Project: Cancer Biology* provides actual replicate/didn't-replicate labels; PubPeer flags community-questioned papers.

---

## Known limitations (read these)

- **Retraction ≠ wrong.** Retractions span honest error to fabrication. The severity tiers exist precisely so harmless retractions don't trigger alarm, but the reason string is surfaced verbatim so a human can overrule the tier.
- **Reliance is a v1 default of `1.0`.** OpenAlex gives no in-text citation context, so v1 cannot yet distinguish a fabricated paper *relied on as the core method* from one *mentioned in passing*. Both score at full weight — conservative (it over-flags rather than misses), and the #1 target for v2.
- **Citation data is imperfect.** Reference lists in OpenAlex have gaps; DOI resolution is messy. Budget for data cleaning.
- **Every coefficient is provisional.** The tier values, the `sqrt` normalizer, the staleness slope, and the display bands are labeled defaults to be tuned against the validation set — not defended as correct.

---

## Roadmap

- **v2 — reliance from full text.** Fetch JATS XML (Europe PMC OA), locate `<xref ref-type="bibr">` for each retracted reference, grade by section and co-citation density, then replace the heuristic with an LLM classifier that quotes the supporting sentence. This is the prerequisite that makes both depth-1 and depth-2 risk weights meaningful (load-bearing vs incidental).
- **Depth-2 propagation** ✅ implemented (`--depth 2`): finds retractions reached through your references and attributes the entry point. Future: extend decay/dedup to depth 3 with a strict node budget.
- **Severity weighting from reason-code research**, staleness from presence/absence of a correction on the citing paper.

---

## Web demo

[`docs/index.html`](docs/index.html) is a self-contained interactive walkthrough of the scoring model — the severity ladder, the `severity × reliance × staleness` factors, the misconduct-undiluted / minor-normalized split, the evidence ledger, and `--depth 2` propagation — running on the same bundled sample data as `--demo`. It's a single static file with no build step and no backend; it mirrors the CLI's output shape rather than calling the live APIs.

Served free via **GitHub Pages** from the `/docs` folder. To publish it: **Settings → Pages → Build and deployment → Source: “Deploy from a branch” → Branch: `main`, folder `/docs`**. The live URL is then `https://jmartucc97.github.io/cic/`.

## License

MIT — see [LICENSE](LICENSE).

Built on open data from [OpenAlex](https://openalex.org) and [Retraction Watch](https://retractionwatch.com) / Crossref.
