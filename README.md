# Search Term Triage Engine (DevSprint, PS-01)

Team: SheCodes

## Problem
Analysts check thousands of Google Ads search terms by hand every week.
We built a tool that decides promote / block / ignore for each term,
explains why, and sends uncertain terms to a human.

## Our 3 ideas
1. **Evidence Gap Advisor** - for unsure terms, shows how many more clicks
   and how much money are needed to decide. The click threshold is
   calculated from the account's own conversion rate, not a fixed number.
2. **Word-Level Evidence Pooling** - combines evidence across terms that
   share a word, so thin terms can borrow strength (e.g. "free").
3. **Decision Memory and Audit Trail** - every human override is saved with
   reason, reviewer and time, and similar terms are re-suggested.

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
Upload `data/ps1.csv` in the browser page that opens.

## Output
Download `triage_final.csv` from the "All decisions" tab. Columns:
decision, confidence, reason, suggested, negative_advice,
more_clicks_needed, cost_to_know, final_decision.

## Method (short)
- Account conversion rate and cost per click are computed from the data.
- A term needs about 3 / CVR clicks before we judge it (about 57 here).
- Promote: enough conversions and the Wilson lower bound beats the account rate.
- Block: enough clicks with zero (or far too few) conversions.
- Needs review: too little data; the reason says what is missing and the cost to find out.
- Word pooling flags poison / winner words from combined evidence.

## Limits (honest)
- Thresholds are rules of thumb; tune per account.
- Word pooling can be confounded by campaign themes (e.g. "financing").
- Learning from overrides works within the session only.
- Negatives are advised at ad group level, exact match; campaign-level
  phrase negatives for poison words are suggested in the Word evidence tab.
