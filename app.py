import math, datetime, os
import numpy as np, pandas as pd

ALIASES = {
    "query": ["query", "search_term", "search_terms", "searchterm", "term", "search_query"],
    "match_type": ["match_type", "matchtype", "match"],
    "campaign": ["campaign", "campaign_name"],
    "ad_group": ["ad_group", "adgroup", "ad_group_name"],
    "impressions": ["impressions", "impr"],
    "clicks": ["clicks"],
    "cost": ["cost", "spend"],
    "conversions": ["conversions", "conv", "conv."],
    "conversion_value": ["conversion_value", "conv_value", "conversion_val"],
}

def load(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    ren = {}
    for std, opts in ALIASES.items():
        for o in opts:
            if o in df.columns:
                ren[o] = std
                break
    df = df.rename(columns=ren)
    for c in ["match_type", "campaign", "ad_group"]:
        if c not in df.columns:
            df[c] = "all"
    for c in ["impressions", "clicks", "cost", "conversions", "conversion_value"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["query"] = df["query"].astype(str).str.lower().str.strip()
    return df

def wilson(conv, clicks, z=1.64):
    if clicks <= 0:
        return 0.0, 1.0
    p = min(conv / clicks, 1)
    d = 1 + z * z / clicks
    c = p + z * z / (2 * clicks)
    m = z * math.sqrt(p * (1 - p) / clicks + z * z / (4 * clicks * clicks))
    return max((c - m) / d, 0), min((c + m) / d, 1)

STOP = {"a", "an", "the", "for", "of", "to", "in", "on", "and", "with", "near", "me", "my", "best"}

def word_table(q, avg_cvr, need):
    w = q.assign(word=q["query"].str.split()).explode("word")
    w = w[~w.word.isin(STOP) & (w.word.str.len() > 2)]
    w = w.groupby("word").agg(terms=("query", "nunique"), clicks=("clicks", "sum"),
                              cost=("cost", "sum"), conv=("conversions", "sum")).reset_index()
    w = w[(w.terms >= 5) & (w.clicks >= need)]
    ub = w.apply(lambda r: wilson(r.conv, r.clicks)[1], axis=1)
    lb = w.apply(lambda r: wilson(r.conv, r.clicks)[0], axis=1)
    w["kind"] = np.where(ub < avg_cvr * 0.5, "poison", np.where(lb > avg_cvr * 1.5, "winner", "neutral"))
    return w[w.kind != "neutral"].sort_values("cost", ascending=False)

def triage(raw):
    df = load(raw)
    q = df.groupby(["campaign", "ad_group", "query"], as_index=False).agg(
        match_type=("match_type", "first"), impressions=("impressions", "sum"),
        clicks=("clicks", "sum"), cost=("cost", "sum"),
        conversions=("conversions", "sum"), conversion_value=("conversion_value", "sum"))
    tc, tconv, tcost = q.clicks.sum(), q.conversions.sum(), q.cost.sum()
    avg_cvr = max(tconv / tc, 1e-4) if tc else 1e-4
    avg_cpc = tcost / tc if tc else 0
    need = math.ceil(3 / avg_cvr)          # clicks needed to expect ~3 conversions
    words = word_table(q, avg_cvr, need)
    poison = dict(zip(words[words.kind == "poison"].word, words[words.kind == "poison"].cost))
    winners = set(words[words.kind == "winner"].word)

    out = []
    for r in q.itertuples():
        lb, ub = wilson(r.conversions, r.clicks)
        more = max(need - r.clicks, 0)
        gap = round(more * avg_cpc, 2)
        toks = r.query.split()
        pw = [t for t in toks if t in poison]
        ww = [t for t in toks if t in winners]
        conf = "high" if r.clicks >= need else ("medium" if r.clicks >= need / 3 else "low")
        sug, neg = "", ""
        if r.clicks >= need and r.conversions == 0:
            d, conf = "block", "high"
            why = f"{int(r.clicks)} clicks, ${r.cost:.0f} spent, 0 conversions; account CVR {avg_cvr:.1%} predicts ~{r.clicks*avg_cvr:.1f}."
            neg = f"exact negative at ad group level ({r.ad_group})"
        elif r.conversions >= 3 and r.clicks >= 30 and lb > avg_cvr:
            d, conf = "promote", "high"
            why = f"{int(r.conversions)} conversions / {int(r.clicks)} clicks; CVR lower bound {lb:.1%} beats account {avg_cvr:.1%}."
        elif r.clicks >= need and ub < avg_cvr * 0.5:
            d, conf = "block", "medium"
            why = f"CVR upper bound {ub:.1%} is far below account {avg_cvr:.1%}; ${r.cost:.0f} spent."
            neg = f"exact negative at ad group level ({r.ad_group})"
        elif r.clicks < need and r.clicks >= 3 and (r.conversions > 0 or r.clicks >= 10 or pw or ww):
            d = "needs_review"
            why = (f"Only {int(r.clicks)} clicks, {int(r.conversions)} conversions. Need ~{int(more)} more clicks "
                   f"(~${gap}) to judge. Wait, or decide now.")
            if pw:
                sug = "block"; why += f" Contains poison word '{pw[0]}' (${poison[pw[0]]:.0f} wasted across many terms)."
            elif ww:
                sug = "promote"; why += f" Contains winning word '{ww[0]}'."
        else:
            d = "ignore"
            why = (f"No clicks yet; nothing to judge." if r.clicks == 0 else f"Only {int(r.clicks)} clicks and ${r.cost:.2f} spent, no conversions; not worth action yet.")
        out.append((d, conf, why, sug, neg, int(more), gap))
    q[["decision", "confidence", "reason", "suggested", "negative_advice", "more_clicks_needed", "cost_to_know"]] = out
    return q, words, dict(avg_cvr=avg_cvr, avg_cpc=avg_cpc, need=need)

def main():
    import streamlit as st
    st.set_page_config(page_title="Search Term Triage", layout="wide")
    st.title("Search Term Triage Engine")
    f = st.file_uploader("Upload search terms CSV", type="csv")
    if not f:
        st.info("Upload the CSV to start."); return
    if "res" not in st.session_state or st.session_state.get("fname") != f.name:
        res, words, stats = triage(pd.read_csv(f))
        res["final_decision"] = res["decision"]
        st.session_state.update(res=res, words=words, stats=stats, fname=f.name)
    res, words, stats = st.session_state.res, st.session_state.words, st.session_state.stats
    c = st.columns(5)
    c[0].metric("Terms", len(res))
    for i, d in enumerate(["promote", "block", "ignore", "needs_review"]):
        c[i + 1].metric(d, int((res.decision == d).sum()))
    st.caption(f"Account CVR {stats['avg_cvr']:.2%} | avg CPC ${stats['avg_cpc']:.2f} | "
               f"data needed per term ~{stats['need']} clicks (adapts to this account)")
    t1, t2, t3, t4 = st.tabs(["Review queue", "All decisions", "Word evidence", "Audit log"])
    with t1:
        who = st.text_input("Reviewer name", "analyst")
        qd = res[(res.decision == "needs_review") & (res.final_decision == "needs_review")].sort_values("cost", ascending=False)
        st.write(f"{len(qd)} terms waiting. Most expensive first.")
        sg = qd[qd.suggested != ""]
        if len(sg) and st.button(f"Accept all {len(sg)} system suggestions (bulk)"):
            log = pd.DataFrame(dict(time=datetime.datetime.now().isoformat(timespec="seconds"), reviewer=who,
                                    term=sg["query"], original=sg["decision"], original_reason=sg["reason"],
                                    final=sg["suggested"], override_reason="bulk accept of word-evidence suggestion"))
            log.to_csv("audit_log.csv", mode="a", index=False, header=not os.path.exists("audit_log.csv"))
            res.loc[sg.index, "final_decision"] = sg["suggested"]
            st.rerun()
        if len(qd):
            r = qd.iloc[0]
            st.subheader(r["query"])
            st.write(r["reason"])
            if r["suggested"]:
                st.warning(f"System suggests: {r['suggested']}")
            note = st.text_input("Override reason (optional)", key="note")
            cols = st.columns(3)
            for col, lab in zip(cols, ["promote", "block", "ignore"]):
                if col.button(lab, key=lab):
                    idx = r.name
                    row = dict(time=datetime.datetime.now().isoformat(timespec="seconds"), reviewer=who,
                               term=r["query"], original=r["decision"], original_reason=r["reason"],
                               final=lab, override_reason=note)
                    pd.DataFrame([row]).to_csv("audit_log.csv", mode="a", index=False, header=not os.path.exists("audit_log.csv"))
                    res.loc[idx, "final_decision"] = lab
                    # learn within session: same-word terms inherit the correction as a suggestion
                    for wd in [x for x in r["query"].split() if x not in STOP and len(x) > 2]:
                        m = (res.decision == "needs_review") & (res.final_decision == "needs_review") & res["query"].str.contains(rf"\b{wd}\b", regex=True)
                        res.loc[m, "suggested"] = lab
                    st.rerun()
    with t2:
        pick = st.multiselect("Filter", ["promote", "block", "ignore", "needs_review"], default=["promote", "block", "needs_review"])
        st.dataframe(res[res.final_decision.isin(pick)], use_container_width=True)
        st.download_button("Download upload-ready CSV", res.to_csv(index=False), "triage_final.csv")
    with t3:
        st.write("Words judged from pooled evidence across many terms. Poison words become campaign-level negatives.")
        st.dataframe(words, use_container_width=True)
    with t4:
        if os.path.exists("audit_log.csv"):
            st.dataframe(pd.read_csv("audit_log.csv"), use_container_width=True)
        else:
            st.write("No overrides yet.")

if __name__ == "__main__":
    main()
