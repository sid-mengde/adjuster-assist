"""B. Retrieval (deterministic)."""
import tools
from evals import fixtures
from evals.common import load, row
from query import search


def run(ctx):
    rows = []
    con, rep = fixtures.contract_collection("base"), fixtures.report_collection(("base",))
    with fixtures.environment("base", ["base"]):
        # B1 recall@k, MRR
        cases = ctx.pick(load("retrieval.jsonl"), key=lambda r: r["collection"])
        hits1 = hits3 = rr = 0
        for c in cases:
            res = search(c["query"], n=3, col=con if c["collection"] == "contract" else rep)
            rank = next((i + 1 for i, h in enumerate(res) if h["id"] in c["expect_ids"]), 0)
            hits1 += rank == 1
            hits3 += rank > 0
            rr += 1 / rank if rank else 0
            rows.append(row("B1", c["id"], rank > 0, query=c["query"], expect=c["expect_ids"], got=[h["id"] for h in res], rank=rank))
        n = max(len(cases), 1)
        ctx.extra.update({"B1.recall_at_1": hits1 / n, "B1.recall_at_3": hits3 / n, "B1.mrr": rr / n})

        # B2 the distance threshold separates in-scope from out-of-scope components
        where = {"$and": [{"section_number": 2}, {"kind": "bullet"}]}
        d = {"in": [], "out": [], "ambiguous": []}
        for c in load("components.jsonl"):
            dist = search(c["component"], n=1, where=where, col=con)[0]["distance"]
            bucket = "ambiguous" if c["group"] == "ambiguous" else "out" if c["group"] == "out_of_scope" else "in"
            d[bucket].append((dist, c["component"]))
            if bucket != "ambiguous":
                rows.append(row("B2", c["component"], (dist <= tools.MATCH_MAX_DISTANCE) == (bucket == "in"),
                                distance=round(dist, 3), threshold=tools.MATCH_MAX_DISTANCE, group=c["group"]))
        max_in, min_out = max(x for x, _ in d["in"]), min(x for x, _ in d["out"])
        ctx.extra.update({"B2.max_in_scope_distance": max_in, "B2.min_out_of_scope_distance": min_out,
                          "B2.separation_margin": min_out - max_in, "B2.threshold": tools.MATCH_MAX_DISTANCE,
                          "B2.threshold_headroom_low": tools.MATCH_MAX_DISTANCE - max_in,
                          "B2.threshold_headroom_high": min_out - tools.MATCH_MAX_DISTANCE,
                          "B2.min_ambiguous_distance": min(x for x, _ in d["ambiguous"]),
                          "B2.ambiguous_inside_threshold": sum(x <= tools.MATCH_MAX_DISTANCE for x, _ in d["ambiguous"])})

        # B3 paraphrases and B4 misspellings reach the same coverage item
        for g in load("paraphrases.jsonl"):
            for v in g["variants"]:
                text, want = (v["text"], v["item"]) if isinstance(v, dict) else (v, g["item"])
                got = tools.match_component(text)["matched_item"]
                rows.append(row(g["eval"], f"{g['group']}:{text}", got == want, want=want, got=got))

        # B5 metadata filters do not leak other sections or kinds
        filt = [("section_number=2", {"section_number": 2}, con, lambda m: m["section_number"] == 2),
                ("kind=bullet", {"kind": "bullet"}, con, lambda m: m["kind"] == "bullet"),
                ("section 3 bullets", {"$and": [{"section_number": 3}, {"kind": "bullet"}]}, con,
                 lambda m: m["section_number"] == 3 and m["kind"] == "bullet"),
                ("kind=declarations", {"kind": "declarations"}, con, lambda m: m["kind"] == "declarations"),
                ("report grand_total", {"kind": "grand_total"}, rep, lambda m: m["kind"] == "grand_total")]
        for label, w, col, ok in filt:
            res = search("anything", n=10, where=w, col=col)
            rows.append(row("B5", label, bool(res) and all(ok(h["metadata"]) for h in res), n=len(res)))

    # B6 the right report is returned when several are loaded
    with fixtures.environment("base", ["base", "m_b", "m_c"]):
        for ref, total, mfr in [("HW-88301-A", 1800.0, "Carrier"), ("HW-77777-B", 950.0, "Trane"), ("HW-55555-C", 2400.0, "Lennox")]:
            q, det = tools.get_repair_quote(ref), tools.get_inspection_details(ref)
            rows.append(row("B6", ref, q.get("grand_total") == total and det.get("equipment", {}).get("manufacturer") == mfr,
                            total=q.get("grand_total"), manufacturer=det.get("equipment", {}).get("manufacturer")))
    return rows
