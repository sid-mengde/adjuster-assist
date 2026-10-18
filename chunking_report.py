"""Chunking for technician inspection / repair reports (one PDF -> many typed chunks)."""
import re
from datetime import datetime

from chunking import BULLET, Chunk, _clean, _window, load_text

HEADING = re.compile(r"^(\d+)\.\s+([A-Z][A-Za-z &,/\-]+)$", re.M)
HEADER_KEYS = ["Service Provider", "Date of Inspection", "License Number", "Work Order / Claim Ref", "Technician Name"]
EQUIPMENT_KEYS = ["System Type", "Manufacturer", "Model Number", "Serial Number", "Age of Unit", "Location"]
TABLE_HEADER = {"item / description", "qty", "rate", "total"}
MONEY = re.compile(r"^\$[\d,]+(?:\.\d\d)?$")
MONEY_ANY = re.compile(r"\$[\d,]+(?:\.\d\d)?")
QTY = re.compile(r"^\d+(?:\.\d+)?(?:\s*(?:Hrs?|Hours?|ea))?$", re.I)


def _money(s):
    return float(s.replace("$", "").replace(",", ""))


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _fields(body, keys):
    """'Key: value' lines; a line without a known key continues the previous value."""
    out, cur = {}, None
    for line in (l.strip() for l in body.splitlines()):
        if not line:
            continue
        key = next((k for k in keys if line.startswith(k + ":")), None)
        if key:
            cur, out[key] = key, line[len(key) + 1:].strip()
        elif cur:
            out[cur] += " " + line
    return out


def _parse_estimate(body):
    """Walk the table one value per line: name, description..., qty, rate, total; then GRAND TOTAL."""
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    while lines and lines[0].lower() in TABLE_HEADER:
        lines.pop(0)
    items, buf, grand, i = [], [], None, 0
    while i < len(lines):
        line = lines[i]
        if line.upper().startswith("GRAND TOTAL"):
            m = MONEY_ANY.search(line) or (MONEY_ANY.search(lines[i + 1]) if i + 1 < len(lines) else None)
            grand = _money(m.group()) if m else None
            break
        if QTY.match(line) and i + 2 < len(lines) and MONEY.match(lines[i + 1]) and MONEY.match(lines[i + 2]) and buf:
            items.append({"name": buf[0], "description": " ".join(buf[1:]), "qty": line,
                          "rate": _money(lines[i + 1]), "total": _money(lines[i + 2])})
            buf, i = [], i + 3
            continue
        buf.append(line)
        i += 1
    return items, grand


def _iso_date(s):
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return None


def chunk_report_text(text, offsets, source):
    """Chunk already-extracted report text. offsets: [(char_offset, page)] as from chunking.load_text."""
    def page_at(pos):
        return max(pg for off, pg in offsets if off <= pos)

    heads = list(HEADING.finditer(text))
    header_end = heads[0].start() if heads else len(text)
    header = _fields(text[:header_end], HEADER_KEYS)
    claim_ref = header.get("Work Order / Claim Ref", "").lstrip("#").strip() or _slug(source)
    doc_id = _slug(claim_ref)
    tag = f"[Report {claim_ref}"
    common = {"source": source, "doc_type": "inspection_report", "doc_id": doc_id, "claim_ref": claim_ref}

    chunks = [Chunk(
        f"{doc_id}-header",
        f"{tag} · Header] " + "; ".join(f"{k}: {v}" for k, v in header.items()),
        {**common, "section_number": 0, "section_title": "Header", "kind": "header", "page": 1,
         "provider": header.get("Service Provider", ""), "technician": header.get("Technician Name", ""),
         "license_number": header.get("License Number", ""),
         "inspection_date": _iso_date(header.get("Date of Inspection", "")) or ""},
    )]

    for i, m in enumerate(heads):
        num, title = int(m.group(1)), m.group(2).strip()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[m.end():end].strip()
        base = {**common, "section_number": num, "section_title": title}
        prefix = f"{tag} · Section {num}: {title}]"
        page = page_at(m.start())
        t = title.lower()

        if "equipment" in t:
            f = _fields(body, EQUIPMENT_KEYS)
            chunks.append(Chunk(f"{doc_id}-equipment", f"{prefix} " + "; ".join(f"{k}: {v}" for k, v in f.items()),
                                {**base, "kind": "equipment", "page": page,
                                 **{k.lower().replace(" ", "_"): v for k, v in f.items()}}))

        elif "diagnostic" in t or "finding" in t:
            head, *bullets = BULLET.split(body)
            if _clean(head):
                chunks.append(Chunk(f"{doc_id}-findings-intro", f"{prefix} {_clean(head)}",
                                    {**base, "kind": "intro", "page": page}))
            for j, b in enumerate(bullets):
                b = _clean(b)
                topic = b.split(":", 1)[0]
                kind = "cause_of_failure" if topic.lower().startswith("cause of failure") else "finding"
                pos = text.find(b[:25], m.end())
                chunks.append(Chunk(f"{doc_id}-finding-{j}", f"{prefix} {b}",
                                    {**base, "kind": kind, "topic": topic, "page": page_at(max(pos, m.start())),
                                     "chunk_index": j}))

        elif "estimate" in t or "quote" in t:
            items, grand = _parse_estimate(body)
            for j, it in enumerate(items):
                pos = text.find(it["name"], m.end())
                desc = f" ({it['description']})" if it["description"] else ""
                chunks.append(Chunk(
                    f"{doc_id}-item-{j}",
                    f"{prefix} {it['name']}{desc}: qty {it['qty']} x ${it['rate']:,.2f} = ${it['total']:,.2f}",
                    {**base, "kind": "line_item", "page": page_at(max(pos, m.start())), "chunk_index": j,
                     "item_name": it["name"], "line_total": it["total"]}))
            if grand is not None:
                total = sum(it["total"] for it in items)
                pos = text.find("GRAND TOTAL", m.end())
                chunks.append(Chunk(
                    f"{doc_id}-grand-total",
                    f"{prefix} GRAND TOTAL of the repair quotation: ${grand:,.2f} "
                    f"(sum of {len(items)} line items: ${total:,.2f})",
                    {**base, "kind": "grand_total", "page": page_at(max(pos, m.start())),
                     "grand_total": grand, "items_sum": total, "line_item_count": len(items),
                     "total_matches_items": abs(total - grand) < 0.005}))

        else:  # notes or any unknown section: one chunk per paragraph (windowed if long)
            for j, para in enumerate(p for p in (_clean(x) for x in re.split(r"\n\s*\n", body)) if p):
                for k, sub in enumerate(_window(para)):
                    chunks.append(Chunk(f"{doc_id}-sec{num}-{j}" + (f"-{k}" if k else ""), f"{prefix} {sub}",
                                        {**base, "kind": "notes" if "note" in t else "paragraph", "page": page,
                                         "chunk_index": j}))
    return chunks


def chunk_inspection_report(path):
    text, offsets = load_text(path)
    return chunk_report_text(text, offsets, path.split("/")[-1])


if __name__ == "__main__":
    import sys
    for c in chunk_inspection_report(sys.argv[1] if len(sys.argv) > 1 else "HVAC_Inspection_Report.pdf"):
        print(f"--- {c.id} | p{c.metadata['page']} | {c.metadata['kind']}")
        print(c.text)
