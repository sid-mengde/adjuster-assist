"""Section-aware chunking for the home warranty agreement."""
import re
from dataclasses import dataclass, field
from pypdf import PdfReader

# "3. MAJOR EXCLUSIONS & LIMITATIONS" -- ALL CAPS, so numbered claim steps
# like "3. Diagnosis & Adjudication: ..." are not mistaken for headings.
HEADING = re.compile(r"^(\d+)\.\s+([A-Z][A-Z &,/\-]+)\s*$", re.M)
BULLET = re.compile(r"^\s*\x7f\s*", re.M)
STEP = re.compile(r"^(?=\d+\.\s+[A-Z])", re.M)
DECLARATION_KEYS = ["Agreement Number", "Coverage Period", "Plan Type", "Service Fee"]
MAX_CHARS, OVERLAP = 1500, 200


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def load_text(path):
    """Return the full text plus (offset, page) pairs so chunks can be mapped to pages."""
    text, offsets = "", []
    for i, page in enumerate(PdfReader(path).pages):
        offsets.append((len(text), i + 1))
        text += (page.extract_text() or "") + "\n"
    return text, offsets


def _clean(s):
    return re.sub(r"\s+", " ", s).strip()


def _window(text):
    """Fallback for oversize pieces: sliding window with overlap."""
    if len(text) <= MAX_CHARS:
        return [text]
    step = MAX_CHARS - OVERLAP
    return [text[i:i + MAX_CHARS] for i in range(0, len(text), step)]


def _parse_declarations(body):
    """The table extracts as 'Key:\\nvalue' with wrapped lines; rebuild key/value pairs."""
    flat = _clean(body)
    pairs = {}
    for i, key in enumerate(DECLARATION_KEYS):
        m = re.search(re.escape(key).replace(r"\ ", r"\s+") + r":\s*", flat)
        if not m:
            continue
        nxt = [re.search(re.escape(k) + r":", flat[m.end():]) for k in DECLARATION_KEYS[i + 1:]]
        ends = [n.start() for n in nxt if n]
        pairs[key] = flat[m.end():m.end() + min(ends)].strip() if ends else flat[m.end():].strip()
    return pairs


def _pieces(num, body):
    """Split a section body into (kind, text) pieces: intro, bullets, or steps."""
    if "\x7f" in body:
        head, *bullets = BULLET.split(body)
        out = [("intro", _clean(head))] if _clean(head) else []
        out += [("bullet", _clean(b)) for b in bullets if _clean(b)]
        return out
    steps = [s for s in STEP.split(body) if s.strip()]
    if len(steps) > 1:
        return [("step", _clean(s)) for s in steps]
    return [("paragraph", _clean(body))]


def chunk_document(path):
    text, offsets = load_text(path)
    source = path.split("/")[-1]

    def page_at(pos):
        return max(pg for off, pg in offsets if off <= pos)

    heads = list(HEADING.finditer(text))
    chunks = []
    for i, m in enumerate(heads):
        num, title = m.group(1), m.group(2).strip()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[m.end():end].strip()
        body_start = m.end() + (len(text[m.end():end]) - len(text[m.end():end].lstrip()))
        base = {"source": source, "section_number": int(num), "section_title": title}

        if title.startswith("DECLARATIONS"):
            intro, _, _ = body.partition("Agreement\nNumber")
            if _clean(intro):
                chunks.append(Chunk(f"sec{num}-intro", f"[Section {num}: {title}] {_clean(intro)}",
                                    {**base, "kind": "intro", "page": page_at(m.start())}))
            decl = _parse_declarations(body)
            chunks.append(Chunk(
                f"sec{num}-declarations",
                f"[Section {num}: {title}] Contract declarations. "
                + "; ".join(f"{k}: {v}" for k, v in decl.items()),
                {**base, "kind": "declarations", "page": page_at(m.start()), **{k.lower().replace(" ", "_"): v for k, v in decl.items()}},
            ))
            continue

        # find each piece's page by locating its first words in the document
        for j, (kind, piece) in enumerate(_pieces(num, body)):
            probe = re.escape(piece.split()[0]) if piece else ""
            pos = text.find(piece.split(":")[0][:30].split(" ")[0], body_start) if probe else m.start()
            for k, sub in enumerate(_window(piece)):
                chunks.append(Chunk(
                    f"sec{num}-{j}" + (f"-{k}" if k else ""),
                    f"[Section {num}: {title}] {sub}",
                    {**base, "kind": kind, "page": page_at(max(pos, m.start())), "chunk_index": j},
                ))
    return chunks


if __name__ == "__main__":
    import sys
    for c in chunk_document(sys.argv[1] if len(sys.argv) > 1 else "home_warranty_contract.pdf"):
        print(f"--- {c.id} | p{c.metadata['page']} | {c.metadata['kind']} | {len(c.text)} chars")
        print(c.text)
