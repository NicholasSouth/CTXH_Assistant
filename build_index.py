import argparse
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", type=Path, default=_HERE / "data")
ap.add_argument("--out", type=Path, default=_HERE / "data" / "index.json")
args = ap.parse_args()

from doc_model import Document
from vector_store import VectorStore

load_dotenv()

SOURCES = [
    {
        "file": "CTXH.md",
        "title": "Quy định tổ chức thực hiện chương trình CTXH (QĐ 224/QĐ-ĐHSPKT-CTHSSV)",
        "date": "2013-12-09",
        "version": "QĐ 224/QĐ-ĐHSPKT-CTHSSV",
        "owner": "Phòng Công tác học sinh sinh viên - Trường Đại học Sư phạm Kỹ thuật TP.HCM"
    },
    {
        "file": "CTXH_v2.md",
        "title": "Quy định tổ chức thực hiện chương trình CTXH (QĐ 225/QĐ-ĐHCNKT-CTSV)",
        "date": "2026-01-03",
        "version": "QĐ 225/QĐ-ĐHCNKT-CTSV",
        "owner": "Phòng Công tác sinh viên - Trường Đại học Công nghệ Kỹ thuật TP.HCM"
    },
]

store = VectorStore()
for src in SOURCES:
    path = args.data_dir / src["file"]
    doc = Document(
        content=path.read_text(encoding="utf-8"),
        title=src["title"],
        source=path.name,
        metadata={
            "title": src["title"],
            "source": path.name,
            "date": src["date"],
            "version": src["version"],
            "owner": src["owner"],
        },
    )
    doc_id = store.add(doc)
    print(f"{doc_id}  {src['version']:26s} {len(store.chunks):>3} chunks so far")

store.save(str(args.out))
print(f"\nsaved {len(store.documents)} documents, {len(store.chunks)} chunks -> {args.out}")