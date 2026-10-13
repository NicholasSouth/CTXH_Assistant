import sys
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

TITLES = {
    "01_quy_dinh_chung.txt": "Quy định chung về chương trình CTXH",
    "02_noi_dung_ctxh.txt": "Nội dung công tác xã hội",
    "03_bang_quy_doi_001.txt": "Bảng mã và quy đổi nhóm 001 (Mùa hè xanh, hiến máu, Xuân tình nguyện, Tiếp sức mùa thi, cứu trợ...)",
    "04_bang_quy_doi_002_004.txt": "Bảng mã và quy đổi nhóm 002-004 (công ích, môi trường, pháp luật, chăm sóc đối tượng chính sách)",
    "05_bang_quy_doi_005_006.txt": "Bảng mã và quy đổi nhóm 005-006 (dạy học, quyên góp, cán bộ đoàn hội, phong trào)",
    "06_tru_ngay_va_quy_doi_diem.txt": "Quy tắc trừ ngày CTXH và quy đổi điểm",
    "07_quy_trinh_buoc_1_3.txt": "Quy trình thực hiện CTXH: bước 1-3",
    "08_quy_trinh_buoc_4_8.txt": "Quy trình thực hiện CTXH: bước 4-8",
    "09_trach_nhiem_don_vi.txt": "Trách nhiệm của các đơn vị",
    "10_trach_nhiem_lop_sv_kinh_phi.txt": "Trách nhiệm của lớp, sinh viên và kinh phí",
    "11_tu_viet_tat.txt": "Từ viết tắt và thuật ngữ",
}

store = VectorStore()
for path in sorted(args.data_dir.glob("*.txt")):
    title = TITLES.get(path.name, path.stem)
    doc = Document(
        content=path.read_text(encoding="utf-8"),
        title=title,
        source=path.name,
        metadata={
            "title": title,
            "source": path.name,
            "date": "2013-12-09",          
            "version": "QĐ 224/QĐ-ĐHSPKT-CTHSSV",
            "owner": "Phòng Công tác sinh viên - Trường Đại học Công nghệ Kỹ thuật TP.HCM",                
        },
    )
    doc_id = store.add(doc)
    print(f"{doc_id}  {title:28s} {len(store.chunks):>3} chunks so far")

store.save(str(args.out))
print(f"\nsaved {len(store.documents)} documents, {len(store.chunks)} chunks -> {args.out}")
