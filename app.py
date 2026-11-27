import copy
import hashlib
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout

import pandas as pd
import streamlit as st
from extractors import (
    load_document, load_history_report_cached, load_combined_invoice_packing_list,
    merge_invoice_packing, merge_declarations, PARSER_VERSION,
)
from checker import run_full_reconciliation, style_result

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache_lichsu")
TASK_TIMEOUT_S = 420   # tổng thời gian chờ tối đa cho bước đọc chứng từ

st.set_page_config(page_title="Đối chiếu Tờ khai nháp & Bộ chứng từ", layout="wide")
st.title("Đối chiếu Tờ khai nháp với Bộ chứng từ (Invoice, PL, B/L, A/N...)")

with st.sidebar:
    st.header("Cấu hình")
    default_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.text_input("Gemini API Key (dùng khi đọc PDF scan)", value=default_key, type="password")
    model_name = st.selectbox("Model đọc PDF scan", ["gemini-3.5-flash-lite", "gemini-3.8-flash"])
    weight_tol = st.number_input("Ngưỡng sai số trọng lượng (kg)", min_value=0.0, value=0.02, step=0.01)
    amount_tol = st.number_input("Ngưỡng sai số trị giá", min_value=0.0, value=0.02, step=0.01)

col1, col2, col3 = st.columns(3)
with col1:
    f_drafts = st.file_uploader(
        "1. Tờ khai nháp (PDF / Excel) * - đính kèm được nhiều tờ khai",
        type=["pdf", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Tờ khai tách nhánh hoặc nhiều tờ khai cùng 1 Bill (khác loại hình, điều kiện thanh toán...): đính kèm tất cả, "
             "chương trình đọc số tờ khai đầu tiên + số nhánh để gộp lại và đối chiếu chung với bộ chứng từ.",
    )
    f_invpk = st.file_uploader(
        "2. Invoice + Packing List (PDF / Excel) - 1 file gộp, chương trình tự tách",
        type=["pdf", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Chỉ cần tải file gộp 1 lần, chương trình quét 1 lần rồi tự tách Invoice và Packing List. "
             "Nếu Invoice và Packing List ở 2 file riêng thì tải cả 2 vào đây.",
    )
    f_debit = st.file_uploader(
        "3. Debit note cước vận chuyển (PDF)",
        type=["pdf"],
        help="So cước vận chuyển quốc tế trên debit note với phí vận chuyển trên tờ khai (quy đổi theo tỷ giá tính thuế trong tờ khai; "
             "1 debit note chia nhiều tờ khai thì phân bổ theo trị giá hàng).",
    )
with col2:
    f_bl = st.file_uploader("4. Bill of Lading - B/L (PDF)", type=["pdf"])
    f_an = st.file_uploader("5. Arrival Notice - A/N (PDF)", type=["pdf"])
    f_barcodes = st.file_uploader(
        "6. Mã vạch - danh sách container qua KVGS (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="So số container / seal trên mã vạch với B/L và Arrival Notice.",
    )
with col3:
    f_hist = st.file_uploader(
        "7. Báo cáo chi tiết hàng hóa - lịch sử khai báo (.xls / .xlsx)",
        type=["xls", "xlsx"],
        help="Dùng làm từ điển: mã hàng -> mã HS, ĐVT... theo lần khai gần nhất. Mỗi khách hàng tải báo cáo của khách đó.",
    )
    f_exports = st.file_uploader(
        "8. Tờ khai xuất (PDF / Excel) - đính kèm được nhiều tờ khai",
        type=["pdf", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Dùng khi tờ khai nhập có mã hiệu phương thức vận chuyển (số đứng cạnh mã loại hình, VD E31 2 [4]) = 9: số quản lý nội bộ "
             "NKTC#&xxxxxxxxxxx hoặc NKPTQ#&xxxxxxxxxxx phải trùng 11 số đầu tiên của số tờ khai xuất.",
    )


def _load_cached(f, doc_type, cache, api_key, model_name):
    """Đọc 1 chứng từ, có nhớ kết quả để lần chạy sau không phải gọi lại Gemini."""
    key = (PARSER_VERSION, doc_type, f.name, hashlib.md5(f.getvalue()).hexdigest(), model_name)
    if key in cache:
        return cache[key]
    data = load_document(f, doc_type, api_key=api_key, model_name=model_name)
    cache[key] = data
    return data


def _load_invpk_cached(f, cache, api_key, model_name):
    """Quét 1 file Invoice + Packing List gộp: 1 lần gọi, tách ra cả Invoice và Packing List."""
    key = (PARSER_VERSION, "INVPK", f.name, hashlib.md5(f.getvalue()).hexdigest(), model_name)
    if key in cache:
        return cache[key]
    data = load_combined_invoice_packing_list(f, api_key=api_key, model_name=model_name)
    cache[key] = data
    return data


def _load_history_cached(f, cache):
    """Đọc báo cáo lịch sử (file nặng): nhớ trong phiên và lưu xuống ổ đĩa cho các lần mở sau."""
    key = (PARSER_VERSION, "HISTORY", f.name, hashlib.md5(f.getvalue()).hexdigest())
    if key in cache:
        return cache[key]
    master = load_history_report_cached(f, CACHE_DIR)
    cache[key] = master
    return master


if st.button("Chạy đối chiếu ngay", type="primary"):
    if not f_drafts:
        st.error("Vui lòng tải lên ít nhất 1 file **Tờ khai nháp** trước.")
        st.stop()

    try:
        cache = st.session_state.setdefault("doc_cache", {})
        t0 = time.time()
        results: dict = {}

        with st.status("Đang đọc chứng từ...", expanded=True) as status:
            tasks = {}   # tên hiển thị -> (khóa kết quả, hàm)
            for i, f in enumerate(f_drafts):
                tasks[f"Tờ khai nháp: {f.name}"] = (("draft", i), lambda f=f: _load_cached(f, "DRAFT_DECL", cache, api_key, model_name))
            for i, f in enumerate(f_invpk or []):
                tasks[f"Invoice + Packing List: {f.name}"] = (("invpk", i), lambda f=f: _load_invpk_cached(f, cache, api_key, model_name))
            for key, f, dt, label in [("bl", f_bl, "BL", "B/L"), ("an", f_an, "ARRIVAL_NOTICE", "Arrival Notice"), ("debit", f_debit, "DEBIT_NOTE", "Debit note")]:
                if f is not None:
                    tasks[f"{label}: {f.name}"] = ((key, 0), lambda f=f, dt=dt: _load_cached(f, dt, cache, api_key, model_name))
            for i, f in enumerate(f_barcodes or []):
                tasks[f"Mã vạch: {f.name}"] = (("barcode", i), lambda f=f: _load_cached(f, "BARCODE", cache, api_key, model_name))
            for i, f in enumerate(f_exports or []):
                tasks[f"Tờ khai xuất: {f.name}"] = (("export", i), lambda f=f: _load_cached(f, "EXPORT_DECL", cache, api_key, model_name))
            if f_hist is not None:
                tasks[f"Báo cáo lịch sử: {f_hist.name} (file nặng, lần đầu có thể mất vài chục giây)"] = (("hist", 0), lambda: _load_history_cached(f_hist, cache))

            status.write(f"Bắt đầu đọc {len(tasks)} file song song...")
            starts = {}
            ex = ThreadPoolExecutor(max_workers=8)
            futs = {}
            try:
                for name, (key, fn) in tasks.items():
                    futs[ex.submit(fn)] = (name, key)
                    starts[name] = time.time()
                try:
                    for fu in as_completed(futs, timeout=TASK_TIMEOUT_S):
                        name, key = futs[fu]
                        try:
                            results[key] = fu.result()
                        except Exception as exc:
                            raise RuntimeError(f"Lỗi khi đọc [{name}]: {exc}") from exc
                        status.write(f"✔ Đã đọc xong {name} ({time.time() - starts[name]:.1f}s)")
                except FuturesTimeout:
                    pending = [futs[x][0] for x in futs if not x.done()]
                    raise RuntimeError(
                        f"Quá {TASK_TIMEOUT_S}s vẫn chưa đọc xong: {'; '.join(pending)}. "
                        "Thường do Gemini phản hồi chậm hoặc mạng - hãy bấm chạy lại."
                    )
            finally:
                ex.shutdown(wait=False, cancel_futures=True)   # không chờ các file bị treo

            status.update(label="Đang gộp tờ khai và đối chiếu...")
            draft_docs = [copy.deepcopy(results[("draft", i)]) for i in range(len(f_drafts))]
            inv_doc, pk_doc = merge_invoice_packing([results[("invpk", i)] for i in range(len(f_invpk or []))]) if f_invpk else (None, None)
            inv_doc, pk_doc = copy.deepcopy(inv_doc), copy.deepcopy(pk_doc)
            bl_doc = copy.deepcopy(results.get(("bl", 0)))
            an_doc = copy.deepcopy(results.get(("an", 0)))
            debit_doc = copy.deepcopy(results.get(("debit", 0)))
            barcode_docs = [copy.deepcopy(results[("barcode", i)]) for i in range(len(f_barcodes or []))]
            export_docs = [results[("export", i)] for i in range(len(f_exports or []))]
            master = results.get(("hist", 0))

            # Gộp các tờ khai (nhánh / nhiều TK cùng bill); dùng số bill trên B/L hoặc A/N để chọn đúng nhóm
            bl_hint = (bl_doc or {}).get("bl_no") or (an_doc or {}).get("bl_no") or ""
            draft_merged = merge_declarations(draft_docs, bl_hint=bl_hint)

            res = run_full_reconciliation(
                draft=draft_merged,
                invoice=inv_doc,
                packing_list=pk_doc,
                bl=bl_doc,
                arrival_notice=an_doc,
                weight_tol=weight_tol,
                amount_tol=amount_tol,
                master=master,
                export_decls=export_docs,
                debit_note=debit_doc,
                barcodes=barcode_docs,
            )
            status.update(label=f"Hoàn tất sau {time.time() - t0:.1f}s", state="complete", expanded=False)

        if master:
            m = getattr(master, "meta", {}) or {}
            st.info(
                f"Từ điển lấy từ báo cáo: **{f_hist.name}** | Doanh nghiệp: **{m.get('ten_dn') or '?'}** "
                f"(MST {m.get('ma_dn') or '?'}) | Khai từ {m.get('tu_ngay') or '?'} đến {m.get('den_ngay') or '?'} | "
                f"{m.get('so_dong', 0):,} dòng hàng -> {len(master):,} mã hàng (lấy theo lần khai gần nhất)."
            )
        else:
            st.warning("Chưa tải báo cáo lịch sử (ô 7) nên chưa đối chiếu mã HS, ĐVT với lần khai gần nhất.")

        decl_df = pd.DataFrame(draft_merged["declarations"])
        with st.expander(f"Bộ tờ khai đã gộp để đối chiếu ({len(decl_df)} tờ khai)", expanded=len(decl_df) > 1):
            st.dataframe(decl_df, use_container_width=True, hide_index=True)
        if debit_doc and debit_doc.get("charges"):
            with st.expander("Chi tiết debit note (khoản nào được tính là cước vận chuyển quốc tế)"):
                st.dataframe(pd.DataFrame(debit_doc["charges"]).rename(columns={
                    "description": "Khoản phí", "amount": "Số tiền", "currency": "Tiền tệ", "is_international_freight": "Tính vào cước quốc tế"}),
                    use_container_width=True, hide_index=True)

        st.subheader("Tóm tắt kết quả")
        st.dataframe(res["summary"], use_container_width=True, hide_index=True)

        st.subheader("Danh sách điểm lệch cần kiểm tra")
        if res["issues"].empty:
            st.success("Không phát hiện sai lệch giữa Tờ khai nháp và các chứng từ đã tải lên!")
        else:
            st.dataframe(res["issues"], use_container_width=True, hide_index=True)

        st.subheader("Bảng đối chiếu chi tiết (mỗi dòng = 1 dòng của tờ khai)")
        tab_names = ["Thông tin chung"] + list(res["line_tables"].keys())
        tabs = st.tabs(tab_names)
        with tabs[0]:
            st.dataframe(res["header_table"], use_container_width=True, hide_index=True)
        for tab, name in zip(tabs[1:], res["line_tables"].keys()):
            with tab:
                st.dataframe(style_result(res["line_tables"][name]), use_container_width=True, hide_index=True)

        # Xuất file Excel để tải về
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            res["summary"].to_excel(writer, sheet_name="Summary", index=False)
            res["issues"].to_excel(writer, sheet_name="Discrepancies", index=False)
            decl_df.to_excel(writer, sheet_name="Declarations", index=False)
            res["header_table"].to_excel(writer, sheet_name="Thong_tin_chung", index=False)
            for name, df_ in res["line_tables"].items():
                sheet = {"Invoice": "TK_vs_Invoice", "Packing List": "TK_vs_PackingList", "Lịch sử khai báo": "TK_vs_LichSu"}.get(name, name[:31])
                style_result(df_).to_excel(writer, sheet_name=sheet, index=False)

        st.download_button(
            label="Tải báo cáo đối soát (.xlsx)",
            data=excel_buf.getvalue(),
            file_name="Bao_cao_doi_soat_to_khai.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as exc:
        st.error(f"Có lỗi xảy ra: {exc}")
