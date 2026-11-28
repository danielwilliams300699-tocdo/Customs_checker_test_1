import copy
import hashlib
import io
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout

import pandas as pd
import streamlit as st
from extractors import (
    load_document, load_history_report_cached, load_combined_invoice_packing_list,
    merge_invoice_packing, merge_declarations, PARSER_VERSION,
)
from checker import (
    run_full_reconciliation, style_result, _convert, CAT_LABEL, CAT_BY_LABEL,
    incoterm3, default_charge_tick, charge_net, collect_notices,
)

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
        "3. Debit note cước vận chuyển (PDF) - tải được nhiều file",
        type=["pdf"],
        accept_multiple_files=True,
        help="Phí OF và phí EXW ở 2 debit note khác nhau thì tải cả 2. Điều kiện F*: mặc định chỉ tick Ocean Freight; "
             "EXW: tick thêm phí EXW/đại lý (tính chưa VAT 8%). Bạn có thể tick lại / thêm dòng ở bảng sau khi đọc.",
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
    f_insurance = st.file_uploader(
        "9. Phí bảo hiểm (PDF) - tải được nhiều file",
        type=["pdf"],
        accept_multiple_files=True,
        help="So phí bảo hiểm (chưa VAT) với phí bảo hiểm trên tờ khai (quy đổi theo tỷ giá tính thuế trong tờ khai).",
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


# ---------------- GIAI ĐOẠN 1: đọc chứng từ (gọi Gemini, có cache) ----------------
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
            for key, f, dt, label in [("bl", f_bl, "BL", "B/L"), ("an", f_an, "ARRIVAL_NOTICE", "Arrival Notice")]:
                if f is not None:
                    tasks[f"{label}: {f.name}"] = ((key, 0), lambda f=f, dt=dt: _load_cached(f, dt, cache, api_key, model_name))
            for i, f in enumerate(f_debit or []):
                tasks[f"Debit note: {f.name}"] = (("debit", i), lambda f=f: _load_cached(f, "DEBIT_NOTE", cache, api_key, model_name))
            for i, f in enumerate(f_insurance or []):
                tasks[f"Bảo hiểm: {f.name}"] = (("insurance", i), lambda f=f: _load_cached(f, "INSURANCE", cache, api_key, model_name))
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
            status.update(label=f"Đã đọc xong sau {time.time() - t0:.1f}s", state="complete", expanded=False)

        st.session_state["parsed"] = {
            "results": results, "stamp": time.time(),
            "n": {"draft": len(f_drafts), "invpk": len(f_invpk or []), "debit": len(f_debit or []),
                  "insurance": len(f_insurance or []), "barcode": len(f_barcodes or []), "export": len(f_exports or [])},
            "hist_name": f_hist.name if f_hist is not None else "",
        }
    except Exception as exc:
        st.session_state.pop("parsed", None)
        st.error(f"Có lỗi xảy ra: {exc}")


# ---------------- GIAI ĐOẠN 2: gộp, chọn khoản phí, đối chiếu (chạy lại mỗi lần tick, KHÔNG gọi lại Gemini) ----------------
parsed = st.session_state.get("parsed")
if parsed:
    try:
        results, n = parsed["results"], parsed["n"]
        draft_docs = [copy.deepcopy(results[("draft", i)]) for i in range(n["draft"])]
        inv_doc, pk_doc = merge_invoice_packing([results[("invpk", i)] for i in range(n["invpk"])]) if n["invpk"] else (None, None)
        inv_doc, pk_doc = copy.deepcopy(inv_doc), copy.deepcopy(pk_doc)
        bl_doc = copy.deepcopy(results.get(("bl", 0)))
        an_doc = copy.deepcopy(results.get(("an", 0)))
        debit_docs = [results[("debit", i)] for i in range(n["debit"])]
        insurance_docs = [copy.deepcopy(results[("insurance", i)]) for i in range(n["insurance"])]
        barcode_docs = [copy.deepcopy(results[("barcode", i)]) for i in range(n["barcode"])]
        export_docs = [results[("export", i)] for i in range(n["export"])]
        master = results.get(("hist", 0))

        # Gộp các tờ khai (nhánh / nhiều TK cùng bill); dùng số bill trên B/L hoặc A/N để chọn đúng nhóm
        bl_hint = (bl_doc or {}).get("bl_no") or (an_doc or {}).get("bl_no") or ""
        draft_merged = merge_declarations(draft_docs, bl_hint=bl_hint)

        # ---- Thông báo cần lưu ý (PCS / LSS trên debit note, A/N; HS đầu 28/29 = hóa chất) ----
        notices = collect_notices(debit_docs, an_doc, draft_merged)
        if notices:
            st.warning("**Thông báo cần lưu ý**\n\n" + "\n".join(f"- {t}" for t in notices))

        # ---- Bảng chọn khoản phí tính vào phí vận chuyển ----
        debit_note = None
        if debit_docs:
            inc = incoterm3(draft_merged.get("incoterms"))
            rows = []
            for dn in debit_docs:
                for c in dn.get("charges") or []:
                    cat = c.get("category", "OTHER")
                    rows.append({
                        "Chọn": default_charge_tick(inc, cat, c.get("is_international_freight")),
                        "Debit note": dn.get("file_name", ""), "Khoản phí": c["description"],
                        "Loại": CAT_LABEL.get(cat, "Khác"), "Số tiền": c["amount"], "Tiền tệ": c["currency"],
                        "Đã gồm VAT 8%": bool(c.get("amount_includes_vat")),
                    })
            st.subheader("Chọn khoản phí tính vào phí vận chuyển")
            st.caption(
                f"Incoterm trên tờ khai: **{inc or '?'}**. "
                + ("F*: mặc định chỉ tick Ocean Freight. " if inc.startswith("F") else "")
                + ("EXW: mặc định tick Ocean Freight + phí EXW/đại lý. " if inc == "EXW" else "")
                + "Bạn tick thêm/bỏ tick, thêm dòng mới (bấm dòng trống cuối bảng) hoặc sửa số tiền tùy ý. "
                  "Dòng tick 'Đã gồm VAT 8%' sẽ được chia 1,08 để lấy số chưa VAT."
            )
            base_df = pd.DataFrame(rows, columns=["Chọn", "Debit note", "Khoản phí", "Loại", "Số tiền", "Tiền tệ", "Đã gồm VAT 8%"])
            edited = st.data_editor(
                base_df, hide_index=True, use_container_width=True, num_rows="dynamic",
                key=f"dn_editor_{parsed['stamp']}",
                disabled=["Debit note"],
                column_config={
                    "Loại": st.column_config.SelectboxColumn("Loại", options=list(CAT_LABEL.values())),
                    "Số tiền": st.column_config.NumberColumn("Số tiền", format="%.2f"),
                },
            )
            edited = edited.fillna({"Chọn": False, "Debit note": "", "Khoản phí": "", "Loại": "Khác",
                                    "Số tiền": 0.0, "Tiền tệ": "", "Đã gồm VAT 8%": False})
            sel = edited[edited["Chọn"].astype(bool)]
            rates = draft_merged.get("exchange_rates") or {}
            cur_main = next((str(x).strip().upper() for x in sel["Tiền tệ"] if str(x).strip()), "")
            total, charges_sel, fails = 0.0, [], []
            for _, r in sel.iterrows():
                net = charge_net(float(r["Số tiền"]), bool(r["Đã gồm VAT 8%"]))
                conv, note = _convert(net, str(r["Tiền tệ"]).strip().upper() or cur_main, cur_main, rates)
                if conv is None:
                    fails.append(f"{r['Khoản phí']}: {note}")
                    continue
                total += conv
                charges_sel.append({"description": r["Khoản phí"], "amount": round(net, 2), "currency": r["Tiền tệ"]})
            if fails:
                st.warning("Không quy đổi được: " + "; ".join(fails))
            st.info(f"Tổng cước tính vào phí vận chuyển: **{total:,.2f} {cur_main}** ({len(charges_sel)} khoản được chọn)")
            debit_note = {
                "doc_type": "DEBIT_NOTE",
                "debit_note_no": ", ".join(d.get("debit_note_no", "") for d in debit_docs if d.get("debit_note_no")),
                "freight_amount": total, "freight_currency": cur_main, "charges": charges_sel,
            }

        res = run_full_reconciliation(
            draft=draft_merged, invoice=inv_doc, packing_list=pk_doc, bl=bl_doc, arrival_notice=an_doc,
            weight_tol=weight_tol, amount_tol=amount_tol, master=master, export_decls=export_docs,
            debit_note=debit_note, barcodes=barcode_docs, insurance=insurance_docs,
        )

        if master:
            m = getattr(master, "meta", {}) or {}
            st.info(
                f"Từ điển lấy từ báo cáo: **{parsed['hist_name']}** | Doanh nghiệp: **{m.get('ten_dn') or '?'}** "
                f"(MST {m.get('ma_dn') or '?'}) | Khai từ {m.get('tu_ngay') or '?'} đến {m.get('den_ngay') or '?'} | "
                f"{m.get('so_dong', 0):,} dòng hàng -> {len(master):,} mã hàng (lấy theo lần khai gần nhất)."
            )
        else:
            st.warning("Chưa tải báo cáo lịch sử (ô 7) nên chưa đối chiếu mã HS, ĐVT với lần khai gần nhất.")

        decl_df = pd.DataFrame(draft_merged["declarations"])
        with st.expander(f"Bộ tờ khai đã gộp để đối chiếu ({len(decl_df)} tờ khai)", expanded=len(decl_df) > 1):
            st.dataframe(decl_df, use_container_width=True, hide_index=True)

        if inv_doc and inv_doc.get("invoice_breakdown"):
            with st.expander(f"Chi tiết từng invoice ({len(inv_doc['invoice_breakdown'])} invoice) - tổng trị giá được cộng từ đây", expanded=True):
                bd = pd.DataFrame(inv_doc["invoice_breakdown"]).rename(columns={
                    "invoice_no": "Số invoice", "invoice_date": "Ngày", "currency": "Tiền tệ", "total_amount": "Tổng ghi trên invoice",
                    "total_items_sum": "Tổng cộng các dòng", "lines": "Số dòng", "is_non_commercial": "Không thương mại (không thanh toán)",
                    "file_name": "File"})
                st.dataframe(bd, use_container_width=True, hide_index=True)

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
            if notices:
                pd.DataFrame({"Thông báo cần lưu ý": notices}).to_excel(writer, sheet_name="Thong_bao", index=False)
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
