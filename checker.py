from dataclasses import dataclass, asdict
import pandas as pd
import re
from extractors import bl_for_compare, strip_bl_suffix, norm_code, norm_text, desc_score, compare_dates, fmt_date, date_candidates, DESC_THRESHOLD, resolve_items_with_master

# =====================================================================
# NGUYÊN TẮC: TỜ KHAI NHÁP LÀ BẢN ĐƯỢC KIỂM TRA.
# Mỗi chứng từ (Invoice, PL, B/L, A/N) và từ điển lịch sử chỉ được đối chiếu
# với tờ khai; KHÔNG đối chiếu chứng từ này với chứng từ kia.
# Mọi điểm lệch đều diễn giải theo hướng: "tờ khai đang ghi gì / đúng phải là gì".
# Tờ khai đưa vào đây là bộ tờ khai ĐÃ GỘP (extractors.merge_declarations).
# =====================================================================


@dataclass
class IssueRow:
    scope: str          # HEADER hoặc LINE_ITEM
    doc_ref: str        # Chứng từ dùng để đối chiếu với tờ khai
    item_code: str      # Mã hàng (hoặc "HEADER" nếu lỗi cấp lô)
    field_name: str     # Mã chỉ tiêu (HS_CODE, QTY...)
    draft_val: str      # Giá trị đang ghi trên Tờ khai nháp
    doc_val: str        # Giá trị theo chứng từ (giá trị đúng cần có)
    severity: str       # ERROR hoặc WARNING
    note: str           # Điểm cần sửa trên tờ khai


_FIELD_LABEL = {
    "BL_NO": "Số vận đơn", "VESSEL_FLIGHT": "Tàu / chuyến", "ETA": "Ngày hàng đến",
    "CONTAINERS": "Container", "TOTAL_GW": "Tổng Gross Weight", "TOTAL_NW": "Tổng Net Weight",
    "TOTAL_PKG": "Tổng số kiện", "TOTAL_AMOUNT": "Tổng trị giá",
    "ITEM_CODE": "Dòng hàng", "HS_CODE": "Mã HS", "QTY": "Số lượng", "UNIT_PRICE": "Đơn giá",
    "AMOUNT": "Trị giá dòng", "NET_WEIGHT": "Net Weight dòng", "GROSS_WEIGHT": "Gross Weight dòng",
    "UNIT": "Đơn vị tính", "MASTER": "Hàng trong lịch sử",
    "CONT_COUNT": "Số lượng container", "FREIGHT": "Phí vận chuyển", "SEAL": "Số seal container", "MST": "Mã số thuế người nhập khẩu", "INV_SUM": "Tổng các dòng Invoice", "INVOICE_NO": "Số hóa đơn", "INVOICE_DATE": "Ngày hóa đơn", "INTERNAL_REF": "Số quản lý nội bộ", "DECL_SET": "Bộ tờ khai", "DECL_BRANCH": "Tờ khai nhánh", "DECL_BL": "Số vận đơn giữa các tờ khai",
}


def _norm(val) -> str:
    return " ".join(str(val or "").strip().upper().split())


def _num(val) -> float:
    try:
        return float(str(val or 0).replace(",", "").strip())
    except ValueError:
        return 0.0


_UNIT_GROUPS = {
    "PCS": {"PCS", "PCE", "PC", "PIECE", "PIECES", "NOS", "EA", "EACH"},
    "SET": {"SET", "SETS"},
    "KG": {"KG", "KGS", "KGM", "KILOGRAM", "KILOGRAMS"},
    "M": {"M", "MTR", "METER", "METERS", "METRE", "METRES"},
    "YD": {"YD", "YRD", "YARD", "YARDS"},
    "PAIR": {"PR", "PRS", "PAIR", "PAIRS"},
    "ROLL": {"ROL", "ROLL", "ROLLS"},
}


def _unit_group(u: str) -> str:
    u = _norm(u)
    for grp, names in _UNIT_GROUPS.items():
        if u in names:
            return grp
    return ""


def _digits(val) -> str:
    return "".join(ch for ch in str(val or "") if ch.isdigit())


def _hs_conflict(a, b) -> bool:
    """Hai mã HS mâu thuẫn khi các chữ số chung đầu tiên khác nhau (6 số vs 8 số vẫn so được)."""
    a, b = _digits(a), _digits(b)
    n = min(len(a), len(b))
    if n < 4:
        return False
    return a[:n] != b[:n]


def _g(x: float) -> str:
    return f"{x:,.4f}".rstrip("0").rstrip(".") if x else ""


def _short(s, n=70) -> str:
    s = str(s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------------
# Gom dòng hàng
# ---------------------------------------------------------------------

def _identity(it: dict) -> str:
    """Khóa nhận diện 1 dòng hàng của tờ khai: ưu tiên mã nội bộ trong lịch sử, sau đó mã hàng, cuối cùng là mô tả."""
    if it.get("rule_applied"):     # khách có quy tắc tiền tố SX/KD/HT: gộp theo mã gốc, bất kể tờ khai / loại hình
        b = it.get("base_code") or norm_code(it.get("item_code"))
        if b:
            return "B:" + b
    if it.get("master_key"):
        return "M:" + it["master_key"]
    c = norm_code(it.get("item_code"))
    if c:
        return "C:" + c
    return "D:" + norm_text(it.get("description"))[:40]


def _build_groups(draft_lines: list) -> dict:
    groups: dict = {}
    for it in draft_lines:
        g = groups.setdefault(_identity(it), {"lines": [], "codes": set(), "descs": []})
        g["lines"].append(it)
        for c in (norm_code(it.get("item_code")), norm_code(it.get("master_code")), norm_code(it.get("base_code"))):
            if c:
                g["codes"].add(c)
        if it.get("description"):
            g["descs"].append(it["description"])
    return groups


def _pair_doc_line(v: dict, groups: dict):
    """Tìm nhóm dòng hàng của tờ khai tương ứng với 1 dòng trên chứng từ. Trả về (khóa nhóm | None, cách ghép)."""
    mk = v.get("master_key")
    if mk and ("M:" + mk) in groups:
        return "M:" + mk, "MÃ LỊCH SỬ"
    vcodes = [c for c in [norm_code(v.get("item_code"))] + [norm_code(a) for a in (v.get("alt_codes") or [])] if c]
    for idk, g in groups.items():
        if any(c in g["codes"] for c in vcodes):
            return idk, "MÃ HÀNG"
    hits = set()
    for c in vcodes:
        if len(c) < 4:
            continue
        for idk, g in groups.items():
            if any(len(gc) >= 4 and (c in gc or gc in c) for gc in g["codes"]):
                hits.add(idk)
    if len(hits) == 1:
        return hits.pop(), "MÃ GẦN GIỐNG"
    if v.get("description"):
        scored = []
        for idk, g in groups.items():
            sc = max((desc_score(v["description"], d) for d in g["descs"]), default=0.0)
            if sc >= DESC_THRESHOLD:
                scored.append((sc, idk))
        if scored:
            top = max(s for s, _ in scored)
            best = [k for s, k in scored if s >= top - 1e-9]
            if len(best) == 1:
                return best[0], "MÔ TẢ"
    return None, ""


def _sum(lines, key) -> float:
    return sum(_num(x.get(key)) for x in lines)


def _first(lines, key) -> str:
    return next((str(x.get(key)).strip() for x in lines if str(x.get(key) or "").strip()), "")


def _label(g) -> str:
    l0 = g["lines"][0]
    return str(l0.get("base_code") or l0.get("item_code") or "").strip().upper() or _short(l0.get("description"), 30) or "(không mã)"


def _result_first(df: pd.DataFrame) -> pd.DataFrame:
    if "Kết quả" in df.columns:
        df = df[["Kết quả"] + [x for x in df.columns if x != "Kết quả"]]
    return df


def style_result(df: pd.DataFrame):
    """Tô màu từng dòng: OK = xanh lá; không OK = nền hồng, chữ đỏ."""
    def _row(r):
        ok = str(r.get("Kết quả", "")).strip().upper().startswith("OK")
        css = "background-color: #C6EFCE; color: #006100" if ok else "background-color: #FFC7CE; color: #9C0006"
        return [css] * len(r)
    return df.style.apply(_row, axis=1)


def _group_text(g: dict) -> str:
    """VD: TK …1221 d3 E31 (60) + TK …1222 d1 A12 (40) = 100"""
    if len(g["lines"]) < 2:
        return ""
    parts = [f"TK …{str(l.get('decl_no') or '')[-4:]} d{l.get('line_no')} {l.get('decl_type') or ''} ({_g(_num(l.get('qty')))})" for l in g["lines"]]
    return " + ".join(parts) + f" = {_g(_sum(g['lines'], 'qty'))}"


def _convert(amount: float, frm: str, to: str, rates: dict):
    """Đổi tiền theo tỷ giá tính thuế trong tờ khai (VND/1 ngoại tệ). Trả về (số tiền, ghi chú) hoặc (None, lý do)."""
    frm, to = (frm or "").upper(), (to or "").upper()
    if not frm or not to or frm == to:
        return amount, ""
    for cur in (frm, to):
        if cur != "VND" and not rates.get(cur):
            return None, f"tờ khai không có tỷ giá {cur}"
    vnd = amount if frm == "VND" else amount * rates[frm]
    res = vnd if to == "VND" else vnd / rates[to]
    used = ", ".join(f"{k} {v:,.0f}" for k, v in rates.items() if k in (frm, to))
    return res, f"quy đổi {amount:,.2f} {frm} → {res:,.2f} {to} (tỷ giá tính thuế {used})"


_FREIGHT_NOT_INCLUDED = {"EXW", "FCA", "FAS", "FOB"}
_FREIGHT_INCLUDED = {"CIF", "CFR", "CNF", "C&F", "CIP", "CPT", "DAP", "DPU", "DAT", "DDP"}


def _total_of(doc: dict, key: str, item_key: str) -> float:
    return _num(doc.get(key)) or sum(_num(x.get(item_key)) for x in (doc.get("items") or []))


# ---------------------------------------------------------------------
# Đối chiếu chính
# ---------------------------------------------------------------------

def run_full_reconciliation(
    draft: dict,
    invoice: dict | None = None,
    packing_list: dict | None = None,
    bl: dict | None = None,
    arrival_notice: dict | None = None,
    weight_tol: float = 0.02,
    amount_tol: float = 0.02,
    master: dict | None = None,
    export_decls: list | None = None,
    debit_note: dict | None = None,
    barcodes: list | None = None,
) -> dict:
    issues: list[IssueRow] = []
    H = "HEADER"

    # Gắn mã lịch sử cho dòng tờ khai + dòng chứng từ (sửa trực tiếp - app đã deepcopy)
    if master:
        resolve_items_with_master(draft, master, is_draft=True)
        resolve_items_with_master(invoice, master, is_draft=False)
        resolve_items_with_master(packing_list, master, is_draft=False)

    # Ghi chú phát sinh khi gộp tờ khai (thiếu nhánh, khác bill...)
    for n in draft.get("merge_notes") or []:
        issues.append(IssueRow(H, "Bộ tờ khai", H, n["field"], str(n["draft_val"]), str(n["doc_val"]), n["severity"], n["note"]))

    # ---------------- TẦNG 1: THÔNG TIN CHUNG & TỔNG LÔ ----------------
    d_bl_raw = _norm(draft.get("bl_no"))
    d_bl_cmp = norm_code(draft.get("bl_no_cmp") or draft.get("bl_no"))
    stripped = bool(d_bl_cmp) and norm_code(d_bl_raw) != d_bl_cmp
    d_bl_show = f"{d_bl_raw} (số bill dùng để đối chiếu: {draft.get('bl_no_cmp')})" if stripped else d_bl_raw

    cls_set = {str(m_.get("transport_mode_code")) for m_ in (draft.get("decl_meta") or [{"transport_mode_code": draft.get("transport_mode_code")}])}
    alt_cmp = norm_code(bl_for_compare(draft.get("bl_no"), "2"))   # số bill nếu bỏ 6 ký tự đầu
    for ref_name, doc in [("B/L", bl), ("Arrival Notice", arrival_notice)]:
        if doc and d_bl_cmp and norm_code(doc.get("bl_no")):
            doc_key = norm_code(strip_bl_suffix(doc.get("bl_no")))
            master_key = norm_code(strip_bl_suffix(doc.get("master_bl_no")))
            if doc.get("bl_is_master"):
                issues.append(IssueRow(H, ref_name, H, "BL_NO", d_bl_show, _norm(doc.get("bl_no")), "WARNING",
                                       f"{ref_name} chỉ đọc được số MASTER bill, không thấy số HOUSE bill - kiểm tra lại chứng từ"))
            if d_bl_cmp != doc_key:
                if master_key and d_bl_cmp == master_key:
                    issues.append(IssueRow(H, ref_name, H, "BL_NO", d_bl_show, _norm(doc.get("bl_no")), "ERROR",
                                           f"Tờ khai đang khai số MASTER bill ({_norm(doc.get('master_bl_no'))}) - phải khai số HOUSE bill ({_norm(doc.get('bl_no'))}) trên {ref_name}"))
                elif not (cls_set & {"2", "3"}) and alt_cmp and alt_cmp == doc_key:
                    issues.append(IssueRow(H, ref_name, H, "BL_NO", d_bl_show, _norm(doc.get("bl_no")), "WARNING",
                                           f"Mã hiệu phương thức vận chuyển đọc được là '{'/'.join(sorted(x for x in cls_set if x)) or 'trống'}' nên chưa bỏ 6 ký tự đầu, "
                                           f"nhưng bỏ 6 ký tự đầu thì khớp {ref_name} - kiểm tra lại mã hiệu phương thức vận chuyển"))
                else:
                    issues.append(IssueRow(H, ref_name, H, "BL_NO", d_bl_show, _norm(doc.get("bl_no")), "ERROR",
                                           f"Số vận đơn trên tờ khai khác {ref_name} - sửa lại cho đúng {ref_name}"))

    ref_year = next((y for _, _, y in date_candidates(draft.get("eta")) if y), None)
    for ref_name, doc in [("B/L", bl), ("Arrival Notice", arrival_notice)]:
        if doc and _norm(draft.get("vessel_flight")) and _norm(doc.get("vessel_flight")):
            if _norm(draft.get("vessel_flight")) != _norm(doc.get("vessel_flight")):
                issues.append(IssueRow(H, ref_name, H, "VESSEL_FLIGHT", _norm(draft.get("vessel_flight")), _norm(doc.get("vessel_flight")), "WARNING",
                                       f"Tên tàu/chuyến trên tờ khai khác {ref_name}"))
    # Ngày hàng đến: chỉ đối chiếu với Arrival Notice (ngày delivery), đưa cả hai về dd/mm/yyyy
    if arrival_notice and _norm(draft.get("eta")) and _norm(arrival_notice.get("eta")):
        ok, a_fmt, b_fmt = compare_dates(draft.get("eta"), arrival_notice.get("eta"))
        if ok is False:
            issues.append(IssueRow(H, "Arrival Notice", H, "ETA", a_fmt, b_fmt, "WARNING",
                                   "Ngày hàng đến trên tờ khai khác ngày delivery trên Arrival Notice (đã đưa cả hai về dd/mm/yyyy)"))
        elif ok is None and _norm(draft.get("eta")) != _norm(arrival_notice.get("eta")):
            issues.append(IssueRow(H, "Arrival Notice", H, "ETA", _norm(draft.get("eta")), _norm(arrival_notice.get("eta")), "WARNING",
                                   "Không đọc được ngày hàng đến để chuẩn hóa - tự kiểm tra Arrival Notice"))

    # Mã vạch (danh sách container đủ điều kiện qua KVGS): coi là danh sách container/seal đã khai của tờ khai
    barcodes = [b_ for b_ in (barcodes or []) if b_]
    member_nos = {str(m_.get("declaration_no")) for m_ in (draft.get("decl_meta") or [])}
    bar_conts, bar_seals = set(), {}
    for b_ in barcodes:
        if member_nos and b_.get("declaration_no") and b_["declaration_no"] not in member_nos:
            issues.append(IssueRow(H, "Mã vạch", H, "CONTAINERS", f"TK {b_['declaration_no']}", "KHÔNG THUỘC BỘ TỜ KHAI", "WARNING",
                                   f"Mã vạch của tờ khai {b_['declaration_no']} không thuộc các tờ khai đã gộp - vẫn dùng để đối chiếu container, hãy kiểm tra có tải nhầm không"))
        if b_.get("manager_no") and d_bl_raw and norm_code(b_["manager_no"]) != norm_code(d_bl_raw):
            issues.append(IssueRow(H, "Mã vạch", H, "BL_NO", d_bl_raw, b_["manager_no"], "WARNING",
                                   "Số quản lý hàng hóa trên mã vạch khác số vận đơn trên tờ khai"))
        for x in b_.get("container_seals") or []:
            bar_conts.add(_norm(x["container_no"]))
            if x.get("seal_no"):
                bar_seals[_norm(x["container_no"])] = _norm(x["seal_no"])
    d_conts = set(_norm(c) for c in (draft.get("containers") or []) if _norm(c)) | bar_conts
    for ref_name, doc in [("B/L", bl), ("Arrival Notice", arrival_notice)]:
        if not doc or not bar_seals:
            continue
        doc_seals = {_norm(x["container_no"]): _norm(x["seal_no"]) for x in (doc.get("container_seals") or []) if x.get("seal_no")}
        diffs = [f"{k}: mã vạch {v} / {ref_name} {doc_seals[k]}" for k, v in bar_seals.items() if k in doc_seals and norm_code(doc_seals[k]) != norm_code(v)]
        if diffs:
            issues.append(IssueRow(H, ref_name, H, "SEAL", "; ".join(f"{k}: {v}" for k, v in bar_seals.items()),
                                   "; ".join(f"{k}: {v}" for k, v in doc_seals.items()), "ERROR",
                                   "Số seal container trên mã vạch khác " + ref_name + " (" + "; ".join(diffs) + ")"))
    for ref_name, doc in [("B/L", bl), ("Arrival Notice", arrival_notice), ("Packing List", packing_list)]:
        if not doc:
            continue
        c_conts = set(_norm(c) for c in (doc.get("containers") or []) if _norm(c))
        if d_conts and c_conts and d_conts != c_conts:
            missing = sorted(c_conts - d_conts)
            extra = sorted(d_conts - c_conts)
            detail = []
            if missing:
                detail.append("tờ khai THIẾU: " + ", ".join(missing))
            if extra:
                detail.append("tờ khai THỪA: " + ", ".join(extra))
            issues.append(IssueRow(
                H, ref_name, H, "CONTAINERS",
                ", ".join(sorted(d_conts)), ", ".join(sorted(c_conts)),
                "ERROR", f"Danh sách container trên tờ khai khác {ref_name} (" + "; ".join(detail) + ")",
            ))

    # Mã hiệu phương thức vận chuyển = 2: đếm container trên vận đơn, đối chiếu với số lượng container khai trên tờ khai
    meta = draft.get("decl_meta") or [{"declaration_no": draft.get("declaration_no", ""), "transport_mode_code": draft.get("transport_mode_code", ""),
                                       "internal_ref_no": draft.get("internal_ref_no", "")}]
    is2 = any(str(m.get("transport_mode_code")) == "2" for m in meta)
    d_cc = int(_num(draft.get("container_count"))) or len(d_conts)
    b_cc = len({_norm(x) for x in ((bl or {}).get("containers") or []) if _norm(x)})
    if bl and is2 and d_cc and b_cc and d_cc != b_cc:
        issues.append(IssueRow(H, "B/L", H, "CONT_COUNT", str(d_cc), str(b_cc), "ERROR",
                               "Số lượng container khai trên tờ khai khác số container đếm được trên B/L"))

    # Mã số thuế người nhập khẩu trên tờ khai phải trùng doanh nghiệp trong báo cáo lịch sử khai báo
    h_mst = re.sub(r"\D", "", str((getattr(master, "meta", {}) or {}).get("ma_dn") or ""))[:10] if master else ""
    d_mst = re.sub(r"\D", "", str(draft.get("importer_tax_code") or ""))[:10]
    if master and h_mst and d_mst and h_mst != d_mst:
        issues.append(IssueRow(H, "Lịch sử khai báo", H, "MST", d_mst, h_mst, "ERROR",
                               "Báo cáo lịch sử khai báo là của doanh nghiệp khác với người nhập khẩu trên tờ khai - tải lại đúng báo cáo của khách hàng, kết quả đối chiếu HS/ĐVT bên dưới không đáng tin"))

    # Phí vận chuyển: theo Incoterm
    for m_ in meta:
        inc = re.sub(r"[^A-Z&]", "", str(m_.get("incoterms") or "").upper())[:3]
        fr = float(m_.get("freight_amount") or 0)
        who = f"TK {m_.get('declaration_no')}: " if len(meta) > 1 else ""
        if inc in _FREIGHT_NOT_INCLUDED and fr <= 0:
            issues.append(IssueRow(H, "Incoterm", H, "FREIGHT", "CHƯA KHAI", f"Điều kiện {inc}", "WARNING",
                                   f"{who}Điều kiện giá {inc} chưa gồm cước vận chuyển quốc tế nhưng tờ khai chưa khai phí vận chuyển"))
        elif inc in _FREIGHT_INCLUDED and fr > 0:
            issues.append(IssueRow(H, "Incoterm", H, "FREIGHT", f"{fr:,.2f}", f"Điều kiện {inc}", "WARNING",
                                   f"{who}Điều kiện giá {inc} đã gồm cước vận chuyển nhưng tờ khai vẫn khai thêm phí vận chuyển"))

    # Debit note: đối chiếu cước với phí vận chuyển trên tờ khai (đổi tiền theo tỷ giá tính thuế; 1 debit note chia nhiều tờ khai thì phân bổ theo trị giá hàng)
    dn_show = ""
    if debit_note:
        dn_amt = _num(debit_note.get("freight_amount"))
        dn_cur = str(debit_note.get("freight_currency") or "").upper() or str(draft.get("currency") or "USD").upper()
        dn_show = f"{dn_amt:,.2f} {dn_cur}"
        if dn_amt <= 0:
            issues.append(IssueRow(H, "Debit note", H, "FREIGHT", "", "KHÔNG ĐỌC ĐƯỢC CƯỚC", "WARNING", "Không đọc được cước vận chuyển quốc tế trên debit note - tự kiểm tra"))
        else:
            total_gv = sum(float(m_.get("goods_value") or 0) for m_ in meta)
            for m_ in meta:
                share = (float(m_.get("goods_value") or 0) / total_gv) if total_gv > 0 else 1.0 / len(meta)
                fr = float(m_.get("freight_amount") or 0)
                tk_cur = (m_.get("freight_currency") or ("VND" if fr > 100000 else (m_.get("currency") or "USD"))).upper()
                rates = {**(draft.get("exchange_rates") or {}), **(m_.get("exchange_rates") or {})}
                alloc = dn_amt * share
                conv, note = _convert(alloc, dn_cur, tk_cur, rates)
                who = f"TK {m_.get('declaration_no')}: " if len(meta) > 1 else ""
                share_txt = f", phân bổ {share:.1%} theo trị giá hàng" if len(meta) > 1 else ""
                if conv is None:
                    issues.append(IssueRow(H, "Debit note", H, "FREIGHT", f"{fr:,.2f} {tk_cur}", dn_show, "WARNING",
                                           f"{who}Không quy đổi được cước debit note để so với tờ khai ({note})"))
                    continue
                if abs(fr - conv) > max(0.01 * max(fr, conv), 0.01):
                    issues.append(IssueRow(H, "Debit note", H, "FREIGHT", f"{fr:,.2f} {tk_cur}", f"{conv:,.2f} {tk_cur}", "ERROR",
                                           f"{who}Phí vận chuyển trên tờ khai khác debit note (debit note {dn_show}{share_txt}" + (f"; {note}" if note else "") + ")"))

    # Số hóa đơn / ngày hóa đơn: đối chiếu với Invoice
    d_inv_nos = draft.get("invoice_nos") or ([re.sub(r"^\s*[A-Z]\s*-\s*", "", str(draft.get("invoice_no") or "")).strip()] if draft.get("invoice_no") else [])
    d_inv_dates = draft.get("invoice_dates") or ([str(draft.get("invoice_date")).strip()] if str(draft.get("invoice_date") or "").strip() else [])
    if invoice and d_inv_nos and norm_code(invoice.get("invoice_no")):
        cn = norm_code(invoice.get("invoice_no"))
        if not any(norm_code(x) == cn or (len(min(norm_code(x), cn, key=len)) >= 4 and (norm_code(x) in cn or cn in norm_code(x))) for x in d_inv_nos):
            issues.append(IssueRow(H, "Invoice", H, "INVOICE_NO", ", ".join(d_inv_nos), _norm(invoice.get("invoice_no")), "ERROR",
                                   "Số hóa đơn trên tờ khai khác Invoice"))
    if invoice and d_inv_dates and _norm(invoice.get("invoice_date")):
        res_d = [compare_dates(x, invoice.get("invoice_date")) for x in d_inv_dates]
        if not any(r[0] for r in res_d):
            if all(r[0] is False for r in res_d):
                issues.append(IssueRow(H, "Invoice", H, "INVOICE_DATE", res_d[0][1], res_d[0][2], "ERROR", "Ngày hóa đơn trên tờ khai khác Invoice (đã đưa về dd/mm/yyyy)"))
            else:
                issues.append(IssueRow(H, "Invoice", H, "INVOICE_DATE", ", ".join(d_inv_dates), _norm(invoice.get("invoice_date")), "WARNING",
                                       "Không đọc được ngày hóa đơn để chuẩn hóa - tự kiểm tra Invoice"))

    # Tờ khai có mã hiệu phương thức vận chuyển = 9 (nhập khẩu tại chỗ): số quản lý nội bộ NKTC#&xxxxxxxxxxx / NKPTQ#&xxxxxxxxxxx
    # phải trùng 11 số đầu tiên của số tờ khai xuất
    exp_nos = [re.sub(r"\D", "", str(e.get("declaration_no") or "")) for e in (export_decls or [])]
    exp_nos = [x for x in exp_nos if x]
    for m in meta:
        if str(m.get("transport_mode_code")) != "9":
            continue
        ref = str(m.get("internal_ref_no") or "").strip()
        who = f"TK {m.get('declaration_no')}: " if len(meta) > 1 else ""
        mm = re.match(r"\s*(NKTC|NKPTQ)\s*#&\s*(\d{11})", ref.upper())
        if not ref:
            issues.append(IssueRow(H, "Tờ khai xuất", H, "INTERNAL_REF", "KHÔNG ĐỌC ĐƯỢC", "", "WARNING",
                                   f"{who}Tờ khai có mã hiệu phương thức vận chuyển = 9 nhưng không đọc được số quản lý nội bộ của doanh nghiệp - tự kiểm tra"))
        elif not mm:
            issues.append(IssueRow(H, "Tờ khai xuất", H, "INTERNAL_REF", ref, "NKTC#&xxxxxxxxxxx / NKPTQ#&xxxxxxxxxxx", "WARNING",
                                   f"{who}Số quản lý nội bộ không đúng dạng NKTC#&/NKPTQ#& + 11 số - tự kiểm tra"))
        elif not exp_nos:
            issues.append(IssueRow(H, "Tờ khai xuất", H, "INTERNAL_REF", ref, "CHƯA TẢI TỜ KHAI XUẤT", "WARNING",
                                   f"{who}Chưa tải tờ khai xuất để đối chiếu số quản lý nội bộ"))
        elif mm.group(2) not in {x[:11] for x in exp_nos}:
            issues.append(IssueRow(H, "Tờ khai xuất", H, "INTERNAL_REF", ref, "11 số đầu TK xuất: " + ", ".join(sorted({x[:11] for x in exp_nos})), "ERROR",
                                   f"{who}Số quản lý nội bộ ({mm.group(2)}) không trùng 11 số đầu tiên của số tờ khai xuất đã tải lên"))

    d_items_all = draft.get("items") or []
    d_gw = _total_of(draft, "total_gross_weight", "gross_weight")
    d_nw = _total_of(draft, "total_net_weight", "net_weight")
    d_pkg = _total_of(draft, "total_pkg_count", "pkg_count")
    d_amt = _total_of(draft, "total_amount", "amount")

    for ref_name, doc in [("B/L", bl), ("Arrival Notice", arrival_notice), ("Packing List", packing_list), ("Invoice", invoice)]:
        if not doc:
            continue
        c_gw = _total_of(doc, "total_gross_weight", "gross_weight")
        if d_gw > 0 and c_gw > 0 and abs(d_gw - c_gw) > weight_tol:
            issues.append(IssueRow(H, ref_name, H, "TOTAL_GW", f"{d_gw:.3f}", f"{c_gw:.3f}", "ERROR",
                                   f"Tổng Gross Weight trên tờ khai khác {ref_name}"))

    for ref_name, doc in [("Packing List", packing_list), ("Invoice", invoice)]:
        if not doc:
            continue
        c_nw = _total_of(doc, "total_net_weight", "net_weight")
        if d_nw > 0 and c_nw > 0 and abs(d_nw - c_nw) > weight_tol:
            issues.append(IssueRow(H, ref_name, H, "TOTAL_NW", f"{d_nw:.3f}", f"{c_nw:.3f}", "ERROR",
                                   f"Tổng Net Weight trên tờ khai khác {ref_name}"))

    for ref_name, doc in [("B/L", bl), ("Arrival Notice", arrival_notice), ("Packing List", packing_list)]:
        if not doc:
            continue
        c_pkg = _total_of(doc, "total_pkg_count", "pkg_count")
        if d_pkg > 0 and c_pkg > 0 and abs(d_pkg - c_pkg) > 1e-6:
            issues.append(IssueRow(H, ref_name, H, "TOTAL_PKG", f"{d_pkg:g}", f"{c_pkg:g}", "ERROR",
                                   f"Tổng số kiện trên tờ khai khác {ref_name}"))

    if invoice:
        c_amt = _total_of(invoice, "total_amount", "amount")
        if d_amt > 0 and c_amt > 0 and abs(d_amt - c_amt) > amount_tol:
            issues.append(IssueRow(H, "Invoice", H, "TOTAL_AMOUNT", f"{d_amt:.2f}", f"{c_amt:.2f}", "ERROR",
                                   "Tổng trị giá trên tờ khai khác Invoice"))

    # ---------------- TẦNG 2: TỪNG DÒNG HÀNG (mỗi chứng từ 1 bảng cột dọc) ----------------
    groups = _build_groups(d_items_all)
    line_tables: dict[str, pd.DataFrame] = {}

    for idk, g in groups.items():
        hs_set = {_digits(x.get("hs_code")) for x in g["lines"] if _digits(x.get("hs_code"))}
        if len(g["lines"]) > 1 and len(hs_set) > 1 and any(_hs_conflict(a_, b_) for a_ in hs_set for b_ in hs_set):
            issues.append(IssueRow("LINE_ITEM", "Các dòng cùng mã", _label(g), "HS_CODE", ", ".join(sorted(hs_set)), "", "WARNING",
                                   "Các dòng cùng mã gốc (khác tờ khai / loại hình) đang khai mã HS khác nhau"))

    if invoice and (invoice.get("items") or []):
        inv_sum = sum(_num(x.get("amount")) for x in invoice["items"])
        inv_tot = _num(invoice.get("total_amount"))
        if inv_sum > 0 and inv_tot > 0 and abs(inv_sum - inv_tot) > max(amount_tol, 0.001 * inv_tot):
            issues.append(IssueRow(H, "Invoice", H, "INV_SUM", f"{inv_sum:,.2f}", f"{inv_tot:,.2f}", "WARNING",
                                   "Tổng các dòng đọc được trên Invoice khác tổng trị giá ghi trên Invoice - có thể đọc thiếu/sai dòng hoặc có chiết khấu/phí, kiểm tra lại Invoice"))

    for ref_name, doc in [("Invoice", invoice), ("Packing List", packing_list)]:
        if not doc or not (doc.get("items") or []):
            continue
        is_inv = ref_name == "Invoice"

        paired: dict[str, list] = {}
        unmatched = []
        for v in doc["items"]:
            idk, how = _pair_doc_line(v, groups)
            if idk is None:
                unmatched.append(v)
            else:
                paired.setdefault(idk, []).append((v, how))

        flags: dict[str, list[str]] = {}
        agg: dict[str, dict] = {}
        for idk, g in groups.items():
            dl = g["lines"]
            code = _label(g)
            vl = [v for v, _ in paired.get(idk, [])]
            if not vl:
                issues.append(IssueRow("LINE_ITEM", ref_name, code, "ITEM_CODE", "CÓ", "KHÔNG CÓ", "ERROR",
                                       f"Tờ khai có dòng hàng này nhưng {ref_name} không có - kiểm tra lại mã hàng hoặc dòng hàng thừa"))
                flags.setdefault(idk, []).append(f"THIẾU trên {ref_name}")
                continue
            a = {"qty": _sum(vl, "qty"), "amount": _sum(vl, "amount"), "nw": _sum(vl, "net_weight"),
                 "gw": _sum(vl, "gross_weight"), "pkg": _sum(vl, "pkg_count")}
            a["up"] = a["amount"] / a["qty"] if a["qty"] > 0 and a["amount"] > 0 else 0.0
            agg[idk] = a

            def flag(field, d_val, c_val, sev, note):
                issues.append(IssueRow("LINE_ITEM", ref_name, code, field, d_val, c_val, sev, note))
                flags.setdefault(idk, []).append(_FIELD_LABEL.get(field, field))

            # HS: so từng dòng tờ khai với các mã HS ghi trên chứng từ
            c_hs = {_digits(v.get("hs_code")) for v in vl if _digits(v.get("hs_code"))}
            if c_hs:
                for hs_d in {_digits(x.get("hs_code")) for x in dl if _digits(x.get("hs_code"))}:
                    if all(_hs_conflict(hs_d, h) for h in c_hs):
                        flag("HS_CODE", hs_d, ", ".join(sorted(c_hs)), "ERROR", f"Mã HS trên tờ khai khác {ref_name}")

            qty_d = _sum(dl, "qty")
            d_units = {_unit_group(x.get("unit")) for x in dl} - {""}
            c_units = {_unit_group(v.get("unit")) for v in vl} - {""}
            if d_units and c_units and not (d_units & c_units):
                flag("UNIT", ", ".join(sorted({_norm(x.get("unit")) for x in dl if x.get("unit")})),
                     ", ".join(sorted({_norm(v.get("unit")) for v in vl if v.get("unit")})), "WARNING",
                     f"Đơn vị tính trên tờ khai khác {ref_name} - không so được số lượng, tự kiểm tra (quy đổi theo trọng lượng nếu cần)")
            elif qty_d > 0 and a["qty"] > 0 and abs(qty_d - a["qty"]) > 1e-6:
                flag("QTY", f"{qty_d:g}", f"{a['qty']:g}", "ERROR",
                     f"Số lượng trên tờ khai khác {ref_name}" + (f" (đã cộng {len(dl)} dòng tờ khai cùng mã)" if len(dl) > 1 else ""))

            if is_inv:
                amt_d = _sum(dl, "amount")
                up_d = amt_d / qty_d if qty_d > 0 and amt_d > 0 else 0.0
                if up_d > 0 and a["up"] > 0 and abs(up_d - a["up"]) > 0.0001:
                    flag("UNIT_PRICE", f"{up_d:.4f}", f"{a['up']:.4f}", "ERROR", "Đơn giá trên tờ khai khác Invoice")
                if amt_d > 0 and a["amount"] > 0 and abs(amt_d - a["amount"]) > amount_tol:
                    flag("AMOUNT", f"{amt_d:.2f}", f"{a['amount']:.2f}", "ERROR", "Trị giá dòng hàng trên tờ khai khác Invoice")
            else:
                # Tờ khai thường không có NW/GW từng dòng (= 0) -> bỏ qua, không báo lệch giả
                nw_d, gw_d = _sum(dl, "net_weight"), _sum(dl, "gross_weight")
                if nw_d > 0 and a["nw"] > 0 and abs(nw_d - a["nw"]) > weight_tol:
                    flag("NET_WEIGHT", f"{nw_d:.3f}", f"{a['nw']:.3f}", "WARNING", "Net Weight dòng hàng trên tờ khai khác Packing List")
                if gw_d > 0 and a["gw"] > 0 and abs(gw_d - a["gw"]) > weight_tol:
                    flag("GROSS_WEIGHT", f"{gw_d:.3f}", f"{a['gw']:.3f}", "WARNING", "Gross Weight dòng hàng trên tờ khai khác Packing List")

        for v in unmatched:
            code = str(v.get("item_code") or "").strip().upper() or _short(v.get("description"), 30)
            issues.append(IssueRow("LINE_ITEM", ref_name, code, "ITEM_CODE", "KHÔNG CÓ", "CÓ", "ERROR",
                                   f"Tờ khai THIẾU dòng hàng này (có trên {ref_name}) - bổ sung hoặc kiểm tra lại mã hàng"))

        # ----- bảng cột dọc: mỗi dòng = 1 dòng hàng của TỜ KHAI, bên cạnh là dòng tương ứng trên chứng từ -----
        rows = []
        for idk, g in groups.items():
            n_grp = len(g["lines"])
            vl_how = paired.get(idk, [])
            vl = [v for v, _ in vl_how]
            how = ", ".join(sorted({h for _, h in vl_how}))
            res = "OK" if idk not in flags else "; ".join(flags[idk])
            if n_grp > 1:
                how = (how + " | " if how else "") + f"so theo tổng {n_grp} dòng TK cùng mã"
            for it in g["lines"]:
                row = {
                    "TK số": it.get("decl_no", ""), "Dòng TK": str(it.get("line_no", "")),
                    "Mã hàng TK": it.get("item_code", ""), "Mô tả TK": _short(it.get("description")),
                    "HS TK": it.get("hs_code", ""), "SL TK": _g(_num(it.get("qty"))), "ĐVT TK": it.get("unit", ""),
                    "Nhóm gộp": _group_text(g),
                }
                if is_inv:
                    row.update({"Đơn giá TK": _g(_num(it.get("unit_price"))), "Trị giá TK": _g(_num(it.get("amount")))})
                else:
                    row.update({"NW TK": _g(_num(it.get("net_weight"))), "GW TK": _g(_num(it.get("gross_weight")))})
                row.update({
                    f"Mã trên {ref_name}": ", ".join(str(v.get("item_code") or "") for v in vl),
                    f"Mô tả {ref_name}": _short(_first(vl, "description")),
                    f"SL {ref_name}": _g(_sum(vl, "qty")) if vl else "",
                })
                if is_inv:
                    row.update({
                        "HS Invoice": ", ".join(sorted({_digits(v.get('hs_code')) for v in vl if _digits(v.get('hs_code'))})),
                        "Đơn giá Invoice": _g(agg[idk]["up"]) if idk in agg else "",
                        "Trị giá Invoice": _g(_sum(vl, "amount")) if vl else "",
                    })
                else:
                    row.update({
                        "NW PL": _g(_sum(vl, "net_weight")) if vl else "", "GW PL": _g(_sum(vl, "gross_weight")) if vl else "",
                        "Số kiện PL": _g(_sum(vl, "pkg_count")) if vl else "",
                    })
                row.update({"Cách ghép": how, "Kết quả": res})
                rows.append(row)
        for v in unmatched:
            row = {"Mã hàng TK": "", "Mô tả TK": ""}
            row.update({
                f"Mã trên {ref_name}": v.get("item_code", ""), f"Mô tả {ref_name}": _short(v.get("description")),
                f"SL {ref_name}": _g(_num(v.get("qty"))),
                "Cách ghép": "", "Kết quả": f"TỜ KHAI THIẾU dòng này (có trên {ref_name})",
            })
            if is_inv:
                row.update({"HS Invoice": _digits(v.get("hs_code")), "Đơn giá Invoice": _g(_num(v.get("unit_price"))),
                            "Trị giá Invoice": _g(_num(v.get("amount")))})
            else:
                row.update({"NW PL": _g(_num(v.get("net_weight"))), "GW PL": _g(_num(v.get("gross_weight"))),
                            "Số kiện PL": _g(_num(v.get("pkg_count")))})
            rows.append(row)
        line_tables[ref_name] = _result_first(pd.DataFrame(rows).fillna(""))

    # ---------------- TẦNG 3: TỜ KHAI vs TỪ ĐIỂN LỊCH SỬ KHAI BÁO (lần khai gần nhất) ----------------
    # Không đối chiếu đơn giá / trị giá vì thay đổi theo từng lô.
    if master:
        hist_rows = []
        warned = set()
        for it in d_items_all:
            code = str(it.get("item_code") or "").strip().upper() or _short(it.get("description"), 30)
            mt = it.get("match_type", "")
            rec = master.get(it.get("master_key")) if it.get("master_key") else None
            how = "Mã NPL" if it.get("is_npl") else "Mô tả hàng"
            row = {
                "TK số": it.get("decl_no", ""), "Dòng TK": str(it.get("line_no", "")), "Loại hình": it.get("decl_type", ""),
                "Mã hàng TK": it.get("item_code", ""), "Mô tả TK": _short(it.get("description")),
                "Tra theo": how, "Kiểu khớp": mt,
                "HS TK": it.get("hs_code", ""), "ĐVT TK": it.get("unit", ""),
            }
            if not rec:
                if code not in warned:
                    warned.add(code)
                    if mt == "AMBIGUOUS":
                        msg = "Có nhiều hàng trong lịch sử gần giống - tự kiểm tra mã HS, tên hàng, ĐVT"
                    else:
                        msg = "Không tìm thấy hàng này trong báo cáo lịch sử (hàng mới?) - tự kiểm tra mã HS, tên hàng, ĐVT"
                    issues.append(IssueRow("LINE_ITEM", "Lịch sử khai báo", code, "MASTER", "CÓ TRÊN TỜ KHAI", "KHÔNG TÌM THẤY", "WARNING", msg))
                row.update({"Mã trong lịch sử": "", "Tên hàng lịch sử": "", "HS lịch sử": "", "ĐVT lịch sử": "", "Lần khai gần nhất": "",
                            "Kết quả": "KHÔNG TÌM THẤY" if mt != "AMBIGUOUS" else "NHIỀU KẾT QUẢ GẦN GIỐNG"})
                hist_rows.append(row)
                continue

            src = f"TK {rec['decl_no']} ngày {rec['decl_date']}".strip()
            res = []
            hs_d, hs_m = _digits(it.get("hs_code")), _digits(rec.get("hs_code"))
            if hs_d and hs_m and _hs_conflict(hs_d, hs_m):
                extra = ""
                if len(rec.get("hs_all") or []) > 1:
                    extra = f" (các lần khai trước từng dùng: {', '.join(rec['hs_all'])})"
                issues.append(IssueRow("LINE_ITEM", "Lịch sử khai báo", code, "HS_CODE", hs_d, hs_m, "ERROR",
                                       f"Mã HS trên tờ khai khác lần khai gần nhất ({src}){extra}"))
                res.append("Mã HS")
            g_d, g_m = _unit_group(it.get("unit")), _unit_group(rec.get("unit"))
            if g_d and g_m and g_d != g_m:
                issues.append(IssueRow("LINE_ITEM", "Lịch sử khai báo", code, "UNIT", _norm(it.get("unit")), _norm(rec.get("unit")), "WARNING",
                                       f"Đơn vị tính trên tờ khai khác lần khai gần nhất ({src})"))
                res.append("Đơn vị tính")
            row.update({"Mã trong lịch sử": rec["code"], "Tên hàng lịch sử": _short(rec.get("ten_hang"), 90),
                        "HS lịch sử": rec.get("hs_code", ""), "ĐVT lịch sử": rec.get("unit", ""),
                        "Lần khai gần nhất": src, "Kết quả": "OK" if not res else "; ".join(res)})
            hist_rows.append(row)
        line_tables["Lịch sử khai báo"] = _result_first(pd.DataFrame(hist_rows).fillna(""))

    # ---------------- BẢNG KẾT QUẢ ----------------
    raw_df = pd.DataFrame([asdict(x) for x in issues]) if issues else pd.DataFrame(
        columns=["scope", "doc_ref", "item_code", "field_name", "draft_val", "doc_val", "severity", "note"]
    )

    # Bảng thông tin chung: mỗi chỉ tiêu 1 dòng, mỗi chứng từ 1 cột
    def _cont(doc):
        return ", ".join(sorted(_norm(c) for c in (doc.get("containers") or []) if _norm(c))) if doc else ""

    ref_cols = {"B/L": "B/L", "Arrival Notice": "Arrival Notice", "Packing List": "Packing List", "Invoice": "Invoice",
                "Debit note": "Debit note", "Mã vạch": "Mã vạch", "Tờ khai xuất": "Tờ khai xuất"}
    docs = {"B/L": bl, "Arrival Notice": arrival_notice, "Packing List": packing_list, "Invoice": invoice,
            "Debit note": debit_note, "Mã vạch": barcodes or None, "Tờ khai xuất": export_decls}

    def _seal_txt(d):
        rows_ = d.get("container_seals") if isinstance(d, dict) else [x for b_ in (d or []) for x in (b_.get("container_seals") or [])]
        return "; ".join(f"{x['container_no']}: {x['seal_no']}" for x in (rows_ or []) if x.get("seal_no"))

    def _bar_conts(d):
        return ", ".join(sorted({_norm(x["container_no"]) for b_ in (d or []) for x in (b_.get("container_seals") or [])}))

    def _bl_txt(d):
        if not d or not d.get("bl_no"):
            return ""
        return _norm(d.get("bl_no")) + (f" (master: {_norm(d.get('master_bl_no'))})" if d.get("master_bl_no") else "")

    d_fr_total = sum(float(m_.get("freight_amount") or 0) for m_ in meta)
    d_fr_cur = next((m_.get("freight_currency") for m_ in meta if m_.get("freight_currency")), "") or ("VND" if d_fr_total > 100000 else "")

    def _tot(doc, key, item_key, fmt):
        if not doc:
            return ""
        v = _total_of(doc, key, item_key)
        return fmt(v) if v else ""

    def _dd(key):
        return str(draft.get(key) or "").strip()

    inv_no = _dd("invoice_no")
    spec = [
        ("EXPORTER", "Người xuất khẩu", _dd("exporter_name"), lambda d: "", ()),
        ("EXPORTER_ADDR", "Địa chỉ người xuất khẩu", _dd("exporter_address"), lambda d: "", ()),
        ("BL_NO", "Số vận đơn (HOUSE bill)", d_bl_show, _bl_txt, ("B/L", "Arrival Notice")),
        ("TOTAL_PKG", "Số kiện", f"{d_pkg:g}" if d_pkg else "", lambda d: _tot(d, "total_pkg_count", "pkg_count", lambda v: f"{v:g}"), ("B/L", "Arrival Notice", "Packing List")),
        ("TOTAL_GW", "Số ký (Gross Weight, kg)", f"{d_gw:.3f}" if d_gw else "", lambda d: _tot(d, "total_gross_weight", "gross_weight", lambda v: f"{v:.3f}"), ("B/L", "Arrival Notice", "Packing List", "Invoice")),
        ("CONT_COUNT", "Số lượng container (PTVT = 2: đối chiếu với số container trên B/L)",
         str(d_cc) if (is2 and d_cc) else "", lambda d: str(b_cc) if (is2 and d is bl and b_cc) else "", ("B/L",)),
        ("STORAGE", "Địa điểm lưu kho", _dd("storage_location"), lambda d: "", ()),
        ("TRANSPORT", "Phương tiện vận chuyển", _dd("transport_means"), lambda d: _norm(d.get("vessel_flight")) if d else "", ("B/L", "Arrival Notice")),
        ("INVOICE_NO", "Số hóa đơn", ", ".join(d_inv_nos), lambda d: _norm(d.get("invoice_no")) if d else "", ("Invoice",)),
        ("INVOICE_DATE", "Ngày hóa đơn", ", ".join(fmt_date(x) for x in d_inv_dates), lambda d: fmt_date(d.get("invoice_date")) if d and d.get("invoice_date") else "", ("Invoice",)),
        ("INCOTERMS", "Incoterm", _dd("incoterms"), lambda d: str(d.get("incoterms") or "") if d else "", ("Invoice",)),
        ("CURRENCY", "Đơn vị tiền tệ", _dd("currency"), lambda d: str(d.get("currency") or "") if d else "", ("Invoice",)),
        ("TOTAL_AMOUNT", "Tổng trị giá hóa đơn", f"{d_amt:.2f}" if d_amt else "", lambda d: _tot(d, "total_amount", "amount", lambda v: f"{v:.2f}"), ("Invoice",)),
        ("INTERNAL_REF", "Số quản lý nội bộ của DN (PTVT = 9)",
         "; ".join(str(m.get("internal_ref_no") or "").strip() for m in meta if str(m.get("transport_mode_code")) == "9" and str(m.get("internal_ref_no") or "").strip()),
         lambda d: ", ".join(sorted({re.sub(r"\D", "", str(e.get("declaration_no") or "")) for e in d if e.get("declaration_no")})) if d else "", ("Tờ khai xuất",)),
        ("ETA", "Ngày hàng đến (A/N: ngày delivery)", fmt_date(draft.get("eta")), lambda d: fmt_date(d.get("eta"), ref_year) if d and d.get("eta") else "", ("Arrival Notice",)),
        ("CONTAINERS", "Danh sách container", ", ".join(sorted(d_conts)),
         lambda d: _bar_conts(d) if isinstance(d, list) else _cont(d), ("B/L", "Arrival Notice", "Packing List", "Mã vạch")),
        ("SEAL", "Số seal container", "", _seal_txt, ("B/L", "Arrival Notice", "Mã vạch")),
        ("FREIGHT", "Phí vận chuyển", f"{d_fr_total:,.2f} {d_fr_cur}".strip() if d_fr_total else "", lambda d: dn_show, ("Debit note",)),
        ("TOTAL_NW", "Tổng Net Weight", f"{d_nw:.3f}" if d_nw else "", lambda d: _tot(d, "total_net_weight", "net_weight", lambda v: f"{v:.3f}"), ("Packing List", "Invoice")),
    ]
    header_rows = []
    for field, label, d_val, getter, applies in spec:
        row = {"Chỉ tiêu": label, "Tờ khai": d_val}
        for col, ref in ref_cols.items():
            row[col] = getter(docs[ref]) if (docs[ref] and ref in applies) else ""
        sub = raw_df[(raw_df["scope"] == "HEADER") & (raw_df["field_name"] == field) & (raw_df["doc_ref"].isin(ref_cols))] if not raw_df.empty else raw_df
        if not sub.empty:
            row["Kết quả"] = "SAI khi đối chiếu " + ", ".join(sub["doc_ref"].drop_duplicates()) if (sub["severity"] == "ERROR").any() \
                else "CẦN KIỂM TRA khi đối chiếu " + ", ".join(sub["doc_ref"].drop_duplicates())
        elif row["Tờ khai"] and any(row[col] for col in ref_cols) :
            row["Kết quả"] = "OK"
        else:
            row["Kết quả"] = "-"
        if row["Tờ khai"] or any(row[col] for col in ref_cols):
            header_rows.append(row)
    header_table = pd.DataFrame(header_rows)

    # Bảng "điểm cần sửa trên tờ khai": sắp theo mức độ, chung lô trước, dòng hàng sau
    if raw_df.empty:
        issues_df = pd.DataFrame(columns=["Mức độ", "Phạm vi", "Đối chiếu với", "Mã hàng", "Chỉ tiêu",
                                          "Tờ khai đang ghi", "Theo chứng từ", "Điểm cần sửa trên tờ khai"])
    else:
        tmp = raw_df.copy()
        tmp["_sev"] = tmp["severity"].map({"ERROR": 0, "WARNING": 1}).fillna(2)
        tmp["_scope"] = tmp["scope"].map({"HEADER": 0, "LINE_ITEM": 1}).fillna(2)
        tmp = tmp.sort_values(["_sev", "_scope", "doc_ref", "item_code"])
        issues_df = pd.DataFrame({
            "Mức độ": tmp["severity"].map({"ERROR": "SAI", "WARNING": "CẦN KIỂM TRA"}),
            "Phạm vi": tmp["scope"].map({"HEADER": "Chung lô", "LINE_ITEM": "Dòng hàng"}),
            "Đối chiếu với": tmp["doc_ref"],
            "Mã hàng": tmp["item_code"].where(tmp["scope"] != "HEADER", ""),
            "Chỉ tiêu": tmp["field_name"].map(lambda f: _FIELD_LABEL.get(f, f)),
            "Tờ khai đang ghi": tmp["draft_val"],
            "Theo chứng từ": tmp["doc_val"],
            "Điểm cần sửa trên tờ khai": tmp["note"],
        }).reset_index(drop=True)

    err_count = int((raw_df["severity"] == "ERROR").sum()) if not raw_df.empty else 0
    warn_count = int((raw_df["severity"] == "WARNING").sum()) if not raw_df.empty else 0

    rows = [
        {"chi_tieu": "Số tờ khai được gộp để đối chiếu", "gia_tri": draft.get("declaration_no", "")},
        {"chi_tieu": "Số dòng hàng trên tờ khai (sau khi gộp)", "gia_tri": len(d_items_all)},
        {"chi_tieu": "Số điểm SAI trên tờ khai", "gia_tri": err_count},
    ]
    if not raw_df.empty:
        for ref in raw_df["doc_ref"].drop_duplicates():
            n_err = int(((raw_df["doc_ref"] == ref) & (raw_df["severity"] == "ERROR")).sum())
            if n_err:
                rows.append({"chi_tieu": f"   - khi đối chiếu với {ref}", "gia_tri": f"{n_err} sai"})
    warn_df = raw_df[raw_df["severity"] == "WARNING"] if not raw_df.empty else raw_df
    rows.append({"chi_tieu": "Các mục CẦN KIỂM TRA lại trên tờ khai", "gia_tri": "Không có" if warn_df.empty else ""})
    if not warn_df.empty:
        for (field, ref), sub in warn_df.groupby(["field_name", "doc_ref"], sort=False):
            label = _FIELD_LABEL.get(field, field)
            if (sub["scope"] == "LINE_ITEM").all():
                val = "Mã hàng: " + ", ".join(sub["item_code"].drop_duplicates())
            else:
                val = f"tờ khai {sub['draft_val'].iloc[0]} / theo chứng từ {sub['doc_val'].iloc[0]}".strip(" /")
            rows.append({"chi_tieu": f"   • {label} (đối chiếu {ref})", "gia_tri": val})
    rows += [
        {"chi_tieu": "Tổng Gross Weight (tờ khai)", "gia_tri": round(d_gw, 3)},
        {"chi_tieu": "Tổng Net Weight (tờ khai)", "gia_tri": round(d_nw, 3)},
        {"chi_tieu": "Tổng trị giá (tờ khai)", "gia_tri": round(d_amt, 2)},
    ]
    summary_df = pd.DataFrame(rows)
    summary_df["gia_tri"] = summary_df["gia_tri"].astype(str)

    return {
        "summary": summary_df,
        "issues": issues_df,
        "header_table": header_table,
        "line_tables": line_tables,
    }
