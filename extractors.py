import datetime
import functools
import hashlib
import io
import json
import math
import numbers
import re
import time
import os
import pickle
import unicodedata
from typing import Literal, Optional
import pandas as pd
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import APIError


class LineItem(BaseModel):
    item_code: str = Field(default="", description="Mã hàng chính trên dòng hàng (Part No / Item No / SKU)")
    description: str = Field(default="", description="Tên hàng / mô tả hàng hóa của dòng hàng (chép nguyên văn, không gồm mã hàng)")
    alt_codes: list[str] = Field(default_factory=list, description="Các mã khác trên CÙNG dòng hàng (Model, Style, Customer item, mã trong tên hàng...), tối đa 4, chỉ lấy chuỗi dạng mã")
    hs_code: str = Field(default="", description="Mã HS (chỉ giữ chữ số)")
    qty: float = Field(default=0.0, description="Số lượng")
    unit: str = Field(default="", description="Đơn vị tính")
    unit_price: float = Field(default=0.0, description="Đơn giá")
    amount: float = Field(default=0.0, description="Thành tiền / trị giá dòng")
    net_weight: float = Field(default=0.0, description="Trọng lượng tịnh (NW)")
    gross_weight: float = Field(default=0.0, description="Trọng lượng cả bì (GW)")
    pkg_count: float = Field(default=0.0, description="Số kiện")


class ExchangeRate(BaseModel):
    currency: str = Field(default="", description="Mã tiền tệ 3 chữ cái, VD USD")
    rate: float = Field(default=0.0, description="Tỷ giá tính thuế (VND cho 1 đơn vị ngoại tệ)")


class ImportDraftData(BaseModel):
    doc_type: Literal["IMPORT_DRAFT"] = "IMPORT_DRAFT"
    declaration_no: str = Field(default="", description="Số tờ khai")
    importer_tax_code: str = Field(default="", description="Mã số thuế người nhập khẩu")
    freight_currency: str = Field(default="", description="Mã tiền tệ của phí vận chuyển (VD USD, VND)")
    insurance_currency: str = Field(default="", description="Mã tiền tệ của phí bảo hiểm")
    exchange_rates: list[ExchangeRate] = Field(default_factory=list, description="Các tỷ giá tính thuế trên tờ khai (mã tiền tệ + tỷ giá VND)")
    exporter_name: str = Field(default="", description="Tên người xuất khẩu")
    exporter_address: str = Field(default="", description="Địa chỉ người xuất khẩu")
    storage_location: str = Field(default="", description="Địa điểm lưu kho (mã + tên)")
    transport_means: str = Field(default="", description="Phương tiện vận chuyển (tên tàu/chuyến)")
    invoice_date: str = Field(default="", description="Ngày phát hành hóa đơn")
    payment_method: str = Field(default="", description="Phương thức thanh toán (KC, TT, ...)")
    container_count: str = Field(default="", description="Số lượng container khai trên tờ khai")
    internal_ref_no: str = Field(default="", description="Số quản lý của nội bộ doanh nghiệp")
    first_declaration_no: str = Field(default="", description="Số tờ khai đầu tiên (nếu tờ khai được tách nhánh / chia nhỏ), chỉ chữ số")
    branch_no: str = Field(default="", description="Số thứ tự nhánh của tờ khai trong 'Số tờ khai đầu tiên x/y' (phần x)")
    total_branches: str = Field(default="", description="Tổng số nhánh / tờ khai chia nhỏ trong 'Số tờ khai đầu tiên x/y' (phần y)")
    check_class_code: str = Field(default="", description="Mã phân loại kiểm tra (1, 2 hoặc 3)")
    transport_mode_code: str = Field(default="", description="Mã hiệu phương thức vận chuyển: chữ số đứng ngay sau mã loại hình, VD 'E31 2 [4]' thì là 2")
    declaration_type_code: str = Field(default="", description="Mã loại hình tờ khai nhập (VD: A12, E31, A42...)")
    invoice_no: str = Field(default="", description="Số hóa đơn")
    bl_no: str = Field(default="", description="Số vận đơn B/L hoặc AWB")
    vessel_flight: str = Field(default="", description="Tên tàu hoặc chuyến bay")
    eta: str = Field(default="", description="Ngày đến dự kiến ETA / ngày hàng đến")
    containers: list[str] = Field(default_factory=list, description="Danh sách số container")
    currency: str = Field(default="", description="Đồng tiền (USD, VND, EUR, JPY...)")
    incoterms: str = Field(default="", description="Điều kiện giao hàng Incoterms (EXW, FOB, FCA, CIF, CIP, CFR, DAP...)")
    freight_amount: float = Field(default=0.0, description="Phí vận chuyển / cước trên tờ khai nếu có")
    insurance_amount: float = Field(default=0.0, description="Phí bảo hiểm trên tờ khai nếu có")
    total_pkg_count: float = Field(default=0.0, description="Tổng số kiện")
    total_net_weight: float = Field(default=0.0, description="Tổng Net Weight")
    total_gross_weight: float = Field(default=0.0, description="Tổng Gross Weight")
    total_amount: float = Field(default=0.0, description="Tổng trị giá hóa đơn / toàn lô")
    items: list[LineItem] = Field(default_factory=list, description="Danh sách dòng hàng")


class SplitInvPkData(BaseModel):
    """Dữ liệu tách từ 1 file gộp gồm cả Commercial Invoice và Packing List."""
    invoice_no: str = Field(default="", description="Số hóa đơn (Invoice No.) trên Commercial Invoice")
    invoice_date: str = Field(default="", description="Ngày hóa đơn (Invoice Date) trên Commercial Invoice, chép đúng như trên chứng từ")
    invoice_pages: list[int] = Field(default_factory=list, description="Danh sách số trang (bắt đầu từ 1) thuộc Commercial Invoice")
    packing_list_pages: list[int] = Field(default_factory=list, description="Danh sách số trang (bắt đầu từ 1) thuộc Packing List")
    currency: str = Field(default="", description="Đồng tiền trên Invoice")
    incoterms: str = Field(default="", description="Incoterms trên Invoice")
    total_amount: float = Field(default=0.0, description="Tổng trị giá trên Invoice")
    is_non_commercial: bool = Field(default=False, description="True nếu Invoice là NON COMMERCIAL INVOICE / hàng không thanh toán (có chữ NON COMMERCIAL, FOC, FREE OF CHARGE, NO PAYMENT / NOPAYMENT, SAMPLE không thanh toán, NO COMMERCIAL VALUE)")
    payment_terms: str = Field(default="", description="Điều khoản thanh toán chép đúng như in (VD: T/T 30 DAYS, NOPAYMENT)")
    total_net_weight: float = Field(default=0.0, description="Tổng Net Weight trên Packing List")
    total_gross_weight: float = Field(default=0.0, description="Tổng Gross Weight trên Packing List")
    total_pkg_count: float = Field(default=0.0, description="Tổng số kiện trên Packing List")
    invoice_items: list[LineItem] = Field(default_factory=list, description="Các dòng hàng lấy từ phần Commercial Invoice")
    packing_list_items: list[LineItem] = Field(default_factory=list, description="Các dòng hàng lấy từ phần Packing List")


class ChemicalDocData(BaseModel):
    chemicals: list[dict] = Field(default_factory=list, description="Danh sách hóa chất khai báo: ten_hoa_chat, ma_hs, so_luong, cas_no")


def clean_hs(hs: str) -> str:
    return "".join(ch for ch in str(hs or "") if ch.isdigit())


def _parse_vnaccs_number(val) -> float:
    """Chuyển chuỗi số VNACCS (VD: '1.460,8574' hoặc '63,45' hoặc '5.281.885') sang float."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0.0
    if isinstance(val, numbers.Real) and not isinstance(val, bool):
        # Đã là số (JSON của Gemini, ô số của Excel): giữ nguyên, KHÔNG đoán dấu chấm/phẩy
        # (trước đây 522.738 bị đọc thành 522738 vì tưởng dấu chấm là ngăn cách hàng nghìn)
        return 0.0 if (isinstance(val, float) and math.isnan(val)) else float(val)
    s = str(val).strip()
    if not s or s == "-":
        return 0.0
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) == 3 and parts[0].isdigit():
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    elif "." in s:
        parts = s.split(".")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) <= 3 and s.count(".") == 1):
            if len(parts[-1]) == 3 and all(p.isdigit() for p in parts):
                s = "".join(parts)
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_vn_text(val) -> float:
    """Số dạng chữ trên tờ khai VNACCS: dấu chấm = ngăn cách hàng nghìn, dấu phẩy = thập phân
    (VD '522,738' = 522.738; '7.578,26' = 7578.26; '25.640' = 25640)."""
    s = re.sub(r"\s", "", str(val if val is not None else ""))
    if not s or s == "-":
        return 0.0
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _extract_incoterm_currency(text: str) -> tuple[str, str]:
    """Tách Incoterms (EXW, FOB, FCA, FAS, CFR, CIF, CPT, CIP, DAP, DPU, DDP...) và mã tiền tệ."""
    up = str(text or "").upper()
    incoterms_list = ["EXW", "FCA", "FAS", "FOB", "CFR", "CIF", "CPT", "CIP", "DAP", "DPU", "DDP"]
    found_inco = ""
    for code in incoterms_list:
        if re.search(rf"\b{code}\b", up):
            found_inco = code
            break
    curr_match = re.search(r"\b(USD|VND|EUR|JPY|CNY|KRW|GBP|SGD|THB|TWD)\b", up)
    found_curr = curr_match.group(1) if curr_match else ""
    return found_inco, found_curr


# Thời gian chờ tối đa cho mỗi lần gọi Gemini (mili giây) - quá thời gian thì báo lỗi thay vì đứng im
GEMINI_TIMEOUT_MS = 150_000


def _call_gemini_json(
    file_bytes: bytes,
    prompt: str,
    schema,
    api_key: str,
    model_name: str = "gemini-3.5-flash-lite",
    max_retries: int = 3,
) -> dict:
    try:
        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS))
    except Exception:
        client = genai.Client(api_key=api_key)

    # Model chọn trước, nếu bận thì lần lượt thử các model dự phòng
    fallback_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
    models_to_try = [model_name] + [m for m in fallback_models if m != model_name]

    retry_codes = {429, 500, 502, 503, 504}
    last_exc = None

    for model in models_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=[
                        types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.0 if attempt == 1 else 0.3,
                        max_output_tokens=65536,
                    ),
                )
                text = (resp.text or "").strip()
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
                try:
                    return json.loads(text)
                except json.JSONDecodeError as jerr:
                    # JSON hỏng/bị cắt: gọi lại, không tự "vá" để tránh mất dòng hàng
                    finish = getattr(resp.candidates[0], "finish_reason", "?") if resp.candidates else "?"
                    last_exc = ValueError(
                        f"JSON không hợp lệ ({jerr}); finish_reason={finish}; "
                        f"độ dài={len(text)}; ĐUÔI={text[-300:]!r}"
                    )
                    if attempt < max_retries:
                        time.sleep(1)
                        continue
                    break  # hết lượt với model này, chuyển model khác
            except APIError as exc:
                last_exc = exc
                code = getattr(exc, "code", None)
                if code in retry_codes:
                    if attempt < max_retries:
                        time.sleep(min(2 * (2 ** (attempt - 1)), 10))  # 2s, 4s
                        continue
                    break  # hết lượt với model này, chuyển model khác
                raise  # lỗi khác (sai key, sai tên model...) thì báo ngay
    raise last_exc


def _normalize_items(raw_items: list[dict]) -> list[dict]:
    cleaned = []
    for it in raw_items or []:
        code = str(it.get("item_code") or "").strip().upper()
        qty = _parse_vnaccs_number(it.get("qty") or 0.0)
        if not code and qty == 0:
            continue
        cleaned.append({
            "item_code": code,
            "alt_codes": [str(a).strip().upper() for a in (it.get("alt_codes") or []) if str(a).strip()][:5],
            "description": str(it.get("description") or "").strip(),
            "hs_code": clean_hs(it.get("hs_code") or ""),
            "qty": qty,
            "unit": str(it.get("unit") or "").strip().upper(),
            "unit_price": _parse_vnaccs_number(it.get("unit_price") or 0.0),
            "amount": _parse_vnaccs_number(it.get("amount") or 0.0),
            "net_weight": _parse_vnaccs_number(it.get("net_weight") or 0.0),
            "gross_weight": _parse_vnaccs_number(it.get("gross_weight") or 0.0),
            "pkg_count": _parse_vnaccs_number(it.get("pkg_count") or 0.0),
        })
    return cleaned


def _parse_vnaccs_import_xls(raw: bytes) -> dict:
    """Đọc tờ khai nhập nháp dạng Excel xuất theo mẫu VNACCS (sheet TKN)."""
    xl = pd.ExcelFile(io.BytesIO(raw))
    sheet_name = "TKN" if "TKN" in xl.sheet_names else xl.sheet_names[0]
    df = xl.parse(sheet_name, header=None)
    n_rows, n_cols = df.shape

    def cell(r: int, c: int) -> str:
        if 0 <= r < n_rows and 0 <= c < n_cols and pd.notna(df.iloc[r, c]):
            return str(df.iloc[r, c]).strip()
        return ""

    def num_at(r: int, c: int) -> float:
        """Đọc số tại ô: ô kiểu số giữ nguyên, ô kiểu chữ đọc theo quy ước VNACCS (chấm = nghìn, phẩy = thập phân)."""
        if not (0 <= r < n_rows and 0 <= c < n_cols):
            return 0.0
        v = df.iloc[r, c]
        if not isinstance(v, str) and pd.isna(v):
            return 0.0
        if isinstance(v, numbers.Real) and not isinstance(v, bool):
            return float(v)
        return _parse_vn_text(v)

    # ---- đọc theo NHÃN (không phụ thuộc cố định vị trí ô) ----
    item_row0 = next((r for r in range(n_rows) if cell(r, 2).lower() == "mã số hàng hóa"), n_rows)
    page1_end = next((r for r in range(1, n_rows) if cell(r, 2).startswith("<IMP>")), item_row0)
    page1_end = min(page1_end, item_row0)

    _nt_cache: dict = {}

    def _nt(r: int, c: int) -> str:
        k = (r, c)
        if k not in _nt_cache:
            _nt_cache[k] = norm_text(cell(r, c))
        return _nt_cache[k]

    def find_label(label: str, last_row: int = None, contains: bool = False):
        """Tìm ô có nhãn - không phân biệt hoa/thường, dấu, kiểu gõ Unicode (NFC/NFD)."""
        last = page1_end if last_row is None else last_row
        key = norm_text(label)
        for r in range(min(last, n_rows)):
            for c in range(n_cols):
                t = _nt(r, c)
                if t and (key in t if contains else t.startswith(key)):
                    return r, c
        return None

    def right_cells(r: int, c: int) -> list:
        return [(cc, cell(r, cc)) for cc in range(c + 1, n_cols) if cell(r, cc)]

    def value_right(label: str, last_row: int = None, pattern: str = "") -> str:
        pos = find_label(label, last_row)
        if not pos:
            return ""
        for _, v in right_cells(*pos):
            if pattern and not re.fullmatch(pattern, v):
                continue
            return v
        return ""

    decl_no = cell(3, 4)
    first_no = re.sub(r"\D", "", cell(3, 17))
    branch_no = re.sub(r"\D", "", cell(3, 21))
    total_branches = re.sub(r"\D", "", cell(3, 23))
    # Mã hiệu phương thức vận chuyển: chữ số đứng sau mã loại hình ("E31  2 [ 4 ]" -> 2)
    _mt = re.match(r"\s*[A-Z]\d{2}\s+(\d)", decl_type_raw.upper()) if (decl_type_raw := cell(5, 15)) else None
    transport_mode = _mt.group(1) if _mt else ""
    # Mã phân loại kiểm tra (ô I6): ưu tiên đọc theo nhãn, không có thì lấy đúng ô I6
    check_class = ""
    _pos = find_label("Mã phân loại kiểm tra", contains=True)
    if _pos:
        for _, v in right_cells(*_pos):
            if re.fullmatch(r"\d(\.0)?", v):
                check_class = v
                break
    if not check_class:
        check_class = cell(5, 8)
    decl_type = cell(5, 15)

    exporter_name, exporter_addr = "", ""
    pos = find_label("Người xuất khẩu")
    if pos:
        r0 = pos[0]
        addr_parts, in_addr = [], False
        for r in range(r0 + 1, min(r0 + 10, page1_end)):
            lab = cell(r, 3).lower()
            if lab.startswith("tên") and not exporter_name:
                exporter_name = " ".join(v for cc, v in right_cells(r, 3))
            elif lab.startswith("địa chỉ"):
                in_addr = True
            elif lab.startswith("mã nước"):
                break
            if in_addr:
                addr_parts += [v for cc, v in right_cells(r, 3)]
        exporter_addr = " ".join(addr_parts)

    def joined_right(label: str, also_next_row: bool = False) -> str:
        pos = find_label(label)
        if not pos:
            return ""
        vals = [v for _, v in right_cells(*pos)]
        if not vals and also_next_row:
            vals = [v for cc, v in right_cells(pos[0] + 1, pos[1] - 1)]
        return " - ".join(vals)

    storage_location = joined_right("Địa điểm lưu kho")
    transport_means = joined_right("Phương tiện vận chuyển", also_next_row=True)
    invoice_date = value_right("Ngày phát hành")
    payment_method = value_right("Phương thức thanh toán")
    container_count = re.sub(r"\D", "", value_right("Số lượng container", pattern=r"\d+(\.0)?").split(".")[0])
    internal_ref = value_right("Số quản lý của nội bộ doanh nghiệp", last_row=item_row0)
    if internal_ref.lower().startswith("số quản lý"):
        internal_ref = ""
    bl_no = cell(30, 3) or cell(30, 2)
    eta = cell(34, 20)
    total_pkg = num_at(35, 10)
    total_gw = num_at(36, 10)
    invoice_no = cell(40, 9)

    inv_meta_str = cell(44, 9)
    incoterms, currency = _extract_incoterm_currency(inv_meta_str)
    total_amount = num_at(44, 15)

    def adjustment(label: str) -> tuple:
        """Phí vận chuyển / bảo hiểm: ô đầu dạng 'D - VND - ' (phân loại - tiền tệ - ...), số tiền nằm ở ô bên phải."""
        pos = find_label(label)
        if not pos:
            return 0.0, ""
        cur, amt = "", 0.0
        for i, (cc, v) in enumerate(right_cells(*pos)):
            if i == 0 and "-" in v:
                m = re.findall(r"\b[A-Z]{3}\b", v.upper())
                cur = m[0] if m else ""
                continue
            if re.fullmatch(r"[\d.,]+", v.strip()):
                amt = num_at(pos[0], cc)
                break
        return amt, cur

    freight_amount, freight_currency = adjustment("Phí vận chuyển")
    insurance_amount, insurance_currency = adjustment("Phí bảo hiểm")

    # Tỷ giá tính thuế: 'USD ... 25.890' (có thể nhiều dòng tiền tệ)
    exchange_rates: dict = {}
    _pos = find_label("Tỷ giá tính thuế")
    if _pos:
        r0, c0 = _pos
        for r in range(r0, min(r0 + 4, n_rows)):
            toks = [cell(r, c) for c in range(c0, n_cols) if cell(r, c)]
            for i, t in enumerate(toks):
                if re.fullmatch(r"[A-Z]{3}", t):
                    for t2 in toks[i + 1:]:
                        if re.fullmatch(r"[\d.,]+", t2):
                            exchange_rates[t] = _parse_vn_text(t2)
                            break

    importer_tax = ""
    _pos = find_label("Người nhập khẩu")
    if _pos:
        for r in range(_pos[0] + 1, min(_pos[0] + 4, page1_end)):
            if cell(r, 3).lower() == "mã":
                importer_tax = re.sub(r"\D", "", " ".join(v for _, v in right_cells(r, 3)))
                break

    items = []
    for r in range(n_rows):
        lbl = cell(r, 2).lower()
        if lbl == "mã số hàng hóa":
            hs_code = clean_hs(cell(r, 6))
            desc_full = cell(r + 1, 6)
            item_code = ""
            description = desc_full
            if "#&" in desc_full:
                parts = desc_full.split("#&", 1)
                item_code = parts[0].strip().upper()
                description = parts[1].strip()

            qty = num_at(r + 4, 21)
            unit = cell(r + 4, 30).upper()
            amount = num_at(r + 6, 8)
            unit_price = num_at(r + 6, 21)

            items.append({
                "item_code": item_code,
                "description": description,
                "hs_code": hs_code,
                "qty": qty,
                "unit": unit,
                "unit_price": unit_price,
                "amount": amount,
                "net_weight": 0.0,
                "gross_weight": 0.0,
                "pkg_count": 0.0,
            })

    return {
        "doc_type": "IMPORT_DRAFT",
        "declaration_no": decl_no,
        "exporter_name": exporter_name,
        "exporter_address": exporter_addr,
        "storage_location": storage_location,
        "transport_means": transport_means,
        "invoice_date": invoice_date,
        "payment_method": payment_method,
        "container_count": container_count,
        "internal_ref_no": internal_ref,
        "first_declaration_no": first_no,
        "branch_no": branch_no,
        "total_branches": total_branches,
        "check_class_code": check_class,
        "transport_mode_code": transport_mode,
        "declaration_type_code": decl_type,
        "invoice_no": invoice_no,
        "bl_no": bl_no,
        "vessel_flight": "",
        "eta": eta,
        "containers": [],
        "currency": currency,
        "incoterms": incoterms,
        "freight_amount": freight_amount,
        "freight_currency": freight_currency,
        "insurance_amount": insurance_amount,
        "insurance_currency": insurance_currency,
        "exchange_rates": exchange_rates,
        "importer_tax_code": importer_tax,
        "total_pkg_count": total_pkg,
        "total_net_weight": 0.0,
        "total_gross_weight": total_gw,
        "total_amount": total_amount,
        "items": items,
    }


def _load_import_draft_raw(
    uploaded_file,
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.5-flash-lite",
) -> dict:
    """Đọc file Tờ khai nhập nháp (Excel mẫu VNACCS/bảng hoặc PDF) và trích xuất dữ liệu."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(io.BytesIO(raw))
        if "TKN" in xl.sheet_names:
            return _parse_vnaccs_import_xls(raw)

        df = xl.parse(xl.sheet_names[0])
        col_map = {
            "item_code": ["item code", "part no", "sku", "mã hàng", "ma hang", "mã sản phẩm"],
            "description": ["description", "tên hàng", "mo ta", "mô tả hàng hóa"],
            "hs_code": ["hs code", "hscode", "mã hs", "ma hs", "mã số hàng hóa"],
            "qty": ["qty", "quantity", "số lượng", "so luong", "lượng"],
            "unit": ["unit", "uom", "đvt", "đơn vị tính"],
            "unit_price": ["unit price", "đơn giá", "don gia"],
            "amount": ["amount", "thành tiền", "trị giá", "tri gia", "trị giá dòng"],
            "net_weight": ["net weight", "nw", "trọng lượng tịnh"],
            "gross_weight": ["gross weight", "gw", "trọng lượng cả bì"],
            "pkg_count": ["pkgs", "số kiện"],
            "incoterms": ["incoterms", "incoterm", "điều kiện giao hàng"],
            "freight_amount": ["freight", "cước", "phí vận chuyển", "cước vận chuyển"],
            "insurance_amount": ["insurance", "bảo hiểm", "phí bảo hiểm"],
        }
        renamed = {}
        for col in df.columns:
            c_low = str(col).strip().lower()
            for std, aliases in col_map.items():
                if c_low in aliases:
                    renamed[col] = std
                    break
        df = df.rename(columns=renamed)
        items = _normalize_items(df.to_dict(orient="records"))
        incoterms = str(df["incoterms"].dropna().iloc[0]).strip().upper() if "incoterms" in df.columns and not df["incoterms"].dropna().empty else ""
        freight = _parse_vnaccs_number(df["freight_amount"].dropna().iloc[0]) if "freight_amount" in df.columns and not df["freight_amount"].dropna().empty else 0.0
        insurance = _parse_vnaccs_number(df["insurance_amount"].dropna().iloc[0]) if "insurance_amount" in df.columns and not df["insurance_amount"].dropna().empty else 0.0

        return {
            "doc_type": "IMPORT_DRAFT",
            "declaration_no": "",
            "declaration_type_code": "",
            "invoice_no": "",
            "bl_no": "",
            "vessel_flight": "",
            "eta": "",
            "containers": [],
            "currency": "",
            "incoterms": incoterms,
            "freight_amount": freight,
            "insurance_amount": insurance,
            "total_pkg_count": sum(x["pkg_count"] for x in items),
            "total_net_weight": sum(x["net_weight"] for x in items),
            "total_gross_weight": sum(x["gross_weight"] for x in items),
            "total_amount": sum(x["amount"] for x in items),
            "items": items,
        }

    if name.endswith(".pdf"):
        if not api_key:
            raise ValueError("Cần nhập Gemini API Key để đọc file PDF tờ khai nhập nháp.")
        prompt = (
            "Đây là TỜ KHAI NHẬP KHẨU NHÁP. Hãy trích xuất chính xác: số tờ khai, mã loại hình, số hóa đơn, số B/L, "
            "tàu/chuyến, ETA/ngày hàng đến, danh sách container, đồng tiền, Incoterms (VD: EXW, FOB, FCA, CIF, CIP, CFR, DAP...), "
            "cước vận chuyển/phí bảo hiểm nếu có trên tờ khai, tổng kiện/NW/GW/trị giá và danh sách dòng hàng "
            "(mã hàng, tên hàng, mã HS, SL, ĐVT, đơn giá, thành tiền). Đọc thêm: mã phân loại kiểm tra (1/2/3), "
            "'Số tờ khai đầu tiên' và phần nhánh x/y đi kèm (x = số thứ tự nhánh, y = tổng số nhánh), "
            "và mã hiệu phương thức vận chuyển (chữ số đứng ngay sau mã loại hình, VD 'E31 2 [4]' thì là 2)."
        )
        data = _call_gemini_json(raw, prompt, ImportDraftData, api_key=api_key, model_name=model_name)
        data["incoterms"] = str(data.get("incoterms") or "").strip().upper()
        data["containers"] = [str(c).strip().upper() for c in (data.get("containers") or []) if str(c).strip()]
        data["items"] = _normalize_items(data.get("items") or [])
        return data

    raise ValueError(f"Chưa hỗ trợ định dạng file tờ khai nháp: {uploaded_file.name}")


_NONCOMM_RX = re.compile(r"NON[\s\-]*COMMERCIAL|NO[\s\-]*COMMERCIAL[\s\-]*VALUE|\bFOC\b|FREE[\s\-]*OF[\s\-]*CHARGE|NO[\s\-]*PAYMENT", re.I)


def _excel_non_commercial(raw: bytes) -> bool:
    """Excel invoice: có chữ NON COMMERCIAL / FOC / NOPAYMENT... ở bất kỳ ô nào không."""
    try:
        xl = pd.ExcelFile(io.BytesIO(raw))
        for sh in xl.sheet_names:
            df_ = xl.parse(sh, header=None, nrows=60)
            for v in df_.astype(str).values.ravel():
                if _NONCOMM_RX.search(v):
                    return True
    except Exception:
        pass
    return False


def load_combined_invoice_packing_list(
    uploaded_file,
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.5-flash-lite",
) -> tuple[dict, dict, dict]:
    """
    Đọc file gộp Invoice + Packing List:
    Tự phân định trang/sheet nào là Invoice, trang/sheet nào là Packing List.
    """
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(io.BytesIO(raw))
        inv_items, pk_items = [], []
        inv_sheets, pk_sheets = [], []
        for s_name in xl.sheet_names:
            df_s = xl.parse(s_name)
            df_s = df_s.rename(columns=lambda c: _EXCEL_ALIAS_FLAT.get(str(c).strip().lower(), c))
            if "item_code" not in df_s.columns:
                continue
            norm_s = [x for x in _normalize_items(df_s.to_dict(orient="records")) if x["item_code"]]
            low_s = s_name.lower()
            if any(k in low_s for k in ("pack", "đóng gói", "dong goi")) or re.search(r"\bpl\b", low_s):
                is_pk = True
            elif any(k in low_s for k in ("invoice", "inv", "hóa đơn", "hoa don")):
                is_pk = False
            else:   # đoán theo cột: có đơn giá / trị giá -> invoice, chỉ có trọng lượng / kiện -> packing list
                has_money = "amount" in df_s.columns or "unit_price" in df_s.columns
                has_weight = any(c in df_s.columns for c in ("net_weight", "gross_weight", "pkg_count"))
                is_pk = has_weight and not has_money
            if is_pk:
                pk_sheets.append(s_name)
                pk_items.extend(norm_s)
            else:
                inv_sheets.append(s_name)
                inv_items.extend(norm_s)

        inv_no, inv_date = _excel_invoice_meta(raw)
        inv_doc = {
            "doc_type": "INVOICE",
            "invoice_no": inv_no,
            "invoice_date": inv_date,
            "containers": [],
            "total_amount": sum(x["amount"] for x in inv_items),
            "total_net_weight": sum(x["net_weight"] for x in inv_items),
            "total_gross_weight": sum(x["gross_weight"] for x in inv_items),
            "total_pkg_count": sum(x["pkg_count"] for x in inv_items),
            "total_items_sum": sum(x["amount"] for x in inv_items),
            "is_non_commercial": _excel_non_commercial(raw),
            "file_name": uploaded_file.name,
            "items": inv_items,
        }
        pk_doc = {
            "doc_type": "PACKING_LIST",
            "containers": [],
            "total_amount": sum(x["amount"] for x in pk_items),
            "total_net_weight": sum(x["net_weight"] for x in pk_items),
            "total_gross_weight": sum(x["gross_weight"] for x in pk_items),
            "total_pkg_count": sum(x["pkg_count"] for x in pk_items),
            "items": pk_items,
        }
        return inv_doc, pk_doc, {"invoice_pages": inv_sheets, "packing_list_pages": pk_sheets}

    if name.endswith(".pdf"):
        if not api_key:
            raise ValueError("Cần nhập Gemini API Key để phân tách và đọc file PDF gộp Invoice + Packing List.")
        prompt = (
            "File PDF này gộp chung chứng từ xuất nhập khẩu (có thể gồm trang Commercial Invoice và trang Packing List). "
            "Hãy quét từng trang (đánh số từ 1) để tự phân định trang nào là Commercial Invoice, trang nào là Packing List "
            "(1 trang có thể chứa cả 2 nếu in chung). Trích xuất riêng invoice_items và packing_list_items, "
            "cùng tổng trị giá, Incoterms, đồng tiền, tổng NW, tổng GW, tổng số kiện. "
            "Với mỗi dòng hàng: item_code là mã hàng chính; alt_codes là các mã khác trên cùng dòng "
            "(Model, Style, Customer item, mã nằm trong tên hàng...), chỉ chép nguyên văn chuỗi mã, không chép mô tả. "
            "Đọc thêm số hóa đơn (Invoice No.) và ngày hóa đơn (Invoice Date) trên Commercial Invoice. "
            "Nếu tiêu đề là NON COMMERCIAL INVOICE hoặc hàng không thanh toán (FOC, FREE OF CHARGE, NO PAYMENT / NOPAYMENT, hàng mẫu không thanh toán) "
            "thì đặt is_non_commercial = true; vẫn đọc số tiền ghi trên invoice (giá trị khai báo) vào amount / total_amount. "
            "Ghi điều khoản thanh toán (PAYMENT) vào payment_terms. "
            "Số có dấu chấm hoặc phẩy phải giữ đúng số thập phân (VD 522.738 là năm trăm hai mươi hai phẩy bảy ba tám, KHÔNG phải 522738)."
        )
        split = _call_gemini_json(raw, prompt, SplitInvPkData, api_key=api_key, model_name=model_name)
        inv_items = _normalize_items(split.get("invoice_items") or [])
        pk_items = _normalize_items(split.get("packing_list_items") or [])

        inv_doc = {
            "doc_type": "INVOICE",
            "invoice_no": str(split.get("invoice_no") or "").strip(),
            "invoice_date": str(split.get("invoice_date") or "").strip(),
            "currency": str(split.get("currency") or "").strip().upper(),
            "incoterms": str(split.get("incoterms") or "").strip().upper(),
            "total_amount": _parse_vnaccs_number(split.get("total_amount") or 0.0) or sum(x["amount"] for x in inv_items),
            "total_net_weight": sum(x["net_weight"] for x in inv_items),
            "total_gross_weight": sum(x["gross_weight"] for x in inv_items),
            "total_pkg_count": sum(x["pkg_count"] for x in inv_items),
            "total_items_sum": sum(x["amount"] for x in inv_items),
            "is_non_commercial": bool(split.get("is_non_commercial")) or bool(_NONCOMM_RX.search(str(split.get("payment_terms") or ""))),
            "payment_terms": str(split.get("payment_terms") or "").strip(),
            "file_name": uploaded_file.name,
            "items": inv_items,
        }
        pk_doc = {
            "doc_type": "PACKING_LIST",
            "total_amount": sum(x["amount"] for x in pk_items),
            "total_net_weight": _parse_vnaccs_number(split.get("total_net_weight") or 0.0) or sum(x["net_weight"] for x in pk_items),
            "total_gross_weight": _parse_vnaccs_number(split.get("total_gross_weight") or 0.0) or sum(x["gross_weight"] for x in pk_items),
            "total_pkg_count": _parse_vnaccs_number(split.get("total_pkg_count") or 0.0) or sum(x["pkg_count"] for x in pk_items),
            "items": pk_items,
        }
        page_info = {
            "invoice_pages": split.get("invoice_pages") or [],
            "packing_list_pages": split.get("packing_list_pages") or [],
        }
        return inv_doc, pk_doc, page_info

    raise ValueError(f"Chưa hỗ trợ định dạng file Invoice/Packing List: {uploaded_file.name}")


def merge_invoice_packing(results: list) -> tuple:
    """Gộp kết quả của nhiều file Invoice/Packing List (mỗi file đã được tách bằng load_combined_invoice_packing_list)."""
    def combine(docs):
        docs = [d for d in docs if d and (d.get("items") or d.get("total_amount"))]
        if not docs:
            return None
        out = dict(docs[0])
        out["items"] = [it for d in docs for it in (d.get("items") or [])]
        for k in ("total_amount", "total_net_weight", "total_gross_weight", "total_pkg_count"):
            out[k] = sum(float(d.get(k) or 0) for d in docs)
        for k in ("invoice_no", "invoice_date", "currency", "incoterms"):
            out[k] = next((d.get(k) for d in docs if str(d.get(k) or "").strip()), "")
        out.setdefault("containers", [])
        # Từng invoice riêng (để biết tổng được cộng từ đâu, invoice nào không thương mại)
        out["invoice_nos"] = [str(d.get("invoice_no") or "").strip() for d in docs if str(d.get("invoice_no") or "").strip()]
        out["invoice_dates"] = [str(d.get("invoice_date") or "").strip() for d in docs if str(d.get("invoice_date") or "").strip()]
        out["invoice_breakdown"] = [{
            "invoice_no": str(d.get("invoice_no") or "").strip(), "invoice_date": str(d.get("invoice_date") or "").strip(),
            "currency": str(d.get("currency") or "").strip().upper(), "total_amount": float(d.get("total_amount") or 0),
            "total_items_sum": float(d.get("total_items_sum") or 0), "lines": len(d.get("items") or []),
            "is_non_commercial": bool(d.get("is_non_commercial")), "file_name": d.get("file_name", ""),
        } for d in docs]
        out["is_non_commercial"] = any(bool(d.get("is_non_commercial")) for d in docs)
        out["mixed_currency"] = len({str(d.get("currency") or "").strip().upper() for d in docs if str(d.get("currency") or "").strip()}) > 1
        return out
    return combine([r[0] for r in results]), combine([r[1] for r in results])


def load_chemical_doc(
    uploaded_file,
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.5-flash-lite",
) -> dict:
    """Đọc file khai báo hóa chất (PDF hoặc Excel) để đối chiếu với các dòng hàng HS đầu 28, 29."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(raw))
        return {"chemicals": df.to_dict(orient="records")}

    if name.endswith(".pdf"):
        if not api_key:
            raise ValueError("Cần nhập Gemini API Key để đọc file PDF Khai báo hóa chất.")
        prompt = "Trích xuất danh sách hóa chất từ tài liệu khai báo hóa chất (tên hóa chất, mã HS, số lượng, CAS nếu có)."
        return _call_gemini_json(raw, prompt, ChemicalDocData, api_key=api_key, model_name=model_name)

    raise ValueError(f"Chưa hỗ trợ định dạng file hóa chất: {uploaded_file.name}")


# ===================== BỔ SUNG: load_document cho app 5 chứng từ =====================

class ContainerSeal(BaseModel):
    container_no: str = Field(default="", description="Số container (4 chữ + 7 số)")
    seal_no: str = Field(default="", description="Số seal (chì) của hãng tàu / seal container, để trống nếu không có")


class TransportDocData(BaseModel):
    """Dữ liệu trích từ Bill of Lading hoặc Arrival Notice."""
    house_bl_no: str = Field(default="", description="Số HOUSE bill (HBL / House B/L No. / HAWB) - số vận đơn do forwarder/công ty giao nhận phát hành")
    master_bl_no: str = Field(default="", description="Số MASTER bill (MBL / Master B/L No. / MAWB / Carrier B/L) - số vận đơn của hãng tàu/hãng bay. Để trống nếu chứng từ không ghi")
    vessel_flight: str = Field(default="", description="Tên tàu + số chuyến (VD: EVER GIVEN 123E) hoặc số chuyến bay")
    eta: str = Field(default="", description="Ngày đến dự kiến ETA, chép đúng như trên chứng từ (kèm năm nếu có)")
    delivery_date: str = Field(default="", description="Ngày DELIVERY (Delivery date / ngày giao hàng / ngày hàng đến theo mục Delivery) trên Arrival Notice, chép đúng như trên chứng từ, kèm năm nếu có")
    containers: list[str] = Field(default_factory=list, description="Danh sách số container (VD: TGHU1234567)")
    container_seals: list[ContainerSeal] = Field(default_factory=list, description="Số container kèm số seal tương ứng (nếu chứng từ có ghi seal)")
    total_pkg_count: float = Field(default=0.0, description="Tổng số kiện / packages")
    total_gross_weight: float = Field(default=0.0, description="Tổng Gross Weight (kg)")
    charge_descriptions: list[str] = Field(default_factory=list, description="Tên TỪNG khoản phí / phụ phí ghi trên chứng từ (nếu có), VD: OCEAN FREIGHT, THC, PCS, LSS, BAF. Để trống nếu chứng từ không liệt kê phí")


class ChargeLine(BaseModel):
    description: str = Field(default="", description="Tên khoản phí trên debit note, chép đúng như in")
    amount: float = Field(default=0.0, description="Số tiền của dòng phí. Nếu chứng từ có cột 'tiền trước thuế' riêng thì lấy số TRƯỚC thuế (VAT); nếu chỉ có một cột số tiền thì lấy đúng số đó")
    currency: str = Field(default="", description="Mã tiền tệ của khoản phí (USD, VND...). Hóa đơn GTGT tiếng Việt thường là VND")
    category: str = Field(default="OTHER", description=(
        "OCEAN_FREIGHT = cước vận chuyển chính (Ocean freight / Sea freight / Air freight / O.F / cước đường biển); "
        "FREIGHT_SURCHARGE = phụ phí đi kèm cước (BAF, CAF, EBS, PSS, GRI, LSS, PCS...); "
        "EXW = phí lấy hàng tại xưởng người bán (EXW charge, EXWORK, ex-works, phí đại lý, agent fee, phí đại lý Trung Quốc, pickup/origin charge...); "
        "OTHER = mọi khoản còn lại (THC, D/O, CIC, handling, tờ khai, lưu bãi, vận chuyển nội địa...)"))
    amount_includes_vat: bool = Field(default=False, description=(
        "True CHỈ KHI số tiền đã điền ở amount còn bao gồm VAT. "
        "False nếu amount là số trước VAT (kể cả khi VAT ghi ở cột/dòng riêng) hoặc khoản phí không chịu VAT"))
    is_international_freight: bool = Field(default=False, description="True nếu category là OCEAN_FREIGHT hoặc FREIGHT_SURCHARGE")


class InsuranceData(BaseModel):
    policy_no: str = Field(default="", description="Số đơn / giấy chứng nhận bảo hiểm")
    premium_amount: float = Field(default=0.0, description="Phí bảo hiểm (premium) CHƯA gồm VAT")
    vat_amount: float = Field(default=0.0, description="Tiền VAT trên phí bảo hiểm nếu có")
    premium_currency: str = Field(default="", description="Mã tiền tệ của phí bảo hiểm")
    insured_value: float = Field(default=0.0, description="Số tiền / giá trị được bảo hiểm")


class DebitNoteData(BaseModel):
    """Debit note của forwarder: dùng để đối chiếu cước vận chuyển trên tờ khai."""
    debit_note_no: str = Field(default="", description="Số debit note")
    freight_amount: float = Field(default=0.0, description="TỔNG cước vận chuyển quốc tế (cộng các khoản is_international_freight)")
    freight_currency: str = Field(default="", description="Mã tiền tệ của freight_amount")
    total_amount: float = Field(default=0.0, description="Tổng cộng toàn bộ debit note")
    charges: list[ChargeLine] = Field(default_factory=list, description="Từng khoản phí trên debit note")


class BarcodeData(BaseModel):
    """Danh sách container đủ điều kiện qua khu vực giám sát (in từ hệ thống hải quan, có mã vạch)."""
    declaration_no: str = Field(default="", description="Số tờ khai (12 chữ số)")
    manager_no: str = Field(default="", description="Số quản lý hàng hóa (mục 9)")
    containers: list[ContainerSeal] = Field(default_factory=list, description="Từng dòng container: số hiệu container và số seal container (cột 2). Không lấy seal hải quan nếu để trống")


_EXCEL_COL_ALIAS = {
    "item_code": ["item code", "item no", "item no.", "part no", "part no.", "part number", "sku", "mã hàng", "ma hang", "mã sản phẩm", "material", "product code"],
    "description": ["description", "description of goods", "tên hàng", "mo ta", "mô tả hàng hóa"],
    "hs_code": ["hs code", "hscode", "hs", "mã hs", "ma hs", "mã số hàng hóa"],
    "qty": ["qty", "quantity", "q'ty", "số lượng", "so luong", "lượng"],
    "unit": ["unit", "uom", "đvt", "đơn vị tính"],
    "unit_price": ["unit price", "price", "đơn giá", "don gia"],
    "amount": ["amount", "total amount", "thành tiền", "trị giá", "tri gia", "trị giá dòng"],
    "net_weight": ["net weight", "n.w", "n.w.", "nw", "net wt", "trọng lượng tịnh"],
    "gross_weight": ["gross weight", "g.w", "g.w.", "gw", "gross wt", "trọng lượng cả bì"],
    "pkg_count": ["pkgs", "pkg", "packages", "ctns", "cartons", "no. of pkgs", "số kiện"],
}
_EXCEL_ALIAS_FLAT = {a: std for std, lst in _EXCEL_COL_ALIAS.items() for a in lst}


def _excel_to_items(raw: bytes) -> list[dict]:
    """Đọc mọi sheet Excel, map tên cột về chuẩn, bỏ dòng không có mã hàng (dòng tổng...)."""
    xl = pd.ExcelFile(io.BytesIO(raw))
    items: list[dict] = []
    for s_name in xl.sheet_names:
        df = xl.parse(s_name)
        df = df.rename(columns=lambda c: _EXCEL_ALIAS_FLAT.get(str(c).strip().lower(), c))
        if "item_code" not in df.columns:
            continue
        items.extend(x for x in _normalize_items(df.to_dict(orient="records")) if x["item_code"])
    return items


def _excel_invoice_meta(raw: bytes) -> tuple[str, str]:
    """Tìm số hóa đơn và ngày hóa đơn trong file Excel Invoice (theo nhãn 'Invoice No', 'Invoice Date', 'Date'...)."""
    try:
        xl = pd.ExcelFile(io.BytesIO(raw))
    except Exception:
        return "", ""
    label_words = ("invoice", "date", "ngày", "số", "no.", "no:", "page", "ref")
    inv_no, inv_date, bare_date = "", "", ""

    def value_of(df, r, c, txt):
        if ":" in txt:
            tail = txt.split(":", 1)[1].strip()
            if tail:
                return tail
        for cc in range(c + 1, min(c + 8, df.shape[1])):
            v = df.iloc[r, cc]
            if pd.notna(v) and str(v).strip():
                t = str(v).strip()
                if t.lower().rstrip(":") in ("date", "no", "no.", "invoice date", "invoice no"):
                    continue
                return t
        return ""

    for sh in xl.sheet_names:
        df = xl.parse(sh, header=None, dtype=object)
        for r in range(min(len(df), 40)):
            for c in range(df.shape[1]):
                v = df.iloc[r, c]
                if pd.isna(v):
                    continue
                txt = str(v).strip()
                low = txt.lower()
                if not inv_no and re.match(r"(commercial\s+)?invoice\s*(no|number|#)|inv\.?\s*no|số hóa đơn", low):
                    cand = value_of(df, r, c, txt)
                    if cand and not cand.lower().startswith(label_words[:1]):
                        inv_no = cand
                elif not inv_date and re.match(r"(invoice|inv\.?)\s*date|ngày hóa đơn|ngày phát hành", low):
                    inv_date = value_of(df, r, c, txt)
                elif not bare_date and re.fullmatch(r"date\s*:?", low):
                    bare_date = value_of(df, r, c, txt)
    return inv_no, inv_date or bare_date


class ExportDeclData(BaseModel):
    declaration_no: str = Field(default="", description="Số tờ khai xuất khẩu (12 chữ số)")


def load_export_declaration(uploaded_file, api_key: Optional[str] = None, model_name: str = "gemini-3.5-flash-lite") -> dict:
    """Đọc tờ khai xuất (Excel VNACCS hoặc PDF) - chỉ cần SỐ TỜ KHAI để đối chiếu số quản lý nội bộ."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    no = ""
    if name.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(io.BytesIO(raw))
        sheet = "TKX" if "TKX" in xl.sheet_names else xl.sheet_names[0]
        df = xl.parse(sheet, header=None, dtype=object)
        for r in range(min(len(df), 20)):
            for c in range(df.shape[1]):
                v = df.iloc[r, c]
                if pd.isna(v):
                    continue
                low = str(v).strip().lower()
                if low.startswith("số tờ khai") and not any(k in low for k in ("đầu tiên", "tạm nhập", "nhánh")):
                    for cc in range(c + 1, df.shape[1]):
                        d = re.sub(r"\D", "", str(df.iloc[r, cc]) if pd.notna(df.iloc[r, cc]) else "")
                        if len(d) >= 11:
                            no = d
                            break
                if no:
                    break
            if no:
                break
        if not no:  # dự phòng: số 12 chữ số đầu tiên trong 20 dòng đầu
            for r in range(min(len(df), 20)):
                for c in range(df.shape[1]):
                    v = df.iloc[r, c]
                    if pd.notna(v) and re.fullmatch(r"\d{12}", str(v).strip().split(".")[0]):
                        no = str(v).strip().split(".")[0]
                        break
                if no:
                    break
    elif name.endswith(".pdf"):
        if not api_key:
            raise ValueError("Cần nhập Gemini API Key để đọc file PDF tờ khai xuất.")
        data = _call_gemini_json(raw, "Đây là TỜ KHAI XUẤT KHẨU. Trích xuất số tờ khai (12 chữ số).", ExportDeclData, api_key=api_key, model_name=model_name)
        no = re.sub(r"\D", "", str(data.get("declaration_no") or ""))
    else:
        raise ValueError(f"Chưa hỗ trợ định dạng file tờ khai xuất: {uploaded_file.name}")
    return {"doc_type": "EXPORT_DECL", "declaration_no": no, "file_name": uploaded_file.name}


def _clean_seals(rows) -> list[dict]:
    out = []
    for r in rows or []:
        c = re.sub(r"\s+", "", str(r.get("container_no") or "")).upper()
        sl = re.sub(r"\s+", "", str(r.get("seal_no") or "")).upper()
        if c:
            out.append({"container_no": c, "seal_no": sl})
    return out


def load_document(
    uploaded_file,
    doc_type: str,
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.5-flash-lite",
) -> dict:
    """
    Hàm đọc chung cho app đối chiếu 5 chứng từ.
    doc_type: DRAFT_DECL | INVOICE | PACKING_LIST | BL | ARRIVAL_NOTICE
    Trả về dict đúng định dạng checker.run_full_reconciliation cần.
    """
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if doc_type == "EXPORT_DECL":
        return load_export_declaration(uploaded_file, api_key=api_key, model_name=model_name)

    # 1. Tờ khai nháp
    if doc_type == "DRAFT_DECL":
        return load_import_draft(uploaded_file, api_key=api_key, model_name=model_name)

    # 2. Invoice / Packing List
    if doc_type in ("INVOICE", "PACKING_LIST"):
        if name.endswith((".xlsx", ".xls")):
            items = _excel_to_items(raw)
            inv_no, inv_date = _excel_invoice_meta(raw) if doc_type == "INVOICE" else ("", "")
            return {
                "doc_type": doc_type,
                "invoice_no": inv_no,
                "invoice_date": inv_date,
                "containers": [],
                "total_amount": sum(x["amount"] for x in items),
                "total_net_weight": sum(x["net_weight"] for x in items),
                "total_gross_weight": sum(x["gross_weight"] for x in items),
                "total_pkg_count": sum(x["pkg_count"] for x in items),
                "items": items,
            }
        if name.endswith(".pdf"):
            inv_doc, pk_doc, _ = load_combined_invoice_packing_list(
                uploaded_file, api_key=api_key, model_name=model_name
            )
            doc = inv_doc if doc_type == "INVOICE" else pk_doc
            other = pk_doc if doc_type == "INVOICE" else inv_doc
            if not doc.get("items") and other.get("items"):
                doc = {**other, "doc_type": doc_type}
            doc.setdefault("containers", [])
            return doc
        raise ValueError(f"Chưa hỗ trợ định dạng file {doc_type}: {uploaded_file.name}")

    # 3. B/L và Arrival Notice (chỉ PDF)
    if doc_type in ("BL", "ARRIVAL_NOTICE"):
        if not name.endswith(".pdf"):
            raise ValueError(f"{doc_type} chỉ hỗ trợ file PDF: {uploaded_file.name}")
        if not api_key:
            raise ValueError(f"Cần nhập Gemini API Key để đọc file PDF {doc_type}.")
        label = "BILL OF LADING (B/L)" if doc_type == "BL" else "ARRIVAL NOTICE (giấy báo hàng đến)"
        prompt = (
            f"Đây là {label}. Trích xuất chính xác:\n"
            "1) SỐ VẬN ĐƠN - PHẢI PHÂN BIỆT HOUSE BILL và MASTER BILL: "
            "house_bl_no là số HOUSE bill (nhãn như House B/L No, HBL No, H.B/L, HAWB, 'B/L No.' của công ty giao nhận/forwarder); "
            "master_bl_no là số MASTER bill (nhãn như Master B/L No, MBL, M.B/L, MAWB, Carrier B/L, số của hãng tàu/hãng bay mà forwarder dùng để gửi hàng). "
            "Chứng từ có thể ghi cả hai số, tuyệt đối không lẫn lộn: số nào đứng cạnh nhãn Master/MBL/MAWB thì KHÔNG được đưa vào house_bl_no. "
            "Nếu chứng từ chỉ có MỘT số vận đơn và không ghi nhãn Master/MBL/MAWB thì đó là house_bl_no và để master_bl_no trống.\n"
            "2) tên tàu kèm số chuyến (hoặc số chuyến bay), ngày ETA.\n"
            "3) danh sách TẤT CẢ số container (11 ký tự, 4 chữ + 7 số) và số seal tương ứng của từng container nếu có (container_seals).\n"
            "4) tổng số kiện (packages) và tổng Gross Weight (kg).\n"
            "Nếu là Arrival Notice thì đọc thêm ngày DELIVERY (Delivery date) vào trường delivery_date, "
            "và liệt kê tên TỪNG khoản phí / phụ phí ghi trên Arrival Notice vào charge_descriptions (đặc biệt phụ phí tắc nghẽn cảng PCS / Port Congestion và phụ phí lưu huỳnh thấp LSS / Low Sulphur nếu có). "
            "Nếu không có trường nào thì để trống / 0."
        )
        data = _call_gemini_json(raw, prompt, TransportDocData, api_key=api_key, model_name=model_name)
        house = re.sub(r"\s+", "", str(data.get("house_bl_no") or "")).upper()
        master = re.sub(r"\s+", "", str(data.get("master_bl_no") or "")).upper()
        if house and house == master:
            master = ""
        return {
            "doc_type": doc_type,
            "house_bl_no": house,
            "master_bl_no": master,
            # số dùng để đối chiếu: luôn là HOUSE bill; chỉ khi chứng từ không có house mới dùng master (và đánh dấu)
            "bl_no": house or master,
            "bl_is_master": bool(master and not house),
            "vessel_flight": str(data.get("vessel_flight") or "").strip().upper(),
            "eta": (str(data.get("delivery_date") or "").strip() or str(data.get("eta") or "").strip()) if doc_type == "ARRIVAL_NOTICE" else str(data.get("eta") or "").strip(),
            "containers": [re.sub(r"\s+", "", str(c)).upper() for c in (data.get("containers") or []) if str(c).strip()],
            "container_seals": _clean_seals(data.get("container_seals")),
            "total_pkg_count": _parse_vnaccs_number(data.get("total_pkg_count") or 0.0),
            "total_gross_weight": _parse_vnaccs_number(data.get("total_gross_weight") or 0.0),
            "charge_descriptions": [str(x).strip() for x in (data.get("charge_descriptions") or []) if str(x).strip()],
            "items": [],
        }

    # 4. Debit note (cước vận chuyển) - có thể có nhiều file (phí OF và phí EXW tách riêng)
    if doc_type == "DEBIT_NOTE":
        if not name.endswith(".pdf"):
            raise ValueError(f"Debit note chỉ hỗ trợ file PDF: {uploaded_file.name}")
        if not api_key:
            raise ValueError("Cần nhập Gemini API Key để đọc file PDF Debit note.")
        prompt = (
            "Đây là DEBIT NOTE hoặc HÓA ĐƠN phí của công ty giao nhận (forwarder). Liệt kê TỪNG khoản phí (charges) kèm số tiền và tiền tệ. "
            "Phân loại mỗi khoản vào category: OCEAN_FREIGHT (cước biển/hàng không chính), FREIGHT_SURCHARGE (BAF, CAF, EBS, PSS, LSS, PCS...), "
            "EXW (phí lấy hàng tại xưởng người bán: có thể ghi là EXW charge, EXWORK, phí đại lý, phí đại lý Trung Quốc, agent fee, pickup...), "
            "OTHER (THC, D/O, CIC, handling, phí tờ khai, vận chuyển nội địa, lưu bãi...). "
            "amount: nếu chứng từ có cột 'tiền trước thuế' thì lấy số TRƯỚC VAT và đặt amount_includes_vat = false; "
            "chỉ đặt amount_includes_vat = true khi bạn buộc phải lấy số đã gồm VAT. "
            "Ghi thêm số debit note (hoặc số hóa đơn) và tổng cộng toàn bộ chứng từ."
        )
        data = _call_gemini_json(raw, prompt, DebitNoteData, api_key=api_key, model_name=model_name)
        valid_cats = {"OCEAN_FREIGHT", "FREIGHT_SURCHARGE", "EXW", "OTHER"}
        charges = []
        for c in (data.get("charges") or []):
            cat = str(c.get("category") or "OTHER").strip().upper()
            cat = cat if cat in valid_cats else "OTHER"
            charges.append({
                "description": str(c.get("description") or "").strip(),
                "amount": _parse_vnaccs_number(c.get("amount") or 0.0),
                "currency": str(c.get("currency") or "").strip().upper(),
                "category": cat,
                "amount_includes_vat": bool(c.get("amount_includes_vat")),
                "is_international_freight": cat in ("OCEAN_FREIGHT", "FREIGHT_SURCHARGE"),
            })
        return {
            "doc_type": "DEBIT_NOTE",
            "debit_note_no": str(data.get("debit_note_no") or "").strip(),
            "total_amount": _parse_vnaccs_number(data.get("total_amount") or 0.0),
            "charges": charges,
            "file_name": uploaded_file.name,
        }

    # 4b. Phí bảo hiểm
    if doc_type == "INSURANCE":
        if not name.endswith(".pdf"):
            raise ValueError(f"Chứng từ bảo hiểm chỉ hỗ trợ file PDF: {uploaded_file.name}")
        if not api_key:
            raise ValueError("Cần nhập Gemini API Key để đọc file PDF bảo hiểm.")
        prompt = (
            "Đây là đơn / giấy chứng nhận bảo hiểm hàng hóa hoặc hóa đơn phí bảo hiểm. Trích xuất số đơn, "
            "phí bảo hiểm (premium) CHƯA gồm VAT, tiền VAT (nếu có), mã tiền tệ của phí và giá trị được bảo hiểm."
        )
        data = _call_gemini_json(raw, prompt, InsuranceData, api_key=api_key, model_name=model_name)
        return {
            "doc_type": "INSURANCE",
            "policy_no": str(data.get("policy_no") or "").strip(),
            "premium_amount": _parse_vnaccs_number(data.get("premium_amount") or 0.0),
            "vat_amount": _parse_vnaccs_number(data.get("vat_amount") or 0.0),
            "premium_currency": str(data.get("premium_currency") or "").strip().upper(),
            "insured_value": _parse_vnaccs_number(data.get("insured_value") or 0.0),
            "file_name": uploaded_file.name,
        }

    # 5. Mã vạch (danh sách container đủ điều kiện qua khu vực giám sát)
    if doc_type == "BARCODE":
        if not name.endswith(".pdf"):
            raise ValueError(f"Mã vạch chỉ hỗ trợ file PDF: {uploaded_file.name}")
        if not api_key:
            raise ValueError("Cần nhập Gemini API Key để đọc file PDF mã vạch.")
        prompt = (
            "Đây là DANH SÁCH CONTAINER ĐỦ ĐIỀU KIỆN QUA KHU VỰC GIÁM SÁT HẢI QUAN (có mã vạch). "
            "Trích xuất số tờ khai (mục 4), số quản lý hàng hóa (mục 9) và TỪNG dòng container: "
            "số hiệu container (cột 1) và số seal container (cột 2, để trống nếu không có). Không lấy cột seal hải quan."
        )
        data = _call_gemini_json(raw, prompt, BarcodeData, api_key=api_key, model_name=model_name)
        seals = _clean_seals(data.get("containers"))
        return {
            "doc_type": "BARCODE",
            "declaration_no": re.sub(r"\D", "", str(data.get("declaration_no") or "")),
            "manager_no": re.sub(r"\s+", "", str(data.get("manager_no") or "")).upper(),
            "containers": [x["container_no"] for x in seals],
            "container_seals": seals,
            "file_name": uploaded_file.name,
        }

    raise ValueError(f"doc_type không hợp lệ: {doc_type}")


# ===================== TỪ ĐIỂN LỊCH SỬ KHAI BÁO =====================

def norm_code(s) -> str:
    """Chuẩn hóa mã hàng để so khớp: in hoa, bỏ mọi ký tự không phải chữ/số."""
    return re.sub(r"[^0-9A-Z]", "", str(s or "").upper())


class HistoryDict(dict):
    """Từ điển lịch sử, kèm thông tin báo cáo (doanh nghiệp, khoảng ngày...) trong thuộc tính .meta."""
    meta: dict = {}


def load_history_report(uploaded_file) -> dict:
    """
    Đọc Báo cáo chi tiết hàng hóa nhập khẩu (xuất từ ECUS, .xls/.xlsx) và tạo từ điển:
    norm_code(Mã NPL/SP) -> thông tin của LẦN KHAI GẦN NHẤT (HS, tên hàng, xuất xứ, ĐVT, thuế suất...).
    """
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    engine = "xlrd" if name.endswith(".xls") else "openpyxl"
    try:
        df = pd.read_excel(io.BytesIO(raw), sheet_name=0, header=None, engine=engine, dtype=object)
    except ImportError as exc:
        raise ValueError("Thiếu thư viện đọc file .xls. Hãy chạy: pip install xlrd") from exc

    hdr = None
    for i in range(min(len(df), 60)):
        vals = [str(v).strip() for v in df.iloc[i].tolist() if pd.notna(v)]
        if "Mã NPL/SP" in vals and "Mã HS" in vals:
            hdr = i
            break
    if hdr is None:
        raise ValueError("Không tìm thấy dòng tiêu đề có cột 'Mã NPL/SP' và 'Mã HS' trong báo cáo lịch sử.")

    cols = [str(v).strip() if pd.notna(v) else f"_c{j}" for j, v in enumerate(df.iloc[hdr].tolist())]
    data = df.iloc[hdr + 1:].copy()
    data.columns = cols
    data = data.loc[:, ~pd.Index(data.columns).duplicated()]

    def col(name_):
        return data[name_] if name_ in data.columns else pd.Series([None] * len(data), index=data.index)

    def _first(name_):
        vals = [str(v).strip() for v in col(name_).dropna().tolist() if str(v).strip()]
        return vals[0] if vals else ""

    meta = {
        "ma_dn": _first("Mã doanh nghiệp"),
        "ten_dn": _first("Tên doanh nghiệp"),
        "so_dong": int(col("Mã NPL/SP").notna().sum()),
    }

    data["_code"] = col("Mã NPL/SP").map(lambda v: str(v).strip().upper() if pd.notna(v) else "")
    data["_key"] = data["_code"].map(norm_code)
    data = data[data["_key"] != ""].copy()
    data["_date"] = pd.to_datetime(col("Ngày ĐK"), errors="coerce", dayfirst=True)
    data["_tk"] = pd.to_numeric(col("Số TK"), errors="coerce")
    data["_stt"] = pd.to_numeric(col("STT hàng"), errors="coerce")
    data["_hs"] = col("Mã HS").map(lambda v: clean_hs(v) if pd.notna(v) else "")

    hs_all = data.groupby("_key")["_hs"].agg(lambda s_: sorted({x for x in s_ if x}))

    data = data.sort_values(["_date", "_tk", "_stt"], na_position="first")
    latest = data.drop_duplicates("_key", keep="last")

    def txt(v):
        return str(v).strip() if pd.notna(v) else ""

    meta["tu_ngay"] = data["_date"].min().strftime("%d/%m/%Y") if data["_date"].notna().any() else ""
    meta["den_ngay"] = data["_date"].max().strftime("%d/%m/%Y") if data["_date"].notna().any() else ""

    master = HistoryDict()
    master.meta = meta
    for _, r in latest.iterrows():
        d = r["_date"]
        tk = r["_tk"]
        master[r["_key"]] = {
            "code": r["_code"],
            "hs_code": r["_hs"],
            "hs_all": hs_all.get(r["_key"], []),
            "ten_hang": txt(r.get("Tên hàng")),
            "origin": txt(r.get("Xuất xứ")).upper(),
            "unit": txt(r.get("Đơn vị tính")).upper(),
            "tax_code": txt(r.get("Mã biểu thuế XNK")),
            "tax_rate": txt(r.get("Thuế suất XNK")),
            "decl_no": str(int(tk)) if pd.notna(tk) else "",
            "decl_date": d.strftime("%d/%m/%Y") if pd.notna(d) else "",
        }
    return master


# Đổi chuỗi này mỗi lần sửa cách đọc chứng từ: app dùng làm một phần khóa cache nên kết quả đọc cũ không bị dùng lại
PARSER_VERSION = "2026-11-28a"


# ===================== NGÀY THÁNG: ĐƯA VỀ dd/mm/yyyy =====================

_MONTH_MAP = {
    "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4, "JUNE": 6, "JULY": 7, "AUGUST": 8,
    "SEPTEMBER": 9, "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12,
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEPT": 9, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_MONTH_RE = re.compile("(" + "|".join(sorted(_MONTH_MAP, key=len, reverse=True)) + ")")


def _valid_dmy(d: int, m: int, y) -> bool:
    if not (1 <= m <= 12 and 1 <= d <= 31):
        return False
    if y is None:
        return True
    try:
        datetime.date(y, m, d)
        return True
    except ValueError:
        return False


def date_candidates(text) -> list[tuple]:
    """
    Đọc 1 chuỗi ngày ở bất kỳ format nào (03/09/2026, 2026-09-03, 03-SEP-2026, SEP 3, 2026, 03SEP26, NOV/16...).
    Trả về danh sách (ngày, tháng, năm|None); nhiều phần tử khi không phân biệt được dd/mm hay mm/dd.
    """
    t = str(text or "").upper()
    t = re.sub(r"\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?", " ", t)  # bỏ giờ
    cands: list[tuple] = []

    mm = _MONTH_RE.search(t)
    if mm:
        month = _MONTH_MAP[mm.group(1)]
        nums = re.findall(r"\d+", t[:mm.start()] + " " + t[mm.end():])
        four = [n for n in nums if len(n) == 4]
        others = [n for n in nums if len(n) != 4]
        year = int(four[0]) if four else None
        if not others:
            return []
        day = int(others[0])
        if year is None and len(others) >= 2:
            y2 = int(others[1])
            year = 2000 + y2 if y2 < 100 else None
        cands.append((day, month, year))
    else:
        g = re.findall(r"\d+", t)
        if len(g) == 1 and len(g[0]) == 8:
            s8 = g[0]
            if s8[:2] in ("19", "20"):
                cands.append((int(s8[6:]), int(s8[4:6]), int(s8[:4])))
            else:
                cands.append((int(s8[:2]), int(s8[2:4]), int(s8[4:])))
        elif len(g) >= 3:
            a, b, c = g[:3]
            if len(a) == 4:                      # yyyy-mm-dd
                cands.append((int(c), int(b), int(a)))
            else:
                y = int(c)
                y = 2000 + y if len(c) <= 2 else y
                cands.append((int(a), int(b), y))       # dd/mm/yyyy (ưu tiên)
                if int(a) != int(b):
                    cands.append((int(b), int(a), y))   # mm/dd/yyyy
        elif len(g) == 2:                        # dd/mm không có năm
            a, b = int(g[0]), int(g[1])
            cands.append((a, b, None))
            if a != b:
                cands.append((b, a, None))

    out = []
    for c in cands:
        if _valid_dmy(*c) and c not in out:
            out.append(c)
    return out


def _fmt_dmy(c: tuple) -> str:
    d, m, y = c
    return f"{d:02d}/{m:02d}/{y}" if y else f"{d:02d}/{m:02d}"


def fmt_date(text, ref_year=None) -> str:
    """Đưa ngày về dd/mm/yyyy. Không đọc được thì trả lại nguyên văn."""
    cands = date_candidates(text)
    if not cands:
        return str(text or "").strip()
    d, m, y = cands[0]
    return _fmt_dmy((d, m, y or ref_year))


def compare_dates(a, b) -> tuple:
    """So 2 ngày sau khi đưa về cùng dạng dd/mm/yyyy. Trả về (khớp: True/False/None, a_dd/mm/yyyy, b_dd/mm/yyyy).
    None = một bên không đọc được thành ngày. Bên nào thiếu năm (VD 'NOV/16') thì mượn năm của bên kia."""
    ca, cb = date_candidates(a), date_candidates(b)
    if not ca or not cb:
        return None, str(a or "").strip(), str(b or "").strip()
    ya = next((y for _, _, y in ca if y), None)
    yb = next((y for _, _, y in cb if y), None)
    ca = [(d, m, y or yb) for d, m, y in ca]
    cb = [(d, m, y or ya) for d, m, y in cb]
    for x in ca:
        if x in cb:
            return True, _fmt_dmy(x), _fmt_dmy(x)
    return False, _fmt_dmy(ca[0]), _fmt_dmy(cb[0])


# ===================== SỐ VẬN ĐƠN & LOẠI HÌNH =====================

# Loại hình nhập liên quan SXXK / chế xuất / gia công (QĐ 1357/QĐ-TCHQ): hàng khai theo MÃ NPL.
#  E11 nhập NL của DNCX từ nước ngoài | E15 nhập NL của DNCX từ nội địa | E21 nhập NL gia công
#  E23 nhập NL gia công từ HĐGC khác chuyển sang | E31 nhập NL sản xuất xuất khẩu | E33 nhập NL SXXK từ kho ngoại quan
NPL_DECL_TYPES = {"E11", "E15", "E21", "E23", "E31", "E33"}


# Quy tắc riêng theo khách hàng (khóa = mã số thuế 10 số của người nhập khẩu).
# prefix_npl  : tiền tố đứng trước mã NPL ở dòng hàng loại hình NPL (E31...)
# prefix_other: tiền tố đứng trước mã NPL ở dòng hàng loại hình khác (A12...)
# Dùng để nhận ra cùng 1 mã hàng bị tách số lượng ra nhiều tờ khai và gộp lại để đối chiếu.
CUSTOMER_RULES = {
    "3600659583": {
        "name": "Công ty TNHH Đông Phương Đồng Nai Việt Nam",
        "prefix_npl": ["SX"],
        "prefix_other": ["KD", "HT"],
    },
}


def customer_rule(tax_code) -> dict:
    return CUSTOMER_RULES.get(re.sub(r"\D", "", str(tax_code or ""))[:10], {})


def base_code_of(item: dict, rule: dict) -> str:
    """Mã gốc (bỏ tiền tố SX/KD/HT theo loại hình của dòng). Rỗng nếu khách không có quy tắc hoặc mã không có tiền tố."""
    if not rule:
        return ""
    code = norm_code(item.get("item_code"))
    prefixes = rule.get("prefix_npl", []) if item.get("is_npl") else rule.get("prefix_other", [])
    for p in prefixes:
        p = norm_code(p)
        if code.startswith(p) and len(code) > len(p):
            return code[len(p):]
    return ""


def _type_token(raw) -> str:
    m = re.match(r"\s*([A-Z]\d{2})", str(raw or "").upper())
    return m.group(1) if m else str(raw or "").strip().upper()


def is_npl_type(type_code) -> bool:
    return _type_token(type_code) in NPL_DECL_TYPES


def strip_bl_suffix(bl_no) -> str:
    """Nhiều tờ khai chung 1 bill: số bill trên tờ khai có đuôi -01, -02, -03... -> bỏ đuôi để ghép đúng bill."""
    return re.sub(r"-\d{1,3}$", "", re.sub(r"\s+", "", str(bl_no or "").upper()))


def bl_for_compare(bl_no, transport_mode) -> str:
    """Số bill dùng để đối chiếu:
    - Mã hiệu phương thức vận chuyển 2, 3 (số cạnh mã loại hình, VD 'E31 2 [4]'): 6 ký tự đầu là ngày tháng (ddmmyy) -> bỏ đi, lấy từ ký tự thứ 7.
    - Mọi trường hợp: bỏ đuôi -01, -02, -03... (tờ khai thứ n chung bill)."""
    s = re.sub(r"\s+", "", str(bl_no or "").upper())
    if str(transport_mode or "").strip() in ("2", "3") and len(s) > 6:
        s = s[6:]
    return strip_bl_suffix(s)


def clean_invoice_no(v) -> str:
    """Số hóa đơn trên tờ khai có dạng 'A - FTFNS95' (A = phân loại): bỏ phần phân loại."""
    return re.sub(r"^\s*[A-Z]\s*-\s*", "", str(v or "").strip())


def stamp_declaration(data: dict) -> dict:
    """Chuẩn hóa thông tin tờ khai và gắn số TK / loại hình vào từng dòng hàng."""
    data["declaration_no"] = re.sub(r"\D", "", str(data.get("declaration_no") or "")) or str(data.get("declaration_no") or "").strip()
    data["first_declaration_no"] = re.sub(r"\D", "", str(data.get("first_declaration_no") or ""))
    data["branch_no"] = re.sub(r"\D", "", str(data.get("branch_no") or ""))
    data["total_branches"] = re.sub(r"\D", "", str(data.get("total_branches") or ""))
    data["check_class_code"] = re.sub(r"\D", "", str(data.get("check_class_code") or ""))[:1]
    _m = re.match(r"\s*[A-Z]\d{2}\s+(\d)", str(data.get("declaration_type_code") or "").upper())
    data["transport_mode_code"] = re.sub(r"\D", "", str(data.get("transport_mode_code") or ""))[:1] or (_m.group(1) if _m else "")
    type_code = _type_token(data.get("declaration_type_code"))
    data["declaration_type_code"] = type_code
    data["bl_no_cmp"] = bl_for_compare(data.get("bl_no"), data["transport_mode_code"])
    data["invoice_no_clean"] = clean_invoice_no(data.get("invoice_no"))
    _m = re.search(r"-(\d{1,3})$", re.sub(r"\s+", "", str(data.get("bl_no") or "")))
    data["bl_suffix"] = _m.group(1) if _m else ""
    data["importer_tax_code"] = re.sub(r"\D", "", str(data.get("importer_tax_code") or ""))
    data["freight_currency"] = str(data.get("freight_currency") or "").strip().upper()[:3]
    data["insurance_currency"] = str(data.get("insurance_currency") or "").strip().upper()[:3]
    rates = data.get("exchange_rates") or {}
    if isinstance(rates, list):
        rates = {str(r.get("currency") or "").strip().upper(): _parse_vnaccs_number(r.get("rate") or 0.0) for r in rates if r.get("currency")}
    data["exchange_rates"] = {k: v for k, v in rates.items() if v}
    rule = customer_rule(data["importer_tax_code"])
    data["customer_rule"] = rule
    for i, it in enumerate(data.get("items") or [], 1):
        it["decl_no"] = data["declaration_no"]
        it["line_no"] = i
        it["decl_type"] = type_code
        it["is_npl"] = is_npl_type(type_code)
        it["base_code"] = base_code_of(it, rule)
        it["rule_applied"] = bool(rule)
    return data


def load_import_draft(uploaded_file, api_key: Optional[str] = None, model_name: str = "gemini-3.5-flash-lite") -> dict:
    data = _load_import_draft_raw(uploaded_file, api_key=api_key, model_name=model_name)
    data["file_name"] = uploaded_file.name
    return stamp_declaration(data)


# ===================== GỘP NHIỀU TỜ KHAI (NHÁNH / NHIỀU TỜ KHAI CÙNG 1 BILL) =====================

def _note(severity, field, draft_val, doc_val, note) -> dict:
    return {"severity": severity, "field": field, "draft_val": draft_val, "doc_val": doc_val, "note": note}


def _sum_by_first_decl(members: list[dict], key: str) -> float:
    """Cộng 1 khoản phí (cước / bảo hiểm) của bộ tờ khai: các nhánh cùng Số TK đầu tiên chỉ lấy 1 lần (số lớn nhất),
    giữa các tờ khai độc lập thì cộng dồn."""
    by_first: dict[str, float] = {}
    for d in members:
        k = d.get("first_declaration_no") or d.get("declaration_no") or str(id(d))
        by_first[k] = max(by_first.get(k, 0.0), float(d.get(key) or 0))
    return sum(by_first.values())


def merge_declarations(decls: list[dict], bl_hint: str = "") -> dict:
    """
    Gộp các tờ khai đính kèm thành 1 bộ để đối chiếu với 1 bộ chứng từ (1 Bill).
    - Cùng 'Số tờ khai đầu tiên' (tách nhánh) hoặc cùng số bill -> gộp chung.
    - Nếu đính kèm tờ khai của nhiều bill khác nhau: chọn nhóm có số bill khớp B/L (bl_hint); nhóm còn lại bị loại và báo.
    """
    notes: list[dict] = []
    uniq, seen = [], set()
    for d in decls:
        if not d:
            continue
        no = d.get("declaration_no") or ""
        if no and no in seen:
            notes.append(_note("WARNING", "DECL_SET", no, "", f"Tờ khai {no} được đính kèm nhiều lần - chỉ tính 1 lần"))
            continue
        seen.add(no)
        uniq.append(d)
    if not uniq:
        raise ValueError("Chưa có tờ khai nào để gộp.")

    # --- gom nhóm (union-find) theo số TK đầu tiên và theo số bill ---
    parent = list(range(len(uniq)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: dict[str, int] = {}
    for i, d in enumerate(uniq):
        keys = ["F:" + (d.get("first_declaration_no") or d.get("declaration_no") or f"?{i}")]
        bc = norm_code(d.get("bl_no_cmp"))
        if bc:
            keys.append("B:" + bc)
        for k in keys:
            if k in owner:
                parent[find(i)] = find(owner[k])
            else:
                owner[k] = i
    groups: dict[int, list[dict]] = {}
    for i, d in enumerate(uniq):
        groups.setdefault(find(i), []).append(d)
    group_list = list(groups.values())

    chosen = group_list[0]
    if len(group_list) > 1:
        hint = norm_code(strip_bl_suffix(bl_hint))
        if hint:
            for g in group_list:
                if any(norm_code(d.get("bl_no_cmp")) == hint or norm_code(strip_bl_suffix(d.get("bl_no"))) == hint for d in g):
                    chosen = g
                    break
        for g in group_list:
            if g is chosen:
                continue
            nos = ", ".join(d.get("declaration_no") or d.get("file_name", "?") for d in g)
            bl_g = next((d.get("bl_no_cmp") for d in g if d.get("bl_no_cmp")), "")
            notes.append(_note(
                "WARNING", "DECL_SET", nos, "KHÔNG ĐƯA VÀO ĐỐI CHIẾU",
                f"Tờ khai {nos} (số bill {bl_g or '?'}) không cùng bill/nhánh với các tờ khai còn lại - đã loại khỏi lần đối chiếu này",
            ))

    members = chosen

    # --- kiểm tra đủ nhánh ---
    by_first: dict[str, list[dict]] = {}
    for d in members:
        by_first.setdefault(d.get("first_declaration_no") or d.get("declaration_no"), []).append(d)
    for first, lst in by_first.items():
        total = max([int(d.get("total_branches") or 0) for d in lst] + [0])
        got = {d.get("declaration_no") for d in lst}
        if total > 1 and len(got) < total:
            notes.append(_note(
                "WARNING", "DECL_BRANCH", f"{len(got)} tờ khai ({', '.join(sorted(x for x in got if x))})", f"{total} tờ khai",
                f"Tờ khai đầu tiên {first} được chia thành {total} tờ khai/nhánh nhưng mới đính kèm {len(got)} - "
                "thiếu tờ khai có thể làm lệch số lượng, trị giá, số dòng hàng so với chứng từ",
            ))

    for d in members:
        if not str(d.get("transport_mode_code") or "").strip():
            notes.append(_note("WARNING", "DECL_SET", d.get("declaration_no", ""), "",
                               f"Không đọc được mã hiệu phương thức vận chuyển (số cạnh mã loại hình) của tờ khai {d.get('declaration_no', '')} - "
                               "số bill chưa được cắt 6 ký tự đầu, số quản lý nội bộ / số container chưa được kiểm tra theo mã này"))

    # --- gộp thông tin chung ---
    def first_nonblank(key):
        return next((d.get(key) for d in members if str(d.get(key) or "").strip()), "")

    bls = {norm_code(d.get("bl_no_cmp")) for d in members if d.get("bl_no_cmp")}
    if len(bls) > 1:
        notes.append(_note("WARNING", "DECL_BL", " | ".join(sorted(d.get("bl_no", "") for d in members)), "",
                           "Các tờ khai đính kèm ghi số vận đơn khác nhau - kiểm tra lại"))
    etas = {fmt_date(d.get("eta")) for d in members if d.get("eta")}
    if len(etas) > 1:
        notes.append(_note("WARNING", "ETA", " | ".join(sorted(etas)), "", "Các tờ khai đính kèm ghi ngày hàng đến khác nhau - kiểm tra lại"))

    def combine(key, label):
        vals = [float(d.get(key) or 0) for d in members]
        vals = [v for v in vals if v > 0]
        if len(members) == 1 or not vals:
            return vals[0] if vals else 0.0
        if max(vals) - min(vals) < 1e-6:        # các nhánh cùng ghi tổng của cả bill
            return vals[0]
        notes.append(_note("WARNING", "DECL_SET", ", ".join(f"{v:g}" for v in vals), f"{sum(vals):g}",
                           f"{label} ở các tờ khai khác nhau - đã cộng dồn để đối chiếu"))
        return sum(vals)

    items = [it for d in members for it in (d.get("items") or [])]
    line_amount = sum(float(it.get("amount") or 0) for it in items)
    if len(members) == 1:
        total_amount = float(members[0].get("total_amount") or 0) or line_amount
    else:
        # Tổng trị giá hóa đơn trên mỗi tờ khai là của cả hóa đơn -> khi gộp phải cộng theo từng dòng hàng
        total_amount = line_amount or sum(float(d.get("total_amount") or 0) for d in members)

    containers = []
    for d in members:
        for c in d.get("containers") or []:
            if c not in containers:
                containers.append(c)

    merged = {
        "doc_type": "IMPORT_DRAFT",
        "declaration_no": ", ".join(d.get("declaration_no", "") for d in members),
        "declaration_type_code": ", ".join(sorted({d.get("declaration_type_code", "") for d in members if d.get("declaration_type_code")})),
        "check_class_code": first_nonblank("check_class_code"),
        "transport_mode_code": first_nonblank("transport_mode_code"),
        "exporter_name": first_nonblank("exporter_name"),
        "exporter_address": first_nonblank("exporter_address"),
        "storage_location": first_nonblank("storage_location"),
        "transport_means": first_nonblank("transport_means"),
        "invoice_date": first_nonblank("invoice_date"),
        "payment_method": first_nonblank("payment_method"),
        "invoice_no": first_nonblank("invoice_no"),
        "internal_ref_no": first_nonblank("internal_ref_no"),
        "invoice_nos": list(dict.fromkeys(d.get("invoice_no_clean") for d in members if d.get("invoice_no_clean"))),
        "invoice_dates": list(dict.fromkeys(str(d.get("invoice_date")).strip() for d in members if str(d.get("invoice_date") or "").strip())),
        "importer_tax_code": first_nonblank("importer_tax_code"),
        "customer_rule": next((d.get("customer_rule") for d in members if d.get("customer_rule")), {}),
        "exchange_rates": {k: v for d in members for k, v in (d.get("exchange_rates") or {}).items()},
        "freight_currency": first_nonblank("freight_currency"),
        "insurance_currency": first_nonblank("insurance_currency"),
        "decl_meta": [{
            "declaration_no": d.get("declaration_no", ""), "check_class_code": d.get("check_class_code", ""),
            # Số TK đầu tiên + nhánh: checker dùng để biết tờ khai nào là NHÁNH của cùng 1 tờ khai
            # (nhánh -> cước/bảo hiểm giữ nguyên, không phân bổ)
            "first_declaration_no": d.get("first_declaration_no") or d.get("declaration_no", ""),
            "branch_no": d.get("branch_no", ""), "total_branches": d.get("total_branches", ""),
            "transport_mode_code": d.get("transport_mode_code", ""),
            "internal_ref_no": d.get("internal_ref_no", ""), "container_count": d.get("container_count", ""),
            "incoterms": d.get("incoterms", ""), "currency": d.get("currency", ""),
            "freight_amount": float(d.get("freight_amount") or 0), "freight_currency": d.get("freight_currency", ""),
            "insurance_amount": float(d.get("insurance_amount") or 0), "insurance_currency": d.get("insurance_currency", ""),
            "payment_method": str(d.get("payment_method") or "").strip().upper(), "invoice_no_clean": d.get("invoice_no_clean") or clean_invoice_no(d.get("invoice_no")),
            "goods_value": sum(float(it.get("amount") or 0) for it in (d.get("items") or [])) or float(d.get("total_amount") or 0),
            "exchange_rates": d.get("exchange_rates") or {},
            "decl_type": d.get("declaration_type_code", ""),
        } for d in members],
        "container_count": combine("container_count", "Số lượng container"),
        "bl_no": first_nonblank("bl_no"),
        "bl_no_cmp": first_nonblank("bl_no_cmp"),
        "vessel_flight": first_nonblank("vessel_flight"),
        "eta": first_nonblank("eta"),
        "containers": containers,
        "currency": first_nonblank("currency"),
        "incoterms": first_nonblank("incoterms"),
        # Nhánh của cùng 1 tờ khai ghi cùng 1 số cước (cước cả bill) -> chỉ tính 1 lần / nhóm nhánh,
        # các tờ khai KHÁC nhau cùng bill thì cộng lại.
        "freight_amount": _sum_by_first_decl(members, "freight_amount"),
        "insurance_amount": _sum_by_first_decl(members, "insurance_amount"),
        "total_pkg_count": combine("total_pkg_count", "Tổng số kiện"),
        "total_gross_weight": combine("total_gross_weight", "Tổng Gross Weight"),
        "total_net_weight": combine("total_net_weight", "Tổng Net Weight"),
        "total_amount": total_amount,
        "items": items,
        "merge_notes": notes,
        "declarations": [{
            "Số tờ khai": d.get("declaration_no", ""),
            "Số TK đầu tiên": d.get("first_declaration_no", ""),
            "Nhánh": f"{d.get('branch_no') or '-'}/{d.get('total_branches') or '-'}",
            "Loại hình": d.get("declaration_type_code", ""),
            "Mã PTVT (cạnh loại hình)": d.get("transport_mode_code", ""),
            "Mã PL kiểm tra (I6)": d.get("check_class_code", ""),
            "Số bill": d.get("bl_no", ""),
            "Số bill dùng để đối chiếu": d.get("bl_no_cmp", ""),
            "Đuôi bill": d.get("bl_suffix", ""),
            "Số dòng hàng": len(d.get("items") or []),
            "Tổng trị giá các dòng": round(sum(float(it.get("amount") or 0) for it in (d.get("items") or [])), 2),
            "File": d.get("file_name", ""),
        } for d in members],
    }
    return merged


# ===================== TỪ ĐIỂN LỊCH SỬ KHAI BÁO: GHÉP DÒNG HÀNG =====================

@functools.lru_cache(maxsize=400000)
def _norm_text_cached(s: str) -> str:
    t = unicodedata.normalize("NFD", str(s or ""))
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn").replace("đ", "d").replace("Đ", "D")
    return " ".join(re.sub(r"[^0-9a-zA-Z]+", " ", t).lower().split())


def norm_text(s) -> str:
    """Chữ thường, bỏ dấu tiếng Việt, bỏ ký tự đặc biệt, gọn khoảng trắng."""
    return _norm_text_cached(str(s or ""))


_STOP = {"hang", "moi", "100", "dung", "de", "san", "xuat", "gia", "cong", "cho", "va", "cua", "la", "co", "khong",
         "new", "for", "the", "and", "of", "used", "use", "quy", "cach", "mm", "loai", "cac"}


@functools.lru_cache(maxsize=400000)
def _tokens(norm: str) -> frozenset:
    return frozenset(t for t in norm.split() if len(t) >= 2 and t not in _STOP)


def desc_score(query, target) -> float:
    """Mức 'tên hàng query nằm trong tên hàng target' (0..1). Không cần giống 100%."""
    qn, tn = norm_text(query), norm_text(target)
    if not qn or not tn:
        return 0.0
    if len(qn) >= 4 and qn in tn:
        return 1.0
    if len(tn) >= 6 and tn in qn:
        return 1.0
    qt = _tokens(qn)
    if not qt:
        return 0.0
    return len(qt & _tokens(tn)) / len(qt)


DESC_THRESHOLD = 0.8


def _ensure_index(master):
    if getattr(master, "_idx_ready", False):
        return
    master._key_lengths = sorted({len(k) for k in master if len(k) >= 4})
    master._desc_idx = [(k, norm_text(r.get("ten_hang")), _tokens(norm_text(r.get("ten_hang")))) for k, r in master.items()]
    master._idx_ready = True


def _code_match(item: dict, master, fuzzy: bool):
    main = norm_code(item.get("item_code"))
    alts = [a for a in (norm_code(x) for x in (item.get("alt_codes") or [])) if a]
    if main and main in master:
        return master[main], "EXACT"
    for a in alts:
        if a in master:
            return master[a], "ALT_CODE"
    for v in item.get("code_variants") or []:   # mã gốc / mã gốc + tiền tố SX, KD, HT (theo quy tắc khách hàng)
        if v in master:
            return master[v], "VARIANT"
    if not fuzzy:
        return None, "NO_MATCH"

    texts = [t for t in ([main] if main else []) + alts if len(t) >= 4]
    # 1) Mã trên chứng từ CHỨA mã NPL của lịch sử (VD: 'PO-HT61283S-X' chứa 'HT61283S')
    found = {}
    for t in texts:
        for L in master._key_lengths:
            if L > len(t):
                break
            for i in range(len(t) - L + 1):
                sub = t[i:i + L]
                if sub in master:
                    found[sub] = master[sub]
    if found:
        longest = max(len(k) for k in found)
        best = [k for k in found if len(k) == longest]
        if len(best) == 1:
            return found[best[0]], "IN_TEXT"
        return None, "AMBIGUOUS"
    # 2) Mã NPL lịch sử CHỨA mã trên chứng từ (VD: chứng từ '3110C', lịch sử 'SX3110C')
    hits = set()
    for t in texts:
        for k in master:
            if t in k:
                hits.add(k)
    if len(hits) == 1:
        return master[hits.pop()], "CONTAINS"
    if len(hits) > 1:
        return None, "AMBIGUOUS"
    return None, "NO_MATCH"


def _digits_only(v) -> str:
    return "".join(ch for ch in str(v or "") if ch.isdigit())


def _desc_match(item: dict, master):
    q = item.get("description") or ""
    if not norm_text(q):
        return None, "NO_MATCH"
    scored = []
    for k, tn, tt in master._desc_idx:
        sc = desc_score(q, tn)
        if sc >= DESC_THRESHOLD:
            scored.append((sc, k))
    if not scored:
        return None, "NO_MATCH"
    top = max(s for s, _ in scored)
    cands = [k for s, k in scored if s >= top - 1e-9]
    if len(cands) == 1:
        return master[cands[0]], "DESC"
    hs = _digits_only(item.get("hs_code"))
    if len(hs) >= 4:
        same = [k for k in cands if _digits_only(master[k].get("hs_code"))[:6] == hs[:6]]
        if len(same) == 1:
            return master[same[0]], "DESC"
        if same:
            cands = same
    if len({_digits_only(master[k].get("hs_code")) for k in cands}) == 1:
        def _dt(k):
            d = master[k].get("decl_date") or ""
            return d[6:] + d[3:5] + d[:2]
        return master[max(cands, key=_dt)], "DESC"
    return None, "AMBIGUOUS"


def match_line_to_master(item: dict, master, code_first: bool):
    """
    Tìm dòng hàng trong lịch sử khai báo.
    code_first=True  (loại hình SXXK/chế xuất/gia công, và dòng chứng từ): ưu tiên MÃ NPL (cho phép mã chứa ký tự khác), không ra thì dùng mô tả.
    code_first=False (loại hình khác): dùng MÔ TẢ HÀNG; chỉ khi không ra mới thử mã khớp chính xác.
    """
    _ensure_index(master)
    if code_first:
        rec, t = _code_match(item, master, fuzzy=True)
        if rec:
            return rec, t
        rec2, t2 = _desc_match(item, master)
        if rec2:
            return rec2, t2
        return None, t if t == "AMBIGUOUS" else t2
    rec, t = _desc_match(item, master)
    if rec:
        return rec, t
    rec2, t2 = _code_match(item, master, fuzzy=False)
    if rec2:
        return rec2, t2
    return None, t if t == "AMBIGUOUS" else t2


def resolve_items_with_master(doc: dict, master, is_draft: bool) -> None:
    """Gắn master_key / master_code / match_type vào từng dòng hàng của chứng từ (sửa trực tiếp, hãy deepcopy trước)."""
    if not doc or not master:
        return
    rule = doc.get("customer_rule") or {}
    prefixes = [norm_code(p) for p in (rule.get("prefix_npl", []) + rule.get("prefix_other", []))]
    for it in doc.get("items") or []:
        base = it.get("base_code") or ""
        if is_draft and base:
            it["code_variants"] = [base] + [p + base for p in prefixes]
        code_first = (bool(it.get("is_npl")) or bool(base)) if is_draft else True
        rec, mt = match_line_to_master(it, master, code_first)
        it["match_type"] = mt
        it["master_key"] = norm_code(rec["code"]) if rec else ""
        it["master_code"] = rec["code"] if rec else ""


def load_history_report_cached(uploaded_file, cache_dir: str):
    """Như load_history_report nhưng lưu kết quả đã xử lý xuống ổ đĩa (file báo cáo nặng ~12MB, lần sau mở gần như tức thì)."""
    h = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    path = os.path.join(cache_dir, f"lichsu_{h}_{PARSER_VERSION}.pkl")
    try:
        if os.path.exists(path):
            with open(path, "rb") as fh:
                pack = pickle.load(fh)
            master = HistoryDict(pack["data"])
            master.meta = pack["meta"]
            return master
    except Exception:
        pass
    master = load_history_report(uploaded_file)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"data": dict(master), "meta": getattr(master, "meta", {})}, fh)
    except Exception:
        pass
    return master
