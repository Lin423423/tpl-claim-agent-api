"""
區塊八：結果驗證與肇責比例後處理
只對 policy_calculation 之外的所有 supported/estimated 項目統一生效——
LLM負責判斷「未套用肇責比例前」的基礎金額，Python這裡統一依肇責比例計算最終金額，
避免LLM在推論時反覆計算肇責比例造成誤差。
"""

ACTUAL_EXPENSE_ITEMS = {"醫療費用", "交通費用", "看護費用", "工作損失", "診斷書費用", "財物損失"}
ESTIMATABLE_ITEMS = {"精神慰撫金", "其他傷害賠償"}


def check_citation_grounded(item, case_context):
    """檢查這個理賠項目引用的案號/判決，是不是真的出現在這次檢索到的
    結果裡。回傳 (有無引用, 是否為捏造引用)。"""
    cited_cases = set((item.get("similar_case_reference") or {}).get("case_nos") or [])
    cited_judg = set((item.get("cited_judgment_reference") or {}).get("chunk_ids") or [])

    if not cited_cases and not cited_judg:
        return False, False

    retrieved_case_nos = {c["case_no"] for c in case_context.get("similar_cases", [])}
    retrieved_judg_ids = {j["source_id"] for j in case_context.get("judgment_excerpts", [])}

    fabricated_case = bool(cited_cases - retrieved_case_nos)
    fabricated_judg = bool(cited_judg - retrieved_judg_ids)

    return True, (fabricated_case or fabricated_judg)


def apply_verification(result: dict, case_context: dict = None) -> dict:
    """保留LLM判斷的基礎金額suggested_amount，由Python統一依被保險人肇責比例計算
    final_amount，並以final_amount加總total_suggested_amount。

    新增：如果 case_context 有提供，會同時檢查每個項目的引用（案號/判決）
    是否真的存在於這次的檢索結果中，捏造引用的項目強制改為 pending_evidence，
    不讓沒有依據的金額進入最終建議總額。"""
    if not isinstance(result, dict):
        return result

    items = result.get("suggested_items", [])
    if not isinstance(items, list):
        result["suggested_items"] = []
        result["total_suggested_amount"] = 0
        return result

    verified_total = 0
    fabrication_flagged = 0

    for item in items:
        if not isinstance(item, dict):
            continue

        # ---- 引用真實性檢查，優先於其他驗算 ----
        if case_context is not None:
            has_citation, fabricated = check_citation_grounded(item, case_context)
            item["_citation_has_reference"] = has_citation
            item["_citation_fabricated"] = fabricated
            if fabricated:
                item["status"] = "pending_evidence"
                item["suggested_amount"] = None
                item["final_amount"] = None
                original_reasoning = item.get("reasoning_summary", "")
                item["reasoning_summary"] = (
                    "【系統自動攔截】模型原始回答引用了本次檢索結果中不存在的案號或判決，"
                    "已強制標記為證據不足，不納入建議金額。原始推理內容：" + original_reasoning
                )
                fabrication_flagged += 1
                continue

        status = item.get("status")
        amount_basis = item.get("amount_basis") or {}
        if not isinstance(amount_basis, dict):
            amount_basis = {}
            item["amount_basis"] = amount_basis

        if status == "pending_evidence":
            # 沒有足夠證據，不納入總額
            item["suggested_amount"] = None
            item["final_amount"] = None
            continue

        if status == "not_applicable":
            # 不適用，不納入總額
            item["suggested_amount"] = 0
            item["final_amount"] = 0
            continue

        if status not in {"supported", "estimated"}:
            item["final_amount"] = None
            continue

        # 取得LLM判斷的基礎金額
        base_amount = item.get("suggested_amount")
        try:
            if base_amount is None:
                item["final_amount"] = None
                continue
            base_amount = float(base_amount)
            if base_amount < 0:
                item["final_amount"] = None
                continue
        except (TypeError, ValueError):
            item["final_amount"] = None
            continue

        # 取得被保險人肇責比例，缺值時預設100%，並限制在合理範圍內
        liability_pct = amount_basis.get("liability_pct")
        try:
            liability_pct = 100.0 if liability_pct is None else float(liability_pct)
            liability_pct = max(0.0, min(100.0, liability_pct))
        except (TypeError, ValueError):
            liability_pct = 100.0

        # suggested_amount 永遠保存「未套用肇責前」的基礎金額，final_amount 才是套用後的結果
        item["suggested_amount"] = int(round(base_amount))
        item["final_amount"] = int(round(base_amount * liability_pct / 100))

        verified_total += item["final_amount"]

    result["total_suggested_amount"] = int(round(verified_total))
    result["_fabrication_flagged_count"] = fabrication_flagged
    return result