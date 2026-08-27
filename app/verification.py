"""
區塊八：結果驗證與肇責比例後處理
只對 policy_calculation 之外的所有 supported/estimated 項目統一生效——
LLM負責判斷「未套用肇責比例前」的基礎金額，Python這裡統一依肇責比例計算最終金額，
避免LLM在推論時反覆計算肇責比例造成誤差。
"""

ACTUAL_EXPENSE_ITEMS = {"醫療費用", "交通費用", "看護費用", "工作損失", "診斷書費用", "財物損失"}
ESTIMATABLE_ITEMS = {"精神慰撫金", "其他傷害賠償"}


def apply_verification(result: dict) -> dict:
    """保留LLM判斷的基礎金額suggested_amount，由Python統一依被保險人肇責比例計算
    final_amount，並以final_amount加總total_suggested_amount。"""
    if not isinstance(result, dict):
        return result

    items = result.get("suggested_items", [])
    if not isinstance(items, list):
        result["suggested_items"] = []
        result["total_suggested_amount"] = 0
        return result

    verified_total = 0

    for item in items:
        if not isinstance(item, dict):
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
    return result