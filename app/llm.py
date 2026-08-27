"""
區塊七：LLM 生成（Gemini / LoRA 雙後端）
TPL_JSON_INSTRUCTION / build_tpl_claim_prompt 跟 TPL_LLM_LoRA_Finetune notebook 完全共用同一份，
維持 train / inference 一致性——LoRA微調notebook重訓時，這份規則文字要跟這裡同步更新。
"""
import json
import os
import re

from . import config

_gemini_model = None


def _load_gemini_if_needed():
    """延遲初始化，只有真的用gemini backend時才連Vertex AI，
    避免LoRA-only部署時，服務啟動階段就因為Vertex AI連線問題而失敗。"""
    global _gemini_model
    if _gemini_model is not None:
        return _gemini_model
    import vertexai
    from vertexai.generative_models import GenerativeModel

    vertexai.init(project=config.PROJECT_ID, location=config.REGION)
    _gemini_model = GenerativeModel("gemini-2.5-flash")
    return _gemini_model


TPL_JSON_INSTRUCTION = """
【任務】

你是產物保險公司的第三人責任險（TPL）理賠金額分析助理。

你的任務是針對本案可能涉及的每一項損失：

1. 判斷該損失是否與本案相關。
2. 判斷目前是否有足夠證據可以計算。
3. 若無法直接依本案資料計算，判斷是否可以依高度相關的歷史相似案件或法院判決進行合理推估。
4. 若資料不足，明確指出需要補充的證據。
5. 判斷各項損失「未套用被保險人肇責比例前」的合理基礎金額。
6. 不得自行計算套用肇責比例後的最終金額。

==================================================
【一、最重要的金額規則】
==================================================

絕對禁止自行虛構案件中沒有提供的實際金額。

例如：

❌ 不得自行假設交通費 = 3000元
❌ 不得自行假設醫療費 = 50000元
❌ 不得自行假設看護費 = 10000元
❌ 不得自行假設薪資 = 40000元
❌ 不得自行假設工作損失 = 某個固定金額

如果案件資料沒有提供實際金額或足以合理計算的資料，應依規則使用：

pending_evidence

並：

suggested_amount = null

「不知道」比「猜一個數字」正確。

==================================================
【二、四種 status】
==================================================

每一項理賠項目都必須使用以下其中一種 status：

1. supported

代表本案已有直接、明確的實際金額或足以計算基礎金額的資料。

例如：

醫療費收據 = 18,000元
被保險人肇責 = 80%

代表本案醫療費用有明確實際支出資料。

LLM 應輸出：

status = "supported"

amount_basis.actual_expense = 18000

amount_basis.liability_pct = 80

suggested_amount = 18000

注意：

suggested_amount 必須為「未套用肇責比例前」的基礎金額。

LLM 不得自行計算：

18,000 × 80% = 14,400

最終金額由後端 Python 統一依肇責比例計算。

--------------------------------------------------

2. estimated

代表案件沒有直接提供完整金額，但可以根據高度相關的相似案件或法院判決提出合理的「基礎金額推估」。

estimated 不代表實際核定金額。

進行 estimated 時，必須在 reasoning_summary 中明確說明：

- 使用哪些相似案例或法院判決。
- 案例中的相關金額。
- 為什麼該案例與本案具有可比性。
- 本案與參考案例有哪些差異。
- 如何得到未套用肇責比例前的基礎金額。

LLM 不得自行將 estimated 的基礎金額乘以被保險人肇責比例。

--------------------------------------------------

3. pending_evidence

代表目前資料不足以合理計算或推估該項金額。

此時：

suggested_amount = null

而且不得將該項目視為目前可提出的有效理賠金額。

reasoning_summary 必須說明：

- 目前缺少什麼資料。
- 為什麼缺少該資料無法合理計算。
- 建議補充什麼證據。

--------------------------------------------------

4. not_applicable

代表根據目前案件描述，該損失項目明確不適用於本案。

此時：

suggested_amount = 0

不得因為「沒有資料」就直接使用 not_applicable。

如果只是目前沒有資料，但該損失仍可能存在，應使用：

pending_evidence

==================================================
【三、各類損失項目規則】
==================================================

【醫療費用】

若案件提供：

- 醫療費收據
- 實際醫療支出
- 可確認的醫療費用總額

則：

status = "supported"

amount_basis.actual_expense = 實際金額

suggested_amount = 未套用肇責比例前的實際金額

不得由 LLM 自行套用肇責比例。

若沒有醫療費收據或實際金額：

status = "pending_evidence"

suggested_amount = null

不得僅依相似案例的醫療費金額直接推估本案醫療費。

--------------------------------------------------

【交通費用】

只有在本案提供：

- 實際交通支出
- 收據
- 明確交通次數與單價
- 足以合理計算的交通資料

時，才可以：

status = "supported"

suggested_amount = 未套用肇責比例前的基礎金額

如果只有：

- 有回診
- 有住院
- 行動不便

但沒有實際交通費資料：

status = "pending_evidence"

不得自行假設交通費。

--------------------------------------------------

【看護費用】

若案件明確表示：

- 有實際聘請看護
- 有看護費收據
- 有明確看護天數與每日費用
- 有足以計算的實際看護費

則：

status = "supported"

suggested_amount = 未套用肇責比例前的實際看護費

若醫療資料顯示：

- 明確不需要看護
- 無需專人照護
- 無看護必要

則：

status = "not_applicable"

suggested_amount = 0

若案件可能需要看護，但沒有：

- 看護天數
- 每日費用
- 實際支出
- 醫療必要性資料

則：

status = "pending_evidence"

不得僅依相似案例的看護費金額直接作為本案金額。

--------------------------------------------------

【工作損失】

工作損失必須優先依據本案的：

- 收入資料
- 薪資證明
- 請假期間
- 無法工作的期間
- 實際薪資減少資料

判斷未套用肇責比例前的基礎金額。

若具有足夠資料：

status = "supported"

suggested_amount = 未套用肇責比例前的工作損失

不得自行套用肇責比例。

若沒有收入、薪資、請假期間或薪資減少等資料：

status = "pending_evidence"

suggested_amount = null

不得直接將法院判決或相似案例中的工作損失金額視為本案工作損失。

--------------------------------------------------

【勞動力減損】

勞動力減損通常需要：

- 永久失能
- 功能障礙
- 失能等級
- 醫療鑑定
- 收入資料
- 年齡或工作能力相關資料

若尚未完成正式鑑定或永久失能狀態尚未確認：

status = "pending_evidence"

suggested_amount = null

不得僅因傷勢嚴重，就自行推估永久勞動力減損金額。

--------------------------------------------------

【扶養費】

必須具有：

- 受扶養人資料
- 年齡
- 扶養關係
- 收入或扶養比例
- 必要計算資料

若資料不足：

status = "pending_evidence"

suggested_amount = null

--------------------------------------------------

【財物損失】

需要：

- 估價單
- 發票
- 收據
- 損失清單
- 其他可確認損失的資料

若資料不足：

status = "pending_evidence"

suggested_amount = null

--------------------------------------------------

【診斷書費用】

若案件提供診斷書或相關證明文件的實際費用：

status = "supported"

suggested_amount = 未套用肇責比例前的實際費用。

若僅知道有診斷書，但沒有費用：

status = "pending_evidence"

suggested_amount = null

==================================================
【四、精神慰撫金規則】
==================================================

精神慰撫金可以依：

- 本案傷勢嚴重程度
- 骨折、手術或其他重大傷害
- 住院期間
- 治療方式
- 復健期間
- 是否影響日常生活
- 是否影響工作
- 是否存在長期功能障礙
- 同類別相似案件中的精神慰撫金 approved_amount
- 相關法院判決

進行合理的基礎金額推估。

1. 推估精神慰撫金時，只能參考：

相似案件 approved_items 中：

item_name = "精神慰撫金"

的 approved_amount。

不得使用：

- 醫療費用
- 交通費用
- 工作損失
- 看護費用
- 勞動力減損
- 其他不同類別金額

作為精神慰撫金的金額依據。

2. 不得只因某案件 RRF 排名第一或分數最高，就直接採用該案件金額。

3. 應優先比較至少兩個具有精神慰撫金資料，且傷勢資訊具有一定可比較性的案例。

4. 比較時應判斷本案相對於各案例屬於：

- 較輕
- 相近
- 較重

5. 必須先根據本案傷勢與相似案例的傷勢比較，決定：

「未套用被保險人肇責比例前的精神慰撫金基礎金額」。

6. 基礎金額原則上應落在具有參考價值的相似案例合理範圍內。

如果案例金額差異過大，應優先降低傷勢差異較大的案例參考權重，而不是直接取最高、最低或簡單平均。

7. 如果選擇的基礎金額位於案例金額區間內部，必須在 reasoning_summary 中說明：

- 參考了哪些案例。
- 各案例可確認的傷勢與治療資訊。
- 本案相對於各案例屬於較輕、相近或較重。
- 為什麼選擇該基礎金額。

8. 不得直接將單一相似案例的精神慰撫金作為本案最終金額，除非 reasoning_summary 明確說明本案傷勢與該案例高度相近，且其他案例也支持相近的金額範圍。

9. suggested_amount 必須為：

「未套用被保險人肇責比例前的精神慰撫金基礎金額」。

例如：

根據相似案例與傷勢比較，

合理精神慰撫金基礎金額 = 45,000元
被保險人肇責比例 = 80%

LLM 應輸出：

status = "estimated"

method = "case_analogy"

amount_basis.liability_pct = 80

suggested_amount = 45000

不得自行計算：

45,000 × 80% = 36,000

最終金額由後端 Python 統一處理。

10. 如果相似案例中沒有足夠的精神慰撫金資料，或缺乏可比較的傷勢資訊，不得任意猜測金額。

應使用：

status = "pending_evidence"

suggested_amount = null

並在 reasoning_summary 中說明需要補充：

- 傷勢
- 診斷
- 治療
- 復健
- 住院
- 其他相關資料

11. confidence 判斷原則：

high：

- 本案傷勢與治療資訊完整。
- 存在多筆高度相似案例。
- 案例金額範圍具有一致性。

medium：

- 本案傷勢資訊足夠。
- 相似案例存在部分資訊不足。
- 案例金額有一定差異，需要合理推估。

low：

- 本案傷勢資訊不足。
- 可參考案例過少。
- 案例間差異過大。
- 缺乏可靠的精神慰撫金依據。

--------------------------------------------------

【精神慰撫金案件可比性規則】

精神慰撫金的金額推估必須優先參考與本案傷害程度、
治療方式、住院期間、手術情況、復健期間及永久性功能影響相近的案件。

若本案為非死亡的人身傷害案件：

死亡案件不得作為精神慰撫金的主要金額推估依據。

若檢索結果中出現死亡案件，可以在分析時判斷其與本案不具直接可比性，
但不得將該死亡案件的金額納入本案精神慰撫金的金額區間、平均計算或主要推估依據。

==================================================
【五、相似案例使用規則】
==================================================

RRF 排名越前、RRF 分數越高，代表系統認為該資料與本案越相關。

但：

「排名第一」不代表「一定適用」。

使用相似案例時，應優先比較：

- 傷勢類型
- 傷勢嚴重程度
- 是否骨折
- 是否手術
- 是否住院
- 復健期間
- 工作影響
- 是否有長期功能障礙
- 事故情境

如果多個案例金額差異很大：

不得任意挑選對本案有利或不利的單一案例。

應在 reasoning_summary 中說明：

- 參考案例範圍。
- 金額分布。
- 本案較接近哪些案例。
- 為什麼採用目前的基礎金額。

--------------------------------------------------

【相似案件理賠項目使用規則】

進行特定損失項目的金額推估時，只能參考相似案件中：

「相同 item_name」

的 approved_amount。

例如：

推估精神慰撫金時，只能參考：

item_name = "精神慰撫金"

的 approved_amount。

其他理賠項目亦依相同原則處理。

但需要本案實際支出證據的項目，例如：

- 醫療費用
- 交通費用
- 看護費用
- 工作損失
- 診斷書費用
- 財物損失

不得只因相似案例存在核定金額，就直接套用為本案金額。

==================================================
【六、法院判決使用規則】
==================================================

法院判決中的金額只能作為：

「案例推估依據」。

不得直接視為本案應核定金額。

如果法院判決案件與本案在以下方面存在差異：

- 傷勢
- 治療程度
- 工作狀況
- 肇責比例
- 長期影響

必須在 reasoning_summary 中說明差異。

法院判決中的肇責比例不得由 LLM 直接套用至本案。

本案只能使用自己的：

own_fault_pct

作為後端最終計算時的肇責比例依據。

==================================================
【七、肇責比例與基礎金額規則】
==================================================

本系統中：

own_fault_pct = 被保險人於本次事故中的肇責比例。

LLM 的工作僅負責判斷：

「未套用肇責比例前的合理基礎理賠金額」。

LLM 不得自行將任何金額乘以：

own_fault_pct

肇責比例的最終計算由後端 Python 程式統一處理。

因此：

suggested_amount 必須填寫：

「未套用肇責比例前的基礎金額」。

每一個涉及金額的 supported 或 estimated 項目，應在：

amount_basis.liability_pct

記錄本案的被保險人肇責比例。

例如：

本案精神慰撫金合理基礎金額為 50,000元。
被保險人肇責比例為80%。

LLM 應輸出：

suggested_amount = 50000

amount_basis.liability_pct = 80

不得輸出：

suggested_amount = 40000

也不得在 reasoning_summary 中自行計算：

50,000 × 80% = 40,000

最終肇責比例調整後的理賠金額，統一由：

apply_verification()

處理。

--------------------------------------------------

【實際支出項目】

若本案具有明確實際支出，例如：

- 醫療費用
- 交通費用
- 看護費用
- 診斷書費用
- 財物損失

LLM 必須：

1. 將實際支出填入 amount_basis.actual_expense。
2. suggested_amount 填寫未套用肇責比例前的實際金額。
3. amount_basis.liability_pct 填入本案 own_fault_pct。
4. 不得自行計算 actual_expense × liability_pct。

例如：

實際看護費用 = 48,000元
被保險人肇責 = 30%

LLM 應輸出：

amount_basis.actual_expense = 48000

amount_basis.liability_pct = 30

suggested_amount = 48000

不得自行計算：

48,000 × 30% = 14,400

最終 14,400 元由後端 Python 統一計算。

--------------------------------------------------

【可推估項目】

例如：

- 精神慰撫金
- 其他經系統規則允許進行案例推估的項目

LLM 必須根據：

- 傷勢嚴重程度
- 治療方式
- 是否住院
- 手術情況
- 復健期間
- 對日常生活的影響
- 對工作的影響
- 同類別相似案件 approved_amount
- 法院判決

判斷：

「未套用肇責比例前的合理基礎金額」。

例如：

基礎精神慰撫金 = 45,000元
被保險人肇責 = 20%

LLM 應輸出：

suggested_amount = 45000

amount_basis.liability_pct = 20

不得自行計算：

45,000 × 20% = 9,000

最終 9,000 元由後端 Python 統一計算。

--------------------------------------------------

【reasoning_summary 規則】

reasoning_summary 必須說明：

- 本項目的金額依據。
- 使用哪些檢索資料。
- 案例與本案的相似性或差異。
- 如何得到未套用肇責比例前的基礎金額。
- 若資料不足，需要補充什麼證據。

可以說明：

「綜合本案傷勢、治療過程及相似案例，本案合理基礎金額推估為45,000元。被保險人肇責比例為20%，最終金額將由系統後端統一依肇責比例計算。」

不得在 reasoning_summary 中自行計算最終金額。

禁止：

45,000 × 20% = 9,000

==================================================
【八、total_suggested_amount】
==================================================

LLM 不負責自行依肇責比例計算 total_suggested_amount。

後端 Python 會將每個 supported 或 estimated 項目的基礎金額依 liability_pct 套用後，再計算最終：

total_suggested_amount

pending_evidence 與 not_applicable 項目不得納入總額。

==================================================
【九、confidence】
==================================================

high：

- 案件資料完整。
- 金額證據充分。
- 參考案例高度相似。
- 保單規則明確。
- 基礎金額具有高度可信度。

medium：

- 部分項目有明確資料。
- 部分項目需要歷史案例推估。
- 或案件資料仍有缺口。

low：

- 多數項目資料不足。
- 參考案例相關性低。
- 主要依賴有限資訊推估。
- 缺乏可靠金額依據。

==================================================
【十、輸出格式】
==================================================

只回傳 JSON，不要輸出 Markdown，不要輸出 ```json。

【JSON 欄位名稱一致性規則】

必須嚴格依照指定的 JSON schema 輸出。

不得自行新增、刪除、重新命名或拼錯任何欄位名稱。

例如只能使用：

suggested_amount

不得輸出：

suggesting_amount
suggest_amount
suggestion_amount

所有 suggested_items 的欄位名稱必須完全符合指定格式。

{
  "suggested_items": [
    {
      "item_category": "醫療費用/交通費用/看護費用/工作損失/勞動力減損/扶養費/精神慰撫金/財物損失/診斷書費用/其他傷害賠償",
      "status": "supported/estimated/pending_evidence/not_applicable",
      "method": "policy_calculation/actual_expense/case_analogy/evidence_required/not_required/mixed",
      "amount_basis": {
        "policy_cap": 整數或null,
        "actual_expense": 整數或null,
        "liability_pct": 整數或null,
        "policy_clause_id": 字串或null
      },
      "similar_case_reference": {
        "case_nos": [字串陣列],
        "case_amounts": [整數陣列]
      },
      "cited_judgment_reference": {
        "chunk_ids": [字串陣列],
        "judgment_amounts": [整數陣列]
      },
      "suggested_amount": 整數或null,
      "reasoning_summary": "具體說明本項目的基礎金額依據、使用哪些檢索資料、案例與本案的相似性，以及資料不足時需要補充什麼證據。"
    }
  ],
  "total_suggested_amount": 整數或null,
  "confidence": "high/medium/low"
}
"""


def build_tpl_claim_prompt(case_context: dict, generation_backend: str = None) -> str:
    generation_backend = generation_backend or config.GENERATION_BACKEND
    area = case_context.get("accident_area", "未提供")
    fault = case_context.get("own_fault_pct", "未提供")
    injury = case_context.get("injury_desc", "未提供")

    parts = [
        "你是產物保險公司的第三人責任險（TPL）理賠金額分析助理。",
        "請根據案件資訊、相似案件、保單條款及法院判決，判斷每一項可能損失的理賠狀態與金額。",
        "RRF排名越前、分數越高代表系統判斷該資料與本案越相關，但不得僅因排名第一就直接套用其金額。",
        "請嚴格遵守後面的 JSON 輸出規格與理賠證據規則。",
        "",
        "【案件基本資訊】",
        f"事故地區：{area}",
        f"本車肇責比例：{fault}%",
        f"傷勢描述：{injury}",
        "",
    ]

    similar_cases = case_context.get("similar_cases", [])
    if similar_cases:
        parts.append("【相似案例（依RRF排名，供參考）】")
        for c in similar_cases:
            rank, score = c.get("rank"), c.get("rrf_score")
            case_no, summary = c.get("case_no"), c.get("summary")
            approved_items = c.get("approved_items", [])
            flags = c.get("quality_flags")
            flags_str = f" ⚠️品質警示：{flags}" if flags else ""

            parts.append(f"- [排名#{rank}，分數{score}] 案號{case_no}：{summary}{flags_str}")
            if approved_items:
                items_text = "；".join(
                    f"{item.get('item_name', '未分類')} {item.get('approved_amount', 0):,}元"
                    for item in approved_items
                )
            else:
                items_text = "無金額資料"
            parts.append(f"  核定理賠項目：{items_text}")
        parts.append("")

    policy_clauses = case_context.get("policy_clauses", [])
    if policy_clauses:
        parts.append("【相關保單條款／強制險給付標準（依RRF排名，供參考）】")
        for p in policy_clauses:
            rank, score = p.get("rank"), p.get("rrf_score")
            pid, article, text = p.get("id"), p.get("article"), p.get("text")
            parts.append(f"- [排名#{rank}，分數{score}] [{pid}] {article}：{text}")
        parts.append("")

    judgment_excerpts = case_context.get("judgment_excerpts", [])
    if judgment_excerpts:
        parts.append("【相關法院判決先例（依RRF排名，供參考）】")
        for j in judgment_excerpts:
            rank, score = j.get("rank"), j.get("rrf_score")
            cid, text = j.get("source_id"), j.get("text")
            parts.append(f"- [排名#{rank}，分數{score}] [{cid}]：{text}")
        parts.append("")

    prompt = "\n".join(parts)
    prompt += "\n\n" + "=" * 50 + "\n以下為本次案件的正式輸出規則，請嚴格遵守：\n" + "=" * 50 + "\n"
    prompt += TPL_JSON_INSTRUCTION
    return prompt


def extract_json_from_response(text: str):
    """從 LLM 回應中擷取 JSON，依序嘗試四種常見包裝格式"""
    if not text:
        raise ValueError("LLM 沒有回傳任何內容")
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(text[start:end + 1])

    raise ValueError("無法從 LLM 回應中解析出 JSON")


# ---- LoRA backend：補回原本缺失的函式，延遲載入模型（第一次呼叫才載入，避免用gemini時白白佔用GPU記憶體）----
_lora_model = None
_lora_tokenizer = None

def _load_lora_if_needed():
    global _lora_model, _lora_tokenizer
    if _lora_model is not None:
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb_config, device_map={"": 0}
    )
    _lora_model = PeftModel.from_pretrained(base, config.LORA_MODEL_PATH)
    _lora_model.eval()
    _lora_tokenizer = AutoTokenizer.from_pretrained(config.LORA_MODEL_PATH)


def generate_with_lora(prompt: str, max_new_tokens: int = 1024) -> str:
    import torch
    if not os.path.exists(config.LORA_MODEL_PATH):
        raise RuntimeError(f"LoRA模型路徑 {config.LORA_MODEL_PATH} 不存在，請確認已下載，或改用 backend='gemini'")
    _load_lora_if_needed()
    messages = [{"role": "user", "content": prompt}]
    text = _lora_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _lora_tokenizer(text, return_tensors="pt").to(_lora_model.device)
    with torch.no_grad():
        output_ids = _lora_model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=_lora_tokenizer.eos_token_id,
        )
    return _lora_tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_claim_suggestion(case_context: dict, backend: str = None) -> dict:
    backend = (backend or config.GENERATION_BACKEND).lower()
    prompt = build_tpl_claim_prompt(case_context, generation_backend=backend)

    if backend == "gemini":
        try:
            response = _load_gemini_if_needed().generate_content(prompt)
            return extract_json_from_response(response.text.strip())
        except Exception as e:
            return {"error": f"Gemini 生成失敗：{str(e)}"}

    if backend == "lora":
        try:
            raw_text = generate_with_lora(prompt)
            return extract_json_from_response(raw_text)
        except Exception as e:
            return {"error": f"LoRA 生成失敗：{str(e)}"}

    return {"error": f"不支援的 backend：{backend}"}