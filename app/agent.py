"""
區塊九：端對端理賠查詢函式
串接 retrieval（Vector/BM25/GraphDB三路檢索+RRF融合） → llm（生成建議） → verification（肇責比例後處理）。
"""
import re
import sqlite3

from . import config
from . import retrieval
from . import llm
from . import verification


def get_case_approved_items(case_no):
    """回頭查claim_items，取得該案件全部核定金額>0的理賠項目，
    並把item_name開頭的編號前綴（如「3.」「01」）清掉，回傳結構化清單"""
    conn = sqlite3.connect(config.TPL_TRAIN_DB)
    cur = conn.cursor()
    cur.execute(
        """SELECT item_name, approved_amount FROM claim_items
           WHERE case_no=? AND approved_amount > 0
           ORDER BY approved_amount DESC""",
        (case_no,)
    )
    rows = cur.fetchall()
    conn.close()

    return [
        {"item_name": re.sub(r"^\d+[\.、]?\s*", "", item_name.strip()), "approved_amount": int(amount)}
        for item_name, amount in rows
    ]


def tpl_claim_agent(accident_area: str, own_fault_pct: float, injury_desc: str, backend: str = None) -> dict:
    query_text = f"{accident_area}發生的車禍，本車肇責{own_fault_pct}%，{injury_desc}"

    similar_cases_raw = retrieval.retrieve(query_text, source="case", top_k=5)
    policy_clauses_raw = retrieval.retrieve(query_text, source="policy", top_k=5)
    judgment_raw = retrieval.retrieve(query_text, source="judgment", top_k=5)

    case_context = {
        "accident_area": accident_area,
        "own_fault_pct": own_fault_pct,
        "injury_desc": injury_desc,
        "similar_cases": [
            {
                "rank": i + 1,
                "rrf_score": round(c["rrf_score"], 5),
                "case_no": c["doc_id"],
                "summary": c["text"][:400],
                "approved_items": get_case_approved_items(c["doc_id"]),
                "quality_flags": c.get("quality_flags", []),
            }
            for i, c in enumerate(similar_cases_raw)
        ],
        "policy_clauses": [
            {
                "rank": i + 1,
                "rrf_score": round(p["rrf_score"], 5),
                "id": p["doc_id"],
                "article": p.get("metadata", {}).get("article", ""),
                "text": p["text"][:500],
            }
            for i, p in enumerate(policy_clauses_raw)
        ],
        "judgment_excerpts": [
            {
                "rank": i + 1,
                "rrf_score": round(j["rrf_score"], 5),
                "source_id": j["doc_id"],
                "text": retrieval.get_judgment_text_for_rag(j["doc_id"]),
            }
            for i, j in enumerate(judgment_raw)
        ],
    }

    result = llm.generate_claim_suggestion(case_context, backend=backend)
    if "error" not in result:
        result = verification.apply_verification(result)

    result["_retrieved_context"] = case_context
    return result