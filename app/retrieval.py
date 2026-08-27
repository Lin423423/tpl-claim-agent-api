"""
區塊四~六：Vector RAG / BM25 / GraphDB 檢索 + RRF融合
所有可變狀態（embedding model、chroma collections、BM25語料庫、policy文字索引）
都透過 init_retrieval() 在服務啟動時（main.py 的 lifespan）初始化一次，
避免每個request重複載入模型/重新讀資料庫。
"""
import os
import re
import sqlite3
import difflib
from collections import defaultdict

import jieba
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb

from . import config

# ---- module-level狀態，由 init_retrieval() 填入 ----
embed_model = None
chroma_client = None
policy_collection = None
case_collection = None
judgment_collection = None
BM25_CORPORA = {}
_POLICY_TEXT_INDEX = None


def init_retrieval():
    """服務啟動時呼叫一次：載入embedding模型、連上ChromaDB、載入BM25語料庫、建立policy文字索引。"""
    global embed_model, chroma_client, policy_collection, case_collection, judgment_collection, BM25_CORPORA

    embed_model = SentenceTransformer(config.FINETUNED_EMBED_MODEL_PATH)

    chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    policy_collection = chroma_client.get_collection("tpl_policy")
    case_collection = chroma_client.get_collection("tpl_case")
    judgment_collection = chroma_client.get_collection("tpl_judgment")
    print(
        f"✅ ChromaDB連線：policy={policy_collection.count()} "
        f"case={case_collection.count()} judgment={judgment_collection.count()}"
    )

    bm25_dir = config.BM25_LOCAL_DIR
    tpl_clause_bm25_db = f"{bm25_dir}/TPL_Clause_BM25_Tuned.db"
    tpl_judgment_bm25_db = f"{bm25_dir}/TPL_Judgment_BM25_Tuned.db"
    tpl_claim_bm25_db = f"{bm25_dir}/TPL_Claim_BM25_Tuned.db"

    BM25_CORPORA = {
        "policy":   _load_policy_corpus(tpl_clause_bm25_db, config.TPL_POLICY_DB),
        "judgment": _load_judgment_corpus(tpl_judgment_bm25_db),
        "case":     _load_case_corpus(tpl_claim_bm25_db),
    }
    for name, corp in BM25_CORPORA.items():
        print(f"✅ BM25 {name} 庫：{len(corp['texts'])} 筆")

    _build_policy_text_index()
    print("✅ retrieval 模組初始化完成")


def embed_query(text):
    emb = embed_model.encode([f"query: {text}"], normalize_embeddings=True)
    return emb[0].tolist()


def vector_search_policy(query_text, top_k=5, exclude_article=("前言",)):
    """exclude_article：過濾掉封面/前言雜訊條目，用迴圈動態加大撈取範圍，
    避免固定倍數在污染筆數未知時不夠用"""
    query_emb = embed_query(query_text)
    max_n = policy_collection.count()
    fetch_n = top_k

    while True:
        fetch_n = min(fetch_n, max_n)
        results = policy_collection.query(query_embeddings=[query_emb], n_results=fetch_n)
        filtered = [
            {"doc_id": doc_id, "metadata": meta, "text": doc, "distance": dist}
            for doc_id, doc, meta, dist in zip(
                results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
            )
            if meta.get("article") not in exclude_article
        ]
        if len(filtered) >= top_k or fetch_n >= max_n:
            return filtered[:top_k]
        fetch_n *= 3

def vector_search_case(query_text, top_k=5):
    query_emb = embed_query(query_text)
    results = case_collection.query(query_embeddings=[query_emb], n_results=top_k)
    return [
        {"doc_id": doc_id, "text": doc, "metadata": meta, "distance": dist}
        for doc_id, doc, meta, dist in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]

def vector_search_judgment(query_text, top_k_chunks=20, top_k_docs=5):
    """先撈較多 chunk（top_k_chunks），再依 case_no 分組取每篇最佳分數（max-pooling），
    最後回傳聚合後的文件層級排名（top_k_docs 篇）。
    doc_id 去除空格，跟 BM25 的案號格式對齊。"""
    query_emb = embed_query(query_text)
    results = judgment_collection.query(query_embeddings=[query_emb], n_results=top_k_chunks)

    by_case = defaultdict(list)
    for doc_id, doc, meta, dist in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        case_no = meta.get("case_no", doc_id.rsplit("__chunk", 1)[0])
        by_case[case_no].append({"chunk_id": doc_id, "text": doc, "distance": dist})

    aggregated = []
    for case_no, chunks in by_case.items():
        best_chunk = min(chunks, key=lambda c: c["distance"])
        aggregated.append({
            "doc_id": case_no.replace(" ", ""),   # 去空格，對齊BM25格式
            "text": best_chunk["text"],
            "matched_chunk_id": best_chunk["chunk_id"],
            "distance": best_chunk["distance"],
        })

    aggregated.sort(key=lambda x: x["distance"])
    return aggregated[:top_k_docs]


def _normalize_key(s):
    """去掉所有標點符號跟底線，只留中文字/英文字母/數字，讓不同符號規則的檔名可以互相比對"""
    return re.sub(r'[^\w\u4e00-\u9fff]', '', s or '').replace('_', '')


def _normalize_article(article):
    if article is None:
        return None
    s = article.strip().replace(' ', '').replace('　', '')
    cn_num_map = {'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10}
    m = re.match(r'第([一二三四五六七八九十]+)條', s)
    if m:
        cn = m.group(1)
        if cn in cn_num_map:
            num = cn_num_map[cn]
        elif len(cn) == 2 and cn[0] == '十':
            num = 10 + cn_num_map.get(cn[1], 0)
        elif len(cn) == 2 and cn[1] == '十':
            num = cn_num_map.get(cn[0], 0) * 10
        else:
            num = cn
        return f"第{num}條"
    return s


def _load_bm25_params(conn):
    try:
        cur = conn.cursor()
        cur.execute("SELECT k1, b FROM bm25_best_params LIMIT 1;")
        row = cur.fetchone()
        if row:
            return {"k1": row[0], "b": row[1]}
    except sqlite3.OperationalError:
        pass
    return {"k1": 1.5, "b": 0.75}


def _load_policy_corpus(bm25_db_path, policy_db_path):
    """source_record_id 直接對應 policy_clauses.id，不用article/檔名比對"""
    conn = sqlite3.connect(bm25_db_path)
    params = _load_bm25_params(conn)
    cur = conn.cursor()
    cur.execute("""
        SELECT source_record_id, index_text
        FROM bm25_ready
        WHERE chunk_type != 'preamble' AND length(index_text) > 40
    """)
    rows = cur.fetchall()
    conn.close()

    ids = [str(r[0]) for r in rows]
    texts = [r[1] for r in rows]
    tokenized = [list(jieba.cut(t)) for t in texts]
    index = BM25Okapi(tokenized, k1=params["k1"], b=params["b"])
    print(f"policy 語料庫載入：{len(ids)} 筆可用（source_record_id直接對應policy_clauses.id）")
    return {"ids": ids, "texts": texts, "index": index, "params": params}


def _load_judgment_corpus(bm25_db_path):
    """title 欄位本身就含完整案號，直接解析當doc_id，不用查別的db"""
    conn = sqlite3.connect(bm25_db_path)
    params = _load_bm25_params(conn)
    cur = conn.cursor()
    cur.execute("SELECT title, index_text FROM bm25_ready")
    rows = cur.fetchall()
    conn.close()

    ids, texts = [], []
    skipped = 0
    for title, text in rows:
        parts = title.split('｜') if title else []
        if len(parts) >= 4:
            ids.append(parts[3])  # 完整案號，例如"臺灣宜蘭地方法院114年度簡上字第37號民事判決"
            texts.append(text)
        else:
            skipped += 1

    tokenized = [list(jieba.cut(t)) for t in texts]
    index = BM25Okapi(tokenized, k1=params["k1"], b=params["b"])
    print(f"judgment 語料庫載入：{len(ids)} 筆可用（{skipped} 筆title格式異常，已跳過）")
    return {"ids": ids, "texts": texts, "index": index, "params": params}


def _load_case_corpus(bm25_db_path):
    """case_no 本身就跟 Vector RAG 格式一致，直接用"""
    conn = sqlite3.connect(bm25_db_path)
    params = _load_bm25_params(conn)
    cur = conn.cursor()
    cur.execute("SELECT case_no, index_text FROM bm25_ready WHERE case_no IS NOT NULL")
    rows = cur.fetchall()
    conn.close()

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    tokenized = [list(jieba.cut(t)) for t in texts]
    index = BM25Okapi(tokenized, k1=params["k1"], b=params["b"])
    print(f"case 語料庫載入：{len(ids)} 筆可用")
    return {"ids": ids, "texts": texts, "index": index, "params": params}


def bm25_search(query_text, source, top_k=5):
    if source not in BM25_CORPORA:
        return []
    corp = BM25_CORPORA[source]
    if not corp["texts"]:
        return []

    tokenized_q = list(jieba.cut(query_text))
    scores = corp["index"].get_scores(tokenized_q)
    wide_top_idx = scores.argsort()[::-1][:top_k * 5]  # judgment可能重複doc_id，多撈一點再去重

    best_by_doc = {}
    for i in wide_top_idx:
        if scores[i] <= 0:
            continue
        doc_id = corp["ids"][i]
        if doc_id not in best_by_doc or scores[i] > best_by_doc[doc_id]["bm25_score"]:
            best_by_doc[doc_id] = {"doc_id": doc_id, "text": corp["texts"][i], "bm25_score": float(scores[i])}

    ranked = sorted(best_by_doc.values(), key=lambda x: -x["bm25_score"])
    return ranked[:top_k]


# policy 路徑：/v1/retrieve/clause，直接查
# judgment 路徑：/v1/answer/freetext，直接查（不用case錨點橋接，維持RRF三路獨立性）
# case 路徑：GraphDB沒有對應端點，永遠回空list
import requests, sqlite3, difflib

GRAPH_BASE_URL = config.GRAPH_BASE_URL
GRAPH_API_KEY = config.GRAPH_API_KEY
GRAPH_HEADERS = {"X-API-Key": GRAPH_API_KEY, "Content-Type": "application/json"}


# ---------- policy 路徑 ----------

def _build_policy_text_index():
    """一次性讀出全部124筆policy_clauses的(id, text)，供內容比對用"""
    global _POLICY_TEXT_INDEX
    if _POLICY_TEXT_INDEX is not None:
        return _POLICY_TEXT_INDEX

    conn = sqlite3.connect(config.TPL_POLICY_DB)
    cur = conn.cursor()
    cur.execute("SELECT id, text FROM policy_clauses")
    _POLICY_TEXT_INDEX = cur.fetchall()
    conn.close()
    print(f"✅ policy 文字索引建立完成，共 {len(_POLICY_TEXT_INDEX)} 筆")
    return _POLICY_TEXT_INDEX


def _find_policy_id_by_content(graphdb_content, min_similarity=0.7):
    """直接拿GraphDB回傳的條文內容，跟policy_clauses裡全部124筆文字做相似度比對，取最相似的那筆"""
    index = _build_policy_text_index()
    best_id, best_score = None, 0.0

    for pc_id, pc_text in index:
        score = difflib.SequenceMatcher(None, graphdb_content[:300], pc_text[:300]).ratio()
        if score > best_score:
            best_score = score
            best_id = pc_id

    if best_score >= min_similarity:
        return best_id, best_score
    return None, best_score


def graphdb_search_policy(query_text, top_k=5):
    """/v1/retrieve/clause：純文字query，不用錨點橋接，直接查。
    對API固定要求至少10筆，避免API內部行為導致回傳筆數少於top_k，最後才截斷成真正要的數量。"""
    request_top_k = max(top_k, 10)
    try:
        resp = requests.post(
            f"{GRAPH_BASE_URL}/v1/retrieve/clause",
            headers=GRAPH_HEADERS,
            json={"query": query_text, "top_k": request_top_k, "insurance_line": "TPL"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ GraphDB clause 呼叫失敗: {e}")
        return []

    results = []
    unmatched = []
    for item in data.get("results", []):
        content = item.get("content", "")
        pc_id, score = _find_policy_id_by_content(content)
        if pc_id is None:
            unmatched.append((item.get("clause_id"), item.get("doc_title"), f"最高相似度僅{score:.2f}"))
            continue
        results.append({
            "doc_id": str(pc_id),
            "text": content,
            "graphdb_score": item.get("score", 0.0),
            "quality_flags": item.get("quality_flags", []),
        })

    if unmatched:
        print(f"⚠️ {len(unmatched)} 筆 GraphDB clause 對不到 policy_clauses.id：")
        for cid, dt, reason in unmatched[:5]:
            print(f"   clause_id={cid} doc_title={dt!r} 原因={reason}")

    return results[:top_k]


# ---------- judgment 路徑（改用 freetext，不用錨點橋接）----------

def get_judgment_summary(case_no_no_space):
    """用judgments.db的結構化欄位組一段精簡摘要（金額明確、缺值時該項略過不寫）"""
    global _JUDGMENT_STRUCTURED_MAP
    if "_JUDGMENT_STRUCTURED_MAP" not in globals():
        conn = sqlite3.connect(config.TPL_JUDGMENTS_DB)
        cur = conn.cursor()
        cur.execute("""
            SELECT case_no, court, year, liability_ratio, medical_fee, nursing_fee,
                   work_loss, labor_loss, mental_damage, vehicle_damage, final_compensation
            FROM judgments
        """)
        _JUDGMENT_STRUCTURED_MAP = {}
        for row in cur.fetchall():
            case_no = row[0].replace(" ", "")
            _JUDGMENT_STRUCTURED_MAP[case_no] = row[1:]
        conn.close()
        print(f"✅ 已建立判決結構化摘要對照表，共 {len(_JUDGMENT_STRUCTURED_MAP)} 篇")

    fields = _JUDGMENT_STRUCTURED_MAP.get(case_no_no_space)
    if fields is None:
        return ""

    court, year, liability_ratio, medical_fee, nursing_fee, work_loss, labor_loss, mental_damage, vehicle_damage, final_compensation = fields

    parts = [f"{court}{year}年度判決" if court else ""]
    if liability_ratio is not None:
        parts.append(f"肇責比例{liability_ratio}%")
    for label, val in [("醫療費用", medical_fee), ("看護費用", nursing_fee), ("工作損失", work_loss),
                        ("勞動力減損", labor_loss), ("精神慰撫金", mental_damage), ("車輛/財損", vehicle_damage)]:
        if val is not None:
            parts.append(f"{label}判賠{val}元")
    if final_compensation is not None:
        parts.append(f"最終判賠總額{final_compensation}元")

    return "，".join(p for p in parts if p)


def get_full_judgment_text(case_no_no_space):
    """撈完整判決全文，不依賴vector的chunk切割，維持GraphDB路徑的完全獨立性"""
    global _FULL_JUDGMENT_TEXT_MAP
    if "_FULL_JUDGMENT_TEXT_MAP" not in globals():
        conn = sqlite3.connect(config.TPL_JUDGMENTS_DB)
        cur = conn.cursor()
        cur.execute("SELECT case_no, judgment_text FROM judgments")
        _FULL_JUDGMENT_TEXT_MAP = {
            case_no.replace(" ", ""): text for case_no, text in cur.fetchall()
        }
        conn.close()
        print(f"✅ 已建立完整判決全文對照表，共 {len(_FULL_JUDGMENT_TEXT_MAP)} 篇")

    return _FULL_JUDGMENT_TEXT_MAP.get(case_no_no_space, "")


def get_judgment_text_for_rag(case_no_no_space, excerpt_length=1000):
    """統一格式：結構化摘要 + 內文節錄，vector/bm25/graphdb三路的judgment候選
    最終都透過這支函式重新撈一次文字，確保呈現格式一致"""
    summary = get_judgment_summary(case_no_no_space)
    full_text = get_full_judgment_text(case_no_no_space)
    excerpt = full_text[:excerpt_length]

    if not summary and not excerpt:
        return ""

    parts = []
    if summary:
        parts.append(f"【判賠金額摘要】{summary}")
    if excerpt:
        parts.append(f"【判決內文節錄】{excerpt}")

    return "\n".join(parts)


def graphdb_search_judgment_freetext(query_text, top_k=5):
    """/v1/answer/freetext：純文字直接查，不依賴case查詢結果當錨點，
    維持RRF三路統計獨立性（排名由GraphDB自己的Neo4j圖譜運算，不受vector/bm25影響）。
    這個端點會呼叫兩次LLM（特徵抽取+生成），比純檢索端點慢，只取precedents部分塞進RRF，
    忽略answer/context/citations（那些是生成結果，不是我們要的檢索候選）。"""
    try:
        resp = requests.post(
            f"{GRAPH_BASE_URL}/v1/answer/freetext",
            headers=GRAPH_HEADERS,
            json={"description": query_text, "question": "有哪些類似的判決先例？", "top_k": top_k},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ GraphDB freetext 呼叫失敗: {e}")
        return []

    results = []
    for p in data.get("precedents", []):
        doc_id = p.get("statute_no", "").replace(" ", "")
        if not doc_id:
            continue
        results.append({
            "doc_id": doc_id,
            "text": get_judgment_text_for_rag(doc_id),
            "graphdb_score": p.get("jaccard", 0.0),
            "strength": p.get("strength"),
            "verified": p.get("verified"),
        })
    return results


# ========== 新增：case 路徑用的 GraphDB 檢索 ==========

def _fetch_tpl_case_text(case_no):
    """用 case_no 回頭查 case_summary 補全文字"""
    conn = sqlite3.connect(config.TPL_TRAIN_DB)
    cur = conn.cursor()
    cur.execute(
        "SELECT text_content FROM embedding_sources WHERE case_no=? AND entity_type='case_summary'",
        (case_no,)
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def graphdb_search_case(query_text, top_k=5, seed_case_nos=None):
    """/v1/answer/tpl_similar：需要case_no當輸入，用多錨點橋接
    （同一輪查詢裡case路徑vector+bm25找到的候選案號當錨點）"""
    if not seed_case_nos:
        return []

    all_hits = {}
    for seed in seed_case_nos:
        try:
            resp = requests.post(
                f"{GRAPH_BASE_URL}/v1/answer/tpl_similar",
                headers=GRAPH_HEADERS,
                json={
                    "case_no": f"TPL:{seed}",
                    "question": "有沒有類似的案件可以參考？",
                    "top_k": top_k,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ GraphDB tpl_similar 呼叫失敗（seed={seed}）: {e}")
            continue

        for sc in data.get("similar_cases", []):
            case_no = sc.get("case_no")
            if not case_no:
                continue
            score = sc.get("final_score", 0.0)
            if case_no not in all_hits or score > all_hits[case_no]["graphdb_score"]:
                all_hits[case_no] = {
                    "case_no": case_no,
                    "graphdb_score": score,
                    "quality_flags": sc.get("quality_flags", []),
                }

    results = []
    for case_no, info in all_hits.items():
        text = _fetch_tpl_case_text(case_no)
        if text:
            results.append({
                "doc_id": case_no,
                "text": text,
                "graphdb_score": info["graphdb_score"],
                "quality_flags": info["quality_flags"],
            })

    results.sort(key=lambda x: -x["graphdb_score"])
    return results[:top_k]


# ---------- 分派入口：改成三個source都有對應處理 ----------

def graphdb_search(query_text, source, top_k=5, seed_case_nos=None):
    """source: "policy" / "case" / "judgment"
    - policy：/v1/retrieve/clause，直接查
    - judgment：/v1/answer/freetext，直接查
    - case：/v1/answer/tpl_similar，需要seed_case_nos橋接"""
    if source == "policy":
        return graphdb_search_policy(query_text, top_k=top_k)
    if source == "judgment":
        return graphdb_search_judgment_freetext(query_text, top_k=top_k)
    if source == "case":
        return graphdb_search_case(query_text, top_k=top_k, seed_case_nos=seed_case_nos)
    return []

print("✅ graphdb_search() 已更新：policy/judgment/case 三路都已接上真實端點")


def _vector_search_judgment_wrapper(query_text, top_k=5):
    return vector_search_judgment(query_text, top_k_docs=top_k)


def rule_agent(query_text, source, top_k=5, seed_case_nos=None):
    """純檢索，回傳三路（vector/bm25/graphdb）各自的原始排名結果，不做融合。
    case路徑：graphdb需要錨點，用同一輪vector+bm25的候選案號自動計算（不用呼叫端手動傳）
    judgment路徑：graphdb已改用freetext，不需要錨點
    policy路徑：graphdb直接查，不需要錨點"""
    vector_fn = {
        "policy": vector_search_policy,
        "case": vector_search_case,
        "judgment": _vector_search_judgment_wrapper,
    }[source]

    vector_results = vector_fn(query_text, top_k=top_k)
    bm25_results = bm25_search(query_text, source, top_k=top_k)

    if source == "case":
        case_seed_ids = list(set(r["doc_id"] for r in vector_results) | set(r["doc_id"] for r in bm25_results))
        graphdb_results = graphdb_search(query_text, source, top_k=top_k, seed_case_nos=case_seed_ids)
    else:
        graphdb_results = graphdb_search(query_text, source, top_k=top_k, seed_case_nos=seed_case_nos)

    return {"vector": vector_results, "bm25": bm25_results, "graphdb": graphdb_results}


def rrf_fusion(rule_agent_result, k=60, weights=None):
    if weights is None:
        weights = {"vector": 1.0, "bm25": 1.0, "graphdb": 1.0}

    scores = defaultdict(float)
    doc_lookup = {}

    for path_name, ranked_list in rule_agent_result.items():
        w = weights.get(path_name, 1.0)
        for rank, item in enumerate(ranked_list, start=1):
            doc_id = item["doc_id"]
            scores[doc_id] += w / (k + rank)
            doc_lookup[doc_id] = item

    fused = sorted(scores.items(), key=lambda x: -x[1])
    return [{"doc_id": doc_id, "rrf_score": score, **doc_lookup[doc_id]} for doc_id, score in fused]


# ---- 校準後的RRF權重（5-fold cross validation結果，2026/08）----
# policy / judgment：CV顯示調權重跟等權重比幾乎沒差（甚至有時更差），維持等權重
# case：CV顯示5折全部穩定提升，採用調校後權重；graphdb固定為0（case類的grid search本來就沒搜graphdb）
FINAL_RRF_WEIGHTS = {
    "policy":   {"vector": 1.0, "bm25": 1.0, "graphdb": 1.0},
    "case":     {"vector": 2.0, "bm25": 0.5, "graphdb": 0.0},
    "judgment": {"vector": 1.0, "bm25": 1.0, "graphdb": 1.0},
}


def retrieve(query_text, source, top_k=5, seed_case_nos=None):
    raw = rule_agent(query_text, source, top_k=top_k, seed_case_nos=seed_case_nos)
    weights = FINAL_RRF_WEIGHTS.get(source)
    return rrf_fusion(raw, weights=weights)[:top_k]
