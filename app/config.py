"""
所有設定值統一從環境變數讀取，不寫死在程式碼裡。
部署到Cloud Run時，這些值透過Cloud Run的環境變數/Secret Manager設定，
不需要任何人手動gcloud login——服務本身用附加的Service Account自動取得
GCS/Vertex AI的存取權限（ADC會自動生效，不需要application_default_credentials.json）。

本地開發時，用python-dotenv自動讀取同資料夾的.env檔案，
這樣Windows使用者不用處理跨平台不一致的環境變數載入指令（cmd跟bash語法不同）。
"""
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "cki-scu-2026")
REGION = os.environ.get("GOOGLE_CLOUD_REGION", "asia-southeast1")

BUCKET_NAME = os.environ.get("BUCKET_NAME", "rsa-cki-scu-2026")
LOCAL_DIR = os.environ.get("LOCAL_DIR", "/data/TPL_RAG_DB")

# GraphDB（萱維護的服務）
GRAPH_BASE_URL = os.environ.get(
    "GRAPH_BASE_URL", "https://graphrag-api-624489584685.asia-east1.run.app"
)
GRAPH_API_KEY = os.environ["GRAPH_API_KEY"]  # 必填，沒設定就直接啟動失敗，不要用預設值悄悄放行

# "gemini" 或 "lora"
GENERATION_BACKEND = os.environ.get("GENERATION_BACKEND", "lora")

# 對外API本身的驗證金鑰（前端/保險公司測試端呼叫這個服務時要帶的key，
# 跟GRAPH_API_KEY是完全不同的東西——那把是「我們的服務去呼叫萱的服務」，
# 這把是「外部呼叫我們的服務」，兩者不要混用同一把）
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY") or None # 若為None，代表本服務先不做外部驗證（僅限內網/測試用）

TPL_TRAIN_DB = f"{LOCAL_DIR}/Vector_DB/TPL_train.db"
TPL_VAL_DB = f"{LOCAL_DIR}/Vector_DB/TPL_val.db"
TPL_POLICY_DB = f"{LOCAL_DIR}/Vector_DB/policy_clauses.db"
TPL_JUDGMENTS_DB = f"{LOCAL_DIR}/Vector_DB/judgments.db"
CHROMA_PERSIST_DIR = f"{LOCAL_DIR}/Vector_DB/chroma_db"
FINETUNED_EMBED_MODEL_PATH = f"{LOCAL_DIR}/Vector_DB/e5_finetuned_tpl"
LORA_MODEL_PATH = f"{LOCAL_DIR}/Vector_DB/qwen_lora_tpl_V1"
BM25_LOCAL_DIR = f"{LOCAL_DIR}/BM25_DB"

BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
