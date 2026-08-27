"""
TPL理賠代理人 API 服務。

啟動流程（lifespan）：
  1. 從GCS下載 TPL_RAG_DB 整包資料到本地磁碟
  2. 初始化 retrieval 模組（embedding模型、ChromaDB、BM25語料庫）
  3. （若GENERATION_BACKEND=lora）LoRA模型延遲載入，第一次真正呼叫時才載入，
     避免容器啟動階段就佔用大量記憶體/GPU，也讓健康檢查能盡快回應。

對外只有一個endpoint：POST /v1/tpl/claim
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import config
from . import gcs_download
from . import retrieval
from . import agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    gcs_download.download_tpl_rag_db()
    retrieval.init_retrieval()
    print(f"✅ TPL理賠代理人服務啟動完成，GENERATION_BACKEND={config.GENERATION_BACKEND}")
    yield


app = FastAPI(title="TPL 理賠代理人 API", lifespan=lifespan)


class ClaimRequest(BaseModel):
    accident_area: str = Field(..., description="事故地區，例如「台北市」")
    own_fault_pct: float = Field(..., ge=0, le=100, description="本車肇責比例（0~100）")
    injury_desc: str = Field(..., description="傷勢描述")
    backend: Optional[str] = Field(
        None, description="覆寫預設的生成後端，'gemini' 或 'lora'，不填則用服務預設值"
    )


def _check_api_key(x_api_key: Optional[str]):
    """對外驗證：如果沒設定SERVICE_API_KEY，代表這個服務先不做外部驗證
    （僅限內網/測試階段），正式對外開放前務必設定這個環境變數。"""
    if config.SERVICE_API_KEY is None:
        return
    if x_api_key != config.SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="無效的 API Key")


@app.get("/healthz")
def healthz():
    """健康檢查，Cloud Run/前端可以用這個確認服務是否就緒"""
    return {"status": "ok", "backend": config.GENERATION_BACKEND}


@app.post("/v1/tpl/claim")
def create_claim_suggestion(req: ClaimRequest, x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)

    result = agent.tpl_claim_agent(
        accident_area=req.accident_area,
        own_fault_pct=req.own_fault_pct,
        injury_desc=req.injury_desc,
        backend=req.backend,
    )

    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])

    return result
