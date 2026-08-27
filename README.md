# TPL 理賠代理人 API

把 `TPL_ClaimAgent_github.ipynb` 的核心邏輯（區塊三～九）改寫成 FastAPI 服務，
供前端/保險公司測試端透過HTTP呼叫，不需要任何人手動`gcloud login`。

## 這份 API 跟原本 notebook 的對應關係

| notebook區塊 | 對應檔案 |
|---|---|
| 區塊一、二（認證、下載GCS資料） | `app/gcs_download.py` |
| 區塊三、四、五、六（Vector/BM25/GraphDB檢索 + RRF融合） | `app/retrieval.py` |
| 區塊七（LLM生成，Gemini/LoRA雙後端） | `app/llm.py` |
| 區塊八（肇責比例後處理） | `app/verification.py` |
| 區塊九（端對端查詢函式） | `app/agent.py` |
| 全部設定值 | `app/config.py`（統一從環境變數讀取，沒有寫死的金鑰） |
| FastAPI進入點 | `app/main.py` |

## 唯一的對外endpoint

```
POST /v1/tpl/claim
Header: X-API-Key: <SERVICE_API_KEY>
Body:
{
  "accident_area": "台北市",
  "own_fault_pct": 70,
  "injury_desc": "事故造成右腳踝骨折...",
  "backend": null   // 選填，覆寫預設backend，"gemini" 或 "lora"
}
```

健康檢查：`GET /healthz`

## 本地測試

```bash
cp .env.example .env
# 編輯 .env，填入真正的 GRAPH_API_KEY

pip install -r requirements.txt

# 本地測試需要能存取GCS bucket跟Vertex AI，用你自己的個人帳號登入即可：
gcloud auth application-default login

export $(cat .env | xargs)
uvicorn app.main:app --reload --port 8080
```

## 部署到Cloud Run（正式環境）

**重要：不要用個人帳號登入的方式部署正式服務。** 正式服務要用 Service Account：

1. 在GCP建一個Service Account，給它：
   - `Storage Object Viewer`（讀GCS bucket）
   - `Vertex AI User`（如果會用到`backend="gemini"`）
2. 部署時把這個Service Account附加給Cloud Run服務（`--service-account`參數），
   不需要在容器裡放任何金鑰檔案，Cloud Run執行環境會自動提供憑證（ADC）。
3. `GRAPH_API_KEY`跟`SERVICE_API_KEY`這兩個敏感值，用 **Secret Manager** 掛進Cloud Run
   （`--set-secrets`），不要用一般的`--set-env-vars`明文帶進去。

```bash
gcloud run deploy tpl-claim-agent \
  --source . \
  --region asia-southeast1 \
  --service-account tpl-claim-agent@cki-scu-2026.iam.gserviceaccount.com \
  --set-secrets GRAPH_API_KEY=graph-api-key:latest,SERVICE_API_KEY=service-api-key:latest \
  --set-env-vars GOOGLE_CLOUD_PROJECT=cki-scu-2026,GOOGLE_CLOUD_REGION=asia-southeast1,BUCKET_NAME=rsa-cki-scu-2026,GENERATION_BACKEND=lora \
  --memory 16Gi \
  --gpu 1 \
  --gpu-type nvidia-l4
```

⚠️ **`GENERATION_BACKEND=lora`需要GPU**（4-bit量化的Qwen2.5-7B仍然需要GPU推論），
Cloud Run的GPU支援目前只有`asia-southeast1`這個region有（跟你們LoRA微調notebook用的
region一致，不是巧合）。如果只需要`backend="gemini"`，可以拿掉`--gpu`相關參數，
用一般的CPU Cloud Run就夠，成本會低很多——如果只是先讓組員/保險公司測試基本流程，
建議先用gemini backend部署，之後LoRA真的要正式上線再切換。

## 目前還沒做、但正式上線前建議補上的部分

- **速率限制/併發控制**：`/v1/tpl/claim`目前沒有限流，LoRA backend單次推論可能要處理較長時間，
  多人同時打會排隊或互搶GPU記憶體，建議之後視情況加上請求佇列或限流。
- **請求記錄/監控**：目前沒有log每次請求的內容跟花費時間，保險公司測試階段建議先加上，
  方便之後追蹤問題案例。
- **BM25/GraphDB連線失敗的重試機制**：目前沿用notebook原本的邏輯（單次timeout後直接跳過該路徑），
  正式服務可以考慮加重試，但要注意不要讓單一慢速路徑拖垮整體回應時間。
