"""
服務啟動時，把 gs://{BUCKET_NAME}/TPL_RAG_DB 整包下載到本地容器磁碟。

用 google-cloud-storage 這個Python SDK，不是shell出去呼叫`gcloud storage cp`——
部署容器不一定會裝gcloud CLI，但一定會裝這個套件（requirements.txt裡列了），
而且在Cloud Run上，這個SDK會自動用附加的Service Account憑證（ADC），
不需要任何人手動`gcloud auth login`。
"""
import os

from google.cloud import storage

from . import config


def download_tpl_rag_db():
    if os.path.exists(config.LOCAL_DIR) and os.listdir(config.LOCAL_DIR):
        print(f"✅ {config.LOCAL_DIR} 已存在且非空，略過下載（重複部署/重啟時常見）")
        return

    print(f"正在從 gs://{config.BUCKET_NAME}/TPL_RAG_DB 下載至 {config.LOCAL_DIR} ...")
    client = storage.Client(project=config.PROJECT_ID)
    bucket = client.bucket(config.BUCKET_NAME)
    prefix = "TPL_RAG_DB/"

    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        raise RuntimeError(
            f"gs://{config.BUCKET_NAME}/{prefix} 底下沒有任何檔案，"
            "請確認bucket路徑正確、且這個服務的Service Account有Storage Object Viewer權限"
        )

    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        relative_path = blob.name[len(prefix):]
        local_path = os.path.join(config.LOCAL_DIR, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)

    print(f"✅ 下載完成，共 {len(blobs)} 個物件")
