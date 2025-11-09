PROJECT_ID=test-vertex-fau
REGION=us-central1
BUCKET="gs://${PROJECT_ID}-llm"
IMAGE_TRAIN="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/train-lora:latest"
IMAGE_SERVE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/serve-flask:latest"

# 1) Build & push image (Cloud Build o local con gcloud builds submit)
gcloud builds submit --tag $IMAGE_URI .

# 2) Sube datos y crea carpetas de artifacts
gsutil cp training/data/data.jsonl $BUCKET/data/data.jsonl

# 3) CustomJob
gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name="lora-dental" \
  --worker-pool-spec=machine-type=a2-highgpu-1g,replica-count=1,container-image-uri=$IMAGE_URI, \
env=BASE_MODEL=mistralai/Mistral-7B-Instruct-v0.2,DATA_PATH=$BUCKET/data/data.jsonl,OUT_DIR=$BUCKET/artifacts/adapters,LORA_R=32,LORA_ALPHA=32,EPOCHS=3,LR=1e-4 \
  --service-account=vertex-sa@$PROJECT_ID.iam.gserviceaccount.com
