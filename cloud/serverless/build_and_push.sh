#!/usr/bin/env bash
# Build + push the serverless worker image. RUN FROM REPO ROOT (build context = repo).
#   ./cloud/serverless/build_and_push.sh <registry>/thaiocr-serverless:v2
set -euo pipefail

IMAGE="${1:?usage: build_and_push.sh <registry>/<name>:<tag>}"

docker build -f cloud/serverless/Dockerfile -t "$IMAGE" .
docker push "$IMAGE"
echo "pushed: $IMAGE"
