#!/bin/bash
# Idempotent docker container launcher for vllm-r3-repro-rebuilt.
# Adapted from verl-grounding/master_setup.sh (single-node, no Ray cluster).
#
# Usage: bash scripts/setup_container.sh
# Override defaults via env vars, e.g.:
#   IMAGE_NAME=verl:h200-snap0-r3fix CONTAINER_NAME=my-test bash scripts/setup_container.sh

# ===== 可配置项（env override） =====
CONTAINER_NAME="${CONTAINER_NAME:-vllm-r3-repro-rebuilt}"
IMAGE_NAME="${IMAGE_NAME:-verl:h200-snap4-h200-2-ready}"
SHM_SIZE="${SHM_SIZE:-32g}"

# Mounts (host:container)。/home/aoshen 用 NFS shared，所有 h200-* 节点共享内容。
HF_CACHE_HOST="${HF_CACHE_HOST:-/mnt/shared/hf-models}"
DATA_HOST="${DATA_HOST:-/home/aoshen/data}"
VLLM_REPRO_HOST="${VLLM_REPRO_HOST:-/home/aoshen/vllm-watchtower/vllm-repro}"
WORKSPACE_HOST="${WORKSPACE_HOST:-/home/aoshen/vllm-watchtower}"
VERL_HOST="${VERL_HOST:-/home/aoshen/verl}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}  vllm-r3-repro container setup${NC}"
echo -e "${GREEN}=====================================${NC}"
echo "Container: ${CONTAINER_NAME}"
echo "Image:     ${IMAGE_NAME}"
echo "Mounts:"
echo "  ${HF_CACHE_HOST}        → /mnt/shared/hf-models (HF model cache)"
echo "  ${DATA_HOST}            → /root/data            (gsm8k etc.)"
echo "  ${VLLM_REPRO_HOST}      → /vllm-src             (vllm-r3rfc source build)"
echo "  ${WORKSPACE_HOST}       → /workspace            (legacy alias)"
echo "  ${VERL_HOST}            → /home/aoshen/verl     (host-style path; verl editable install root)"
echo ""

# Ensure image exists
ensure_image_exists() {
    if docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
        echo -e "${GREEN}  ✓ Image '${IMAGE_NAME}' found locally${NC}"
        return 0
    fi
    echo -e "${RED}  ✗ Image '${IMAGE_NAME}' not found locally.${NC}"
    echo -e "${YELLOW}  Try: docker load -i /home/aoshen/h200-snap4.tar${NC}"
    echo -e "${YELLOW}  (NFS-shared 14GB tarball, restorable on any h200 node)${NC}"
    return 1
}

# Run container with full mount + ulimit setup
create_main_container() {
    docker run -d --name "${CONTAINER_NAME}" \
        --gpus all \
        --network host \
        --ipc=host \
        --shm-size="${SHM_SIZE}" \
        --ulimit memlock=-1 \
        -e HF_HOME=/mnt/shared/hf-models \
        -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
        -v "${HF_CACHE_HOST}:/mnt/shared/hf-models" \
        -v "${DATA_HOST}:/root/data" \
        -v "${VLLM_REPRO_HOST}:/vllm-src" \
        -v "${WORKSPACE_HOST}:/workspace" \
        -v "${VERL_HOST}:/home/aoshen/verl" \
        "${IMAGE_NAME}" \
        sleep infinity
}

# nvidia-smi healthcheck (NVML can be flaky on container start)
check_nvidia_smi() {
    local max_retries=3
    local retry=0
    while [ $retry -lt $max_retries ]; do
        local out
        out=$(docker exec "${CONTAINER_NAME}" nvidia-smi 2>&1)
        if echo "$out" | grep -qiE "NVML|Unknown Error|failed"; then
            echo -e "${YELLOW}  Warning: nvidia-smi issue detected, restarting container... retry $((retry+1))/${max_retries}${NC}"
            docker restart "${CONTAINER_NAME}" >/dev/null
            sleep 5
            retry=$((retry + 1))
        else
            echo -e "${GREEN}  ✓ nvidia-smi check passed${NC}"
            echo "$out" | grep -E "^GPU [0-9]" | head -3 | sed 's/^/    /'
            return 0
        fi
    done
    echo -e "${RED}  ✗ nvidia-smi still failing after ${max_retries} retries${NC}"
    return 1
}

# Sanity-check vllm + verl import inside container
check_imports() {
    docker exec "${CONTAINER_NAME}" bash -lc '
        python -c "import vllm; print(\"  vllm:\", vllm.__version__)" 2>&1 | tail -1
        python -c "import verl; print(\"  verl:\", verl.__file__)" 2>&1 | tail -1
        pip show verl 2>/dev/null | grep -E "^Editable" | sed "s/^/  /"
    ' 2>&1
}

# === Main ===
echo -e "${YELLOW}[1/3] Ensuring image is available...${NC}"
if ! ensure_image_exists; then
    echo -e "${RED}Cannot proceed without docker image.${NC}"
    exit 1
fi
echo ""

echo -e "${YELLOW}[2/3] Container lifecycle...${NC}"
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "  Container already running"
    else
        echo "  Container exists but stopped, starting..."
        docker start "${CONTAINER_NAME}" >/dev/null
    fi
else
    echo "  Container does not exist, creating..."
    create_main_container >/dev/null
fi
echo -e "${GREEN}  ✓ Container is up${NC}"
echo ""

echo -e "${YELLOW}[3/3] Health checks...${NC}"
check_nvidia_smi || exit 1
check_imports
echo ""

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}  Container ready${NC}"
echo -e "${GREEN}=====================================${NC}"
echo "Enter container: docker exec -it ${CONTAINER_NAME} bash"
echo "Run R3-on:       docker exec -it ${CONTAINER_NAME} bash -c 'cd /home/aoshen/verl/examples/router_replay && bash run_r3_example_qwen30_a3b.sh'"
echo "Run R3-off:      docker exec -it ${CONTAINER_NAME} bash -c 'cd /home/aoshen/verl/examples/router_replay && bash run_r3_example_qwen30_a3b_R3OFF.sh'"
