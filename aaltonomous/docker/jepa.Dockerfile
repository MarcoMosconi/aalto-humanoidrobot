# JEPA world-model policy (aaltonomous/baselines/jepa) as a policy server. GPU (V-JEPA 2 ViT-L encoder + CEM).
#
#   mkdir -p jepa_weights && cp runs/jepa/ckpt/best.pt jepa_weights/ && cp -r runs/jepa/latents jepa_weights/bank
#   docker build -f aaltonomous/docker/jepa.Dockerfile -t myteam/jepa:v1 .            # from the repo root
#   docker run --rm --gpus all -p 8000:8000 myteam/jepa:v1
#   python -m aaltonomous.deploy.replay --synthetic 100 --container myteam/jepa:v1
#
# The V-JEPA 2 weights are downloaded at build time into the image (HF_HOME) and the server runs offline.
FROM nvidia/cuda:12.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HOME=/opt/hf
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install --no-cache-dir numpy==2.2.6 pyyaml msgpack "websockets>=12" pin==3.3.1 "transformers>=4.53" \
        safetensors huggingface_hub
ARG VJEPA2_ID=facebook/vjepa2-vitl-fpc64-256
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${VJEPA2_ID}')"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

WORKDIR /app
COPY third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf third_party/xr_teleoperate/assets/g1/
COPY aaltonomous/__init__.py aaltonomous/
COPY aaltonomous/deploy/__init__.py aaltonomous/deploy/protocol.py aaltonomous/deploy/policy_server.py aaltonomous/deploy/
COPY aaltonomous/interface/__init__.py aaltonomous/interface/safety.py aaltonomous/interface/tasks.py \
     aaltonomous/interface/joint_spec.yaml aaltonomous/interface/
COPY aaltonomous/baselines/__init__.py aaltonomous/baselines/
# hand postures + home pose come from the classical config (read only)
COPY aaltonomous/baselines/classical/__init__.py aaltonomous/baselines/classical/config.py \
     aaltonomous/baselines/classical/config.yaml aaltonomous/baselines/classical/
COPY aaltonomous/baselines/jepa/*.py aaltonomous/baselines/jepa/config.yaml aaltonomous/baselines/jepa/
COPY jepa_weights/best.pt /app/weights/best.pt
COPY jepa_weights/bank /app/weights/bank
ENV JEPA_CKPT=/app/weights/best.pt JEPA_BANK=/app/weights/bank

EXPOSE 8000
CMD ["python3", "-m", "aaltonomous.baselines.jepa.policy_server"]
