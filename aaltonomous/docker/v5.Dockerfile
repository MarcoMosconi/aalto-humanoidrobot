# v5 (classical + Dex3 finger check + V-JEPA 2 grasp verifier, aaltonomous/baselines/jepa/v5_server.py) as a policy
# server. GPU for the verifier (~30 ms per crop); it also runs on CPU, slower, only on picks the fingers can't decide.
#
#   mkdir -p jepa_weights && cp runs/jepa/held_verifier_all_fa0.10.pt runs/jepa/variant_prior_dlv.json jepa_weights/
#   docker build -f aaltonomous/docker/v5.Dockerfile -t myteam/v5:v1 .          # from the repo root
#   python -m aaltonomous.sim.evaluate --container myteam/v5:v1 --team v5 --trials 8
#
# The V-JEPA 2 weights are downloaded at build time (HF_HOME) and the server runs offline. Calibration and the
# classical config.yaml are baked in: rebuild after changing them.
FROM nvidia/cuda:12.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HOME=/opt/hf
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install --no-cache-dir numpy==2.2.6 scipy==1.15.2 opencv-python-headless==4.10.0.84 pyyaml msgpack \
        "websockets>=12" pin==3.3.1 "transformers>=4.53" safetensors huggingface_hub
ARG VJEPA2_ID=facebook/vjepa2-vitl-fpc64-256
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${VJEPA2_ID}')"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

WORKDIR /app
COPY third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf third_party/xr_teleoperate/assets/g1/
COPY aaltonomous/__init__.py aaltonomous/
COPY aaltonomous/deploy/__init__.py aaltonomous/deploy/protocol.py aaltonomous/deploy/policy_server.py aaltonomous/deploy/
COPY aaltonomous/interface/__init__.py aaltonomous/interface/safety.py aaltonomous/interface/calibration.py \
     aaltonomous/interface/tasks.py aaltonomous/interface/joint_spec.yaml aaltonomous/interface/calibration.yaml \
     aaltonomous/interface/
COPY aaltonomous/scene/__init__.py aaltonomous/scene/scene.py aaltonomous/scene/scene.yaml aaltonomous/scene/
COPY aaltonomous/baselines/__init__.py aaltonomous/baselines/
COPY aaltonomous/baselines/classical/*.py aaltonomous/baselines/classical/config.yaml aaltonomous/baselines/classical/
COPY aaltonomous/baselines/jepa/*.py aaltonomous/baselines/jepa/config.yaml aaltonomous/baselines/jepa/
COPY jepa_weights/held_verifier_all_fa0.10.pt jepa_weights/variant_prior_dlv.json /app/weights/
ENV V5_VERIFIER=/app/weights/held_verifier_all_fa0.10.pt V5_FAMILY_PRIOR=/app/weights/variant_prior_dlv.json

EXPOSE 8000
CMD ["python3", "-m", "aaltonomous.baselines.jepa.v5_server"]
