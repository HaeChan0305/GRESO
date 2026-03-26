#!/bin/bash
# Environment setup script for GRESO
# Base image: whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3

set -e

# Fix apt mirror (base image uses Chinese mirror that may be unreachable)
sudo sed -i 's|https://mirrors.tuna.tsinghua.edu.cn|http://mirror.kakao.com|g' /etc/apt/sources.list
sudo apt-get update -qq
sudo apt-get install -y tmux poppler-utils

# Create venv with system site-packages (reuse torch, vllm, flash-attn, etc.)
python3 -m venv ~/greso-env --system-site-packages --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py | ~/greso-env/bin/python3

# Activate and install GRESO
source ~/greso-env/bin/activate
pip install -e . --no-deps
pip install wandb IPython matplotlib ipdb latex2sympy2-extended math-verify torchdata pylatexenc liger-kernel
pip install "gguf>=0.13.0" "lm-format-enforcer>=0.10.11,<0.11"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source ~/greso-env/bin/activate"
echo "Run with: export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\" && bash train-scripts/math_qwen_1_5b_dm_greso.sh"
