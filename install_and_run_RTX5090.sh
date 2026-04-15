cat << 'EOF' > install_and_run_5090.sh
#!/bin/bash
set -e

echo "==================================================="
echo "🚀 開始自動建置 PaperReviewMultiSystem (RTX 5090 專屬版)"
echo "==================================================="

echo -e "\n[1/6] 更新系統與安裝底層依賴套件..."
sudo apt update
sudo apt install -y git python3-venv python3-full nvidia-cuda-toolkit build-essential cmake

echo -e "\n[2/6] 下載專案程式碼..."
if [ ! -d "PaperReviewMultiSystem" ]; then
    git clone https://github.com/wangsungwen/PaperReviewMultiSystem.git
fi
cd PaperReviewMultiSystem

echo -e "\n[3/6] 建立並啟動 Python 虛擬環境..."
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel "huggingface_hub[cli]"

echo -e "\n[4/6] 配置 RTX 5090 (sm_89) 專屬編譯環境並安裝套件..."
export CUDACXX=/usr/bin/nvcc
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89"
export FORCE_CMAKE="1"

echo "⏳ 正在編譯 llama-cpp-python..."
pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
pip install -r requirements.txt

echo "⏳ 正在強制安裝 CUDA 12.4 版 PyTorch (確保不被 requirements 洗掉)..."
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo -e "\n[5/6] Hugging Face 認證與模型下載..."
echo "👉 請貼上您的 Hugging Face Access Token (貼上後按 Enter):"
read -s HF_TOKEN
hf auth login --token $HF_TOKEN --add-to-git-credential False

mkdir -p local_models
echo "📥 正在下載 Gemma-3-TAIDE-12B 模型..."
hf download nctu6/Gemma-3-TAIDE-12b-Chat-GGUF Gemma-3-TAIDE-12b-Chat-Q4_K_M.gguf --local-dir ./local_models
echo "📥 正在下載 Meta-Llama-3-8B 模型..."
hf download QuantFactory/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct.Q4_K_M.gguf --local-dir ./local_models

echo -e "\n[6/6] 🎉 系統建置大功告成！正在為您啟動伺服器..."
echo "==================================================="
streamlit run app.py
EOF

chmod +x install_and_run_5090.sh
./install_and_run_5090.sh