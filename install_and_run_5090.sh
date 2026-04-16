cat << 'EOF' > run_5090_final.sh
#!/bin/bash
set -e

echo "==================================================="
echo "🚀 開始自動建置 PaperReviewMultiSystem (終極完整版)"
echo "==================================================="

echo -e "\n[1/6] 更新系統與安裝底層依賴套件..."
sudo apt update
sudo apt install -y git python3-venv python3-full nvidia-cuda-toolkit build-essential cmake

echo -e "\n[2/6] 下載專案程式碼..."
cd ~
if [ ! -d "PaperReviewMultiSystem" ]; then
    git clone https://github.com/wangsungwen/PaperReviewMultiSystem.git
fi

# 進入專案資料夾
cd PaperReviewMultiSystem
echo "目前工作目錄: $(pwd)"

echo -e "\n[3/6] 建立並啟動 Python 虛擬環境..."
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel "huggingface_hub[cli]"

echo -e "\n[4/6] 配置 RTX 5090 專屬環境..."
# 解除 requirements.txt 的 PyTorch 封印
echo "⏳ 正在解除 requirements.txt 的 PyTorch 封印..."
cp requirements.txt requirements.txt.bak
sed -i -E '/^(torch|torchvision|torchaudio)(==|>=|<=|$)/Id' requirements.txt

# 安裝穩定相容的 PyTorch 2.6.0 (CUDA 12.4)
echo "⏳ 正在安裝與系統相容的 PyTorch 2.6.0 (CUDA 12.4)..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 編譯 llama-cpp-python
export CUDACXX=/usr/bin/nvcc
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89"
export FORCE_CMAKE="1"

echo "⏳ 正在為 RTX 5090 編譯 LLM 推論引擎..."
pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

echo "⏳ 正在安裝其餘專案套件..."
pip install -r requirements.txt

echo -e "\n[5/6] Hugging Face 認證與模型下載..."
echo "👉 請貼上您的 Hugging Face Access Token (輸入時畫面不會顯示，貼上後請按 Enter):"
read -s HF_TOKEN
hf auth login --token $HF_TOKEN --add-to-git-credential False

mkdir -p local_models
echo "📥 正在下載 Meta-Llama-3-8B 模型..."
hf download QuantFactory/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct.Q4_K_M.gguf --local-dir ./local_models
echo "📥 正在下載 Gemma-3-TAIDE-12B 模型..."
hf download nctu6/Gemma-3-TAIDE-12b-Chat-GGUF Gemma-3-TAIDE-12b-Chat-Q4_K_M.gguf --local-dir ./local_models

echo -e "\n[6/6] 🎉 系統建置與模型下載大功告成！正在為您啟動伺服器..."
echo "==================================================="
streamlit run app.py
EOF

# 賦予權限並執行
chmod +x run_5090_final.sh
./run_5090_final.sh
