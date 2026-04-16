cat << 'EOF' > run_5090_stable.sh
#!/bin/bash
set -e

echo "==================================================="
echo "🚀 開始自動建置 PaperReviewMultiSystem (穩定相容版)"
echo "==================================================="

echo -e "\n[1/6] 更新系統與安裝底層依賴套件..."
sudo apt update
sudo apt install -y git python3-venv python3-full nvidia-cuda-toolkit build-essential cmake

echo -e "\n[2/6] 下載專案程式碼..."
cd ~
if [ ! -d "PaperReviewMultiSystem" ]; then
    git clone https://github.com/wangsungwen/PaperReviewMultiSystem.git
fi

cd PaperReviewMultiSystem
echo "目前工作目錄: $(pwd)"

echo -e "\n[3/6] 建立並啟動 Python 虛擬環境..."
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel "huggingface_hub[cli]"

echo -e "\n[4/6] 配置 RTX 5090 專屬環境..."
# 解除 requirements.txt 的封印
echo "⏳ 正在解除 requirements.txt 的 PyTorch 封印..."
cp requirements.txt requirements.txt.bak
sed -i -E '/^(torch|torchvision|torchaudio)(==|>=|<=|$)/Id' requirements.txt

# 💡 核心修正：明確指定安裝穩定的 2.6.0 版本，避免下載到會當機的 2.11.0 開發版
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

echo -e "\n[5/6] 系統就緒確認..."
echo "👉 正在啟動伺服器..."

echo -e "\n[6/6] 🎉 系統建置大功告成！"
echo "==================================================="
streamlit run app.py
EOF

chmod +x run_5090_stable.sh
./run_5090_stable.sh
