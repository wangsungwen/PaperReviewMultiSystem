太神啦！能在 Windows 環境下把充滿 C++ 與編碼地雷的底層套件編譯成功，代表您已經完全掌握了這套系統的硬體調校精髓了！🎉

這是一份專為 Windows + RTX 5090 量身打造的完整部署手冊（SOP）。這份指南已經整合了我們剛剛踩過的所有坑，包含「MSVC 繁體中文 Big5 編碼衝突」與「RTX 50 系列架構相容性」的終極解決方案。

請將以下內容存檔，未來在任何 Windows 電腦上部署時，只需開啟 PowerShell 照著貼上即可：

🚀 多代理人論文審查系統：Windows 部署全手冊 (SOP)
🛠️ 階段一：系統層級環境準備 (前置作業)
在 Windows 上編譯 C++ 專案，必須先準備好微軟的編譯工具與 NVIDIA 開發套件。如果您是一台全新的電腦，請確認已安裝以下軟體：

Python 3.10+ (安裝時務必勾選 Add Python to PATH)。

Visual Studio 2022 Community：

下載並啟動 Visual Studio Installer。

在工作負載（Workloads）頁面，務必勾選「使用 C++ 的桌面開發」 (Desktop development with C++)，這是編譯 llama.cpp 的核心。

NVIDIA CUDA Toolkit (建議安裝 12.4 或更新版本)。

💻 階段二：建立與進入專案環境
請開啟 PowerShell (建議以系統管理員身分執行一次，以防權限問題)，然後依序輸入：

PowerShell

# 1. 進入您的專案資料夾 (請替換為您的實際路徑)

cd C:\Users\wangs\paper_review_system_multi

# 2. 建立 Python 虛擬環境

python -m venv .venv

# 3. 啟動虛擬環境 (若遇執行原則錯誤，請先執行 Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser)

.\.venv\Scripts\activate
(啟動成功後，指令列最前方會出現 (.venv) 的字樣)

🔥 階段三：RTX 5090 專屬編譯配置 (核心步驟)
這個步驟將同時解決 Windows 繁體中文語系特有的 C4819 編碼亂碼報錯，並強制系統以高相容性的 RTX 40 系列 (sm_89) 標準來編譯 RTX 5090 不被識別的指令集。

請在 (.venv) 環境下的 PowerShell，逐行複製並執行：

PowerShell

# 1. 強制 MSVC 編譯器使用 UTF-8 讀取 C++ 原始碼，解決 Big5 亂碼報錯

$env:CFLAGS="/utf-8"
$env:CXXFLAGS="/utf-8"

# 2. 開啟 CUDA 支援，並強制指定架構碼為 89 (RTX 40 系列標準，5090 可完美向下相容)

$env:CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89"
$env:FORCE_CMAKE="1"

# 3. 從原始碼重新建構並安裝 llama-cpp-python

pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

# 4. 安裝專案的其他必要套件

pip install -r requirements.txt
📥 階段四：模型檔案下載 (Hugging Face)
使用最新版的 Hugging Face CLI 工具，精準下載 GGUF 格式的本地模型。

PowerShell

# 1. 確保已安裝 HF Hub 工具

pip install -U "huggingface_hub[cli]"

# 2. 登入 Hugging Face (請貼上您的 Read 權限 Token，貼上後按 Enter)

hf auth login

# 3. 建立模型存放資料夾 (若已存在會自動略過錯誤)

New-Item -ItemType Directory -Force -Path .\local_models

# 4. 下載 Gemma-3-TAIDE-12B (交大繁中版)

hf download nctu6/Gemma-3-TAIDE-12b-Chat-GGUF Gemma-3-TAIDE-12b-Chat-Q4_K_M.gguf --local-dir .\local_models

# 5. 下載 Meta-Llama-3-8B

hf download QuantFactory/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct.Q4_K_M.gguf --local-dir .\local_models
🚀 階段五：啟動系統與驗證
模型下載完成後，即可啟動多代理人網頁伺服器：

PowerShell

# 啟動 Streamlit 服務

streamlit run app.py
📂 檔案結構檢查
部署完成後，您的專案目錄結構應該會長這樣：

Plaintext
paper_review_system_multi\
├── .venv\                  # Python 虛擬環境
├── local_models\           # 模型存放區 (必須在最外層)
│   ├── Gemma-3-TAIDE-12b-Chat-Q4_K_M.gguf
│   └── Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
├── app.py                  # 啟動主程式
├── requirements.txt        # 套件清單
└── ...
現在，無論是 Linux 還是 Windows，您的 RTX 5090 都能在這套系統上發揮 100% 的 AI 推論戰力了！

========================================================================================

這是一份針對 RTX 5090 硬體架構與 Ubuntu 系統環境量身打造的完整部署指南。本指南已整合先前解決的編碼問題、CUDA 編譯路徑修正以及最新版的 Hugging Face 下載指令。

🚀 多代理人論文審查系統：Linux 部署全手冊 (SOP)

1. 系統層級環境準備
首先，安裝編譯原始碼所需的基礎工具與 NVIDIA CUDA 開發套件。

Bash

# 更新套件清單

sudo apt update

# 安裝 Python 虛擬環境工具、編譯器與 CUDA Toolkit

sudo apt install -y python3-venv python3-full nvidia-cuda-toolkit build-essential cmake
2. 建立與進入專案環境
進入專案目錄，並建立隔離的 Python 環境以避免干擾系統套件。

Bash

# 進入專案資料夾

cd ~/PaperReviewMultiSystem

# 建立虛擬環境

python3 -m venv .venv

# 啟動虛擬環境 (啟動後提示字元前方會出現 (.venv))

source .venv/bin/activate
3. RTX 5090 專屬編譯配置
這是啟用 GPU 加速最關鍵的一步。由於 RTX 5090 架構極新，需強制指定編譯參數。

Bash

# 1. 定位 CUDA 編譯器路徑

export CUDACXX=/usr/bin/nvcc

# 2. 設定編譯參數 (強制使用 sm_89 規格以確保 RTX 50 系列向下相容)

export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89"
export FORCE_CMAKE="1"

# 3. 從原始碼重新安裝 llama-cpp-python

pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

# 4. 安裝其餘專案依賴項

pip install -r requirements.txt
4. 模型檔案下載 (Hugging Face)
使用最新版的 hf 工具下載您指定的繁體中文模型。

Bash

# 登入 Hugging Face (請貼上您的 Read 權限 Token)

hf auth login

# 建立模型目錄

mkdir -p local_models

# 下載 Gemma-3-TAIDE-12B (交大版)

hf download nctu6/Gemma-3-TAIDE-12b-Chat-GGUF Gemma-3-TAIDE-12b-Chat-Q4_K_M.gguf --local-dir ./local_models

# 下載 Meta-Llama-3-8B

hf download QuantFactory/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct.Q4_K_M.gguf --local-dir ./local_models
5. 啟動系統
確保模型已就位後，即可執行網頁介面。

Bash

# 啟動 Streamlit 服務

streamlit run app.py
🛠️ 部署流程導航 (互動式進度追蹤)
為了幫助您追蹤部署進度，下方提供了一個互動式檢查清單：

顯示圖表

📂 檔案結構檢查
部署完成後，您的目錄結構應如下所示：

Plaintext
PaperReviewMultiSystem/
├── .venv/                  # Python 虛擬環境
├── local_models/           # 模型存放區
│   ├── Gemma-3-TAIDE-12b-Chat-Q4_K_M.gguf
│   └── Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
├── app.py                  # 啟動主程式
├── requirements.txt        # 套件清單
└── ...
如果您在執行 streamlit run app.py 後遇到任何硬體調用問題（例如顯存溢出或 CUDA 版本不匹配），請隨時告訴我。
