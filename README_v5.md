# 多代理人論文審查系統 - 使用說明 (v5.0)

本系統是一個專業級的多代理人 AI 論文審查平台，已全面升級為 **網頁伺服器架構**。特別針對高階硬體 (如 NVIDIA RTX 5090 Blackwell) 進行深度優化，支援大規模併發連線、會話隔離隱私保護與全域模型快取。

---

## 🌟 核心功能說明 (v5.0 伺服器版更新)

### 1. 🌐 多人伺服器架構 (Server-Ready)

- **記憶體隔離防護**：使用者的 API Key 與模型偏好完全儲存在瀏覽器會話 (`st.session_state`) 中。**隨改隨套用，且絕不會被寫入伺服器硬碟**，確保多人同時使用時金鑰不外洩。
- **全網域單例快取 (Global Singleton Cache)**：本地大型模型 (Llama GGUF, Ollama, HF) 採用「全域載入一次」機制，防止多人連線導致顯存/記憶體溢出 (OOM) 崩潰。

### 2. 🔍 進化版 AI 寫作偵測

- **Hugging Face 精準推論**：整合 `desklib/ai-text-detector-v1.01`。
- **Blackwell 硬體相容性**：內建 UI 切換開關，一鍵開啟 `force_cpu` 模式解決 50 系顯卡 CUDA 錯誤。

### 3. 🤖 本地 LLM 多引擎驅動

- **🐑 Ollama API (強烈推薦)**：提供極佳的連線穩定性。**[新功能]** 內建 Ollama 服務自動偵測機制；若尚未安裝，只需在設定介面一鍵點擊，系統即可在背景自動下載並安裝 Ollama，體驗開箱即用。
- **💻 Llama-cpp (GGUF)**：支援伺服器端管理員統一配置 GGUF 模型路徑與上下文窗口。

### 4. 📄 跨格式讀取與持久化報告

- 支援 `.pdf`、`.docx` 及 `.txt`。
- 支援匯出為 **JSON** 或 **Markdown** 與熱力圖分析。

---

## ⚙️ 系統設定與部署

### 1. 管理員靜態配置 (config.json)

管理員可透過專案目錄下的 `config.json` 預設伺服器端的本地模型路徑。使用者無權修改物理檔案路徑，確保伺服器穩定。

### 2. 跨平台快速啟動與部署

#### 【選項 A】Windows 環境部署 (原生)

```powershell
# 1. 建立並啟動環境
python -m venv .venv
.\.venv\Scripts\activate

# 2. 安裝 CUDA 版 Torch 與其他套件
pip install torch --extra-index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 3. 執行伺服器
streamlit run app.py
```

#### 【選項 B】Linux (Ubuntu) 原生一鍵部署 **[新]**

只需透過內建的腳本即可自動建立環境並啟動：

```bash
# 賦予執行權限
chmod +x setup_linux.sh run_linux.sh

# 執行自動建置 (將安裝系統級 Python 工具與 .venv 並下載 requirements)
./setup_linux.sh

# 啟動伺服器
./run_linux.sh
```

#### 【選項 C】Docker 容器化無痛部署 (推薦) **[新]**

如果您希望完全隔絕環境污染，可以直接使用 Docker 化架構一鍵升空：

```bash
# 背景部署容器 (包含自動透通掛載 config 與 local_models)
docker compose up -d

# 若要在 Docker 內啟用 NVIDIA GPU 加速，請先取消 docker-compose.yml 內 deploy 區塊的註解。
```

### 3. 打包與發佈 (EXE 版)

如果您希望將此系統作為無須安裝環境的可攜式軟體分發：

```powershell
python build_exe.py
```

執行完後，結果將輸出於 `dist/PaperReviewSystem/` 內。
**注意**：為避免執行檔體積過大，`local_models` 模型資料夾與裡面的 `.gguf` 並未被打包。請務必在發布前，手動將 `local_models` 資料夾複製並放置於 `PaperReviewSystem.exe` 同一層級！

---

## ⚠️ 常見問題

- **隱私安全性**：畫面上輸入的所有 API 金鑰僅存在於您的瀏覽器視窗中，登出或重新整理即清除，後台管理員無法窺視。
- **CUDA Error**：若遇到 `no kernel image`，請至「⚙️ 參數設定」勾選「強制使用 CPU 進行 AI 偵測」。

---
*本系統由 AI 協作開發，旨在提升學術論文的審查效率與品質。*
