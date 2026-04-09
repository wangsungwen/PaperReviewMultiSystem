 # 打包執行檔步驟 (Packaging Guide)

本專案使用 **PyInstaller** 將 Streamlit 應用程式打包為獨立的 Windows 執行檔 (.exe)。為確保本地模型 (Local LLM) 正常運作，請務必遵循以下環境規範。

## 1. 環境準備 (Environment Setup)

必須使用 **Python 3.11** 環境 (專案內附的 `.venv` 即為此版本)。

### 安裝必要依賴
在您的虛擬環境中執行以下命令：

```powershell
pip install streamlit pandas requests numpy pyinstaller diskcache
```

### 本地模型依賴 (llama-cpp-python)
由於 `llama-cpp-python` 需要編譯器，建議直接使用原本開發環境中的版本，或從專案 `.venv` 中拷貝。

## 2. 打包腳本 (Build Scripts)

確保專案根目錄包含以下檔案：
- `run_app.py`: 程式啟動入口點。
- `build_exe.py`: 自動化打包腳本 (已設置絕對路徑與元數據包含)。
- `hook-streamlit.py`: Streamlit 的 PyInstaller 專用 Hook。
- `app.py`: Streamlit 介面邏輯。

## 3. 執行打包命令

在啟動虛擬環境後，執行：

```powershell
python build_exe.py
```

## 4. 打包後處理 (Post-Build)

打包完成後，系統會生成 `dist/PaperReviewSystem` 資料夾。為了讓使用者更方便操作，請手動確認以下檔案位置：

1. **設定檔**: 將根目錄的 `config.json` 複製一份到 `dist/PaperReviewSystem/` 根目錄，方便使用者直接修改 API Key。
2. **說明文件**: 將 `README.md` (或專屬說明) 放入 `dist/PaperReviewSystem/`。

## 5. 常見問題 (Troubleshooting)

- **PackageNotFoundError**: 如果啟動時提示找不到 `streamlit` 元數據，請確保 `build_exe.py` 中有包含 `--additional-hooks-dir=.` 且執行環境中有 `streamlit` 的 `.dist-info` 資料夾。
- **遺漏本地模型**: 檢查 `dist/PaperReviewSystem/_internal/local_models` 是否包含 `.gguf` 檔案。
- **權限錯誤 (WinError 5)**: 打包前請確保舊版的 `PaperReviewSystem.exe` 已完全關閉。
