# GPU 加速啟用指南 (NVIDIA RTX 5090 Blackwell) - 最後步驟

您目前的 **Python 3.11** 環境已經正確裝好 Torch 了！現在只差最後一個編譯成功的 `llama-cpp-python`。

請按照以下順序執行，確保編譯器不會因為編碼問題報錯：

### 1. 設定環境變數 (解決 error C2001)
請在 PowerShell 中執行這行指令（這會告訴編譯器使用 UTF-8）：
```powershell
$env:CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=100 -DCMAKE_CXX_FLAGS='/utf-8' -DCMAKE_C_FLAGS='/utf-8'"
```

### 2. 手動編譯安裝 llama-cpp-python
執行這行指令進行編譯（這步可能需要 5-10 分鐘，請耐心等候）：
```powershell
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

### 3. 完成其餘套件安裝
編譯成功後，最後一次跑完其餘依賴：
```powershell
pip install -r requirements.txt
```

---

### ✅ 驗證
全部裝完後，在 Streamlit 側邊欄點選 **「🔍 檢測推論硬體狀態」**。
*   **LLM 推論：** 應顯示 `💻 Local GPU (CUDA)`
*   **AI 偵測：** 應顯示 `💻 Local GPU (CUDA)`

> [!TIP]
> 如果 `pip install llama-cpp-python` 再次失敗，請檢查輸出的錯誤訊息。若看到 "Visual Studio" 相關錯誤，請確認您已安裝 **Visual Studio 2022 Build Tools** 並勾選了「使用 C++ 的桌面開發」。
