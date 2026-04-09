# Ollama 安裝與設定教學 (針對 RTX 5090)

既然 `llama-cpp-python` 的編譯太過繁瑣，使用 **Ollama** 是目前的最佳解。它在 Windows 上非常穩定，且能自動優化 50 系列顯效。

### 1. 下載與安裝
*   請前往 [Ollama 官網 (ollama.com)](https://ollama.com/download/windows) 下載 Windows 安裝程式。
*   安裝後，您的右下角工作列會出現一個「羊駝」圖示。

### 2. 下載模型
開啟 PowerShell，執行以下指令下載預設的模型 (Llama 3.1 8B)：
```powershell
ollama pull llama3.1
```
*(下載速度取決於網路，模型大小約 4.7GB)*

### 3. 如何在專案中使用
1.  確保右下角 Ollama 羊駝圖示顯示為 **Running**。
2.  啟動本專案的 Streamlit 介面。
3.  在側邊欄 **「🤖 LLM 快速切換」** 中，選擇 **「🐑 Ollama API (推薦)」**。
4.  點選 **「🔍 檢測推論硬體狀態」**，如果顯示 `🐑 Ollama (Running)`，即表示設定成功！

---

### 💡 進階技巧
如果您想嘗試更強大的模型（例如 `llama3.1:70b` ），因為您的 RTX 5090 有 32GB VRAM，您可以嘗試下載更大的型號：
```powershell
ollama pull llama3.1:70b
```
然後在專案的 [config.json](file:///c:/Users/wangs/paper_review_system_single/config.json) 中將 `"model_name": "llama3.1"` 改為 `"llama3.1:70b"` 即可。
