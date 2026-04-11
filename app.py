# app.py

import streamlit as st
import asyncio
import json
import os
import sys
# 新增解析套件
import pypdf
import docx
from io import BytesIO

# ----------------- 處理 Tkinter (雲端環境不支援) -----------------
try:
    import tkinter as tk
    from tkinter import filedialog
    TK_AVAILABLE = True
except (ImportError, Exception):
    TK_AVAILABLE = False
# -----------------------------------------------------------

# 匯入自定義模組
from models.paper import Paper
from models.reviewer import ReviewerAgent
from llm.interface import LLMInterface
from core.orchestrator import PaperReviewOrchestrator
from core.ai_detector import AIDetector

# ----------------- 工具函式 -----------------

def select_file(current_path=""):
    """ 開啟檔案選擇視窗並返回路徑 """
    if not TK_AVAILABLE:
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        initial_dir = os.path.dirname(current_path) if current_path and os.path.exists(os.path.dirname(current_path)) else os.getcwd()
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="選擇 GGUF 模型檔案",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
        )
        root.destroy()
        return file_path
    except Exception:
        return None

def resource_path(relative_path):
    """ 取得相對於執行路徑的絕對路徑 (支援 PyInstaller 打包環境) """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# 配置文件路徑處理 (確保打包後設定可持久化)
config_name = "config.json"
if getattr(sys, 'frozen', False):
    # 如果是打包後的執行檔，將設定檔放在 exe 旁邊，而不是 _internal 資料夾內
    base_dir = os.path.dirname(sys.executable)
    config_path = os.path.join(base_dir, config_name)
else:
    # 開發模式下使用當前目錄
    config_path = os.path.abspath(config_name)

def extract_text_from_file(uploaded_file):
    """ 根據檔案副檔名提取文字內容 [新功能] """
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # 確保檔案指標在起始位置
    uploaded_file.seek(0)
    
    try:
        if file_extension == "txt":
            return uploaded_file.getvalue().decode("utf-8")
        
        elif file_extension == "pdf":
            pdf_reader = pypdf.PdfReader(BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
            return text
        
        elif file_extension == "docx":
            try:
                doc = docx.Document(BytesIO(uploaded_file.read()))
                return "\n".join([para.text for para in doc.paragraphs])
            except Exception as e:
                # 專門處理 docx 結構錯誤
                if "no relationship of type" in str(e):
                    raise ValueError("該 Word 檔案結構不完整。請嘗試在 Word 中「另存新檔」為標準 .docx 格式後再次上傳。")
                raise e
        
    except Exception as e:
        raise Exception(f"解析 {file_extension.upper()} 失敗：{str(e)}")
    
    return ""

# ----------------- 設定檔管理 -----------------

st.set_page_config(page_title="多代理人論文審查系統", page_icon="🎓", layout="wide")

config_name = "config.json"
if os.path.exists(config_name):
    config_path = config_name
else:
    config_path = resource_path(config_name)

if not os.path.exists(config_path):
    default_config = {
        "llm_mode": "mock",
        "cloud": {"api_key": "", "model_name": "gpt-4o", "api_url": "https://api.openai.com/v1/chat/completions"},
        "local": {"model_path": "./local_models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", "n_ctx": 4096, "max_tokens": 1024},
        "ai_detector": {"api_key": "", "api_url": "https://api.gptzero.me/v2/predict/text"}
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=4)

def load_config():
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_config(config):
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

app_config = load_config()

# 初始化 Session State
if "review_history" not in st.session_state:
    st.session_state.review_history = None

if "ai_report" not in st.session_state:
    st.session_state.ai_report = None

if "reviewers" not in st.session_state:
    st.session_state.reviewers = [
        ReviewerAgent("Dr. Alan", "電腦視覺與深度學習", "輕量化神經網路", "嚴格，要求完整數據驗證。"),
        ReviewerAgent("Prof. Lin", "嵌入式與邊緣運算", "微控制器整合", "務實，重視硬體資源消耗。")
    ]

if "config" not in st.session_state:
    st.session_state.config = load_config()

# ----------------- 側邊欄 -----------------

with st.sidebar:
    st.header("🎮 功能選單")
    app_mode = st.radio("目前工作區", ["論文審查與分析", "⚙️ 參數設定"])
    
    st.divider()
    st.subheader("🤖 LLM 快速切換")
    llm_modes = {
        "local": "💻 本地落地模型 (llama-cpp)",
        "ollama": "🐑 Ollama API (推薦)",
        "cloud": "☁️ 雲端大型模型",
        "mock": "🛠️ 模擬測試模式"
    }
    
    current_mode = st.session_state.config.get("llm_mode", "mock")
    # 確保當前模式在選項中，不在則預設為 mock
    if current_mode not in llm_modes:
        current_mode = "mock"
        
    selected_mode = st.selectbox(
        "切換 LLM 推論模式",
        options=list(llm_modes.keys()),
        index=list(llm_modes.keys()).index(current_mode),
        format_func=lambda x: llm_modes[x]
    )

    if selected_mode != current_mode:
        st.session_state.config["llm_mode"] = selected_mode
        save_config(st.session_state.config)
        st.success(f"已切換至 {llm_modes[selected_mode]}")
        st.rerun()

    if st.session_state.config.get("llm_mode", "mock") == "ollama":
        import requests
        try:
            requests.get("http://localhost:11434/api/tags", timeout=1)
        except requests.exceptions.RequestException:
            st.error("⚠️ 偵測不到 Ollama 服務正在執行。")
            st.markdown("請確保您已安裝並啟動 Ollama。若尚未安裝：")
            st.markdown("1. 前往 [Ollama 官網](https://ollama.com/) 下載。")
            st.markdown("2. 或點擊下方按鈕自動下載並啟動安裝程式。")
            if st.button("🚀 自動下載並啟動 Ollama 安裝"):
                with st.spinner("正在下載 Ollama 安裝檔，檔案較大請耐心稍候... (下載完成後將自動啟動安裝)"):
                    try:
                        import urllib.request
                        import subprocess
                        import sys
                        
                        setup_path = os.path.abspath("OllamaSetup.exe")
                        url = "https://ollama.com/download/OllamaSetup.exe"
                        
                        print(f"\n[Ollama 輔助程式] 開始從 {url} 下載...", flush=True)
                        
                        def report_progress(block_num, block_size, total_size):
                            if total_size > 0:
                                percent = min(100, block_num * block_size * 100 / total_size)
                                sys.stdout.write(f"\r[Ollama 輔助程式] 下載進度：{percent:.1f}%")
                                sys.stdout.flush()
                        
                        urllib.request.urlretrieve(url, setup_path, reporthook=report_progress)
                        print(f"\n[Ollama 輔助程式] 下載完成！儲存於 {setup_path}", flush=True)
                        
                        print("[Ollama 輔助程式] 準備啟動安裝程式...", flush=True)
                        if os.name == 'nt':
                            os.startfile(setup_path)
                        else:
                            subprocess.Popen([setup_path])
                            
                        st.success("✅ 下載完成！已為您彈出安裝程式視窗，請依畫面指示完成安裝。安裝完成後請**重新整理網頁**或**重啟系統**。")
                        print("[Ollama 輔助程式] 安裝程式啟動指令已送出。\n", flush=True)
                    except Exception as e:
                        st.error(f"自動下載或啟動失敗：{e}")
                        print(f"\n[Ollama 輔助程式] 發生錯誤：{e}\n", flush=True)

    st.divider()
    if st.button("🔍 檢測推論硬體狀態"):
        with st.status("正在檢測硬體資源...", expanded=True) as status:
            # 僅在按下按鈕時初始化介面以讀取狀態
            llm_service = LLMInterface(config_path="config.json")
            detector = AIDetector(config_path="config.json")
            
            st.write(f"🔹 **LLM 推論：** {llm_service.hardware_info}")
            st.write(f"🔹 **AI 偵測：** {detector.hardware_info}")
            status.update(label="檢測完成！", state="complete", expanded=True)

# ----------------- 分頁 1：參數設定 -----------------

if app_mode == "⚙️ 參數設定":
    st.title("⚙️ 系統參數設定")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("☁️ 雲端 LLM 設定")
        
        # 核心提供者配置
        provider_options = ["OpenAI-Compatible", "Gemini"]
        current_provider = app_config["cloud"].get("provider", "openai")
        provider_index = 1 if current_provider == "gemini" else 0
        
        selected_provider_type = st.radio("API 類型", provider_options, index=provider_index, horizontal=True)
        app_config["cloud"]["provider"] = "gemini" if selected_provider_type == "Gemini" else "openai"

        if selected_provider_type == "OpenAI-Compatible":
            # 預設提供者清單 (OpenAI 格式)
            openai_providers = {
                "OpenAI": "https://api.openai.com/v1/chat/completions",
                "DeepSeek": "https://api.deepseek.com/v1/chat/completions",
                "Groq": "https://api.groq.com/openai/v1/chat/completions",
                "OpenRouter": "https://openrouter.ai/api/v1/chat/completions",
                "Custom": ""
            }
            
            current_api_url = app_config["cloud"].get("api_url", "https://api.openai.com/v1/chat/completions")
            
            # 找出當前 URL 對應的提供者
            provider_key = "Custom"
            for k, v in openai_providers.items():
                if v == current_api_url and k != "Custom":
                    provider_key = k
                    break
            
            selected_preset = st.selectbox("模型來源預設 (OpenAI 格式)", list(openai_providers.keys()), index=list(openai_providers.keys()).index(provider_key))
            
            if selected_preset != "Custom":
                app_config["cloud"]["api_url"] = openai_providers[selected_preset]
                current_api_url = openai_providers[selected_preset]

            app_config["cloud"]["api_url"] = st.text_input("API 端點 (Endpoint)", value=current_api_url)
            app_config["cloud"]["model_name"] = st.text_input("模型名稱 (Model Name)", value=app_config["cloud"].get("model_name", "gpt-4o"))
        else:
            # Gemini 專屬設定
            st.info("Gemini 模式將使用 Google Generative AI REST API。")
            
            # (原偵測按鈕已移至 API Key 下方)

            # Gemini 常見模型清單
            gemini_presets = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-8b", "gemini-2.0-flash-exp", "Custom"]
            current_gemini_model = app_config["cloud"].get("model_name", "gemini-1.5-flash")
            
            # 找出目前模型是否在預設清單中
            model_key = "Custom"
            if current_gemini_model in gemini_presets:
                model_key = current_gemini_model
                
            selected_gemini_preset = st.selectbox("選取 Gemini 模型", gemini_presets, index=gemini_presets.index(model_key))
            
            if selected_gemini_preset != "Custom":
                app_config["cloud"]["model_name"] = selected_gemini_preset
                current_gemini_model = selected_gemini_preset

            app_config["cloud"]["model_name"] = st.text_input("模型名稱 (Model Name)", value=current_gemini_model)

        # API Key 輸入
        app_config["cloud"]["api_key"] = st.text_input("API Key", value=app_config["cloud"].get("api_key", ""), type="password")

        # --- Gemini 專屬：模型偵測按鈕 (放置在此以支援即時測試新 Key) ---
        if selected_provider_type == "Gemini":
            if st.button("🔍 偵測目前填入 API Key 可用的模型"):
                if app_config["cloud"]["api_key"] == "YOUR_NEW_GEMINI_API_KEY" or not app_config["cloud"]["api_key"]:
                    st.error("請先填入您從 Google AI Studio 取得的新 API Key！")
                else:
                    llm_service = LLMInterface(config_path="config.json")
                    # 使用目前畫面上填寫的 API Key 進行偵測，不需先儲存
                    available_models = llm_service.list_models(api_key=app_config["cloud"]["api_key"])
                    st.write(f"**可用模型清單：**\n{available_models}")
        # ---------------------------------------------------------
        
        st.divider()
        st.subheader("🔍 AI Detector 設定")
        
        detector_modes = ["Hugging Face 神經網路 (推薦)", "GPTZero API (雲端)", "本地落地模型 (Local LLM)"]
        current_detector_mode = app_config["ai_detector"].get("mode", "hf_model")
        
        if current_detector_mode == "hf_model":
            detector_index = 0
        elif current_detector_mode == "cloud":
            detector_index = 1
        else:
            detector_index = 2
        
        selected_detector_mode = st.radio("偵測模式", detector_modes, index=detector_index, horizontal=False)
        
        if "Hugging Face" in selected_detector_mode:
            app_config["ai_detector"]["mode"] = "hf_model"
        elif "本地" in selected_detector_mode:
            app_config["ai_detector"]["mode"] = "local"
        else:
            app_config["ai_detector"]["mode"] = "cloud"

        if app_config["ai_detector"]["mode"] == "cloud":
            app_config["ai_detector"]["api_key"] = st.text_input("GPTZero API Key", value=app_config["ai_detector"].get("api_key", ""), type="password")
            app_config["ai_detector"]["api_url"] = st.text_input("API 端點", value=app_config["ai_detector"].get("api_url", "https://api.gptzero.me/v2/predict/text"))
        elif app_config["ai_detector"]["mode"] == "hf_model":
            st.info("Hugging Face 模式將不需聯網，直接使用 Desklib 神經網路模型處理 (效能與精準度最佳)。")
            
            # 針對 Blackwell (RTX 50 系列) 的硬體相容性開關
            app_config["ai_detector"]["force_cpu"] = st.checkbox(
                "強制使用 CPU 進行 AI 偵測", 
                value=app_config["ai_detector"].get("force_cpu", False),
                help="如果您的顯卡 (如 RTX 5090) 遇到 CUDA no kernel image 錯誤，請開啟此選項。CPU 跑小模型依然很快。"
            )
        else:
            st.info("本地模式將優先使用您在「💻 本地 LLM 設定」中載入的模型。注意：大模型極度消耗運算資源，且請確保上下文窗口 (n_ctx) 足夠容納全文。")

    with col_b:
        st.subheader("💻 本地 LLM 設定 (GGUF)")
        
        if "model_path_widget" not in st.session_state:
            st.session_state.model_path_widget = app_config["local"].get("model_path", "")
            
        def on_browse_click():
            picked_path = select_file(st.session_state.model_path_widget)
            if picked_path:
                st.session_state.model_path_widget = picked_path

        path_col1, path_col2 = st.columns([0.85, 0.15])
        with path_col1:
            model_path_input = st.text_input("模型檔案路徑", key="model_path_widget")
        with path_col2:
            st.write(" ")
            st.write(" ")
            if TK_AVAILABLE:
                st.button("📂", help="瀏覽本地模型檔案", on_click=on_browse_click)
            else:
                st.button("📂", help="雲端環境不支援瀏覽本地檔案", disabled=True)

        app_config["local"]["model_path"] = model_path_input
        app_config["local"]["n_ctx"] = st.number_input("上下文窗口 (n_ctx)", value=app_config["local"].get("n_ctx", 4096), step=1024)
        app_config["local"]["max_tokens"] = st.number_input("最大輸出 (max_tokens)", value=app_config["local"].get("max_tokens", 1024), step=256)
        
        # 增加 GPU 卸載開關避免 Blackwell 5090 直接閃退
        app_config["local"]["use_gpu"] = st.checkbox(
            "啟用 GPU 顯示卡加速 (若閃退請關閉)", 
            value=app_config["local"].get("use_gpu", True),
            help="取消勾選將以純 CPU 執行本地大模型，可解決部分顯卡不相容導致程式無預警閃退中斷的問題。"
        )

    if st.button("💾 儲存並套用設定", type="primary"):
        save_config(app_config)
        st.success("設定檔已成功更新！")
        st.rerun()

# ----------------- 分頁 2：主頁面 -----------------

else:
    st.title("🎓 多代理人 AI 論文審查系統")

    # 1. 論文資料設定
    st.header("📄 1. 論文資料設定")
    col1, col2 = st.columns(2)
    with col1:
        paper_title = st.text_input("論文標題", placeholder="請輸入論文標題...")
    with col2:
        paper_field = st.text_input("主題領域", placeholder="例如：電腦視覺、人工智慧")

    paper_content = ""
    # 更新支援格式：txt, pdf, docx
    uploaded_file = st.file_uploader("上傳論文檔案 (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
    
    if uploaded_file is not None:
        with st.spinner("正在解析檔案內容..."):
            try:
                paper_content = extract_text_from_file(uploaded_file)
                if paper_content.strip():
                    st.success(f"檔案「{uploaded_file.name}」解析成功！")
                else:
                    st.warning("檔案解析完成，但未偵測到有效文字。")
            except Exception as e:
                st.error(f"解析失敗：{e}")
        
        # 當上傳新檔案時，清除舊的 AI 偵測結果
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.ai_report = None
            st.session_state.last_uploaded_file = uploaded_file.name
    
    paper_content_input = st.text_area("或直接貼上/編輯論文內容", value=paper_content, height=200)
    final_paper_content = paper_content_input

    # 2. AI 寫作偵測
    st.header("🔍 2. AI 寫作偵測")
    col_btn1, col_btn2 = st.columns([0.2, 0.8])
    with col_btn1:
        execute_btn = st.button("執行 AI 寫作分析", icon="🔎", type="primary")
    with col_btn2:
        clear_btn = st.button("🧼 清除分析結果")
    
    if clear_btn:
        st.session_state.ai_report = None
        st.rerun()

    if execute_btn:
        if not final_paper_content.strip():
            st.warning("請先輸入或上傳論文內容！")
        else:
            llm_service = LLMInterface(config_path="config.json")
            detector = AIDetector(config_path="config.json")
            with st.spinner("AI 偵測分析中..."):
                 # 傳遞正在使用的 LLM 介面給偵測器 (僅在本地模式需要)
                 report = detector.analyze(final_paper_content, llm_interface=llm_service)
                 st.session_state.ai_report = report
                 st.success("分析完成！")

    # 顯示分析結果 (渲染邏輯移出按鈕判斷區，以支援 Session State 持久化)
    if st.session_state.ai_report:
        report = st.session_state.ai_report
        
        if "notice" in report:
            st.warning(f"⚠️ 注意：{report['notice']}")
        
        # 顯示推論模型名稱 [新功能]
        st.success(f"🤖 **推論模型：** {report.get('model_name', '未知模型')}")
        
        st.subheader(f"📊 偵測報告 (AI 比例：{report['ai_ratio']}%)")
        
        if report.get("summary"):
            st.info(f"📝 **分析摘要：** {report['summary']}")

        # 渲染顏色標示 (包含 Tooltip 提示理由)
        highlighted_html = "<div style='line-height:1.8; border:1px solid #ddd; padding:20px; border-radius:10px; background-color:#fafafa; color:#333; font-size:16px;'>"
        
        found_ai = False
        for seg in report['segments']:
            if seg['type'] == 'AI':
                found_ai = True
                reason = seg.get('reason', 'AI 生成嫌疑')
                highlighted_html += f"<span style='background-color:{seg['color']}; border-radius:3px; cursor:help; margin-right:2px;' title='{reason}'>{seg['text']}</span> "
            else:
                highlighted_html += f"<span>{seg['text']}</span> "
        
        highlighted_html += "</div>"
        st.markdown(highlighted_html, unsafe_allow_html=True)
        
        if found_ai:
            st.caption("💡 提示：將滑鼠移至紅色標記文字上可查看詳細判定理由。")
        else:
            st.caption("✅ 未偵測到明顯 AI 生成嫌疑句。")
        
        # 5. 詳細數據表格呈現
        with st.expander("📊 查看詳細偵測數據表格", expanded=False):
            import pandas as pd
            df_data = []
            for seg in report['segments']:
                df_data.append({
                    "類型": "🤖 AI" if seg['type'] == 'AI' else "👤 人類",
                    "機率": f"{seg.get('prob', 0)*100:.1f}%" if seg.get('prob') is not None else "-",
                    "內容片段": seg['text'],
                    "判定理由": seg.get('reason', '-')
               })
            if df_data:
                st.dataframe(pd.DataFrame(df_data), use_container_width=True)

        # 增加匯出功能
        st.divider()
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            json_report = json.dumps(report, indent=4, ensure_ascii=False)
            st.download_button(
                label="📥 匯出完整 JSON 報告",
                data=json_report,
                file_name=f"AI_Detector_Report_{paper_title if paper_title else 'Untitled'}.json",
                mime="application/json",
                key="json_download_btn" # 增加 key 以防衝突
            )
        with col_exp2:
            md_report = f"# AI 寫作偵測報告\n\n"
            md_report += f"- **論文標題：** {paper_title if paper_title else '未命名'}\n"
            md_report += f"- **推論模型：** {report.get('model_name', '未知')}\n"
            md_report += f"- **AI 比例：** {report['ai_ratio']}%\n"
            md_report += f"- **分析摘要：** {report.get('summary', '無')}\n\n"
            md_report += "## 詳細分析\n\n"
            for seg in report['segments']:
                md_report += f"- [{seg['type']}] {seg['text']}\n"
                if seg.get('reason'):
                     md_report += f"  - *原因：{seg['reason']}*\n"
            
            st.download_button(
                label="📄 匯出 Markdown 摘要",
                data=md_report,
                file_name=f"AI_Detector_Report_{paper_title if paper_title else 'Untitled'}.md",
                mime="text/markdown",
                key="md_download_btn" # 增加 key
            )

    # 3. 審查委員配置
    st.header("👥 3. 審查委員配置")
    with st.expander("➕ 管理審查委員", expanded=False):
        for i, reviewer in enumerate(st.session_state.reviewers):
            st.text(f"委員 {i+1}: {reviewer.name} ({reviewer.expertise})")
        
        with st.form("add_reviewer_form"):
            r_name = st.text_input("委員名稱")
            r_expertise = st.text_input("專業領域")
            r_focus = st.text_input("研究重心")
            r_style = st.text_input("學術審查風格")
            if st.form_submit_button("加入名單") and r_name and r_expertise:
                st.session_state.reviewers.append(ReviewerAgent(r_name, r_expertise, r_focus, r_style))
                st.rerun()

    # 4. 執行多代理人審查
    st.header("🚀 4. 執行多代理人學術審查")

    async def run_review_process():
        if not paper_title or not final_paper_content.strip():
            st.error("請提供論文標題與內容")
            return

        my_paper = Paper(title=paper_title, field=paper_field, content=final_paper_content)
        llm_service = LLMInterface(config_path="config.json")
        orchestrator = PaperReviewOrchestrator(paper=my_paper, reviewers=st.session_state.reviewers, llm=llm_service)

        st.divider()
        st.subheader(f"審查進行中：{paper_title}")
        
        with st.status("第一輪：獨立深度審查...", expanded=True) as s1:
            await orchestrator.run_round_1()
            s1.update(label="✅ 第一輪完成", state="complete")
        
        with st.status("第二輪：交叉辯論...", expanded=True) as s2:
            await orchestrator.run_round_2()
            s2.update(label="✅ 第二輪完成", state="complete")
            
        with st.status("第三輪：最終裁決...", expanded=True) as s3:
            await orchestrator.run_round_3()
            s3.update(label="✅ 第三輪完成", state="complete")
        
        st.session_state.review_history = orchestrator.history
        st.rerun()

    if st.button("啟動多輪 AI 審查", type="primary"):
        asyncio.run(run_review_process())

    # 結果展示與匯出
    if st.session_state.review_history:
        st.divider()
        st.header("📋 審查結果展示")
        
        export_text = f"# 論文審查報告：{paper_title}\n領域：{paper_field}\n\n"
        for rnd in ["round_1", "round_2", "round_3"]:
            title = "第一輪：獨立審查" if rnd=="round_1" else "第二輪：交叉辯論" if rnd=="round_2" else "第三輪：最終裁決"
            export_text += f"## {title}\n"
            for name, content in st.session_state.review_history[rnd].items():
                export_text += f"### {name}\n{content}\n\n"
        
        st.download_button(
            label="📥 下載完整 Markdown 報告",
            data=export_text,
            file_name=f"Review_Report_{paper_title}.md",
            mime="text/markdown"
        )

        for rnd_id, rnd_name in [("round_1", "第一輪意見"), ("round_2", "第二輪辯論"), ("round_3", "最終裁決")]:
            with st.expander(rnd_name, expanded=(rnd_id == "round_3")):
                for name, content in st.session_state.review_history[rnd_id].items():
                    st.markdown(f"**{name}：**")
                    st.write(content)
                    st.divider()