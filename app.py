# app.py

import streamlit as st
import asyncio
import json
import os
import sys
import re
import tempfile
import copy
import subprocess
import platform
import urllib.request
import requests
from io import BytesIO

# 文字解析套件
import pypdf
import docx

# ----------------- 處理 Tkinter (雲端環境不支援) -----------------
try:
    import tkinter as tk
    from tkinter import filedialog
    TK_AVAILABLE = True
except (ImportError, Exception):
    TK_AVAILABLE = False
# -----------------------------------------------------------

# ----------------- 處理 llama-cpp (檢查本地模型套件) -----------------
try:
    import llama_cpp
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
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

def extract_text_from_file(uploaded_file):
    """ 根據檔案副檔名提取文字內容 """
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

def get_temp_user_config_path(user_conf):
    """ 為線上使用者生成臨時設定檔路徑，保護伺服器實體 config.json 不被污染 """
    if "config_temp_path" not in st.session_state:
        fd, path = tempfile.mkstemp(prefix="user_config_", suffix=".json")
        os.close(fd)
        st.session_state.config_temp_path = path
        
    with open(st.session_state.config_temp_path, 'w', encoding='utf-8') as f:
        json.dump(user_conf, f, indent=4, ensure_ascii=False)
        
    return st.session_state.config_temp_path

# ----------------- 設定檔管理 -----------------

st.set_page_config(page_title="多代理人論文審查系統", page_icon="🎓", layout="wide")

# 配置文件路徑處理 (確保打包後設定可持久化)
config_name = "config.json"
if getattr(sys, 'frozen', False):
    # 如果是打包後的執行檔，將設定檔放在 exe 旁邊
    base_dir = os.path.dirname(sys.executable)
    config_path = os.path.join(base_dir, config_name)
else:
    # 開發模式下使用當前目錄
    config_path = os.path.abspath(config_name)

if not os.path.exists(config_path):
    default_config = {
        "llm_mode": "mock",
        "cloud": {
            "provider": "openai",
            "api_key": "", 
            "model_name": "gpt-4o", 
            "api_url": "https://api.openai.com/v1/chat/completions"
        },
        "gemini_native": {
            "api_key": "", 
            "model_name": "gemini-1.5-flash"
        },
        "local": {
            "model_path": "./local_models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", 
            "n_ctx": 4096, 
            "max_tokens": 1024, 
            "n_gpu_layers": -1
        },
        "ollama": {
            "model_name": "llama3.1", 
            "base_url": "http://localhost:11434"
        },
        "ai_detector": {
            "api_key": "", 
            "api_url": "https://api.gptzero.me/v2/predict/text", 
            "mode": "Hugging Face 神經網路 (推薦)",
            "force_cpu": False
        }
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=4)

def load_global_config():
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_global_config(config):
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

# ==========================================
# 初始化 Session State (多租戶與狀態隔離)
# ==========================================

if "global_config_cache" not in st.session_state:
    st.session_state.global_config_cache = load_global_config()

if "user_config" not in st.session_state:
    st.session_state.user_config = copy.deepcopy(st.session_state.global_config_cache)

if "review_history" not in st.session_state:
    st.session_state.review_history = None

if "review_stats" not in st.session_state:
    st.session_state.review_stats = {}

if "ai_report" not in st.session_state:
    st.session_state.ai_report = None

if "reviewers" not in st.session_state:
    st.session_state.reviewers = [
        ReviewerAgent("Dr. Alan", "電腦視覺與深度學習", "輕量化神經網路架構", "極度嚴格，要求完整數據驗證。"),
        ReviewerAgent("Prof. Lin", "嵌入式與邊緣運算", "微控制器整合與邊緣AI", "務實且具建設性，重視系統現場落地性。")
    ]

if "paper_content_text_area" not in st.session_state:
    st.session_state.paper_content_text_area = ""

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if "exclusion_keywords" not in st.session_state:
    st.session_state.exclusion_keywords = ""

if "manual_exclusions" not in st.session_state:
    st.session_state.manual_exclusions = []

# ----------------- 側邊欄：雙入口身分切換 -----------------

with st.sidebar:
    st.header("🎮 工作區切換")
    entry_mode = st.radio(
        "選擇進入入口", 
        ["🌐 線上使用者", "💻 管理員 (單機推論)", "⚙️ 管理員 (參數設定)"]
    )
    
    st.divider()

    is_admin = False
    active_config = None
    active_config_path = ""

    # ==========================================
    # 模式一：線上使用者 (安全隔離區)
    # ==========================================
    if entry_mode == "🌐 線上使用者":
        st.success("🔒 **安全隔離模式已啟動**\n您的設定僅存於記憶體中，關閉視窗即自動清除。")
        st.subheader("🔑 推論引擎選擇")
        
        st.caption("雲端/本地引擎")
        user_provider = st.radio("選擇引擎", ["Gemini (推薦)", "OpenAI 相容", "Ollama (本地高併發)"], label_visibility="collapsed")
        
        if user_provider == "Ollama (本地高併發)":
            st.session_state.user_config["llm_mode"] = "ollama"
            st.info("Ollama 模式將使用伺服器本地資源進行高效能併發推論，不需輸入 API Key。")
            
            ollama_base_url = st.session_state.user_config.get("ollama", {}).get("base_url", "http://localhost:11434")
            try:
                requests.get(f"{ollama_base_url}/api/tags", timeout=1)
                st.success("✅ Ollama 伺服器連線正常，可開始推論！")
            except requests.exceptions.RequestException:
                st.error("⚠️ 伺服器端的 Ollama 服務似乎未啟動。請通知管理員。")
                
        else:
            user_key = st.text_input("輸入您的 API Key", type="password")
            
            if user_provider == "Gemini (推薦)":
                st.session_state.user_config["llm_mode"] = "gemini_native"
                st.session_state.user_config.setdefault("gemini_native", {})
                st.session_state.user_config["gemini_native"]["api_key"] = user_key
                user_model = st.selectbox("選擇 Gemini 模型", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-8b", "gemini-2.0-flash-exp"])
                st.session_state.user_config["gemini_native"]["model_name"] = user_model
            else:
                st.session_state.user_config["llm_mode"] = "cloud"
                st.session_state.user_config.setdefault("cloud", {})
                st.session_state.user_config["cloud"]["api_key"] = user_key
                st.session_state.user_config["cloud"]["provider"] = "openai"
                user_model = st.text_input("輸入 OpenAI 模型名稱", value="gpt-4o")
                st.session_state.user_config["cloud"]["model_name"] = user_model
        
        active_config = st.session_state.user_config
        active_config_path = get_temp_user_config_path(active_config)

    # ==========================================
    # 模式二與三：管理員 (最高權限區)
    # ==========================================
    else:
        pwd = st.text_input("請輸入系統管理員密碼", type="password")
        
        if pwd == "admin":
            is_admin = True
            active_config = st.session_state.global_config_cache
            active_config_path = config_path
            st.success("身分驗證成功！擁有伺服器完整存取權。")
            
            if entry_mode == "💻 管理員 (單機推論)":
                st.subheader("🤖 LLM 快速切換")
                
                llm_modes = {
                    "local": "💻 本地落地模型 (llama-cpp)",
                    "ollama": "🐑 Ollama API (推薦)",
                    "cloud": "☁️ 雲端大型模型",
                    "gemini_native": "✨ Google Gemini API",
                    "mock": "🛠️ 模擬測試模式"
                }
                
                current_mode = active_config.get("llm_mode", "mock")
                if current_mode not in llm_modes:
                    current_mode = "mock"
                    
                selected_mode = st.selectbox(
                    "切換 LLM 推論模式",
                    options=list(llm_modes.keys()),
                    index=list(llm_modes.keys()).index(current_mode),
                    format_func=lambda x: llm_modes[x]
                )
                
                if selected_mode != current_mode:
                    active_config["llm_mode"] = selected_mode
                    save_global_config(active_config)
                    st.success(f"已切換至 {llm_modes[selected_mode]}")
                    st.rerun()

                if active_config.get("llm_mode") == "ollama":
                    ollama_base_url = active_config.get("ollama", {}).get("base_url", "http://localhost:11434")
                    try:
                        requests.get(f"{ollama_base_url}/api/tags", timeout=1)
                    except requests.exceptions.RequestException:
                        st.error("⚠️ 偵測不到 Ollama 服務正在執行。")
                        st.markdown("請確保您已安裝並啟動 Ollama。若尚未安裝：")
                        st.markdown("1. 前往 [Ollama 官網](https://ollama.com/) 下載。")
                        st.markdown("2. 或點擊下方按鈕自動下載並啟動安裝程式。")
                        
                        if st.button("🚀 自動下載並啟動 Ollama 安裝"):
                            with st.spinner("正在下載 Ollama 安裝檔，檔案較大請耐心稍候... (下載完成後將自動啟動安裝)"):
                                try:
                                    if sys.platform != "win32":
                                        st.info("偵測為非 Windows 環境，正在執行官方自動安裝腳本... (可能需要 sudo 權限)")
                                        print("\n[Ollama 輔助程式] 開始執行 Linux/macOS 安裝腳本...", flush=True)
                                        res = subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, capture_output=True, text=True)
                                        if res.returncode == 0:
                                            st.success("Ollama 安裝完成！建議您透過終端機執行 `ollama serve` 來確保背景服務活著。")
                                            print("[Ollama 輔助程式] 安裝腳本執行成功！", flush=True)
                                        else:
                                            st.error(f"安裝腳本執行失敗，請手動在終端機執行: curl -fsSL https://ollama.com/install.sh | sh\n\n{res.stderr}")
                                    else:
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
                                        
                                        st.success("下載完成！正在為您啟動安裝程式...請注意快顯視窗")
                                        subprocess.Popen([setup_path])
                                        print("[Ollama 輔助程式] 安裝程式啟動指令已送出。\n", flush=True)
                                except Exception as e:
                                    st.error(f"自動下載或啟動失敗：{e}")
                                    print(f"\n[Ollama 輔助程式] 發生錯誤：{e}\n", flush=True)
        else:
            st.warning("⚠️ 請輸入正確密碼以解鎖管理員權限。(預設：admin)")
            st.stop()


# ----------------- 分頁 1：參數設定 (完美還原) -----------------

if entry_mode == "⚙️ 管理員 (參數設定)":
    st.title("⚙️ 系統參數設定")
    st.info("變更設定後，請點擊下方的「💾 儲存並套用設定」按鈕。")
    
    active_config.setdefault("cloud", {})
    active_config.setdefault("local", {})
    active_config.setdefault("ai_detector", {})
    active_config.setdefault("gemini_native", {})
    active_config.setdefault("ollama", {"model_name": "llama3.1", "base_url": "http://localhost:11434"})
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("☁️ 雲端 LLM 設定")
        
        provider_options = ["OpenAI-Compatible", "Gemini"]
        current_llm_mode = active_config.get("llm_mode", "mock")
        provider_index = 1 if current_llm_mode == "gemini_native" else 0
        
        selected_provider_type = st.radio("API 類型", provider_options, index=provider_index, horizontal=True)

        if selected_provider_type == "OpenAI-Compatible":
            active_config["llm_mode"] = "cloud"
            active_config["cloud"]["provider"] = "openai"
            
            openai_providers = {
                "OpenAI": "https://api.openai.com/v1/chat/completions",
                "DeepSeek": "https://api.deepseek.com/v1/chat/completions",
                "Groq": "https://api.groq.com/openai/v1/chat/completions",
                "OpenRouter": "https://openrouter.ai/api/v1/chat/completions",
                "Custom": ""
            }
            current_api_url = active_config["cloud"].get("api_url", "https://api.openai.com/v1/chat/completions")
            provider_key = "Custom"
            for k, v in openai_providers.items():
                if v == current_api_url and k != "Custom":
                    provider_key = k
                    break
            
            selected_preset = st.selectbox("模型來源預設 (OpenAI 格式)", list(openai_providers.keys()), index=list(openai_providers.keys()).index(provider_key))
            if selected_preset != "Custom":
                active_config["cloud"]["api_url"] = openai_providers[selected_preset]
                current_api_url = openai_providers[selected_preset]

            active_config["cloud"]["api_url"] = st.text_input("API 端點 (Endpoint)", value=current_api_url)
            active_config["cloud"]["model_name"] = st.text_input("模型名稱 (Model Name)", value=active_config["cloud"].get("model_name", "gpt-4o"))
            
            cloud_api_show = st.checkbox("顯示 Key", key="show_cloud_api")
            active_config["cloud"]["api_key"] = st.text_input("API Key", value=active_config["cloud"].get("api_key", ""), type="default" if cloud_api_show else "password")

        else:
            active_config["llm_mode"] = "gemini_native"
            st.info("Gemini 模式將使用 Google Generative AI REST API。")
            
            gemini_presets = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-8b", "gemini-2.0-flash-exp", "Custom"]
            current_gemini_model = active_config["gemini_native"].get("model_name", "gemini-1.5-flash")
            
            model_key = "Custom"
            if current_gemini_model in gemini_presets:
                model_key = current_gemini_model
                
            selected_gemini_preset = st.selectbox("選取 Gemini 模型", gemini_presets, index=gemini_presets.index(model_key))
            if selected_gemini_preset != "Custom":
                active_config["gemini_native"]["model_name"] = selected_gemini_preset
                current_gemini_model = selected_gemini_preset

            active_config["gemini_native"]["model_name"] = st.text_input("模型名稱 (Model Name)", value=current_gemini_model)

            cloud_api_show = st.checkbox("顯示 Key", key="show_cloud_api")
            active_config["gemini_native"]["api_key"] = st.text_input("API Key", value=active_config["gemini_native"].get("api_key", ""), type="default" if cloud_api_show else "password")

            if st.button("🔍 偵測目前填入 API Key 可用的模型"):
                if not active_config["gemini_native"]["api_key"] or "YOUR" in active_config["gemini_native"]["api_key"]:
                    st.error("請先填入您從 Google AI Studio 取得的 API Key！")
                else:
                    llm_service = LLMInterface(config_path="config.json")
                    available_models = llm_service.list_models(api_key=active_config["gemini_native"]["api_key"])
                    st.write(f"**可用模型清單：**\n{available_models}")

        st.divider()
        st.subheader("🔍 AI Detector 設定")
        detector_modes = ["Hugging Face 神經網路 (推薦)", "GPTZero API (雲端)", "本地落地模型 (Local LLM)"]
        current_det_mode = active_config["ai_detector"].get("mode", "Hugging Face 神經網路 (推薦)")
        det_index = detector_modes.index(current_det_mode) if current_det_mode in detector_modes else 0
        
        selected_det_mode = st.radio("偵測模式", detector_modes, index=det_index)
        active_config["ai_detector"]["mode"] = selected_det_mode

        if selected_det_mode == "Hugging Face 神經網路 (推薦)":
            st.info("Hugging Face 模式將不需聯網，直接使用 Desklib 神經網路模型處理 (效能與精準度最佳)。")
            active_config["ai_detector"]["force_cpu"] = st.checkbox(
                "強制使用 CPU 進行 AI 偵測", 
                value=active_config["ai_detector"].get("force_cpu", False)
            )
        elif selected_det_mode == "GPTZero API (雲端)":
            detector_api_show = st.checkbox("顯示 GPTZero Key", key="show_detector_api")
            active_config["ai_detector"]["api_key"] = st.text_input("GPTZero API Key", value=active_config["ai_detector"].get("api_key", ""), type="default" if detector_api_show else "password")
            active_config["ai_detector"]["api_url"] = st.text_input("GPTZero API Endpoint", value=active_config["ai_detector"].get("api_url", "https://api.gptzero.me/v2/predict/text"))
        elif selected_det_mode == "本地落地模型 (Local LLM)":
            st.info("本地模式將優先使用您在「💻 本地 LLM 設定」中載入的模型。注意：大模型極度消耗運算資源，且請確保上下文窗口 (n_ctx) 足夠容納全文。")

    with col_b:
        st.subheader("💻 本地 LLM 設定 (GGUF)")
        if "model_path_widget" not in st.session_state:
            st.session_state.model_path_widget = active_config["local"].get("model_path", "")
            
        def on_browse_click():
            picked_path = select_file(st.session_state.model_path_widget)
            if picked_path:
                st.session_state.model_path_widget = picked_path

        path_col1, path_col2 = st.columns([0.85, 0.15])
        with path_col1:
            model_path_input = st.text_input("GGUF 模型檔案路徑", key="model_path_widget")
        with path_col2:
            st.write(" ") 
            st.write(" ") 
            if TK_AVAILABLE:
                st.button("📂", help="瀏覽檔案系統", on_click=on_browse_click)
            else:
                st.button("📂", disabled=True)

        active_config["local"]["model_path"] = model_path_input
        active_config["local"]["n_ctx"] = st.number_input("上下文窗口大小 (n_ctx)", value=active_config["local"].get("n_ctx", 4096), step=1024)
        active_config["local"]["max_tokens"] = st.number_input("最大輸出 Token", value=active_config["local"].get("max_tokens", 1024), step=256)
        
        current_gpu_setting = active_config["local"].get("n_gpu_layers", -1)
        use_gpu = st.checkbox("啟用 GPU 顯示卡加速 (若閃退請關閉)", value=(current_gpu_setting != 0))
        active_config["local"]["n_gpu_layers"] = -1 if use_gpu else 0

        # Ollama 設定安插於右下角，不破壞經典版面
        st.divider()
        st.subheader("🦙 Ollama 伺服器設定")
        active_config["ollama"]["model_name"] = st.text_input("Ollama 模型名稱 (例如: llama3.1)", value=active_config["ollama"].get("model_name", "llama3.1"))
        active_config["ollama"]["base_url"] = st.text_input("Ollama 伺服器網址", value=active_config["ollama"].get("base_url", "http://localhost:11434"))

    st.divider()
    if st.button("💾 儲存並套用設定", type="primary"):
        save_global_config(active_config)
        st.success("設定檔已成功更新！")
        st.rerun()

# ----------------- 分頁 2：主頁面 -----------------

else:
    col_t1, col_t2 = st.columns([0.8, 0.2])
    with col_t1:
        st.title("🎓 多代理人 AI 論文審查系統")
        
        # 資源監控與狀態列
        c1, c2 = st.columns([0.3, 0.7])
        with c1: 
            st.markdown(f"**目前推論模式**：<span style='color:#FF4B4B'>{active_config.get('llm_mode')}</span>", unsafe_allow_html=True)
        with c2: 
            if st.button("🔍 檢測推論硬體狀態", help="檢測底層神經網路與 LLM 引擎狀態", use_container_width=True):
                with st.spinner("正在檢測硬體資源..."):
                    det = AIDetector(config_path=active_config_path)
                    llm_inf = LLMInterface(config_path=active_config_path)
                    
                    llm_mode = active_config.get("llm_mode", "mock")
                    if llm_mode == "gemini_native": 
                        llm_hw_status = "☁️ Cloud API (Gemini Native)"
                    elif llm_mode == "cloud": 
                        llm_hw_status = "☁️ Cloud API (OpenAI 相容)"
                    elif llm_mode == "ollama": 
                        llm_hw_status = "🦙 Ollama Server (高併發)"
                    elif llm_mode == "local":
                        n_gpu = active_config.get("local", {}).get("n_gpu_layers", 0)
                        if n_gpu != 0:
                            llm_hw_status = f"💻 GPU (Offloaded {n_gpu} layers)"
                        else:
                            llm_hw_status = "💻 CPU"
                    else: 
                        llm_hw_status = "🛠️ Mock Engine"
                        
                    st.success(f"✔️ **檢測完成！**\n\n🔹 **LLM 推論**： {llm_hw_status}\n\n🔹 **AI 偵測**： {det.hardware_info}")

    with col_t2:
        with st.popover("⚙️ 當前參數"):
            st.json(active_config)

    # ==========================================
    # 1. 論文資料設定
    # ==========================================
    st.header("📄 1. 論文基本資料輸入")
    col_f1, col_f2 = st.columns([0.6, 0.4])
    with col_f1:
        paper_title = st.text_input("論文標題", placeholder="基於深度學習的機械缺陷檢測系統...")
    with col_f2:
        paper_field = st.text_input("主題領域", placeholder="人工智慧、智慧製造")

    up_col1, up_col2 = st.columns([0.15, 0.85])
    with up_col1:
        if st.button("❌ 清除原始內文", help="清空下方的原始文字域"):
            st.session_state.paper_content_text_area = ""
            st.session_state.last_uploaded_file = None
            st.session_state.ai_report = None
            st.rerun()
            
    with up_col2:
        uploaded_file = st.file_uploader("上傳文檔解析內文 (.pdf, .txt, .docx)", type=["pdf", "txt", "docx"])
        
        if uploaded_file is not None and uploaded_file.name != st.session_state.get("last_uploaded_file"):
            with st.spinner("正在解析檔案內容，請稍候..."):
                try:
                    parsed_text = extract_text_from_file(uploaded_file)
                    if parsed_text.strip():
                        st.session_state.paper_content_text_area = parsed_text
                        st.session_state.last_uploaded_file = uploaded_file.name
                        st.success(f"檔案「{uploaded_file.name}」解析成功！")
                        st.rerun()
                    else:
                        st.warning("未偵測到文字內容。")
                except Exception as e:
                    st.error(f"檔案解析失敗：{e}")
    
    raw_paper_content = st.text_area(
        "原始論文內容 (上傳後將顯示於此，或可直接貼上)", 
        height=200, 
        key="paper_content_text_area"
    )

    # ==========================================
    # 1.5 內容預處理與範圍界定
    # ==========================================
    st.header("✂️ 1.5 內容預處理與資料清洗")
    
    total_words = len(raw_paper_content)
    filtered_text = raw_paper_content

    col_set1, col_set2 = st.columns([0.5, 0.5])
    
    with col_set1:
        st.subheader("設定排除規則")
        
        exclude_quotes = st.checkbox("排除標記為引用的文字 (包含中英文引號內文)", value=True, help="標記為引用的文字將從後續的 AI 審查與抄襲比對中剔除。")
        
        if st.button("一鍵套用常見參考文獻/附錄建議"):
            st.session_state.exclusion_keywords = "Bibliography\nReferences\nAppendix\n參考文獻\n參考書目\n附錄"
            st.rerun()
            
        exclusion_keywords = st.text_area(
            "自訂排除截斷關鍵字 (每行一個)：\n⚠️ 系統若偵測到以下標題，將自動刪除該標題至文末的所有內容。",
            value=st.session_state.exclusion_keywords,
            height=120
        )
        st.session_state.exclusion_keywords = exclusion_keywords
        
        st.markdown("##### 📝 手動新增欲排除的特定字串")
        col_m1, col_m2 = st.columns([0.7, 0.3])
        with col_m1:
            manual_exclude_input = st.text_input("在預覽區複製欲排除的句子貼於此：", key="manual_exclude_input")
        with col_m2:
            st.write(" ")
            st.write(" ")
            if st.button("加入排除"):
                if manual_exclude_input and manual_exclude_input not in st.session_state.manual_exclusions:
                    st.session_state.manual_exclusions.append(manual_exclude_input)
                    st.rerun()
        
        if st.session_state.manual_exclusions:
            st.caption("目前手動排除清單：")
            for i, me in enumerate(st.session_state.manual_exclusions):
                col_i1, col_i2 = st.columns([0.9, 0.1])
                col_i1.code(me[:50] + "..." if len(me)>50 else me)
                if col_i2.button("🗑️", key=f"del_me_{i}"):
                    st.session_state.manual_exclusions.pop(i)
                    st.rerun()

    # --- 執行文字過濾邏輯 ---
    if filtered_text:
        for me in st.session_state.manual_exclusions:
            filtered_text = filtered_text.replace(me, "")
            
        if exclude_quotes:
            filtered_text = re.sub(r'["“”「」](.*?)["“”「」]', '', filtered_text)
            
        if exclusion_keywords.strip():
            keywords = [k.strip() for k in exclusion_keywords.split('\n') if k.strip()]
            for kw in keywords:
                idx = filtered_text.lower().find(kw.lower())
                if idx != -1:
                    filtered_text = filtered_text[:idx] 

    compared_words = len(filtered_text)
    excluded_words = total_words - compared_words

    with col_set2:
        st.subheader("字數統計儀表板")
        
        # 這裡 100% 保留了多行 HTML 字串與原始排版
        st.markdown(f"""
        <div style="border:1px solid #e0e0e0; border-radius:10px; padding:15px; text-align:center; margin-bottom:12px; background-color:#ffffff; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
            <div style="color:#5f6368; font-size:16px; font-weight:500;">📄 文件原始總字數</div>
            <div style="font-size:32px; font-weight:800; color:#202124;">{total_words:,}</div>
        </div>
        <div style="border:1px solid #fad2cf; border-radius:10px; padding:15px; text-align:center; margin-bottom:12px; background-color:#fce8e6; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
            <div style="color:#d93025; font-size:16px; font-weight:500;">❌ 總計排除字數</div>
            <div style="font-size:32px; font-weight:800; color:#d93025;">{excluded_words:,}</div>
        </div>
        <div style="border:1px solid #ceead6; border-radius:10px; padding:15px; text-align:center; background-color:#e6f4ea; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
            <div style="color:#137333; font-size:16px; font-weight:500;">✅ 最終比對/審查字數</div>
            <div style="font-size:32px; font-weight:800; color:#137333;">{compared_words:,}</div>
        </div>
        """, unsafe_allow_html=True)

        if compared_words > 30000:
            st.warning("⚠️ 注意：最終比對字數超過 30,000 字，可能因超過 Token 限制而報錯，建議進一步精簡。")

    st.markdown("##### 🔍 內容預覽 (分析前檢視您的文件)")
    preview_limit = 1000
    display_text = filtered_text[:preview_limit] + ("\n\n... (內容已截斷，僅顯示前 1000 字)" if len(filtered_text) > preview_limit else "")
    st.info(display_text if display_text else "尚無內容。")

    final_paper_content_for_llm = filtered_text

    # ==========================================
    # 2. AI 寫作偵測
    # ==========================================
    st.header("🔍 2. AI 寫作偵測")
    
    col_btn1, col_btn2 = st.columns([0.2, 0.8])
    with col_btn1:
        run_ai_btn = st.button("🔎 執行 AI 寫作分析", type="primary")
    with col_btn2:
        if st.button("🧽 清除分析結果"):
            st.session_state.ai_report = None
            st.rerun()

    if run_ai_btn:
        if not final_paper_content_for_llm.strip():
            st.warning("清洗後的文本為空，請確認排除設定是否誤刪全文。")
        else:
            llm_service = LLMInterface(config_path=active_config_path)
            detector = AIDetector(config_path=active_config_path)
            with st.spinner("正在呼叫 AI 引擎進行特徵比對..."):
                report = detector.analyze(final_paper_content_for_llm, llm_interface=llm_service)
                st.session_state.ai_report = report

    if st.session_state.ai_report:
        report = st.session_state.ai_report
        
        if "notice" in report:
            st.warning(f"⚠️ 注意：{report['notice']}")
            
        st.success(f"🤖 **推論模型：** {report.get('model_name', '未知模型')}")
        
        st.markdown(f"### 📊 分析報告 (AI 生成機率：<span style='color:#FF4B4B'>{report['ai_ratio']}%</span>)", unsafe_allow_html=True)
        
        if report.get("summary"):
            st.info(f"📝 **分析摘要：** {report['summary']}")
            
        st.markdown("##### 📝 文本風格標示 (顏色越深表示機率越高)：")
        st.markdown(
            "圖例： <span style='background-color:#ffcccc; padding:2px; border-radius:3px;'>淺紅 (疑似 AI)</span> | "
            "<span style='background-color:#ff9999; padding:2px; border-radius:3px;'>深紅 (極高機率 AI)</span>", 
            unsafe_allow_html=True
        )
        
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

        with st.expander("📊 查看詳細偵測數據表格"):
            import pandas as pd
            table_data = []
            for seg in report['segments']:
                table_data.append({
                    "類型": "🤖 AI" if seg['type'] == 'AI' else "👤 人類",
                    "機率": f"{seg.get('prob', 0)*100:.1f}%" if seg.get('prob') is not None else "-",
                    "內容片段": seg['text'],
                    "判定理由": seg.get('reason', '-')
                })
            df = pd.DataFrame(table_data)
            st.dataframe(df, width="stretch")

        st.divider()

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            json_str = json.dumps(report, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 匯出完整 JSON 報告",
                data=json_str,
                file_name="AI_Detection_Report.json",
                mime="application/json"
            )
        with col_exp2:
            md_str = f"# AI 寫作偵測報告\n\n**總體 AI 生成機率**：{report['ai_ratio']}%\n\n## 詳細內容分析\n\n"
            for seg in report['segments']:
                marker = "🤖 [AI 生成]" if seg['type'] == 'AI' else "👤 [人類撰寫]"
                md_str += f"- **{marker}**: {seg['text']}\n"
                if seg.get('reason'):
                    md_str += f"  - *原因：{seg['reason']}*\n"
                
            st.download_button(
                label="📄 匯出 Markdown 摘要",
                data=md_str,
                file_name="AI_Detection_Summary.md",
                mime="text/markdown"
            )

    # ==========================================
    # 3. 審查委員配置
    # ==========================================
    st.header("👥 3. 審查委員配置")
    with st.expander("➕ 新增/管理代理人委員", expanded=False):
        st.markdown("##### 當前委員陣容：")
        cols = st.columns(len(st.session_state.reviewers))
        for i, reviewer in enumerate(st.session_state.reviewers):
            with cols[i]:
                st.info(f"**{reviewer.name}**\n\n專業：{reviewer.expertise}\n\n風格：{reviewer.style}")
        st.divider()
        
        st.markdown("##### 新增自訂委員")
        with st.form("add_reviewer_form_simplified"):
            r_name = st.text_input("姓名 (如: Dr. Lee)")
            r_expertise = st.text_input("專業領域 (如: 硬體加密、工業資安)")
            r_prompt_simplfied = st.text_area("審查風格提示詞 (Agent Definition Prompt)", value="務實的學術標準，強調系統在工業 4.0 環境下的安全性與硬體資源消耗，拒絕理論推導不完整的內容。", height=100)
            
            submitted = st.form_submit_button("加入名單")
            if submitted and r_name and r_expertise:
                st.session_state.reviewers.append(ReviewerAgent(
                    name=r_name, 
                    expertise=r_expertise, 
                    research_focus="根據風格自適應", 
                    style=r_prompt_simplfied
                ))
                st.success(f"已加入委員：{r_name}")
                st.rerun()

    # ==========================================
    # 4. 執行多代理人審查
    # ==========================================
    st.header("🚀 4. 執行多代理人學術審查")

    async def run_review_process():
        if not paper_title or not final_paper_content_for_llm.strip():
            st.error("請提供論文標題與清洗後的內容。")
            return
        if len(st.session_state.reviewers) < 2:
            st.error("請至少設定兩位以上委員以進行交叉辯論。")
            return

        my_paper = Paper(title=paper_title, field=paper_field, content=final_paper_content_for_llm)
        llm_service = LLMInterface(config_path=active_config_path)
        orchestrator = PaperReviewOrchestrator(paper=my_paper, reviewers=st.session_state.reviewers, llm=llm_service)

        st.divider()
        st.subheader(f"專案審查中：《{paper_title}》")
        
        with st.status("執行第一輪：獨立深度審查...", expanded=True) as s1:
            await orchestrator.run_round_1()
            s1.update(label="✅ 第一輪：獨立審查完成", state="complete")
        
        with st.status("執行第二輪：交叉辯論與反駁...", expanded=True) as s2:
            await orchestrator.run_round_2()
            s2.update(label="✅ 第二輪：交叉辯論完成", state="complete")
            
        with st.status("執行第三輪：最終共識與裁決 (JSON 解析數據)...", expanded=True) as s3:
            await orchestrator.run_round_3()
            s3.update(label="✅ 第三輪：最終裁決完成", state="complete")
        
        # 加入 getattr 安全取值，防止模型輸出無效 JSON 導致系統崩潰
        st.session_state.review_history = getattr(orchestrator, "history", {})
        st.session_state.review_stats = getattr(orchestrator, "review_stats", {
            "avg_contribution": 0.0,
            "avg_deficiencies": 0.0,
            "avg_robustness": 0.0
        })
        st.balloons()
        st.rerun()

    btn_disabled = (not HAS_LLAMA_CPP and active_config.get("llm_mode") == "local")
    if st.button("啟動多輪 AI 審查", type="primary", disabled=btn_disabled):
        asyncio.run(run_review_process())

    # ==========================================
    # 5. 結果展示與儀表板
    # ==========================================
    if st.session_state.review_history:
        st.divider()
        st.header("📋 審查結果展示")
        
        if st.session_state.review_stats:
            stats = st.session_state.review_stats
            st.subheader(f"📈 綜合統計儀表板 (基於《{paper_title}》)")
            c1, c2, c3 = st.columns(3)
            with c1:
                val = stats.get('avg_contribution', 0)
                color = "green" if val >= 8 else "orange" if val >= 6 else "red"
                st.markdown(f"**總體貢獻度** (高分為佳)")
                st.progress(val / 10.0, text=f"{val}/10 ({color})")
            with c2:
                val = stats.get('avg_deficiencies', 0)
                color = "green" if val <= 4 else "orange" if val <= 6 else "red"
                st.markdown(f"**主要缺陷密度** (低分為佳)")
                st.progress(val / 10.0, text=f"{val}/10 ({color})")
            with c3:
                val = stats.get('avg_robustness', 0)
                color = "green" if val >= 8 else "orange" if val >= 6 else "red"
                st.markdown(f"**方法論強健度** (高分為佳)")
                st.progress(val / 10.0, text=f"{val}/10 ({color})")
            
            st.divider()

        export_text = f"# 論文審查報告：{paper_title}\n領域：{paper_field}\n\n"
        if st.session_state.review_stats:
            stats = st.session_state.review_stats
            export_text += "## 📈 統計報告\n"
            export_text += f"Contribution: {stats.get('avg_contribution')}/10\n"
            export_text += f"Deficiencies: {stats.get('avg_deficiencies')}/10 (低為佳)\n"
            export_text += f"Robustness: {stats.get('avg_robustness')}/10\n\n"

        for rnd in ["round_1", "round_2", "round_3"]:
            title = "第一輪意见" if rnd=="round_1" else "第二輪交叉辯論" if rnd=="round_2" else "第三輪最終裁決"
            export_text += f"## {title}\n"
            for name, content in st.session_state.review_history[rnd].items():
                export_text += f"### {name}\n{content}\n\n"
        
        st.download_button(
            label="📥 下載完整報告 (Markdown格式)",
            data=export_text,
            file_name=f"Review_Report_{paper_title}.md",
            mime="text/markdown"
        )

        rnd_names = [("round_1", "第一輪意見 (獨立)"), ("round_2", "第二輪交叉辯論"), ("round_3", "第三輪最終裁決")]
        for rnd_id, rnd_name in rnd_names:
            with st.expander(rnd_name, expanded=(rnd_id == "round_3")):
                for name, content in st.session_state.review_history[rnd_id].items():
                    st.markdown(f"**{name}：**")
                    if rnd_id == "round_3":
                        st.markdown(content)
                    else:
                        st.write(content)
                    st.divider()