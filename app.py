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

# -----------------------------------------------------------
# 處理 Tkinter (雲端環境不支援時的防呆機制)
# -----------------------------------------------------------
try:
    import tkinter as tk
    from tkinter import filedialog
    TK_AVAILABLE = True
except (ImportError, Exception):
    TK_AVAILABLE = False

# -----------------------------------------------------------
# 處理 llama-cpp (檢查本地模型套件是否已安裝)
# -----------------------------------------------------------
try:
    import llama_cpp
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

# -----------------------------------------------------------
# 匯入自定義模組
# -----------------------------------------------------------
from models.paper import Paper
from models.reviewer import ReviewerAgent
from llm.interface import LLMInterface
from core.orchestrator import PaperReviewOrchestrator
from core.ai_detector import AIDetector


# ===========================================================
# 工具函式區塊
# ===========================================================

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

def check_model_exists(path):
    """ 檢查本地模型路徑是否有效 (含相對路徑轉換) """
    if not path: 
        return False
    if os.path.exists(path): 
        return True
    if os.path.exists(resource_path(path)): 
        return True
    return False

def extract_text_from_file(uploaded_file):
    """ 根據檔案副檔名提取文字內容 """
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # 確保檔案指標在起始位置，防止二次讀取變為空
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


# ===========================================================
# 系統設定檔管理初始化
# ===========================================================

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
            "mode": "hf_model",
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


# ===========================================================
# 核心狀態初始化 (Session State) - 多租戶隔離關鍵
# ===========================================================

# 處理強制重載設定檔的邏輯 (確保 UI 切換立刻生效)
if "force_reload_config" in st.session_state and st.session_state.force_reload_config:
    st.session_state.global_config_cache = load_global_config()
    st.session_state.user_config = copy.deepcopy(st.session_state.global_config_cache)
    st.session_state.force_reload_config = False

# 1. 讀取並暫存伺服器全域設定 (最高權限)
if "global_config_cache" not in st.session_state:
    st.session_state.global_config_cache = load_global_config()

# 2. 發配一份獨立的設定檔給當前連線的網頁使用者
if "user_config" not in st.session_state:
    st.session_state.user_config = copy.deepcopy(st.session_state.global_config_cache)

# 3. 各項面板與紀錄狀態
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


# ===========================================================
# 側邊欄：雙入口身分切換與控制面板
# ===========================================================

with st.sidebar:
    st.header("🎮 工作區切換")
    
    entry_mode = st.radio(
        "選擇進入入口", 
        ["🌐 線上使用者", "💻 管理員 (單機推論)", "⚙️ 管理員 (參數設定)"]
    )
    
    st.divider()

    # 記錄當前身分與對應的設定檔路徑
    is_admin = False
    active_config = None
    active_config_path = ""

    # -------------------------------------------------------
    # 模式一：線上使用者 (安全隔離區)
    # -------------------------------------------------------
    if entry_mode == "🌐 線上使用者":
        st.success("🔒 **安全隔離模式已啟動**\n您的設定僅存於記憶體中，關閉視窗即自動清除。")
        
        st.subheader("🔑 推論引擎選擇")
        st.caption("雲端/本地引擎")
        
        user_provider = st.radio(
            "選擇引擎", 
            ["Gemini (推薦)", "OpenAI 相容", "Ollama (本地高併發)"], 
            label_visibility="collapsed"
        )
        
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
            st.caption("輸入您的 API Key")
            user_key = st.text_input("輸入您的 API Key", type="password", label_visibility="collapsed")
            
            if user_provider == "Gemini (推薦)":
                st.session_state.user_config["llm_mode"] = "cloud"
                st.session_state.user_config.setdefault("cloud", {})
                st.session_state.user_config["cloud"]["provider"] = "gemini"
                st.session_state.user_config["cloud"]["api_key"] = user_key
                
                # ==========================================
                # 【新增功能】線上使用者專屬：偵測可用模型清單
                # ==========================================
                if st.button("🔍 偵測可用模型", key="online_detect_models_btn"):
                    if not user_key:
                        st.error("請先填入您的 API Key！")
                    else:
                        with st.spinner("正在與 Google 伺服器驗證金鑰..."):
                            llm_service = LLMInterface(config_path="config.json")
                            available_models = llm_service.list_models(api_key=user_key)
                            
                            if "錯誤" in available_models or "失敗" in available_models:
                                st.error(f"**金鑰無效或連線失敗：**\n\n{available_models}")
                            else:
                                st.success(f"**✅ 驗證成功！可用模型清單：**\n\n{available_models}")
                # ==========================================

                st.caption("選擇 Gemini 模型")
                user_model = st.selectbox(
                    "選擇 Gemini 模型", 
                    ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-8b", "gemini-2.0-flash-exp", "gemini-2.5-flash"],
                    label_visibility="collapsed"
                )
                st.session_state.user_config["cloud"]["model_name"] = user_model
            else:
                st.session_state.user_config["llm_mode"] = "cloud"
                st.session_state.user_config.setdefault("cloud", {})
                st.session_state.user_config["cloud"]["provider"] = "openai"
                st.session_state.user_config["cloud"]["api_key"] = user_key
                
                st.caption("輸入 OpenAI 模型名稱")
                user_model = st.text_input("輸入 OpenAI 模型名稱", value="gpt-4o", label_visibility="collapsed")
                st.session_state.user_config["cloud"]["model_name"] = user_model
        
        # 綁定給後續推論流程使用
        active_config = st.session_state.user_config
        active_config_path = get_temp_user_config_path(active_config)

    # -------------------------------------------------------
    # 模式二與三：管理員 (最高權限區)
    # -------------------------------------------------------
    else:
        st.caption("請輸入系統管理員密碼")
        pwd = st.text_input("請輸入系統管理員密碼", type="password", label_visibility="collapsed")
        
        if pwd == "admin":
            is_admin = True
            # 管理員模式直接讀取全域快取
            active_config = st.session_state.global_config_cache
            active_config_path = config_path
            st.success("身分驗證成功！擁有伺服器完整存取權。")
            
            if entry_mode == "💻 管理員 (單機推論)":
                st.subheader("🤖 LLM 快速切換")
                
                llm_modes = {
                    "cloud": "☁️ 雲端 API (含 Gemini/OpenAI)",
                    "gemini_native": "✨ Google Gemini 原生 SDK",
                    "local": "💻 本地落地模型 (llama-cpp)",
                    "ollama": "🐑 Ollama API (推薦)",
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
                    # 強制更新快取，確保一致性
                    st.session_state.global_config_cache = copy.deepcopy(active_config)
                    st.success(f"已切換至 {llm_modes[selected_mode]}")
                    st.rerun()

                # ===============================================
                # Ollama 原汁原味的自動檢查與下載腳本 (僅管理員可用)
                # ===============================================
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
            st.warning("⚠️ 此區塊僅限管理員存取。請輸入正確密碼。")
            st.stop()


# ===========================================================
# 分頁 1：管理員系統參數設定
# ===========================================================

if entry_mode == "⚙️ 管理員 (參數設定)":
    
    st.title("⚙️ 系統參數設定")
    st.info("變更設定後，請點擊下方的「💾 儲存並套用設定」按鈕。")
    
    active_config.setdefault("cloud", {})
    active_config.setdefault("local", {})
    active_config.setdefault("ai_detector", {})
    active_config.setdefault("ollama", {"model_name": "llama3.1", "base_url": "http://localhost:11434"})
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("☁️ 雲端 LLM 設定")
        
        provider_options = ["OpenAI-Compatible", "Gemini"]
        current_provider = active_config["cloud"].get("provider", "openai")
        provider_index = 1 if current_provider == "gemini" else 0
        
        st.caption("API 類型")
        selected_provider_type = st.radio(
            "API 類型", 
            provider_options, 
            index=provider_index, 
            horizontal=True,
            label_visibility="collapsed"
        )
        
        active_config["cloud"]["provider"] = "gemini" if selected_provider_type == "Gemini" else "openai"
        active_config["llm_mode"] = "cloud"

        if selected_provider_type == "OpenAI-Compatible":
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
            
            selected_preset = st.selectbox(
                "模型來源預設 (OpenAI 格式)", 
                list(openai_providers.keys()), 
                index=list(openai_providers.keys()).index(provider_key)
            )
            
            if selected_preset != "Custom":
                active_config["cloud"]["api_url"] = openai_providers[selected_preset]
                current_api_url = openai_providers[selected_preset]

            active_config["cloud"]["api_url"] = st.text_input("API 端點 (Endpoint)", value=current_api_url)
            active_config["cloud"]["model_name"] = st.text_input("模型名稱 (Model Name)", value=active_config["cloud"].get("model_name", "gpt-4o"))
            
        else:
            st.info("Gemini 模式將使用 Google Generative AI REST API。")
            
            gemini_presets = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-8b", "gemini-2.0-flash-exp", "Custom"]
            current_gemini_model = active_config["cloud"].get("model_name", "gemini-1.5-flash")
            
            model_key = "Custom"
            if current_gemini_model in gemini_presets:
                model_key = current_gemini_model
                
            selected_gemini_preset = st.selectbox(
                "選取 Gemini 模型", 
                gemini_presets, 
                index=gemini_presets.index(model_key)
            )
            
            if selected_gemini_preset != "Custom":
                active_config["cloud"]["model_name"] = selected_gemini_preset
                current_gemini_model = selected_gemini_preset

            active_config["cloud"]["model_name"] = st.text_input("模型名稱 (Model Name)", value=current_gemini_model)

        cloud_api_show = st.checkbox("顯示 Key", key="show_cloud_api")
        active_config["cloud"]["api_key"] = st.text_input(
            "API Key", 
            value=active_config["cloud"].get("api_key", ""), 
            type="default" if cloud_api_show else "password"
        )

        if selected_provider_type == "Gemini":
            if st.button("🔍 偵測目前填入 API Key 可用的模型"):
                if not active_config["cloud"]["api_key"] or "YOUR" in active_config["cloud"]["api_key"]:
                    st.error("請先填入您從 Google AI Studio 取得的 API Key！")
                else:
                    llm_service = LLMInterface(config_path="config.json")
                    available_models = llm_service.list_models(api_key=active_config["cloud"]["api_key"])
                    st.write(f"**可用模型清單：**\n{available_models}")

        st.divider()
        
        st.subheader("🔍 AI Detector 設定")
        
        detector_options = {
            "Hugging Face 神經網路 (推薦)": "hf_model",
            "GPTZero API (雲端)": "cloud",
            "本地落地模型 (Local LLM)": "local"
        }
        
        current_det_mode = active_config["ai_detector"].get("mode", "hf_model")
        det_index = 0
        for i, (ui_text, sys_code) in enumerate(detector_options.items()):
            if sys_code == current_det_mode:
                det_index = i
                break
        
        st.caption("偵測模式")
        selected_det_ui = st.radio(
            "偵測模式", 
            list(detector_options.keys()), 
            index=det_index, 
            label_visibility="collapsed"
        )
        
        active_config["ai_detector"]["mode"] = detector_options[selected_det_ui]

        if active_config["ai_detector"]["mode"] == "hf_model":
            st.info("Hugging Face 模式將不需聯網，直接使用 Desklib 神經網路模型處理 (效能與精準度最佳)。")
            active_config["ai_detector"]["force_cpu"] = st.checkbox(
                "強制使用 CPU 進行 AI 偵測", 
                value=active_config["ai_detector"].get("force_cpu", False)
            )
            
        elif active_config["ai_detector"]["mode"] == "cloud":
            detector_api_show = st.checkbox("顯示 GPTZero Key", key="show_detector_api")
            active_config["ai_detector"]["api_key"] = st.text_input(
                "GPTZero API Key", 
                value=active_config["ai_detector"].get("api_key", ""), 
                type="default" if detector_api_show else "password"
            )
            active_config["ai_detector"]["api_url"] = st.text_input(
                "GPTZero API Endpoint", 
                value=active_config["ai_detector"].get("api_url", "https://api.gptzero.me/v2/predict/text")
            )
            
        elif active_config["ai_detector"]["mode"] == "local":
            st.info("本地模式將優先使用右方「💻 本地 LLM 設定」中載入的模型。注意：大模型極度消耗運算資源，且請確保上下文窗口 (n_ctx) 足夠容納全文。")

    with col_b:
        st.subheader("💻 本地 LLM 設定 (GGUF)")
        
        if "model_path_widget" not in st.session_state:
            st.session_state.model_path_widget = active_config["local"].get("model_path", "")
            
        def on_browse_click():
            picked_path = select_file(st.session_state.model_path_widget)
            if picked_path:
                st.session_state.model_path_widget = picked_path

        st.caption("GGUF 模型檔案路徑")
        path_col1, path_col2 = st.columns([0.85, 0.15])
        
        with path_col1:
            model_path_input = st.text_input("GGUF 模型檔案路徑", key="model_path_widget", label_visibility="collapsed")
        with path_col2:
            if TK_AVAILABLE:
                st.button("📂", help="瀏覽檔案系統", on_click=on_browse_click)
            else:
                st.button("📂", disabled=True)

        active_config["local"]["model_path"] = model_path_input
        
        st.caption("上下文窗口大小 (n_ctx)")
        active_config["local"]["n_ctx"] = st.number_input(
            "上下文窗口大小 (n_ctx)", 
            value=active_config["local"].get("n_ctx", 4096), 
            step=1024,
            label_visibility="collapsed"
        )
        
        st.caption("最大輸出 Token")
        active_config["local"]["max_tokens"] = st.number_input(
            "最大輸出 Token", 
            value=active_config["local"].get("max_tokens", 1024), 
            step=256,
            label_visibility="collapsed"
        )
        
        current_gpu_setting = active_config["local"].get("n_gpu_layers", -1)
        use_gpu = st.checkbox("啟用 GPU 顯示卡加速 (若閃退請關閉)", value=(current_gpu_setting != 0))
        active_config["local"]["n_gpu_layers"] = -1 if use_gpu else 0

        st.divider()
        st.subheader("🦙 Ollama 伺服器設定")
        active_config["ollama"]["model_name"] = st.text_input(
            "Ollama 模型名稱 (例如: llama3.1)", 
            value=active_config["ollama"].get("model_name", "llama3.1")
        )
        active_config["ollama"]["base_url"] = st.text_input(
            "Ollama 伺服器網址", 
            value=active_config["ollama"].get("base_url", "http://localhost:11434")
        )

    st.divider()
    if st.button("💾 儲存並套用設定", type="primary"):
        save_global_config(active_config)
        # 強制更新快取標記
        st.session_state.force_reload_config = True
        st.success("設定檔已成功更新！系統已同步最新參數。")
        st.rerun()


# ===========================================================
# 分頁 2：推論主畫面 (論文分析與審查)
# ===========================================================

else:
    col_t1, col_t2 = st.columns([0.8, 0.2])
    with col_t1:
        st.title("🎓 多代理人 AI 論文審查系統")
    with col_t2:
        with st.popover("⚙️ 當前參數 (除錯專用)"):
            st.json(active_config)
            
            st.divider()
            st.markdown("**🐞 系統內部狀態**")
            st.text(f"AI 偵測器模式: {active_config.get('ai_detector', {}).get('mode', '未知')}")
            st.text(f"LLM 推論模式: {active_config.get('llm_mode', '未知')}")

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
                if llm_mode == "cloud": 
                    provider = active_config.get("cloud", {}).get("provider", "openai")
                    llm_hw_status = "☁️ Cloud API (Gemini)" if provider == "gemini" else "☁️ Cloud API (OpenAI 相容)"
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

    # ==========================================
    # 🚨 智慧防呆：偵測本地模型路徑錯誤
    # ==========================================
    current_llm_mode = active_config.get("llm_mode", "mock")
    current_det_mode = active_config.get("ai_detector", {}).get("mode", "hf_model")
    current_model_path = active_config.get("local", {}).get("model_path", "")
    
    if (current_llm_mode == "local" or current_det_mode == "local") and not check_model_exists(current_model_path):
        st.error(
            f"🚨 **嚴重路徑錯誤**：系統偵測到您的本地 GGUF 模型路徑無效或不存在！\n\n"
            f"目前系統讀取到的路徑為：`{current_model_path}`\n\n"
            f"👉 **解決方案**：請至左側切換為「⚙️ 管理員 (參數設定)」，"
            f"將路徑修改為 Linux 伺服器上的絕對路徑（例如：`/home/wangs/PaperReviewMultiSystem/local_models/...`）並儲存，否則本地推論將被強制降級為模擬模式。"
        )

    # ==========================================
    # 1. 論文資料設定
    # ==========================================
    st.header("📄 1. 論文資料設定")
    
    col_title, col_field = st.columns(2)
    with col_title:
        paper_title = st.text_input("論文標題", placeholder="請輸入論文標題...")
    with col_field:
        paper_field = st.text_input("主題領域", placeholder="例如：電腦視覺、人工智慧")

    paper_content = ""
    
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
        
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.ai_report = None
            st.session_state.last_uploaded_file = uploaded_file.name
            
    paper_content_input = st.text_area("或直接貼上/編輯論文內容", value=paper_content, height=200)
    raw_paper_content = paper_content_input


    # ==========================================
    # 1.5 內容預處理與範圍界定
    # ==========================================
    st.header("✂️ 1.5 內容預處理與清洗")
    
    total_words = len(raw_paper_content)
    filtered_text = raw_paper_content

    col_set1, col_set2 = st.columns([0.5, 0.5])
    
    with col_set1:
        exclude_quotes = st.checkbox("排除引用文字", value=True)
        
        if st.button("套用參考文獻截斷"):
            st.session_state.exclusion_keywords = "Bibliography\nReferences\nAppendix\n參考文獻\n附錄"
            st.rerun()
            
        exclusion_keywords = st.text_area(
            "截斷關鍵字", 
            value=st.session_state.exclusion_keywords,
            height=100
        )
        st.session_state.exclusion_keywords = exclusion_keywords
        
        st.markdown("##### 📝 手動新增欲排除的特定字串")
        
        c_m1, c_m2 = st.columns([0.7, 0.3])
        with c_m1:
            manual_exclude_input = st.text_input("貼入欲排除句子：", key="manual_exclude_input", label_visibility="collapsed")
        with c_m2:
            if st.button("加入排除"):
                if manual_exclude_input and manual_exclude_input not in st.session_state.manual_exclusions:
                    st.session_state.manual_exclusions.append(manual_exclude_input)
                    st.rerun()
                    
        if st.session_state.manual_exclusions:
            for i, me in enumerate(st.session_state.manual_exclusions):
                c_i1, c_i2 = st.columns([0.9, 0.1])
                c_i1.code(me[:50] + "..." if len(me) > 50 else me) 
                if c_i2.button("🗑️", key=f"del_me_{i}"):
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
    
    current_det_mode = active_config.get("ai_detector", {}).get("mode", "hf_model")
    
    if current_det_mode == "hf_model" and is_admin:
        st.caption("💡 提示：Hugging Face 模式初次執行時，會從網路下載神經網路模型至伺服器。如果您的 Linux 網路受阻或連線逾時，系統會自動啟動保護機制，降級為本地模擬模式。")
    
    col_btn1, col_btn2 = st.columns([0.2, 0.8])
    with col_btn1:
        execute_btn = st.button("執行分析", icon="🔎", type="primary")
    with col_btn2:
        clear_btn = st.button("🧼 清除分析結果")
        
    if clear_btn:
        st.session_state.ai_report = None
        st.rerun()

    if execute_btn:
        if not final_paper_content_for_llm.strip():
            st.warning("請先輸入或上傳論文內容！")
        else:
            llm_service = LLMInterface(config_path=active_config_path)
            detector = AIDetector(config_path=active_config_path)
            with st.spinner("AI 偵測分析中 (若是初次運行 Hugging Face 模型，可能需要較長下載時間，請耐心等候)..."):
                 report = detector.analyze(final_paper_content_for_llm, llm_interface=llm_service)
                 st.session_state.ai_report = report
                 st.success("分析完成！")

    if st.session_state.ai_report:
        report = st.session_state.ai_report
        
        if "模擬" in report.get('model_name', ''):
            st.error(
                "🚨 **系統已觸發底層安全降級機制 (Fail-safe)** 🚨\n\n"
                "系統無法啟動您指定的 AI 偵測引擎，為避免程式崩潰，已自動連鎖降級為模擬模式。\n\n"
                "**🔍 常見除錯方向 (Linux 環境)：**\n"
                "1. **Hugging Face 失敗**：伺服器可能無法連上 Hugging Face，或 `~/.cache/huggingface` 目錄無權限寫入。\n"
                "2. **本地模型 (GGUF) 失敗**：伺服器實體記憶體 (RAM) 不足，或 Linux 未安裝 `gcc`/`g++` 導致 `llama-cpp` 無法啟動。\n"
                "3. **雲端 API 失敗**：未填寫有效的 GPTZero API Key。\n\n"
                "👉 **建議**：請查看伺服器終端機 (Terminal) 的詳細報錯日誌以鎖定具體原因。"
            )
            
        if "notice" in report:
            st.warning(f"⚠️ 注意：{report['notice']}")
            
        st.success(f"🤖 **推論模型：** {report.get('model_name', '未知模型')}")
        st.subheader(f"📊 偵測報告 (AI 比例：{report['ai_ratio']}%)")
        
        if report.get("summary"):
            st.info(f"📝 **分析摘要：** {report['summary']}")

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

        st.divider()

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            json_report = json.dumps(report, indent=4, ensure_ascii=False)
            st.download_button(
                label="📥 匯出完整 JSON 報告",
                data=json_report,
                file_name=f"AI_Detector_Report_{paper_title if paper_title else 'Untitled'}.json",
                mime="application/json",
                key="json_download_btn"
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
                key="md_download_btn"
            )


    # ==========================================
    # 3. 審查委員配置
    # ==========================================
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


    # ==========================================
    # 4. 執行多代理人審查
    # ==========================================
    st.header("🚀 4. 執行多代理人學術審查")

    async def run_review_process():
        if not paper_title or not final_paper_content_for_llm.strip():
            st.error("請提供論文標題與內容")
            return

        my_paper = Paper(title=paper_title, field=paper_field, content=final_paper_content_for_llm)
        llm_service = LLMInterface(config_path=active_config_path)
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
        
        # 安全取值，防止 JSON 崩潰
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
    # 5. 結果展示與匯出
    # ==========================================
    if st.session_state.review_history:
        st.divider()
        st.header("📋 審查結果展示")
        
        if st.session_state.review_stats:
            stats = st.session_state.review_stats
            c1, c2, c3 = st.columns(3)
            c1.metric("總體貢獻度", f"{stats.get('avg_contribution', 0)}/10")
            c2.metric("主要缺陷密度", f"{stats.get('avg_deficiencies', 0)}/10", delta_color="inverse")
            c3.metric("方法論強健度", f"{stats.get('avg_robustness', 0)}/10")
            st.write("")
        
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

        for rnd_id, rnd_name in [("round_1", "第一輪意見 (獨立)"), ("round_2", "第二輪辯論"), ("round_3", "最終裁決")]:
            with st.expander(rnd_name, expanded=(rnd_id == "round_3")):
                for name, content in st.session_state.review_history[rnd_id].items():
                    st.markdown(f"**{name}：**")
                    st.write(content)
                    st.divider()