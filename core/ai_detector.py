# core/ai_detector.py

import re
import json
import requests
import asyncio
import warnings

# --- 引入 Hugging Face 與 PyTorch 依賴 ---
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel

# 抑制一些 transformers 無關緊要的警告
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# 定義 Desklib AI 偵測專用模型架構
# ==========================================
class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig
    
    # 修正 transformers 新版本 AttributeError ('all_tied_weights_keys') 問題
    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config):
        super().__init__(config)
        # 初始化底層 Transformer 模型
        self.model = AutoModel.from_config(config)
        # 定義分類層 (輸出單一 Logit)
        self.classifier = nn.Linear(config.hidden_size, 1)
        # 初始化權重
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        
        # Mean pooling (平均池化) - 修正精度不匹配問題
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(last_hidden_state.dtype)
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # 通過分類器
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output

# ==========================================
# 主要偵測器類別
# ==========================================
class AIDetector:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        # 預設模式改為 'hf_model' 以追求最高效能
        self.mode = self.config.get("ai_detector", {}).get("mode", "hf_model")
        self.api_key = self.config.get("ai_detector", {}).get("api_key", "")
        self.api_url = self.config.get("ai_detector", {}).get("api_url", "https://api.gptzero.me/v2/predict/text")
        
        # 如果是 HF 模式，初始化載入模型與分詞器 (避免每次分析重複載入)
        self.hf_model = None
        self.tokenizer = None
        self.device = None
        if self.mode == "hf_model":
            self._init_hf_model()

    def _load_config(self, path: str) -> dict:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _init_hf_model(self):
        """初始化 Hugging Face 本地神經網路模型"""
        print("[System] 正在載入 Desklib AI 偵測神經網路模型...")
        model_directory = "desklib/ai-text-detector-v1.01"
        
        # --- 診斷與強制作為 ---
        print(f"[System] PyTorch 版本: {torch.__version__}")
        print(f"[System] PyTorch 是否編譯了 CUDA 支援: {torch.version.cuda}")
        print(f"[System] 系統是否偵測到可用的 GPU: {torch.cuda.is_available()}")
        
        force_cpu = self.config.get("ai_detector", {}).get("force_cpu", False)
        
        if force_cpu:
            print("[System] 使用者於設定中勾選了強制使用 CPU。")
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0") # 明確指定第一張顯卡
            # 清除 GPU 快取，確保載入模型時有乾淨的空間
            torch.cuda.empty_cache()
        else:
            print("[System] 警告：無法偵測到可用的 GPU，將退回使用 CPU。請檢查 PyTorch 安裝版本。")
            self.device = torch.device("cpu")
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_directory)
        # 加上 torch_dtype 參數，確保在 GPU 上使用更節省記憶體的精度
        self.hf_model = DesklibAIDetectionModel.from_pretrained(model_directory, torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32).to(self.device)
        self.hf_model.eval() # 設置為評估模式
        print(f"[System] 模型載入完成！使用運算裝置: {self.device}")

    @property
    def hardware_info(self) -> str:
        """ 返回目前使用的推論硬體 """
        if self.mode == "cloud":
            return "☁️ Cloud API (GPTZero)"
        if self.mode == "hf_model" and self.device:
            dev_name = "GPU (CUDA)" if "cuda" in str(self.device).lower() else "CPU"
            return f"💻 Local {dev_name}"
        if self.mode == "local":
            return "💻 Local LLM Shared"
        return "❓ 未知"

    def analyze(self, text: str, llm_interface=None) -> dict:
        """
        多重模式偵測架構：依據設定檔切換推論引擎
        """
        if not text.strip():
            return {"ai_ratio": 0.0, "segments": [], "summary": "無有效輸入文本。"}
        
        if self.mode == "hf_model":
            return self._hf_analyze(text)
        elif self.mode == "local" and llm_interface:
            return self._local_analyze(text, llm_interface)
        else:
            return self._cloud_analyze(text)

    # ==========================================
    # 模式 1: Hugging Face 神經網路精準推論 (最高效能)
    # ==========================================
    def _hf_analyze(self, text: str) -> dict:
        """使用 Desklib 神經網路模型進行整體與逐句評估"""
        if not self.hf_model or not self.tokenizer:
            return self._mock_analyze(text, "Hugging Face 模型未正確載入。")

        try:
            # 1. 評估整篇文章的總 AI 機率 (Max length 768)
            overall_prob = self._predict_single_text(text, max_len=768)
            
            # 2. 為了維持 UI 體驗，將文章拆分逐句評估 (產生熱力圖)
            sentences = re.split(r'(?<=[。！？.!?\n])', text)
            segments = []
            
            for sent in sentences:
                if not sent.strip():
                    continue
                    
                # 逐句預測機率 (句子較短，max_len 可設小一點加快速度)
                sent_prob = self._predict_single_text(sent, max_len=128)
                
                # 依照機率決定顏色
                color = "transparent"
                sent_type = "Human"
                if sent_prob > 0.8:
                    color = "#ff9999" # 高度確信 AI
                    sent_type = "AI"
                elif sent_prob > 0.5:
                    color = "#ffcccc" # 疑似 AI
                    sent_type = "AI"
                    
                segments.append({
                    "text": sent,
                    "type": sent_type,
                    "color": color,
                    "reason": f"神經網路判定機率：{sent_prob*100:.1f}%"
                })

            return {
                "ai_ratio": round(overall_prob * 100, 2),
                "model_name": "Desklib AI Text Detector v1.01 (Neural Network)",
                "task_type": "Neural Network Analysis",
                "summary": f"由 Desklib 專用神經網路模型進行精準量化分析。整體 AI 生成機率為 {overall_prob*100:.1f}%。",
                "segments": segments
            }

        except Exception as e:
            return self._mock_analyze(text, f"HF 模型推論失敗：{str(e)}")

    def _predict_single_text(self, text: str, max_len: int = 768) -> float:
        """核心推論函數：將文本轉換為張量並通過模型獲取機率"""
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.hf_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            probability = torch.sigmoid(logits).item()
            
        return probability

    # ==========================================
    # 模式 2: Local LLM 語義推論 (保留作為備援/風格分析)
    # ==========================================
    def _local_analyze(self, text: str, llm_interface) -> dict:
        system_prompt = (
            "你是一個高級 AI 文本偵測引擎。請對下方文字進行 AI 寫作判定。\n"
            "你必須「嚴格」且「僅」以 JSON 格式輸出，不要包含任何開場白或解釋。\n"
            "所需的 JSON 格式如下：\n"
            "{\n"
            "  \"ai_ratio\": (數字，0到100，表示整體 AI 生成的機率),\n"
            "  \"summary\": \"(一段簡短的繁體中文分析結論)\",\n"
            "  \"segments\": [\n"
            "    {\"text\": \"(原文句子)\", \"type\": \"(填寫 AI 或 Human)\", \"color\": \"(如果是 AI 填 #ffcccc，如果是 Human 填 transparent)\"}\n"
            "  ]\n"
            "}"
        )
        user_prompt = f"分析目標文本：\n\n{text}"
        
        try:
            response_text = asyncio.run(llm_interface.generate_response(system_prompt, user_prompt))
            
            # 如果回傳的是錯誤訊息，直接呈現給使用者
            if response_text.startswith("【"):
                return self._mock_analyze(text, response_text)

            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    # 確保必要欄位存在，防止 app.py KeyError
                    data.setdefault("ai_ratio", 0.0)
                    data.setdefault("segments", [])
                    data.setdefault("summary", "本地模型成功完成分析，但未提供摘要。")
                    data["model_name"] = self.config.get("local", {}).get("model_path", "Local LLM").split("/")[-1].split("\\")[-1]
                    return data
                except json.JSONDecodeError:
                    return self._mock_analyze(text, "模型輸出了不合法的 JSON。")
            return self._mock_analyze(text, f"模型解析失敗：無法找到有效 JSON 資料。模型回傳為：{response_text[:50]}...")
        except Exception as e:
            return self._mock_analyze(text, f"本地偵測異常：{str(e)}")

    # ==========================================
    # 模式 3: Cloud API (GPTZero)
    # ==========================================
    def _cloud_analyze(self, text: str) -> dict:
        if not self.api_key: return self._mock_analyze(text, "未設定 API Key")
        try:
            headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
            res = requests.post(self.api_url, headers=headers, json={"document": text}, timeout=15)
            res.raise_for_status()
            doc = res.json().get("documents", [{}])[0]
            
            segments = []
            for sent in doc.get("sentences", []):
                p = sent.get("generated_prob", 0)
                segments.append({
                    "text": sent.get("sentence", ""),
                    "type": "AI" if p > 0.7 else "Human",
                    "color": "#ffcccc" if p > 0.7 else "transparent"
                })
            return {
                "ai_ratio": round(doc.get("completely_generated_prob", 0) * 100, 2), 
                "segments": segments,
                "model_name": "GPTZero API (Cloud)"
            }
        except Exception as e:
            return self._mock_analyze(text, f"Cloud 服務異常：{str(e)}")

    def _mock_analyze(self, text: str, reason: str) -> dict:
        return {"ai_ratio": 0.0, "segments": [{"text": text, "type": "Error", "color": "transparent"}], "notice": reason, "model_name": "Mock Detection"}