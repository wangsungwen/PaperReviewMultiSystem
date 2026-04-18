# core/ai_detector.py

import re
import json
import requests
import asyncio
import warnings
import random
import os
import sys

# --- 引入 Hugging Face 與 PyTorch 依賴 ---
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel

# 抑制一些 transformers 無關緊要的警告
warnings.filterwarnings("ignore", category=FutureWarning)

def resource_path(relative_path):
    """ 取得相對於執行路徑的絕對路徑 """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ==========================================
# 定義 Desklib AI 偵測專用模型架構
# ==========================================
class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig
    
    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        
        # Mean pooling (平均池化)
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
        if not os.path.exists(config_path):
            config_path = resource_path(config_path)
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception:
            self.config = {}

        # 取得設定檔中的 mode (預設為 hf_model)
        raw_mode = self.config.get("ai_detector", {}).get("mode", "hf_model")
        
        # 建立相容性映射：確保舊版中文與新版系統代碼都能被正確解析
        if "Hugging Face" in raw_mode or raw_mode == "hf_model":
            self.mode = "hf_model"
        elif "GPTZero" in raw_mode or raw_mode == "cloud":
            self.mode = "cloud"
        elif "本地" in raw_mode or "Local" in raw_mode or raw_mode == "local":
            self.mode = "local"
        else:
            self.mode = "hf_model" # 最終防呆預設值

        self.api_key = self.config.get("ai_detector", {}).get("api_key", "")
        self.api_url = self.config.get("ai_detector", {}).get("api_url", "https://api.gptzero.me/v2/predict/text")
        
        self.hf_model = None
        self.tokenizer = None
        self.device = None
        
        if self.mode == "hf_model":
            self._init_hf_model()

    def _init_hf_model(self):
        """初始化 Hugging Face 本地神經網路模型"""
        model_directory = resource_path("desklib/ai-text-detector-v1.01")
        
        if not os.path.exists(model_directory):
            return

        try:
            force_cpu = self.config.get("ai_detector", {}).get("force_cpu", False)
            
            if force_cpu or not torch.cuda.is_available():
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda:0")
                torch.cuda.empty_cache()
                
            self.tokenizer = AutoTokenizer.from_pretrained(model_directory)
            self.hf_model = DesklibAIDetectionModel.from_pretrained(
                model_directory, 
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            ).to(self.device)
            self.hf_model.eval()
        except Exception as e:
            self.hf_model = None

    @property
    def hardware_info(self) -> str:
        """ 返回目前使用的推論硬體 """
        if self.mode == "cloud":
            return "☁️ Cloud API (GPTZero)"
        if self.mode == "hf_model":
            if self.device and self.device.type == 'cuda':
                return "💻 Local GPU (CUDA) - Desklib"
            else:
                return "💻 Local CPU - Desklib"
        if self.mode == "local":
            return "💻 Local LLM Shared"
        return "🛠️ Mock Engine"

    def analyze(self, text: str, llm_interface=None) -> dict:
        """多重模式偵測架構"""
        if not text.strip():
            return {"ai_ratio": 0.0, "segments": [], "summary": "無有效輸入文本。", "model_name": "N/A"}
        
        # 使用精確的系統代碼進行判斷
        if self.mode == "hf_model":
            return self._hf_analyze(text)
        elif self.mode == "local" and llm_interface:
            return self._local_analyze(text, llm_interface)
        else:
            return self._cloud_analyze(text)

    def _hf_analyze(self, text: str) -> dict:
        """使用 Desklib 神經網路模型進行整體與逐句評估"""
        if not self.hf_model or not self.tokenizer:
            return self._mock_analyze(text, "Desklib 模型尚未下載或載入失敗，已切換為本地特徵模擬引擎。", "Desklib (降級模擬)")

        try:
            overall_prob = self._predict_single_text(text, max_len=768)
            
            sentences = re.split(r'(?<=[。！？.!?\n])\s*', text)
            segments = []
            
            for sent in sentences:
                if not sent.strip():
                    if sent:
                        segments.append({"text": sent, "type": "Human", "color": "transparent"})
                    continue
                    
                sent_prob = self._predict_single_text(sent, max_len=128)
                
                color = "transparent"
                sent_type = "Human"
                if sent_prob > 0.8:
                    color = "#ff9999"
                    sent_type = "AI"
                elif sent_prob > 0.5:
                    color = "#ffcccc"
                    sent_type = "AI"
                    
                segments.append({
                    "text": sent,
                    "type": sent_type,
                    "color": color,
                    "reason": f"神經網路判定機率：{sent_prob*100:.1f}%"
                })

            return {
                "ai_ratio": round(overall_prob * 100, 2),
                "model_name": "Desklib AI Text Detector v1.01",
                "task_type": "Neural Network Analysis",
                "summary": f"由 Desklib 專用神經網路模型進行精準量化分析。整體 AI 生成機率為 {overall_prob*100:.1f}%。",
                "segments": segments
            }

        except Exception as e:
            return self._mock_analyze(text, f"HF 模型推論失敗：{str(e)}", "Desklib (降級模擬)")

    def _predict_single_text(self, text: str, max_len: int = 768) -> float:
        """核心推論函數"""
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

    def _local_analyze(self, text: str, llm_interface) -> dict:
        return self._mock_analyze(text, "本地模型分析尚未配置完整，暫使用特徵模擬。", "Local LLM (模擬)")

    def _cloud_analyze(self, text: str) -> dict:
        return self._mock_analyze(text, "未設定有效的 GPTZero 金鑰。", "GPTZero (模擬)")

    def _mock_analyze(self, text: str, reason: str, model_name: str) -> dict:
        """備援引擎：種子特徵模擬"""
        sentences = re.split(r'(?<=[。！？.!?\n])\s*', text)
        segments = []
        ai_char_count = 0
        total_char_count = len(text.replace(" ", "").replace("\n", ""))

        for sent in sentences:
            if not sent.strip():
                if sent:
                    segments.append({"text": sent, "type": "Human", "color": "transparent"})
                continue
                
            seed_val = sum(ord(c) for c in sent.strip())
            random.seed(seed_val)
            ai_probability = random.random()
            
            if ai_probability > 0.7:  
                char_len = len(sent.replace(" ", "").replace("\n", ""))
                ai_char_count += char_len
                color = "#ff9999" if ai_probability > 0.85 else "#ffcccc"
                segments.append({"text": sent, "type": "AI", "color": color})
            else:
                segments.append({"text": sent, "type": "Human", "color": "transparent"})

        ai_ratio = (ai_char_count / total_char_count) * 100 if total_char_count > 0 else 0.0

        return {
            "ai_ratio": round(ai_ratio, 2),
            "segments": segments,
            "notice": reason,
            "model_name": model_name
        }