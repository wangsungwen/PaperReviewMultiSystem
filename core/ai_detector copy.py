# core/ai_detector.py

import re
import json
import requests

class AIDetector:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.mode = self.config.get("ai_detector", {}).get("mode", "cloud")
        self.api_key = self.config.get("ai_detector", {}).get("api_key", "")
        self.api_url = self.config.get("ai_detector", {}).get("api_url", "https://api.gptzero.me/v2/predict/text")

    def _load_config(self, path: str) -> dict:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}

    def analyze(self, text: str, llm_interface=None) -> dict:
        """
        分析文本的 AI 寫作比例。支援 GPTZero API (cloud) 或 本地 LLM (local) 模式。
        """
        if not text.strip():
            return {"ai_ratio": 0.0, "segments": []}
        
        # 根據模式切換
        if self.mode == "local" and llm_interface:
            return self._local_analyze(text, llm_interface)

        # Cloud 模式 (GPTZero)
        if not self.api_key or "YOUR_DETECTOR_API_KEY_HERE" in self.api_key:
            return self._mock_analyze(text, "請在「⚙️ 參數設定」中填入有效的 GPTZero API Key (或切換至本地模式)。")

        try:
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            payload = {"document": text}
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # 解析 GPTZero V2 回傳結構
            doc = data.get("documents", [{}])[0]
            ai_ratio = doc.get("completely_generated_prob", 0) * 100
            gpt_sentences = doc.get("sentences", [])
            
            segments = []
            if gpt_sentences:
                for sent in gpt_sentences:
                    prob = sent.get("generated_prob", 0)
                    text_sent = sent.get("sentence", "")
                    
                    if prob > 0.7:
                        color = "#ff9999" if prob > 0.9 else "#ffcccc"
                        segments.append({"text": text_sent, "type": "AI", "color": color})
                    else:
                        segments.append({"text": text_sent, "type": "Human", "color": "transparent"})
            else:
                segments.append({"text": text, "type": "Mixed", "color": "transparent"})

            return {
                "ai_ratio": round(ai_ratio, 2),
                "segments": segments
            }

        except requests.exceptions.RequestException as e:
            return self._mock_analyze(text, f"API 連線或請求失敗：{str(e)} (切換回模擬模式)")
        except Exception as e:
            return self._mock_analyze(text, f"偵測過程發生未知錯誤：{str(e)} (切換回模擬模式)")

    def _local_analyze(self, text: str, llm_interface) -> dict:
        """
        使用本地 LLM 進行偵測。
        """
        import asyncio
        
        system_prompt = (
            "你是一個專門偵測 AI 生成內容的助手。請分析使用者提供的文本，並決定其為 AI 生成的可能性。\n"
            "請嚴格按照以下 JSON 格式回傳，不要有其他解釋文字：\n"
            "{\n"
            '  "ai_ratio": 數字 (0-100),\n'
            '  "segments": [\n'
            '    {"text": "句子內容", "prob": 0-1 之間的浮點數}\n'
            '  ]\n'
            "}"
        )
        
        user_prompt = f"請分析以下文本：\n\n{text}"
        
        try:
            # 調用 LLMInterface 的非同步方法 (在同步環境中需注意)
            response_text = asyncio.run(llm_interface.generate_response(system_prompt, user_prompt))
            
            # 嘗試解析 JSON
            # 輔助：提取 JSON 區塊
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                ai_ratio = data.get("ai_ratio", 0)
                raw_segments = data.get("segments", [])
                
                segments = []
                for item in raw_segments:
                    p = item.get("prob", 0)
                    txt = item.get("text", "")
                    if p > 0.7:
                        color = "#ff9999" if p > 0.9 else "#ffcccc"
                        segments.append({"text": txt, "type": "AI", "color": color})
                    else:
                        segments.append({"text": txt, "type": "Human", "color": "transparent"})
                
                return {
                    "ai_ratio": ai_ratio,
                    "segments": segments
                }
            else:
                return self._mock_analyze(text, f"本地模型回傳格式不正確：{response_text[:200]}...")
        except Exception as e:
            return self._mock_analyze(text, f"本地模型偵測失敗：{str(e)}")

    def _mock_analyze(self, text: str, reason: str) -> dict:
        # 保留原有的模擬邏輯作為備援或未設定 Key 時的呈現
        sentences = re.split(r'(?<=[。！？.!?\n])', text)
        segments = []
        for sent in sentences:
            if not sent.strip():
                segments.append({"text": sent, "type": "Human", "color": "transparent"})
                continue
            import random
            ai_probability = random.random()
            if ai_probability > 0.8:
                color = "#ff9999" if ai_probability > 0.95 else "#ffcccc"
                segments.append({"text": sent, "type": "AI", "color": color})
            else:
                segments.append({"text": sent, "type": "Human", "color": "transparent"})
        
        return {
            "ai_ratio": 10.0, # 固定一個模擬數值代表模式
            "segments": segments,
            "notice": reason
        }