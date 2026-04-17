# core/orchestrator.py

import asyncio
import json
import re

class PaperReviewOrchestrator:
    def __init__(self, paper, reviewers, llm):
        self.paper = paper
        self.reviewers = reviewers
        self.llm = llm
        
        # 【關鍵修復】：在物件建立的瞬間，立刻初始化這些屬性
        # 這樣就算後續模型斷線或降級，app.py 也絕對抓得到預設值 0.0，不會報錯！
        self.history = {"round_1": {}, "round_2": {}, "round_3": {}}
        self.review_stats = {
            "avg_contribution": 0.0,
            "avg_deficiencies": 0.0,
            "avg_robustness": 0.0
        }

    async def run_round_1(self):
        """第一輪：獨立審查"""
        tasks = []
        for reviewer in self.reviewers:
            system_prompt = (
                f"You are {reviewer.name}, an expert in {reviewer.expertise}. "
                f"Your style is: {reviewer.style}. "
                "Please review the following paper."
            )
            # 截斷過長文本以確保安全
            user_prompt = f"Title: {self.paper.title}\nContent:\n{self.paper.content[:5000]}..." 
            tasks.append(self.llm.generate_response(system_prompt, user_prompt))

        responses = await asyncio.gather(*tasks)
        for idx, reviewer in enumerate(self.reviewers):
            self.history["round_1"][reviewer.name] = responses[idx]
        
        return self.history["round_1"]

    async def run_round_2(self):
        """第二輪：交叉辯論"""
        tasks = []
        for reviewer in self.reviewers:
            # 整理其他委員的意見
            other_reviews = "\n\n".join([
                f"{r.name}'s review:\n{self.history['round_1'][r.name]}" 
                for r in self.reviewers if r != reviewer
            ])
            
            system_prompt = (
                f"You are {reviewer.name}. Read the reviews of your colleagues and provide your rebuttal or agreement. "
                f"Maintain your persona: {reviewer.expertise}, {reviewer.style}."
            )
            user_prompt = f"Colleague Reviews:\n{other_reviews}\n\nYour rebuttal:"
            tasks.append(self.llm.generate_response(system_prompt, user_prompt))

        responses = await asyncio.gather(*tasks)
        for idx, reviewer in enumerate(self.reviewers):
            self.history["round_2"][reviewer.name] = responses[idx]
            
        return self.history["round_2"]

    async def run_round_3(self):
        """第三輪：最終共識與評分 (強制要求 JSON)"""
        # 指定第一位委員作為主席 (Chair)
        chair = self.reviewers[0]
        
        all_prior_context = "Round 1:\n" + "\n".join([f"{k}: {v}" for k, v in self.history["round_1"].items()]) + \
                            "\n\nRound 2:\n" + "\n".join([f"{k}: {v}" for k, v in self.history["round_2"].items()])

        system_prompt = (
            f"You are {chair.name}, acting as the Area Chair. "
            "Synthesize the debate and output the final decision STRICTLY as a JSON object. "
            "Do not include Markdown blocks like ```json. Just output the raw JSON.\n\n"
            "Format:\n"
            "{\n"
            '  "summary": "String summarizing the final decision",\n'
            '  "avg_contribution": float (0.0 to 10.0),\n'
            '  "avg_deficiencies": float (0.0 to 10.0),\n'
            '  "avg_robustness": float (0.0 to 10.0)\n'
            "}"
        )
        user_prompt = f"Debate History:\n{all_prior_context}\n\nProvide the final JSON verdict:"
        
        response = await self.llm.generate_response(system_prompt, user_prompt)
        
        # 預設先將原始文字存入歷史紀錄
        self.history["round_3"]["Final Verdict"] = response

        # 強健的 JSON 解析機制
        try:
            # 嘗試擷取大括號內的 JSON 內容 (防止 LLM 雞婆加上 Markdown 標記)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            json_str = json_match.group() if json_match else response
            
            parsed_data = json.loads(json_str)
            
            # 安全提取數據，若缺少則給予預設值 0.0
            self.review_stats["avg_contribution"] = float(parsed_data.get("avg_contribution", 0.0))
            self.review_stats["avg_deficiencies"] = float(parsed_data.get("avg_deficiencies", 0.0))
            self.review_stats["avg_robustness"] = float(parsed_data.get("avg_robustness", 0.0))
            
            # 將 UI 顯示的結論替換為 JSON 中的 summary
            self.history["round_3"]["Final Verdict"] = parsed_data.get("summary", "無法從 JSON 中提取 summary。")

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # 若模型輸出非 JSON 格式 (例如降級為模擬模式時)，保留預設的 0.0 分並顯示錯誤提示
            self.history["round_3"]["Final Verdict"] = f"⚠️ 模型未輸出標準 JSON 格式，無法解析評分。\n\n**原始輸出內容：**\n{response}"

        return self.history["round_3"]