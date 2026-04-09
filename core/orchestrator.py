# core/orchestrator.py

import asyncio
from typing import List, Dict
from models.paper import Paper
from models.reviewer import ReviewerAgent
from llm.interface import LLMInterface

class PaperReviewOrchestrator:
    def __init__(self, paper: Paper, reviewers: List[ReviewerAgent], llm: LLMInterface):
        self.paper = paper
        self.reviewers = reviewers
        self.llm = llm
        self.history = {
            "round_1": {},
            "round_2": {},
            "round_3": {}
        }

    async def run_round_1(self) -> Dict[str, str]:
        user_prompt = (
            f"請審查以下論文。\n標題：{self.paper.title}\n內容：{self.paper.content}\n"
            f"請給出初步審查意見，包含：1. 論文貢獻評估 2. 主要缺陷 3. 研究方法檢驗。"
        )
        
        # 改為完全序列執行並增加間隔，以應對嚴格的 TPM (Tokens Per Minute) 限制
        for reviewer in self.reviewers:
            response = await self.llm.generate_response(reviewer.get_system_prompt(), user_prompt)
            self.history["round_1"][reviewer.name] = response
            await asyncio.sleep(2) # 每個審稿人間隔 2 秒
            
        return self.history["round_1"]

    async def run_round_2(self) -> Dict[str, str]:
        r1_summary = "\n".join([f"[{name}] 的意見:\n{opinion}" for name, opinion in self.history["round_1"].items()])
        user_prompt = (
            f"進行第二輪審查。以下是第一輪的意見彙整：\n{r1_summary}\n\n"
            f"請根據你的立場對其他人進行點評，可以認同、反駁或針對矛盾之處提出新疑問。"
        )
        
        for reviewer in self.reviewers:
            response = await self.llm.generate_response(reviewer.get_system_prompt(), user_prompt)
            self.history["round_2"][reviewer.name] = response
            await asyncio.sleep(2)
            
        return self.history["round_2"]

    async def run_round_3(self) -> Dict[str, str]:
        r2_summary = "\n".join([f"[{name}] 在第二輪的討論:\n{opinion}" for name, opinion in self.history["round_2"].items()])
        user_prompt = (
            f"最後一輪審查。以下是前一輪的討論紀錄：\n{r2_summary}\n\n"
            f"請給出：1. 給作者的最終修改清單 2. 最終裁決結論 (Accept / Minor Revision / Major Revision / Reject)，並附上理由。"
        )
        
        for reviewer in self.reviewers:
            response = await self.llm.generate_response(reviewer.get_system_prompt(), user_prompt)
            self.history["round_3"][reviewer.name] = response
            await asyncio.sleep(2)
            
        return self.history["round_3"]