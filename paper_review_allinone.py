import time
from typing import List, Dict

# ==========================================
# 1. 資料結構與模型層
# ==========================================
class Paper:
    def __init__(self, title: str, field: str, content: str):
        self.title = title
        self.field = field
        self.content = content

class ReviewerAgent:
    def __init__(self, name: str, expertise: str, research_focus: str, style: str):
        self.name = name
        self.expertise = expertise
        self.research_focus = research_focus
        self.style = style

    def get_system_prompt(self) -> str:
        return (
            f"你現在扮演一位頂尖的學術論文審查委員。你的名字是 {self.name}。\n"
            f"你的專業領域是：{self.expertise}。\n"
            f"你的研究重心在於：{self.research_focus}。\n"
            f"你的學術審查風格是：{self.style}。\n"
            f"請務必保持這個角色設定，在接下來的所有審查與討論中，展現出你的專業與獨特視角。"
        )

# ==========================================
# 2. 模型介接層 (LLM Interface)
# ==========================================
class LLMInterface:
    def __init__(self, api_key: str = "YOUR_API_KEY"):
        self.api_key = api_key
        # 在此處初始化您的 LLM 客戶端，例如：
        # import google.generativeai as genai
        # genai.configure(api_key=self.api_key)
        # self.model = genai.GenerativeModel('gemini-pro')

    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        將系統提示詞與使用者請求發送給 LLM。
        (此處為模擬回應，請替換為真實的 API 呼叫)
        """
        # 範例真實 API 呼叫邏輯 (Gemini):
        # response = self.model.generate_content(f"System: {system_prompt}\n\nUser: {user_prompt}")
        # return response.text
        
        print(f"  [系統正呼叫 LLM 模擬 {system_prompt.split('。')[1]} 的思考...]")
        time.sleep(1) # 模擬 API 延遲
        
        # 模擬的回應內容
        return f"【{system_prompt.split('。')[1]} 的見解】：\n基於我的專業，我認為這篇論文在某些方面表現不錯，但仍需根據我的學術風格進行嚴格檢視..."

# ==========================================
# 3. 審查流程控制器 (Orchestrator)
# ==========================================
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

    def run_round_1(self):
        print("\n=== 第一輪：獨立深度審查 (Initial Review) ===")
        for reviewer in self.reviewers:
            print(f"-> 委員 {reviewer.name} 正在審閱論文...")
            user_prompt = (
                f"請審查以下論文。\n"
                f"標題：{self.paper.title}\n"
                f"內容：{self.paper.content}\n"
                f"請給出你獨立的初步審查意見，包含：1. 論文貢獻評估 2. 主要缺陷 3. 研究方法檢驗。"
            )
            response = self.llm.generate_response(reviewer.get_system_prompt(), user_prompt)
            self.history["round_1"][reviewer.name] = response
            print(f"{reviewer.name} 審查完成。\n")

    def run_round_2(self):
        print("\n=== 第二輪：交叉辯論與討論 (Cross-Discussion) ===")
        # 彙整第一輪意見
        r1_summary = "\n".join([f"[{name}] 的意見:\n{opinion}" for name, opinion in self.history["round_1"].items()])
        
        for reviewer in self.reviewers:
            print(f"-> 委員 {reviewer.name} 正在閱讀其他委員的意見並準備回應...")
            user_prompt = (
                f"我們正在進行第二輪審查。以下是第一輪所有委員的意見彙整：\n"
                f"{r1_summary}\n\n"
                f"請根據你 ({reviewer.name}) 的立場，對其他人的意見進行點評。你可以認同他們、反駁他們，或者針對矛盾之處提出更深入的問題。"
            )
            response = self.llm.generate_response(reviewer.get_system_prompt(), user_prompt)
            self.history["round_2"][reviewer.name] = response
            print(f"{reviewer.name} 回應完成。\n")

    def run_round_3(self):
        print("\n=== 第三輪：最終共識與裁決 (Final Consensus & Decision) ===")
        # 彙整第二輪討論
        r2_summary = "\n".join([f"[{name}] 在第二輪的討論:\n{opinion}" for name, opinion in self.history["round_2"].items()])
        
        for reviewer in self.reviewers:
            print(f"-> 委員 {reviewer.name} 正在撰寫最終裁決...")
            user_prompt = (
                f"這是最後一輪審查。以下是前一輪的交叉討論紀錄：\n"
                f"{r2_summary}\n\n"
                f"綜合以上所有資訊與你自身的專業判斷，請給出：\n"
                f"1. 對作者的最終修改清單 (Actionable items)\n"
                f"2. 最終裁決結論 (Accept / Minor Revision / Major Revision / Reject)，並給出簡短理由。"
            )
            response = self.llm.generate_response(reviewer.get_system_prompt(), user_prompt)
            self.history["round_3"][reviewer.name] = response
            print(f"{reviewer.name} 裁決完成。\n")

    def execute_full_review(self):
        print(f"啟動《{self.paper.title}》的專案審查流程\n論文領域：{self.paper.field}")
        self.run_round_1()
        self.run_round_2()
        self.run_round_3()
        print("\n=== 審查流程全數結束 ===")

# ==========================================
# 4. 系統執行入口
# ==========================================
if __name__ == "__main__":
    # 步驟 1: 上傳論文
    my_paper = Paper(
        title="基於邊緣運算與機器視覺的工業瑕疵檢測系統",
        field="人工智慧與先進製造",
        content="本論文提出了一種基於邊緣運算設備（如 Raspberry Pi 與微控制器）的輕量化深度學習模型，用於即時檢測生產線上的精密機械零件瑕疵。實驗結果顯示..."
    )

    # 步驟 2: 自訂審查委員 (可根據論文領域動態生成或手動指定)
    committee = [
        ReviewerAgent(
            name="Dr. Alan",
            expertise="電腦視覺與深度學習演算法",
            research_focus="輕量化神經網路架構",
            style="極度嚴格，對數學推導與模型效能數據要求極高，喜歡挑剔演算法的創新性。"
        ),
        ReviewerAgent(
            name="Prof. Lin",
            expertise="嵌入式系統與硬體架構",
            research_focus="Raspberry Pi 與微控制器系統整合",
            style="務實且具建設性，重視系統在工業現場的可落地性與硬體資源消耗。"
        ),
        ReviewerAgent(
            name="Dr. Chen",
            expertise="精密機械與製造工程",
            research_focus="工業 4.0 產線整合",
            style="宏觀視角，看重研究成果對產業帶來的實際經濟價值，容忍理論上的微小瑕疵，但要求應用場景必須合理。"
        )
    ]

    # 步驟 3: 初始化 LLM 與系統並執行
    llm_service = LLMInterface(api_key="YOUR_REAL_API_KEY_HERE")
    system = PaperReviewOrchestrator(paper=my_paper, reviewers=committee, llm=llm_service)
    
    # 執行三輪審查
    system.execute_full_review()