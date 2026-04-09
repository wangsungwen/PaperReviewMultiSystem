# models/reviewer.py

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