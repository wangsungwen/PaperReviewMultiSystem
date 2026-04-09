# models/paper.py

class Paper:
    def __init__(self, title: str, field: str, content: str):
        self.title = title
        self.field = field
        self.content = content