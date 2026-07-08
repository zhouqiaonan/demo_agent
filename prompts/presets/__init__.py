"""预设模板包 — 提供开箱即用的 PromptTemplate 实例。"""

from prompts.presets.code_review import CodeReviewer
from prompts.presets.translator import Translator
from prompts.presets.summarizer import Summarizer
from prompts.presets.classifier import Classifier
from prompts.presets.roleplay import RolePlayer

#: 所有预设模板的列表，方便批量操作。
ALL_PRESETS = [
    CodeReviewer,
    Translator,
    Summarizer,
    Classifier,
    RolePlayer,
]

__all__ = [
    "CodeReviewer",
    "Translator",
    "Summarizer",
    "Classifier",
    "RolePlayer",
    "ALL_PRESETS",
]
