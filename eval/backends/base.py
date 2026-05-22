from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PromptRecord:
    dataset: str
    prompt: str
    answers: list[str]
    all_classes: list[str] = field(default_factory=list)
    meta: dict | None = None


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] | None = None


class BaseBackend(ABC):
    name: str = ""

    @classmethod
    def add_cli_args(cls, parser):
        return parser

    def get_prompt_formatter(self):
        return None

    def close(self):
        return None

    @abstractmethod
    def build(self, args) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        records: list[PromptRecord],
        gen_config: GenerationConfig,
    ) -> list[dict]:
        raise NotImplementedError
