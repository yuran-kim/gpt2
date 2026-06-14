from pathlib import Path

import torch
from torch.utils.data import Dataset


class CharacterTokenizer:
    """텍스트의 각 문자를 정수 토큰으로 변환하는 문자 단위 토크나이저."""

    def __init__(self, text: str):
        chars = sorted(set(text))

        self.stoi = {char: index for index, char in enumerate(chars)}
        self.itos = {index: char for char, index in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        """문자열을 정수 인덱스 목록으로 변환한다."""
        unknown_chars = {char for char in text if char not in self.stoi}

        if unknown_chars:
            raise ValueError(
                f"학습 데이터에 없는 문자가 포함되어 있습니다: {unknown_chars}"
            )

        return [self.stoi[char] for char in text]

    def decode(self, token_ids: list[int]) -> str:
        """정수 인덱스 목록을 문자열로 변환한다."""
        return "".join(self.itos[token_id] for token_id in token_ids)


class NextTokenDataset(Dataset):
    """입력 문장과 한 칸 이동한 정답 문장을 생성한다."""

    def __init__(self, data: torch.Tensor, block_size: int):
        if len(data) <= block_size:
            raise ValueError("데이터 길이는 block_size보다 커야 합니다.")

        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index : index + self.block_size]
        y = self.data[index + 1 : index + self.block_size + 1]

        return x, y


def load_text(path: str) -> str:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")

    return file_path.read_text(encoding="utf-8")