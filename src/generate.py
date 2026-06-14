import argparse

import torch
import torch.nn.functional as F

from src.model import TinyGPT


def build_tokenizer(chars: list[str]):
    """체크포인트에 저장된 문자 목록으로 stoi와 itos를 복원한다."""
    stoi = {
        char: index
        for index, char in enumerate(chars)
    }

    itos = {
        index: char
        for index, char in enumerate(chars)
    }

    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    """문자열을 토큰 인덱스 목록으로 변환한다."""
    unknown_chars = {
        char for char in text
        if char not in stoi
    }

    if unknown_chars:
        raise ValueError(
            f"학습 데이터에 없던 문자가 포함되어 있습니다: {unknown_chars}"
        )

    return [stoi[char] for char in text]


def decode(
    token_ids: list[int],
    itos: dict[int, str],
) -> str:
    """토큰 인덱스 목록을 문자열로 변환한다."""
    return "".join(
        itos[token_id]
        for token_id in token_ids
    )


@torch.no_grad()
def generate(
    model: TinyGPT,
    context: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
) -> torch.Tensor:
    """입력 context 뒤에 새로운 문자를 반복적으로 생성한다."""
    model.eval()

    for _ in range(max_new_tokens):
        # block_size를 넘지 않도록 최근 토큰만 사용
        context_condition = context[:, -model.block_size:]

        # logits shape: (B, T, V)
        logits = model(context_condition)

        # 마지막 위치에서 다음 문자에 대한 점수만 선택
        logits = logits[:, -1, :]

        # temperature 적용
        logits = logits / temperature

        # 점수를 확률로 변환
        probabilities = F.softmax(logits, dim=-1)

        # 확률에 따라 다음 토큰 하나를 sampling
        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        # 새 토큰을 기존 context 뒤에 붙임
        context = torch.cat(
            [context, next_token],
            dim=1,
        )

    return context


def main(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    chars = checkpoint["chars"]
    model_config = checkpoint["model_config"]

    stoi, itos = build_tokenizer(chars)

    model = TinyGPT(**model_config).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    prompt_tokens = encode(
        args.prompt,
        stoi,
    )

    context = torch.tensor(
        [prompt_tokens],
        dtype=torch.long,
        device=device,
    )

    generated = generate(
        model=model,
        context=context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    generated_text = decode(
        generated[0].tolist(),
        itos,
    )

    print(generated_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="학습된 TinyGPT 모델로 텍스트를 생성합니다."
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/tiny_gpt.pt",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Elizabeth",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())