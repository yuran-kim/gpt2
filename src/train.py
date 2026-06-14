import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import CharacterTokenizer, NextTokenDataset, load_text
from src.model import TinyGPT


def set_seed(seed: int) -> None:
    """실행할 때마다 비슷한 결과가 나오도록 난수를 고정한다."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sequence_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    전체 시퀀스의 다음 토큰 예측 손실을 계산한다.

    logits shape:  (B, T, V)
    targets shape: (B, T)
    """
    return F.cross_entropy(
        logits.transpose(1, 2),
        targets,
    )


def evaluate(
    model: TinyGPT,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    """검증 데이터의 평균 loss를 계산한다."""
    model.eval()

    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch_index, (x, y) in enumerate(data_loader):
            if batch_index >= max_batches:
                break

            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = sequence_cross_entropy(logits, y)

            batch_size = x.size(0)

            total_loss += loss.item() * batch_size
            total_count += batch_size

    model.train()

    if total_count == 0:
        raise RuntimeError("검증할 데이터가 없습니다.")

    return total_loss / total_count


def save_checkpoint(
    path: str,
    model: TinyGPT,
    tokenizer: CharacterTokenizer,
    model_config: dict,
    training_config: dict,
    step: int,
) -> None:
    """나중에 학습된 모델을 다시 불러올 수 있도록 저장한다."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "chars": list(tokenizer.stoi.keys()),
        "model_config": model_config,
        "training_config": training_config,
        "step": step,
    }

    torch.save(checkpoint, checkpoint_path)

    print(f"모델 체크포인트를 저장했습니다: {checkpoint_path}")


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"사용 장치: {device}")

    # 1. 텍스트 읽기
    text = load_text(args.data_path)

    # 2. 문자 단위 토크나이저 생성
    tokenizer = CharacterTokenizer(text)

    # 3. 전체 텍스트를 정수 토큰으로 변환
    encoded = torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long,
    )

    # 4. 학습 데이터와 검증 데이터 분리
    split_index = int(len(encoded) * args.train_ratio)

    train_data = encoded[:split_index]
    val_data = encoded[split_index:]

    print(f"전체 문자 수: {len(encoded):,}")
    print(f"학습 문자 수: {len(train_data):,}")
    print(f"검증 문자 수: {len(val_data):,}")
    print(f"vocab size: {tokenizer.vocab_size}")

    # 5. 입력 x와 한 칸 이동한 정답 y를 생성하는 Dataset
    train_dataset = NextTokenDataset(
        train_data,
        block_size=args.block_size,
    )

    val_dataset = NextTokenDataset(
        val_data,
        block_size=args.block_size,
    )

    # 6. mini-batch 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "block_size": args.block_size,
        "emb_dim": args.emb_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }

    # 7. 모델 생성
    model = TinyGPT(**model_config).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"모델 파라미터 수: {parameter_count:,}")

    # 8. optimizer 생성
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    training_config = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
    }

    model.train()

    train_iterator = iter(train_loader)

    # 9. 학습 반복
    for step in range(1, args.max_steps + 1):
        try:
            x, y = next(train_iterator)

        except StopIteration:
            # DataLoader를 한 번 모두 사용하면 새 iterator를 만든다.
            train_iterator = iter(train_loader)
            x, y = next(train_iterator)

        x = x.to(device)
        y = y.to(device)

        # forward
        logits = model(x)

        # loss 계산
        loss = sequence_cross_entropy(logits, y)

        # 이전 step의 gradient 제거
        optimizer.zero_grad(set_to_none=True)

        # backward
        loss.backward()

        # parameter 업데이트
        optimizer.step()

        # 일정 간격마다 학습 및 검증 결과 출력
        if step == 1 or step % args.eval_interval == 0:
            val_loss = evaluate(
                model=model,
                data_loader=val_loader,
                device=device,
                max_batches=args.eval_iters,
            )

            print(
                f"step {step}: "
                f"train_loss={loss.item():.4f}, "
                f"val_loss={val_loss:.4f}"
            )

    # 10. 최종 모델 저장
    save_checkpoint(
        path=args.checkpoint_path,
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        training_config=training_config,
        step=args.max_steps,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="문자 단위 TinyGPT 모델을 학습합니다."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/pride_and_prejudice.txt",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/tiny_gpt.pt",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--emb-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--eval-iters",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())