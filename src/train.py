import torch
import random

import numpy as np
from torch.utils.data import DataLoader

from perceiver_ar_pytorch import PerceiverAR
from perceiver_ar_pytorch.autoregressive_wrapper import AutoregressiveWrapper
from dataset import HuggingDataset

import argparse
import wandb
import os

from rich.table import Table
from rich.console import Console
from rich.progress import track
from rich import box

from accelerate import Accelerator


def cycle(loader):
    while True:
        for data in loader:
            yield data


def train(
    dataset_name,
    text_field,
    seq_len,
    tokenizer,
    separate_token,
    batch_size,
    cpu,
    mixed_precision,
    lr,
    epochs,
    generate_every,
    save_every,
    log_every,
    output_dir,
    num_tokens,
    dim,
    depth,
    heads,
    dim_head,
    cross_attn_dropout,
    cross_attn_seq_len,
    use_wandb,
    wandb_project,
):
    console = Console()
    accelerator = Accelerator(cpu=cpu, mixed_precision=mixed_precision)

    if use_wandb:
        wandb.init(project=wandb_project)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = HuggingDataset(
        dataset_name,
        text_field,
        seq_len,
        accelerator.device,
        separate_token,
        tokenizer,
    )

    data_loader = DataLoader(dataset, batch_size=batch_size)

    model = PerceiverAR(
        num_tokens=num_tokens,
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        cross_attn_dropout=cross_attn_dropout,
        max_seq_len=seq_len,
        cross_attn_seq_len=cross_attn_seq_len,
    )

    model = AutoregressiveWrapper(model)
    model.to(accelerator.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_table = Table(
        title="Model",
        show_header=True,
        header_style="bold magenta",
        box=box.SIMPLE,
        show_lines=True,
    )
    model_table.add_column("Parameter", style="dim", width=12)
    model_table.add_column("Value", justify="right")
    model_table.add_row("Num params", f"{n_params:,}")
    console.print(model_table)

    table = Table(
        title="Training",
        show_header=True,
        header_style="bold magenta",
        box=box.SIMPLE,
        show_lines=True,
    )
    table.add_column("Epoch")
    table.add_column("Step")
    table.add_column("Loss")
    table.add_column("PPL")

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    model, optim, data_loader = accelerator.prepare(model, optim, data_loader)

    model.train()
    loader = cycle(data_loader)

    for epoch in range(epochs):
        for i in track(range(len(dataset)), description=f"Epoch {epoch+1}/{epochs}"):
            losses = []
            for _ in range(4):
                loss = model(next(loader))
                accelerator.backward(loss)
                losses.append(loss.item())
            if i % generate_every == 0:
                model.eval()
                inp = random.choice(dataset)[:-1]

                sample = model.generate(inp[None, ...], 512)
                text = dataset.tokenizer.decode(sample[0])
                if use_wandb:
                    wandb.log({"text": text, "step": i})
                console.print(text)
            if i % save_every == 0:
                torch.save(model.state_dict(), f"{output_dir}/model_{i}.pt")
            if i % log_every == 0:
                table.add_row(
                    str(epoch + 1),
                    str(i),
                    f"{sum(losses)/len(losses):.3f}",
                    f"{np.exp(sum(losses)/len(losses)):.3f}",
                )
                console.print(table)
                if use_wandb:
                    wandb.log(
                        {
                            "loss": sum(losses) / len(losses),
                            "ppl": np.exp(sum(losses) / len(losses)),
                            "step": i,
                        }
                    )

            optim.step()
            optim.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="openwebtext")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--separate_token", type=str, default="<|endoftext|>")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--generate_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="output")
    # Model
    parser.add_argument("--num_tokens", type=int, default=50400)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--cross_attn_dropout", type=int, default=0.5)
    parser.add_argument("--cross_attn_seq_len", type=int, default=3584)
    # Logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="text-perceiver")

    args = parser.parse_args()

    train(
        args.dataset_name,
        args.text_field,
        args.seq_len,
        args.tokenizer,
        args.separate_token,
        args.batch_size,
        args.cpu,
        args.mixed_precision,
        args.lr,
        args.epochs,
        args.generate_every,
        args.save_every,
        args.log_every,
        args.output_dir,
        args.num_tokens,
        args.dim,
        args.depth,
        args.heads,
        args.dim_head,
        args.cross_attn_dropout,
        args.cross_attn_seq_len,
        args.wandb,
        args.wandb_project,
    )
