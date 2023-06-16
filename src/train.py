import argparse
import os
import random

import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from perceiver_ar_pytorch import PerceiverAR
from perceiver_ar_pytorch.autoregressive_wrapper import AutoregressiveWrapper
from rich import box
from rich.console import Console
from rich.progress import track
from rich.table import Table
from torch.utils.data import DataLoader

from dataset import HuggingDataset

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.experimental.pjrt as pjrt


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

    if use_wandb:
        wandb.init(project=wandb_project)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = xm.xla_device()
    #dist.init_process_group('xla', init_method='pjrt://')

    dataset = HuggingDataset(
        dataset_name,
        text_field,
        seq_len,
        device,
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
    # model = DDP(model)
    model.to(device)
    pjrt.broadcast_master_param(model)

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
    scheduler = lr_scheduler.LinearLR(optim, start_factor=1, total_iters=len(data_loader) * epochs)

    model.train()
    loader = cycle(data_loader)

    for epoch in range(epochs):
        for i in track(range(len(dataset) // batch_size), description=f"Epoch {epoch + 1}/{epochs}"):
            losses = []
            for _ in range(4):
                loss = model(next(loader))
                loss.backward()
                losses.append(loss.item())
            if i % generate_every == 0:
                """model.eval()
                inp = random.choice(dataset)[:-1]
                sample = model.generate(inp[None, ...], 64)
                text = dataset.tokenizer.decode(sample[0])
                console.print(text)
                with open(f"{output_dir}/sample_{i}.txt", "w", encoding="utf-8") as f:
                    f.write(text)
                model.train()"""
                ...
            if i % save_every == 0:
                torch.save(model.state_dict(), f"{output_dir}/model_{i}.pt")
            if i % log_every == 0:
                table.add_row(
                    str(epoch + 1),
                    str(i),
                    f"{sum(losses) / len(losses):.3f}",
                    f"{np.exp(sum(losses) / len(losses)):.3f}",
                )
                if use_wandb:
                    wandb.log(
                        {
                            "loss": sum(losses) / len(losses),
                            "ppl": np.exp(sum(losses) / len(losses)),
                            "step": i,
                            "processed_tokens": i * batch_size * seq_len,
                            "lr": scheduler.get_last_lr()[0],
                        }
                    )
                console.print(table)

            optim.step()
            optim.zero_grad()
            scheduler.step()
            xm.mark_step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="openwebtext")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--seq_len", type=int, default=4096)  # 131072
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--separate_token", type=str, default="<|endoftext|>")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--generate_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="output")
    # Model
    parser.add_argument("--num_tokens", type=int, default=50400)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim_head", type=int, default=128)
    parser.add_argument("--cross_attn_dropout", type=int, default=0.5)
    parser.add_argument("--cross_attn_seq_len", type=int, default=3584)  # 114688
    # Logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="text-perceiver")

    args = parser.parse_args()

    os.environ['PJRT_DEVICE'] = 'TPU'

    xmp.spawn(
        train(
            args.dataset,
            args.text_field,
            args.seq_len,
            args.tokenizer,
            args.separate_token,
            args.batch_size,
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
    )
