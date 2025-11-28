import os
import torch
import math
import tiktoken
from previous_chapters import GPTModel, create_dataloader_v1, generate_text_simple

def load_domain_text(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def main(gpt_config, settings):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = "../../data/cooking_corpus.txt"  
    text_data = load_domain_text(data_path)

    model = GPTModel(gpt_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )

    # 3) Tạo dataloader train/val
    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_text = text_data[:split_idx]
    val_text = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_text,
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        val_text,
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    tokenizer = tiktoken.get_encoding("gpt2")

    # 4) Train + log loss & perplexity
    return train_loop(
        model, train_loader, val_loader, optimizer,
        device, settings, tokenizer
    )

def calc_loss_batch(model, x, y, device):
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1),
    )
    return loss

def eval_loss(model, data_loader, device, max_batches=None):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if max_batches is not None and i >= max_batches:
                break
            loss = calc_loss_batch(model, x, y, device)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def train_loop(model, train_loader, val_loader, optimizer,
               device, settings, tokenizer):
    num_epochs = settings["num_epochs"]
    eval_freq = settings["eval_freq"]
    eval_iter = settings["eval_iter"]
    tokens_seen = 0
    global_step = 0

    train_losses, val_losses, token_track = [], [], []

    for epoch in range(num_epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(model, x, y, device)
            loss.backward()
            optimizer.step()

            tokens_seen += x.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss = eval_loss(model, train_loader, device, eval_iter)
                val_loss = eval_loss(model, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                token_track.append(tokens_seen)

                # Perplexity = exp(loss)
                train_ppl = math.exp(train_loss)
                val_ppl = math.exp(val_loss)
                print(
                    f"Ep {epoch+1} | step {global_step:05d} | "
                    f"train loss {train_loss:.3f} (ppl {train_ppl:.1f}) | "
                    f"val loss {val_loss:.3f} (ppl {val_ppl:.1f})"
                )

        # Sau mỗi epoch, in mẫu text
        generate_sample(model, tokenizer, device)

    return train_losses, val_losses, token_track, model

def generate_sample(model, tokenizer, device, start="In this recipe"):
    model.eval()
    context_len = model.pos_emb.weight.size(0)
    idx = torch.tensor(
        [tokenizer.encode(start)],
        dtype=torch.long,
    ).to(device)
    with torch.no_grad():
        from previous_chapters import generate_text_simple
        out_ids = generate_text_simple(
            model=model,
            idx=idx,
            max_new_tokens=80,
            context_size=context_len,
        )
    text = tokenizer.decode(out_ids[0].tolist())
    print("=== SAMPLE TEXT ===")
    print(text.replace("\n", " "))
    print("===================")
    model.train()

GPT_CONFIG_TINY = {
    "vocab_size": 50257,
    "context_length": 128,
    "emb_dim": 256,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

OTHER_SETTINGS = {
    "learning_rate": 3e-4,
    "num_epochs": 5,
    "batch_size": 8,
    "weight_decay": 0.1,
    "eval_freq": 50,   # mỗi 50 step eval
    "eval_iter": 5,    # dùng 5 batch khi eval
}

if __name__ == "__main__":
    train_losses, val_losses, tokens_seen, model = main(
        GPT_CONFIG_TINY,
        OTHER_SETTINGS,
    )

    torch.save(model.state_dict(), "tiny_gpt_domain.pth")

