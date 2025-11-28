import torch
import math
import tiktoken

from previous_chapters import GPTModel, create_dataloader_v1
from my_train_domain import GPT_CONFIG_TINY, load_domain_text


def build_model(device):
    """Load Tiny GPT model đã train từ file .pth"""
    model = GPTModel(GPT_CONFIG_TINY)
    state = torch.load("tiny_gpt_domain.pth", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def compute_perplexity(model, text, device, batch_size=8):
    """Tính loss & perplexity trên một đoạn text (ví dụ: validation set)."""
    tokenizer = tiktoken.get_encoding("gpt2")
    context_len = GPT_CONFIG_TINY["context_length"]

    loader = create_dataloader_v1(
        text,
        batch_size=batch_size,
        max_length=context_len,
        stride=context_len,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    losses = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )
            losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)
    return mean_loss, perplexity


# ====== SAMPLING FUNCTIONS (không dùng TensorFlow, không import file khác) ======

def sample_next_token(logits, temperature=1.0, top_k=None):
    """Lấy 1 token tiếp theo từ logits với temperature + top-k sampling."""
    # Áp dụng temperature
    logits = logits / temperature

    # Áp dụng top-k filtering
    if top_k is not None:
        values, _ = torch.topk(logits, top_k)
        min_topk = values.min()
        logits[logits < min_topk] = -float("inf")

    # Softmax → xác suất
    probs = torch.softmax(logits, dim=-1)

    # Lấy mẫu 1 token theo phân phối xác suất
    next_id = torch.multinomial(probs, num_samples=1)
    return next_id.item()


def generate(model, idx, max_new_tokens, context_length,
             temperature=1.0, top_k=None):
    """
    Generate text token-by-token với sampling.
    idx: tensor shape (1, T) chứa prompt đã encode.
    """
    for _ in range(max_new_tokens):
        # Giới hạn context cho phù hợp context_length
        idx_cond = idx[:, -context_length:]

        with torch.no_grad():
            logits = model(idx_cond)          # (B, T, vocab)
            logits = logits[:, -1, :]         # chỉ lấy token cuối (B, vocab)

        next_token_id = sample_next_token(
            logits[0],
            temperature=temperature,
            top_k=top_k,
        )

        next_token = torch.tensor(
            [[next_token_id]],
            dtype=torch.long,
            device=idx.device,
        )
        idx = torch.cat((idx, next_token), dim=1)

    return idx


def generate_with_temperature(model, tokenizer, device, prompt,
                              temperature=1.0, top_k=40, max_new=80):
    """Wrapper tiện dùng để sinh text từ 1 prompt."""
    model.eval()
    context_len = GPT_CONFIG_TINY["context_length"]

    # Encode prompt thành token ids
    idx = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
    ).to(device)

    out_ids = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new,
        context_length=context_len,
        temperature=temperature,
        top_k=top_k,
    )

    text = tokenizer.decode(out_ids[0].tolist())
    return text


# ================== MAIN ==================

if __name__ == "__main__":
    # 1) Thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 2) Load dữ liệu domain
    # 💡 ĐỔI LẠI tên file cho đúng nếu bạn không dùng cooking_corpus.txt
    data_path = "../../data/cooking_corpus.txt"
    full_text = load_domain_text(data_path)

    # Tách 90% train, 10% val (giống lúc train trong my_train_domain.py)
    split_idx = int(0.9 * len(full_text))
    val_text = full_text[split_idx:]

    # 3) Load model đã train
    model = build_model(device)

    # 4) Tính loss + perplexity trên validation
    val_loss, val_ppl = compute_perplexity(model, val_text, device)
    print(f"\nValidation loss: {val_loss:.3f}")
    print(f"Validation perplexity: {val_ppl:.2f}")

    # 5) Sinh text với nhiều prompt và temperature khác nhau
    tokenizer = tiktoken.get_encoding("gpt2")

    prompts = [
        "In this recipe, we will",
        "To prepare this dish, first",
        "For a healthy breakfast,",
        "This Vietnamese dish is",
        "For a simple dinner, you can",
    ]

    temperatures = [0.7, 1.0, 1.3]

    for temp in temperatures:
        print("\n" + "=" * 60)
        print(f"TEXT GENERATION WITH TEMPERATURE = {temp}")
        print("=" * 60)
        for p in prompts:
            print(f"\nPrompt: {p}")
            generated_text = generate_with_temperature(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=p,
                temperature=temp,
                top_k=40,          # top-k sampling
                max_new=80,        # số token sinh thêm
            )
            print("Output:")
            print(generated_text)
            print("-" * 40)
