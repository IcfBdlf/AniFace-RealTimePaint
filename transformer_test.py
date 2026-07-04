import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------- 1. 位置编码 Positional Encoding ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# ---------------------- 2. 多头注意力 MultiHeadAttention（修复维度bug） ----------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        assert d_model % nhead == 0
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def scaled_dot_product(self, q, k, v, mask=None):
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            # 使用 ~mask（逻辑取反）替代 mask == 0，更直观且兼容各版本 PyTorch
            attn_score = attn_score.masked_fill(~mask, -1e9)
        attn_weight = F.softmax(attn_score, dim=-1)
        out = torch.matmul(attn_weight, v)
        return out, attn_weight

    def forward(self, q, k, v, mask=None):
        seq_q, batch, _ = q.shape
        seq_k, _, _ = k.shape

        q_proj = self.w_q(q)
        k_proj = self.w_k(k)
        v_proj = self.w_v(v)

        # (seq, batch, head, dk) -> (batch, head, seq, dk)
        q_proj = q_proj.view(seq_q, batch, self.nhead, self.d_k).permute(1, 2, 0, 3)
        k_proj = k_proj.view(seq_k, batch, self.nhead, self.d_k).permute(1, 2, 0, 3)
        v_proj = v_proj.view(seq_k, batch, self.nhead, self.d_k).permute(1, 2, 0, 3)

        attn_out, _ = self.scaled_dot_product(q_proj, k_proj, v_proj, mask)

        # 还原维度
        attn_out = attn_out.permute(2, 0, 1, 3).contiguous()
        attn_out = attn_out.view(seq_q, batch, self.nhead * self.d_k)

        return self.w_o(attn_out)

# ---------------------- 3. FFN 前馈网络 ----------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
    def forward(self, x):
        return self.net(x)

# ---------------------- 4. Encoder Layer 单层编码器 ----------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        attn_out = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout1(attn_out))
        ffn_out = self.ffn(src)
        src = self.norm2(src + self.dropout2(ffn_out))
        return src

# ---------------------- 5. Decoder Layer 单层解码器 ----------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # decoder自注意力
        attn1 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(attn1))
        # 交叉注意力
        attn2 = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout2(attn2))
        # FFN
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout3(ffn_out))
        return tgt

# ---------------------- 6. 完整 Transformer 模型 ----------------------
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask=None):
        x = self.src_emb(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pos_enc(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = self.tgt_emb(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pos_enc(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encode(src, src_mask)
        dec_out = self.decode(tgt, memory, tgt_mask, memory_mask)
        logits = self.fc_out(dec_out)
        return logits

# ---------------------- 测试运行 Demo ----------------------
def generate_tgt_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask == 0
    return mask

if __name__ == "__main__":
    # 超参
    SRC_VOCAB = 1000
    TGT_VOCAB = 1000
    BATCH_SIZE = 2
    SEQ_SRC = 10
    SEQ_TGT = 12
    EPOCHS = 10  # 训练轮数

    # 1. 初始化模型、损失、优化器
    model = Transformer(SRC_VOCAB, TGT_VOCAB)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 生成解码器掩码（遮挡未来token）
    tgt_mask = generate_tgt_mask(SEQ_TGT)

    # ====================== 模拟训练集 ======================
    # 这里只是模拟真实数据集，实际要替换成你的文本翻译数据
    # 造50条训练样本：每条src长10，tgt长12
    train_src_dataset = [torch.randint(0, SRC_VOCAB, (SEQ_SRC,)) for _ in range(50)]
    train_tgt_dataset = [torch.randint(0, TGT_VOCAB, (SEQ_TGT,)) for _ in range(50)]

    # 简易分批模拟dataloader
    def get_batch():
        idx = torch.randint(0, len(train_src_dataset), (BATCH_SIZE,))
        src_batch = torch.stack([train_src_dataset[i] for i in idx]).transpose(0,1) # [seq, batch]
        tgt_batch = torch.stack([train_tgt_dataset[i] for i in idx]).transpose(0,1)
        # 训练输入tgt：去掉最后一位；标签label：去掉第一位（右移）
        tgt_input = tgt_batch[:-1, :]
        label = tgt_batch[1:, :]
        return src_batch, tgt_input, label

    # ====================== 完整训练循环 ======================
    for epoch in range(EPOCHS):
        model.train()  # 开启训练模式(dropout生效)
        total_loss = 0.0
        # 每轮取20个batch训练
        for _ in range(20):
            src, tgt_in, label = get_batch()

            # 前向传播
            logits = model(src, tgt_in, tgt_mask=tgt_mask[:tgt_in.size(0),:tgt_in.size(0)])

            # 计算损失：reshape成二维 [seq*batch, vocab]
            loss = criterion(
                logits.reshape(-1, TGT_VOCAB),
                label.reshape(-1)
            )

            # 反向传播更新权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / 20
        print(f"Epoch {epoch+1}/{EPOCHS}, 平均损失: {avg_loss:.4f}")

    # 训练完成后测试一次前向输出
    model.eval()
    src_test = torch.randint(0, SRC_VOCAB, (SEQ_SRC, BATCH_SIZE))
    tgt_test = torch.randint(0, TGT_VOCAB, (SEQ_TGT-1, BATCH_SIZE))
    with torch.no_grad():
        logits = model(src_test, tgt_test, tgt_mask=tgt_mask[:SEQ_TGT-1,:SEQ_TGT-1])
    print("训练后输出logits shape:", logits.shape)