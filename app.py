import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# ---------------------------------------
# Model Definition

class NextWordMLP(nn.Module):
    def __init__(self, vocab_size, emb_dim, context_length, hidden1, hidden2=None, activation='relu', dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        activation_layer = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'leakyrelu': nn.LeakyReLU()
        }.get(activation.lower(), nn.ReLU())

        layers = [
            nn.Flatten(),
            nn.Linear(context_length * emb_dim, hidden1),
            activation_layer,
            nn.Dropout(dropout)
        ]

        if hidden2:
            layers += [
                nn.Linear(hidden1, hidden2),
                activation_layer,
                nn.Dropout(dropout),
                nn.Linear(hidden2, vocab_size)
            ]
        else:
            layers += [nn.Linear(hidden1, vocab_size)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.emb(x)
        logits = self.mlp(x)
        return logits

# ---------------------------------------
# Text Generation Utility

def generate_next_k_words(model, seed_text, stoi, itos, device, k, context_length, temperature=1.0):
    model.eval()
    words_in = seed_text.lower().split()
    context = (
        [0] * max(0, context_length - len(words_in)) +
        [stoi.get(w, 1) for w in words_in]
    )[-context_length:]
    generated = words_in.copy()

    for _ in range(k):
        x = torch.tensor([context], dtype=torch.long).to(device)  # [1, context_length]
        with torch.no_grad():
            logits = model(x)[0]  # [context_length, vocab_size] → last position selected
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1).cpu()

            if probs.dim() != 1:
                probs = probs.flatten()

            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_word = itos.get(next_idx, "<UNK>")
            generated.append(next_word)
            context = context[1:] + [next_idx]

    return ' '.join(generated)

# ---------------------------------------
# Streamlit UI

st.title("🪶 Next Word Generator using MLP (Word-Level)")
st.write("Enter context text, select your pre-trained model, and generate next *k* words.")

sample_text = "Sherlock Holmes was"
text = st.text_area("Input Context Text", sample_text, height=100)

k = st.slider("Predict Next k Words", min_value=1, max_value=100, value=20)
seed = st.number_input("Random Seed", min_value=0, value=42)
temperature = st.slider("Sampling Temperature", min_value=0.5, max_value=2.0, value=1.0, step=0.05)

# ---------------------------------------
# Model Registry and interlinked dropdowns

model_registry = {
    "Holmes 64 ReLU": {"path": "models/holmes_64_relu_mlp.pt", "activation": "relu", "embedding": 64},
    "Shakespeare 32 Tanh": {"path": "models/shakespeare_32_tanh_mlp.pt", "activation": "tanh", "embedding": 32},
    "Tolstoy 128 GELU": {"path": "models/tolstoys_128_gelu_mlp.pt", "activation": "gelu", "embedding": 128},
}

activations = sorted(list(set(info["activation"] for info in model_registry.values())))
activation_fn = st.selectbox("Select Activation Function", activations)

valid_embeddings = sorted(
    list(set(info["embedding"] for info in model_registry.values() if info["activation"] == activation_fn))
)
embedding_dim = st.selectbox("Select Embedding Dimension", valid_embeddings)

valid_models = {
    name: info for name, info in model_registry.items()
    if info["activation"] == activation_fn and info["embedding"] == embedding_dim
}
model_name = st.selectbox("Select Pre-trained Model", list(valid_models.keys()))

# ---------------------------------------
# Generate Button

if st.button("Generate Text"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not text.strip():
        st.error("Please enter some input text.")
        st.stop()

    with st.spinner("Loading model and generating text..."):
        selected_model = valid_models[model_name]
        checkpoint = torch.load(selected_model["path"], map_location=device)

        model_state_dict = checkpoint["model_state_dict"]
        stoi = checkpoint["stoi"]
        itos = checkpoint["itos"]
        config = checkpoint["config"]

        # ✅ Use actual vocab size from model weights to avoid mismatch errors
        vocab_size = next(iter(model_state_dict.values())).shape[0]


        emb_dim = config["emb_dim"]
        context_length = config["context_length"]
        hidden1 = config["hidden1"]
        hidden2 = config["hidden2"]
        activation = config["activation"]

        model = NextWordMLP(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            context_length=context_length,
            hidden1=hidden1,
            hidden2=hidden2,
            activation=activation
        ).to(device)

        model.load_state_dict(model_state_dict)

        generated_text = generate_next_k_words(
            model=model,
            seed_text=text,
            stoi=stoi,
            itos=itos,
            device=device,
            k=k,
            context_length=context_length,
            temperature=temperature
        )

        st.subheader("📝 Generated Text:")
        st.write(generated_text)

