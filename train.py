"""
train.py
Trains RNN and LSTM models for Fake News Detection.

Usage:
    python train.py
    python train.py --model lstm   (default)
    python train.py --model rnn
    python train.py --model bilstm
"""

import os
import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, SimpleRNN, Bidirectional, Dense,
    Dropout, SpatialDropout1D, GlobalMaxPooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from utils import clean_text, save_tokenizer

# ─── Hyperparameters ────────────────────────────────────────────────────────
MAX_WORDS   = 20000   # vocabulary size
MAX_LEN     = 300     # max sequence length (tokens)
EMBED_DIM   = 128     # embedding dimension
LSTM_UNITS  = 64      # LSTM / RNN units
BATCH_SIZE  = 64
EPOCHS      = 10
TEST_SIZE   = 0.2
VAL_SIZE    = 0.1
RANDOM_SEED = 42

os.makedirs("model", exist_ok=True)
os.makedirs("dataset", exist_ok=True)


# ─── Data Loading ────────────────────────────────────────────────────────────
def load_data(csv_path: str = "dataset/news.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"[!] Dataset not found at {csv_path}")
        print("[*] Generating sample dataset …")
        import subprocess, sys
        subprocess.run([sys.executable, "generate_dataset.py"], check=True)

    df = pd.read_csv(csv_path)
    print(f"[+] Loaded {len(df)} rows. Columns: {list(df.columns)}")

    # Normalise column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Combine title + text if both exist
    if "title" in df.columns and "text" in df.columns:
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "text" in df.columns:
        df["content"] = df["text"].fillna("")
    else:
        raise ValueError("Dataset must contain a 'text' column.")

    # Normalise label: accept 0/1 or FAKE/REAL strings
    if df["label"].dtype == object:
        df["label"] = df["label"].str.upper().map({"FAKE": 1, "REAL": 0, "0": 0, "1": 1})
    df["label"] = df["label"].astype(int)

    df.dropna(subset=["content", "label"], inplace=True)
    print(f"[+] Label distribution:\n{df['label'].value_counts()}\n")
    return df


# ─── Preprocessing ───────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    print("[*] Cleaning text …")
    df["clean"] = df["content"].apply(clean_text)

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["clean"])
    save_tokenizer(tokenizer)

    sequences = tokenizer.texts_to_sequences(df["clean"])
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
    y = np.array(df["label"])

    print(f"[+] Vocabulary size : {len(tokenizer.word_index)}")
    print(f"[+] Sequence shape  : {X.shape}")
    return X, y, tokenizer


# ─── Model Builders ──────────────────────────────────────────────────────────
def build_lstm(vocab_size: int) -> Sequential:
    model = Sequential([
        Embedding(vocab_size, EMBED_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.2),
        LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])
    return model


def build_bilstm(vocab_size: int) -> Sequential:
    model = Sequential([
        Embedding(vocab_size, EMBED_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2)),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])
    return model


def build_rnn(vocab_size: int) -> Sequential:
    model = Sequential([
        Embedding(vocab_size, EMBED_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.2),
        SimpleRNN(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])
    return model


MODEL_BUILDERS = {
    "lstm":   build_lstm,
    "bilstm": build_bilstm,
    "rnn":    build_rnn,
}


# ─── Training ────────────────────────────────────────────────────────────────
def train(model_type: str = "lstm"):
    print(f"\n{'='*60}")
    print(f"  Fake News Detection — Training [{model_type.upper()}]")
    print(f"{'='*60}\n")

    df = load_data()
    X, y, tokenizer = preprocess(df)

    # Train / Val / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_SEED, stratify=y_train
    )
    print(f"[+] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}\n")

    # Build model
    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
    builder = MODEL_BUILDERS.get(model_type, build_lstm)
    model = builder(vocab_size)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # Callbacks
    model_path = f"model/{model_type}_model.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    # ── Evaluation ──────────────────────────────────────────────────────────
    print("\n[*] Evaluating on test set …")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = (model.predict(X_test, verbose=0) >= 0.5).astype(int).flatten()

    print(f"\n{'─'*40}")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  Test Loss     : {loss:.4f}")
    print(f"{'─'*40}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

    # ── Save training plot ───────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"],   label="Train Acc")
    ax1.plot(history.history["val_accuracy"], label="Val Acc")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.set_xlabel("Epoch")

    ax2.plot(history.history["loss"],     label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Loss"); ax2.legend(); ax2.set_xlabel("Epoch")

    plt.suptitle(f"{model_type.upper()} Training History", fontsize=14)
    plt.tight_layout()
    plot_path = f"model/{model_type}_training_plot.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"\n[+] Training plot saved to {plot_path}")
    print(f"[+] Model saved to {model_path}\n")
    return model, history


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fake News Detection Model")
    parser.add_argument(
        "--model",
        choices=["lstm", "rnn", "bilstm"],
        default="lstm",
        help="Model architecture to train (default: lstm)",
    )
    args = parser.parse_args()
    train(args.model)
