from __future__ import annotations

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class FinBERTScorer:
    """
    Batched GPU/CPU FinBERT scorer (`ProsusAI/finbert`) returning full
    [positive, negative, neutral] probabilities for financial headlines.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str | None = None,
        batch_size: int = 128,
        max_length: int = 128,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = (
            torch.bfloat16
            if (self.device.type == "cuda" and torch.cuda.is_bf16_supported())
            else (torch.float16 if self.device.type == "cuda" else torch.float32)
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            dtype=dtype,
        ).to(self.device)
        self.model.eval()

        # Map label indices dynamically from model config (ProsusAI/finbert: positive, negative, neutral)
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.pos_idx = next(k for k, v in id2label.items() if "pos" in v)
        self.neg_idx = next(k for k, v in id2label.items() if "neg" in v)
        self.neu_idx = next(k for k, v in id2label.items() if "neu" in v)

    @torch.no_grad()
    def score_texts(self, texts: list[str]) -> np.ndarray:
        """
        Scores a list of texts in batches and returns an array of shape (N, 3)
        with columns [p_pos, p_neg, p_neu].
        """
        if not texts:
            return np.empty((0, 3), dtype=np.float32)

        all_probs: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**encoded).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            ordered = probs[:, [self.pos_idx, self.neg_idx, self.neu_idx]]
            all_probs.append(ordered.astype(np.float32))

        return np.vstack(all_probs)

    def score_dataframe(self, df_mapped: pl.DataFrame) -> pl.DataFrame:
        """
        Scores a currency-mapped DataFrame. Deduplicates unique titles before
        running FinBERT so multi-currency headlines are only inferred once.
        """
        if df_mapped.is_empty():
            return df_mapped.with_columns(
                [
                    pl.lit(0.0, dtype=pl.Float32).alias("p_pos"),
                    pl.lit(0.0, dtype=pl.Float32).alias("p_neg"),
                    pl.lit(0.0, dtype=pl.Float32).alias("p_neu"),
                    pl.lit(0.0, dtype=pl.Float32).alias("net_sentiment"),
                    pl.lit(0.0, dtype=pl.Float32).alias("weighted_sentiment"),
                ]
            )

        unique_titles = df_mapped.select("title").unique()["title"].to_list()
        probs = self.score_texts(unique_titles)

        score_df = pl.DataFrame(
            {
                "title": unique_titles,
                "p_pos": probs[:, 0],
                "p_neg": probs[:, 1],
                "p_neu": probs[:, 2],
            }
        ).with_columns(
            (pl.col("p_pos") - pl.col("p_neg")).cast(pl.Float32).alias("net_sentiment")
        )

        return (
            df_mapped.join(score_df, on="title", how="left")
            .with_columns(
                (pl.col("net_sentiment") * pl.col("importance"))
                .cast(pl.Float32)
                .alias("weighted_sentiment")
            )
            .sort(["published_utc", "currency"])
        )
