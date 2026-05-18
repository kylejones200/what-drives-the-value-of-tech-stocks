"""Generated from Jupyter notebook: 2025-05-20 OR tools example

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import os

import numpy as np
import pandas as pd
import torch
from ortools.linear_solver import pywraplp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val.squeeze(0) for key, val in self.tokens[idx].items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def tokenize_time_series(series):
    series_str = " ".join(map(str, series))
    return tokenizer(
        series_str,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )


def parameters() -> None:
    demand = [20, 30, 40, 35, 25]
    holding_cost = 2
    order_cost = 50
    capacity = 100
    solver = pywraplp.Solver.CreateSolver("SCIP")
    order = [solver.BoolVar(f"order_{t}") for t in range(len(demand))]
    inventory = [
        solver.IntVar(0, capacity, f"inventory_{t}") for t in range(len(demand))
    ]
    solver.Add(inventory[0] == 0)
    for t in range(1, len(demand)):
        solver.Add(inventory[t] == inventory[t - 1] + 100 * order[t] - demand[t])

    total_cost = sum(
        (
            order_cost * order[t] + holding_cost * inventory[t]
            for t in range(len(demand))
        )
    )
    solver.Minimize(total_cost)
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print("Optimal solution found!")
        for t in range(len(demand)):
            print(
                f"Day {t + 1}: Order = {order[t].solution_value()}, Inventory = {inventory[t].solution_value()}"
            )
    else:
        print("No optimal solution found.")


def generate_synthetic_time_series_data() -> None:
    np.random.seed(42)
    n_samples = 100
    n_timestamps = 50
    class_0 = np.random.normal(0, 1, (n_samples // 2, n_timestamps))
    class_1 = np.random.normal(2, 1, (n_samples // 2, n_timestamps))
    X = np.vstack((class_0, class_1))
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    df = pd.DataFrame(X)
    df["label"] = y
    print(df.head())
    os.environ["WANDB_DISABLED"] = "true"
    AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = [tokenize_time_series(row[:-1].values) for _, row in df.iterrows()]
    labels = df["label"].values
    train_tokens, test_tokens, train_labels, test_labels = train_test_split(
        tokens, labels, test_size=0.2, random_state=42
    )
    train_dataset = TimeSeriesDataset(train_tokens, train_labels)
    test_dataset = TimeSeriesDataset(test_tokens, test_labels)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_steps=10,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()
    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    accuracy = accuracy_score(test_labels, predicted_labels)
    print(f"Test Accuracy: {accuracy:.2f}")


def notebook_step_005() -> None:
    np.random.seed(42)
    n_samples = 100
    n_timestamps = 50
    class_0 = np.random.normal(0, 1, (n_samples // 2, n_timestamps))
    class_1 = np.random.normal(2, 1, (n_samples // 2, n_timestamps))
    X = np.vstack((class_0, class_1))
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    df = pd.DataFrame(X)
    df["label"] = y
    print(df.head())
    AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = [tokenize_time_series(row[:-1].values) for _, row in df.iterrows()]
    labels = df["label"].values
    train_tokens, test_tokens, train_labels, test_labels = train_test_split(
        tokens, labels, test_size=0.2, random_state=42
    )
    train_dataset = TimeSeriesDataset(train_tokens, train_labels)
    test_dataset = TimeSeriesDataset(test_tokens, test_labels)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_steps=10,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()
    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    accuracy = accuracy_score(test_labels, predicted_labels)
    print(f"Test Accuracy: {accuracy:.2f}")


def main() -> None:
    parameters()
    generate_synthetic_time_series_data()
    notebook_step_005()


if __name__ == "__main__":
    main()
