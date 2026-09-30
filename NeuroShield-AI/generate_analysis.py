from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE = Path(__file__).parent

DATA = BASE / "data" / "messages.csv"
ARTIFACTS = BASE / "artifacts"

ARTIFACTS.mkdir(exist_ok=True)


df = pd.read_csv(DATA)

df = df.dropna(
    subset=["text", "label"]
)

df["message_length"] = (
    df["text"]
    .astype(str)
    .str.len()
)


statistics = {
    "total_messages": len(df),

    "normal_messages":
        int((df["label"] == 0).sum()),

    "suspicious_messages":
        int((df["label"] == 1).sum()),

    "average_message_length":
        round(
            df["message_length"].mean(),
            2
        ),

    "median_message_length":
        round(
            df["message_length"].median(),
            2
        ),

    "std_message_length":
        round(
            df["message_length"].std(),
            2
        ),
}


pd.DataFrame(
    [statistics]
).to_csv(
    ARTIFACTS / "dataset_statistics.csv",
    index=False
)


# Class distribution
plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="label"
)

plt.title(
    "Message Class Distribution"
)

plt.xlabel(
    "0 = Normal | 1 = Suspicious"
)

plt.ylabel(
    "Messages"
)

plt.tight_layout()

plt.savefig(
    ARTIFACTS / "class_distribution.png",
    dpi=160
)

plt.close()


# Message length distribution
plt.figure(figsize=(7, 4))

sns.histplot(
    data=df,
    x="message_length",
    hue="label",
    bins=50,
    element="step"
)

plt.title(
    "Message Length Distribution"
)

plt.xlabel(
    "Message Length (characters)"
)

plt.tight_layout()

plt.savefig(
    ARTIFACTS /
    "message_length_distribution.png",
    dpi=160
)

plt.close()


print(
    "EDA and visualizations generated successfully."
)