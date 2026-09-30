from pathlib import Path
from datetime import datetime

import pandas as pd


BASE = Path(__file__).parent

MONITORING_DIR = (
    BASE / "monitoring"
)

LOG_FILE = (
    MONITORING_DIR /
    "prediction_log.csv"
)

MONITORING_DIR.mkdir(
    exist_ok=True
)


def log_prediction(
    language,
    ml_prediction,
    dl_prediction,
    anomaly_prediction,
):

    row = {
        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "language":
            language,

        "ml_prediction":
            int(ml_prediction),

        "dl_prediction":
            int(dl_prediction),

        "anomaly_prediction":
            int(anomaly_prediction),

        "models_agreed":
            int(
                ml_prediction
                == dl_prediction
            ),
    }


    df = pd.DataFrame(
        [row]
    )


    if LOG_FILE.exists():

        df.to_csv(
            LOG_FILE,
            mode="a",
            header=False,
            index=False,
        )

    else:

        df.to_csv(
            LOG_FILE,
            index=False,
        )


def get_monitoring_stats():

    if not LOG_FILE.exists():

        return {
            "total_predictions": 0,
            "agreement_rate": 0.0,
            "suspicious_rate": 0.0,
        }


    df = pd.read_csv(
        LOG_FILE
    )


    if df.empty:

        return {
            "total_predictions": 0,
            "agreement_rate": 0.0,
            "suspicious_rate": 0.0,
        }


    return {
        "total_predictions":
            len(df),

        "agreement_rate":
            round(
                df[
                    "models_agreed"
                ].mean() * 100,
                1
            ),

        "suspicious_rate":
            round(
                df[
                    "ml_prediction"
                ].mean() * 100,
                1
            ),
    }