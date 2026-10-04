"""Central configuration: persisted app settings editable from the Settings page.

All timeframes/thresholds/model params referenced by the feature engine and
models flow from here, so the Settings page is a real control surface rather
than a display.
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

ALL_TIMEFRAMES = ["5m", "15m", "30m", "1H", "2H", "4H"]
TREND_TIMEFRAMES = ["15m", "30m", "1H", "2H", "4H"]
GAP_TIMEFRAMES = ["5m", "15m", "30m", "1H", "2H", "4H"]

TIMEFRAME_MINUTES = {"5m": 5, "15m": 15, "30m": 30, "1H": 60, "2H": 120, "4H": 240}


@dataclasses.dataclass
class BollingerConfig:
    period: int = 20
    std_multiplier: float = 2.0
    timeframes: list[str] = dataclasses.field(default_factory=lambda: list(ALL_TIMEFRAMES))


@dataclasses.dataclass
class VolatilityConfig:
    atr_period: int = 14
    realized_vol_window: int = 20
    percentile_lookback: int = 252
    low_pct: float = 25.0
    high_pct: float = 75.0
    extreme_pct: float = 95.0


@dataclasses.dataclass
class GapConfig:
    timeframes: list[str] = dataclasses.field(default_factory=lambda: list(GAP_TIMEFRAMES))
    min_gap_pct: float = 0.05
    cluster_lookback: int = 5
    closure_horizon_bars: int = 20


@dataclasses.dataclass
class TrendConfig:
    timeframes: list[str] = dataclasses.field(default_factory=lambda: list(TREND_TIMEFRAMES))
    fast_ma: int = 10
    slow_ma: int = 50
    roc_period: int = 10


@dataclasses.dataclass
class ModelConfig:
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.05
    transformer_seq_len: int = 24
    transformer_d_model: int = 48
    transformer_nhead: int = 4
    transformer_layers: int = 2
    transformer_epochs: int = 8
    train_split: float = 0.7
    val_split: float = 0.15
    forward_horizon_bars: int = 10


@dataclasses.dataclass
class DataConfig:
    symbol: str = "SYNTH/USD"
    base_timeframe: str = "5m"
    lookback_bars: int = 3000
    random_seed: int = 42
    session_open_hour: int = 9
    session_open_minute: int = 30
    session_close_hour: int = 16
    session_close_minute: int = 0


@dataclasses.dataclass
class AppConfig:
    enabled_timeframes: list[str] = dataclasses.field(default_factory=lambda: list(ALL_TIMEFRAMES))
    bollinger: BollingerConfig = dataclasses.field(default_factory=BollingerConfig)
    volatility: VolatilityConfig = dataclasses.field(default_factory=VolatilityConfig)
    gaps: GapConfig = dataclasses.field(default_factory=GapConfig)
    trend: TrendConfig = dataclasses.field(default_factory=TrendConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)


def _dataclass_from_dict(cls, data: dict[str, Any]):
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for name, ftype in field_types.items():
        if name not in data:
            continue
        val = data[name]
        nested = {"BollingerConfig": BollingerConfig, "VolatilityConfig": VolatilityConfig,
                  "GapConfig": GapConfig, "TrendConfig": TrendConfig,
                  "ModelConfig": ModelConfig, "DataConfig": DataConfig}.get(ftype)
        kwargs[name] = _dataclass_from_dict(nested, val) if nested else val
    return cls(**kwargs)


def load_config() -> AppConfig:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                raw = json.load(f)
            return _dataclass_from_dict(AppConfig, raw)
        except Exception:
            pass
    return AppConfig()


def save_config(cfg: AppConfig) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
