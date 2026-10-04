"""Shared app state: config, cached data/feature generation, and the trained
model slot in session_state so every page sees the same run without
recomputing it."""
from __future__ import annotations

import dataclasses
import hashlib
import json

import streamlit as st

from core.config import AppConfig, load_config, save_config
from core.data import generate_base_data, get_multi_timeframe_data, generate_external_sr_levels
from core.features.pipeline import build_feature_dataset
from core.logging_utils import get_logger

logger = get_logger("state")


def get_config() -> AppConfig:
    if "cfg" not in st.session_state:
        st.session_state["cfg"] = load_config()
    return st.session_state["cfg"]


def persist_config(cfg: AppConfig) -> None:
    st.session_state["cfg"] = cfg
    save_config(cfg)
    logger.info("Settings saved")
    get_dataset.clear()


def _cfg_hash(cfg: AppConfig) -> str:
    return hashlib.md5(json.dumps(dataclasses.asdict(cfg), sort_keys=True).encode()).hexdigest()


@st.cache_data(show_spinner="Generating market data...")
def _get_ohlcv(cfg_hash: str, _cfg: AppConfig):
    base = generate_base_data(_cfg.data)
    tfs = get_multi_timeframe_data(_cfg.data, _cfg.enabled_timeframes, override_5m=base)
    base_tf = _cfg.data.base_timeframe if _cfg.data.base_timeframe in tfs else next(iter(tfs))
    sr_by_tf = {base_tf: generate_external_sr_levels(tfs[base_tf], base_tf)}
    logger.info(f"Generated synthetic market data: {len(base)} base bars across {len(tfs)} timeframes")
    return tfs, sr_by_tf


@st.cache_data(show_spinner="Building unified feature dataset...")
def _get_dataset(cfg_hash: str, _cfg: AppConfig, _tfs, _sr_by_tf):
    dataset, events = build_feature_dataset(_cfg, _tfs, _sr_by_tf)
    logger.info(f"Built unified feature dataset: {dataset.shape[0]} bars x {dataset.shape[1]} features")
    return dataset, events


def get_ohlcv():
    cfg = get_config()
    override = st.session_state.get("csv_override_5m")
    if override is not None:
        tfs = get_multi_timeframe_data(cfg.data, cfg.enabled_timeframes, override_5m=override)
        base_tf = cfg.data.base_timeframe if cfg.data.base_timeframe in tfs else next(iter(tfs))
        sr_by_tf = {base_tf: generate_external_sr_levels(tfs[base_tf], base_tf)}
        return tfs, sr_by_tf
    return _get_ohlcv(_cfg_hash(cfg), cfg)


def get_dataset():
    cfg = get_config()
    tfs, sr_by_tf = get_ohlcv()
    if st.session_state.get("csv_override_5m") is not None:
        return build_feature_dataset(cfg, tfs, sr_by_tf)
    return _get_dataset(_cfg_hash(cfg), cfg, tfs, sr_by_tf)


get_dataset.clear = lambda: (_get_ohlcv.clear(), _get_dataset.clear())


def get_training_result():
    return st.session_state.get("training_result")


def set_training_result(result) -> None:
    st.session_state["training_result"] = result


def get_predictions():
    return st.session_state.get("predictions")


def set_predictions(preds) -> None:
    st.session_state["predictions"] = preds


def get_backtest_result():
    return st.session_state.get("backtest_result")


def set_backtest_result(result) -> None:
    st.session_state["backtest_result"] = result
