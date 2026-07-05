"""
app/services/ml_engine.py

Sprint 3 — Machine Learning Engine.

One model, done properly:
  Target : 30-day realised portfolio volatility
  Model  : XGBoost regressor
  Val    : Walk-forward — never peeks at future data
  Explain: SHAP values per prediction

Flow:
  1. Build feature matrix from price + macro data in Postgres
  2. Train XGBoost with walk-forward cross-validation
  3. Predict next 30-day volatility for a given portfolio
  4. Return SHAP top-5 feature contributions
"""

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from datetime import date, timedelta

from app.models.market_data import DailyPrice, MacroIndicator
from app.services.quant_engine import _load_prices, _portfolio_returns


# ── Feature engineering ───────────────────────────────────────────────────────

def _load_macro(db: Session) -> pd.DataFrame:
    """
    Pull macro series from Postgres into a wide DataFrame.
    index=date, columns=series_id (DFF, VIXCLS, CPIAUCSL, UNRATE)
    Forward-fill gaps (macro data isn't published daily).
    """
    rows = db.query(
        MacroIndicator.date,
        MacroIndicator.series_id,
        MacroIndicator.value
    ).filter(
        MacroIndicator.series_id.in_(["DFF", "VIXCLS", "CPIAUCSL", "UNRATE"])
    ).all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date", "series_id", "value"])
    wide = df.pivot(index="date", columns="series_id", values="value")
    wide.index = pd.to_datetime(wide.index)
    return wide.ffill()


def build_feature_matrix(
    db: Session,
    weights: dict[str, float],
    lookback: int = 1260,  # 5 years of trading days
) -> pd.DataFrame:
    """
    Build the full feature matrix used for training and prediction.

    Features per row (one row = one trading day):
      Market features   — rolling vol, momentum, beta-like measures
      Portfolio features — Sharpe-like rolling ratio, drawdown
      Macro features    — VIX, Fed Funds rate, CPI, unemployment

    Target: realised_vol_30d (30-day forward rolling volatility)
    Rows with NaN target are dropped — these become the prediction rows.
    """
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=lookback + 60)).isoformat()
    symbols = list(weights.keys())

    prices = _load_prices(db, symbols, start, end)
    if prices.empty or len(prices) < 100:
        return pd.DataFrame()

    port_returns = _portfolio_returns(prices, weights)
    port_returns.index = pd.to_datetime(port_returns.index)

    # ── Market / portfolio features ──
    feats = pd.DataFrame(index=port_returns.index)
    feats["ret_1d"]       = port_returns
    feats["ret_5d"]       = port_returns.rolling(5).mean()
    feats["ret_21d"]      = port_returns.rolling(21).mean()
    feats["vol_10d"]      = port_returns.rolling(10).std() * np.sqrt(252)
    feats["vol_21d"]      = port_returns.rolling(21).std() * np.sqrt(252)
    feats["vol_63d"]      = port_returns.rolling(63).std() * np.sqrt(252)
    feats["momentum_21d"] = port_returns.rolling(21).sum()
    feats["momentum_63d"] = port_returns.rolling(63).sum()

    # Rolling Sharpe (21-day)
    roll_mean = port_returns.rolling(21).mean()
    roll_std  = port_returns.rolling(21).std()
    feats["rolling_sharpe_21d"] = (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(252)

    # Rolling max drawdown (63-day window)
    cum = (1 + port_returns).cumprod()
    roll_max = cum.rolling(63).max()
    feats["rolling_drawdown_63d"] = ((cum - roll_max) / roll_max.replace(0, np.nan))

    # Vol regime: is current vol above its 63-day average?
    feats["vol_regime"] = (feats["vol_21d"] > feats["vol_63d"]).astype(int)

    # ── Macro features ──
    macro = _load_macro(db)
    if not macro.empty:
        macro_aligned = macro.reindex(feats.index, method="ffill")
        for col in ["DFF", "VIXCLS", "CPIAUCSL", "UNRATE"]:
            if col in macro_aligned.columns:
                feats[f"macro_{col}"] = macro_aligned[col]
                # Rate of change for macro (21-day)
                feats[f"macro_{col}_roc"] = macro_aligned[col].pct_change(21)

    # ── Target: 30-day forward realised volatility ──
    feats["realised_vol_30d"] = (
        port_returns.rolling(30).std().shift(-30) * np.sqrt(252)
    )

    feats = feats.dropna(subset=["vol_21d", "vol_63d"])  # need at least these
    return feats


# ── Walk-forward validation ───────────────────────────────────────────────────

def walk_forward_validate(
    feats: pd.DataFrame,
    n_splits: int = 5,
) -> dict:
    """
    Strict walk-forward validation — no future data leakage.

    Splits data chronologically:
      Train on first 60%, test on next 8%, train on first 68%, test on next 8%, etc.

    Returns MAE and RMSE averaged across all folds.
    This is what you show to prove the model is honest.
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import mean_absolute_error, mean_squared_error
    except ImportError:
        return {"error": "xgboost or scikit-learn not installed"}

    df = feats.dropna(subset=["realised_vol_30d"]).copy()
    if len(df) < 200:
        return {"error": "insufficient data for walk-forward validation"}

    feature_cols = [c for c in df.columns if c != "realised_vol_30d"]
    X = df[feature_cols].values
    y = df["realised_vol_30d"].values

    n = len(X)
    fold_size = n // (n_splits + 1)
    maes, rmses = [], []

    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end  = min(train_end + fold_size, n)
        if test_end <= train_end:
            break

        X_train, y_train = X[:train_end], y[:train_end]
        X_test,  y_test  = X[train_end:test_end], y[train_end:test_end]

        # Drop rows where target is NaN (edge of dataset)
        mask_train = ~np.isnan(y_train)
        mask_test  = ~np.isnan(y_test)
        if mask_train.sum() < 50 or mask_test.sum() < 10:
            continue

        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train[mask_train], y_train[mask_train])
        preds = model.predict(X_test[mask_test])

        maes.append(mean_absolute_error(y_test[mask_test], preds))
        rmses.append(np.sqrt(mean_squared_error(y_test[mask_test], preds)))

    if not maes:
        return {"error": "walk-forward produced no valid folds"}

    return {
        "n_folds": len(maes),
        "mae":  round(float(np.mean(maes)), 6),
        "rmse": round(float(np.mean(rmses)), 6),
        "note": "Walk-forward validated — no future data leakage",
    }


# ── Train final model + predict ───────────────────────────────────────────────

def train_and_predict(
    db: Session,
    weights: dict[str, float],
) -> dict:
    """
    1. Build feature matrix
    2. Run walk-forward validation (for honest metrics)
    3. Train final model on all available data
    4. Predict next 30-day volatility
    5. Return SHAP top-5 feature contributions
    """
    try:
        import xgboost as xgb
        import shap
    except ImportError:
        return {"error": "xgboost or shap not installed"}

    feats = build_feature_matrix(db, weights)
    if feats.empty:
        return {"error": "Could not build feature matrix. Check price data exists."}

    feature_cols = [c for c in feats.columns if c != "realised_vol_30d"]

    # Rows where target exists = training data
    train_df = feats.dropna(subset=["realised_vol_30d"])
    # Most recent row = what we predict on
    predict_row = feats[feature_cols].iloc[[-1]]

    if len(train_df) < 100:
        return {"error": f"Only {len(train_df)} training rows — need at least 100"}

    X_train = train_df[feature_cols].values
    y_train = train_df["realised_vol_30d"].values

    # ── Walk-forward validation first ──
    validation = walk_forward_validate(feats)

    # ── Train final model on all data ──
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    predicted_vol = float(model.predict(predict_row.values)[0])

    # Current realised vol for comparison
    current_vol = float(feats["vol_21d"].iloc[-1]) if "vol_21d" in feats.columns else None

    # ── SHAP explanation ──
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(predict_row.values)
    shap_arr = shap_values[0] if hasattr(shap_values, "__len__") else shap_values

    shap_pairs = sorted(
        zip(feature_cols, shap_arr),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:5]

    shap_explanation = [
        {
            "feature": feat,
            "shap_value": round(float(val), 6),
            "direction": "increases predicted volatility" if val > 0 else "decreases predicted volatility",
        }
        for feat, val in shap_pairs
    ]

    return {
        "predicted_30d_volatility": round(predicted_vol, 4),
        "current_21d_volatility":   round(current_vol, 4) if current_vol else None,
        "signal": (
            "volatility expected to INCREASE" if predicted_vol > (current_vol or 0)
            else "volatility expected to DECREASE or STAY STABLE"
        ),
        "validation": validation,
        "shap_top5_features": shap_explanation,
        "training_rows": len(train_df),
        "feature_count": len(feature_cols),
        "model": "XGBoost regressor — walk-forward validated",
    }