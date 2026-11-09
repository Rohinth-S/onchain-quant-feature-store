from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class FeatureConfig:
    horizon: int = 3
    vol_window: int = 8

def synthetic_events(n: int = 240, seed: int = 7) -> pd.DataFrame:
    rng=np.random.default_rng(seed)
    t=pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
    mid=100*np.exp(np.cumsum(rng.normal(0,0.002,n)))
    volume=rng.lognormal(7.0,0.35,n)
    buy=rng.binomial(1,0.52,n)
    gas=rng.lognormal(3.5,0.25,n)
    return pd.DataFrame({"timestamp":t,"pool":"WETH-USDC","mid_price":mid,"volume":volume,"side":np.where(buy,"buy","sell"),"gas_gwei":gas,"liquidity_usd":rng.lognormal(14,0.18,n),"tick":np.floor(np.log(mid)*100).astype(int)})

def build_features(events: pd.DataFrame, cfg: FeatureConfig=FeatureConfig()) -> pd.DataFrame:
    df=events.sort_values("timestamp").copy()
    df["log_return"]=np.log(df.mid_price).diff()
    df["realized_vol"]=df.log_return.rolling(cfg.vol_window,min_periods=3).std()*np.sqrt(1440)
    df["signed_volume"]=np.where(df.side.eq("buy"),df.volume,-df.volume)
    df["order_flow_imbalance"]=df.signed_volume.rolling(cfg.vol_window,min_periods=3).sum()/df.volume.rolling(cfg.vol_window,min_periods=3).sum()
    df["liquidity_concentration"]=df.groupby("pool")["volume"].transform(lambda s:s/s.sum())
    df["price_impact_bps"]=1e4*(df.mid_price-df.mid_price.shift(1)).abs()/df.mid_price.shift(1)
    df["slippage_bps"]=df.price_impact_bps*(1+df.volume/df.liquidity_usd*1e6)
    df["gas_cost_usd"]=df.gas_gwei*210_000*1e-9*df.mid_price
    df["target_cost_bps"]=df.slippage_bps.shift(-cfg.horizon)
    return df.dropna().reset_index(drop=True)

def walk_forward_rmse(features: pd.DataFrame, train_size: int=80) -> dict:
    x=features[["realized_vol","order_flow_imbalance","gas_cost_usd"]].to_numpy()
    y=features.target_cost_bps.to_numpy()
    preds=[]; actual=[]
    for i in range(train_size,len(features)):
        X=x[:i]; Y=y[:i]
        X1=np.c_[np.ones(len(X)),X]
        beta=np.linalg.lstsq(X1,Y,rcond=None)[0]
        preds.append(np.r_[1,x[i]]@beta); actual.append(y[i])
    err=np.asarray(preds)-np.asarray(actual)
    return {"rmse_bps":float(np.sqrt(np.mean(err**2))),"observations":len(err)}
