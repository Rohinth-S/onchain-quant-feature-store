from feature_store.core import synthetic_events, build_features, walk_forward_rmse
def test_features_are_reproducible():
    a=build_features(synthetic_events(120, 11)); b=build_features(synthetic_events(120, 11))
    assert a.equals(b)
    assert {"realized_vol","order_flow_imbalance","slippage_bps"}.issubset(a.columns)
def test_walk_forward_has_holdout():
    f=build_features(synthetic_events(150))
    out=walk_forward_rmse(f, train_size=50)
    assert out["observations"]>0 and out["rmse_bps"]>=0
