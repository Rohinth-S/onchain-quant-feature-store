from feature_store.rpc import JsonRpcClient

def test_rpc_payload_is_offline_and_network_selectable(monkeypatch):
    monkeypatch.setenv("RPC_NETWORK", "polygon")
    client = JsonRpcClient(network="arbitrum")
    payload = client.request_payload("eth_blockNumber", [])
    assert b"eth_blockNumber" in payload and client.network == "polygon" and client.url.startswith("https://")
