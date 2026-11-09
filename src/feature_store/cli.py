from .core import synthetic_events, build_features, walk_forward_rmse
def main():
    f=build_features(synthetic_events())
    print(f.tail(3).to_string(index=False))
    print(walk_forward_rmse(f))
if __name__ == "__main__": main()
