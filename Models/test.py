import xgboost as xgb
import numpy as np

# Create dummy data
X = np.random.rand(50, 2)
y = np.random.rand(50)

# Force it to use the GPU
try:
    model = xgb.XGBRegressor(tree_method='hist', device='cuda')
    model.fit(X, y)
    print("\n✅ SUCCESS: XGBoost is successfully using your RTX 5070!")
except Exception as e:
    print(f"\n❌ ERROR: GPU not detected. {e}")