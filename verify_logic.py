# verify_logic.py
import numpy as np
from hybrid_predictor import predict_hybrid, FEATURE_COLUMNS

def test_prediction():
    print("Testing hybrid predictor...")
    print(f"Feature columns count: {len(FEATURE_COLUMNS)}")
    
    # Create a dummy feature vector with 15 elements
    dummy_features = [30.0, 24.0, 27.0, 10.0, 8.0, 70.0, 1013.0, 50.0, 22.0, 6.0, 29.7, 10.0, 30.0, 5, 120]
    
    try:
        result = predict_hybrid(dummy_features)
        print("Prediction successful!")
        print(f"Result: {result['final_prediction']}")
        print(f"Risk Score: {result['risk_score']}")
    except Exception as e:
        print(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    test_prediction()
