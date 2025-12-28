from src.encode_features import encode_features

def test_encode_features(vectorization_mode: str = 'separate') -> bool:
    """
    Tests the feature encoding process for refined CV JSONs.
    
    Args:
        vectorization_mode (str): 'separate' for per-section TF-IDF, 'combined' for single text.
    
    Returns:
        bool: True if encoding succeeded, False otherwise.
    """
    print(f"Testing feature encoding with vectorization mode: {vectorization_mode}")
    success = encode_features(vectorization_mode)
    if success:
        print(f"Feature encoding test completed successfully for mode: {vectorization_mode}")
    else:
        print(f"Feature encoding test failed for mode: {vectorization_mode}")
    return success

# Executable code for Spyder debugging
test_encode_features(vectorization_mode='separate')
test_encode_features(vectorization_mode='combined')