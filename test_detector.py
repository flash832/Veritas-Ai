
try:
    from src.ai_detector import AIDetector
    print("Import successful")
    # We won't instantiate it fully because it might try to download 400MB+ model which is slow and might fail in this environment if no internet or huggingface access.
    # But the user code says:
    # try: ... download ... except: ... backup ...
    # So it should handle failure.
    
    detector = AIDetector()
    print("Instantiation successful")
    
    res = detector.analyze_text("This is a test sentence to check perplexity.")
    print("Analysis result:", res)
    
except Exception as e:
    print(f"Error: {e}")
