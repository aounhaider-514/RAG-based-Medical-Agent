import os
from xray_analysis import analyze_xray

def main():
    image_path = input("Enter the path to your X-ray image: ").strip()
    if not os.path.exists(image_path):
        print("❌ File does not exist. Please check the path and try again.")
        return
    try:
        result = analyze_xray(image_path)
        print("\n--- X-ray Analysis Result ---")
        print(f"Predicted Condition: {result['condition']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("Class Probabilities:")
        for cond, prob in result['class_probs'].items():
            print(f"  {cond}: {prob:.2f}")
    except Exception as e:
        print(f"Error analyzing X-ray: {e}")

if __name__ == "__main__":
    main()