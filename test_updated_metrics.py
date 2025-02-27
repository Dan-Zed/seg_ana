"""
Simple test script to verify the updated metrics implementation.
"""
from seg_ana.core.synthetic import create_circle_mask, create_mathematical_circle
from seg_ana.core.metrics import calculate_all_metrics

def main():
    print("\nTesting updated metrics implementation")
    print("--------------------------------------")
    
    # Test with OpenCV circle
    circle_opencv = create_circle_mask(size=(256, 256), radius=50, noise=0.0)
    metrics_opencv = calculate_all_metrics(circle_opencv)
    
    print("\nOpenCV Circle Metrics:")
    print(f"  Roundness:    {metrics_opencv['roundness']}")
    print(f"  Ellipticity:  {metrics_opencv['ellipticity']}")
    print(f"  Solidity:     {metrics_opencv['solidity']}")
    
    # Test with mathematical circle
    circle_math = create_mathematical_circle(size=(256, 256), radius=50)
    metrics_math = calculate_all_metrics(circle_math)
    
    print("\nMathematical Circle Metrics:")
    print(f"  Roundness:    {metrics_math['roundness']}")
    print(f"  Ellipticity:  {metrics_math['ellipticity']}")
    print(f"  Solidity:     {metrics_math['solidity']}")
    
    print("\nBoth approaches should now show roundness values of 1.0 for perfect circles.")
    print("If needed, use mathematical circles for more consistent theoretical roundness values.\n")

if __name__ == "__main__":
    main()
