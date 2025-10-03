import glob
import numpy as np
from zhang_calibration import ZhangCalibration
from opencv_calibration import OpenCVCalibration


def compare_results(zhang_K, zhang_error, opencv_K, opencv_error):
    """
    Compare Zhang's method and OpenCV calibration results
    
    Parameters:
    -----------
    zhang_K : ndarray (3x3)
        Intrinsic matrix from Zhang's method
    zhang_error : float
        Mean reprojection error from Zhang's method
    opencv_K : ndarray (3x3)
        Intrinsic matrix from OpenCV
    opencv_error : float
        Mean reprojection error from OpenCV
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "CALIBRATION COMPARISON")
    print("=" * 80)
    
    # Intrinsic parameters comparison
    print("\n[Intrinsic Parameters Comparison]")
    print("-" * 80)
    print(f"{'Parameter':<12} {'Zhang Method':<18} {'OpenCV':<18} {'Difference':<18} {'Diff %':<14}")
    print("-" * 80)
    
    params = [
        ('fx', zhang_K[0, 0], opencv_K[0, 0]),
        ('fy', zhang_K[1, 1], opencv_K[1, 1]),
        ('cx', zhang_K[0, 2], opencv_K[0, 2]),
        ('cy', zhang_K[1, 2], opencv_K[1, 2]),
        ('skew', zhang_K[0, 1], opencv_K[0, 1])
    ]
    
    for name, zhang_val, opencv_val in params:
        diff = zhang_val - opencv_val
        diff_percent = abs(diff) / opencv_val * 100 if opencv_val != 0 else 0
        print(f"{name:<12} {zhang_val:<18.4f} {opencv_val:<18.4f} {diff:<18.4f} {diff_percent:<14.2f}%")
    
    # Reprojection error comparison
    print("\n[Reprojection Error Comparison]")
    print("-" * 80)
    print(f"{'Method':<20} {'Mean Error (pixels)':<25} {'Quality':<20}")
    print("-" * 80)
    
    # Quality assessment
    def assess_quality(error):
        if error < 0.5:
            return "Excellent"
        elif error < 1.0:
            return "Good"
        elif error < 2.0:
            return "Acceptable"
        else:
            return "Needs improvement"
    
    zhang_quality = assess_quality(zhang_error)
    opencv_quality = assess_quality(opencv_error)
    
    print(f"{'Zhang Method':<20} {zhang_error:<25.4f} {zhang_quality:<20}")
    print(f"{'OpenCV':<20} {opencv_error:<25.4f} {opencv_quality:<20}")
    print(f"{'Difference':<20} {abs(zhang_error - opencv_error):<25.4f}")
    print("-" * 80)
    
    # Analysis
    print("\n[Analysis]")
    print("-" * 80)
    
    # Check if results are similar
    fx_diff_percent = abs(zhang_K[0,0] - opencv_K[0,0]) / opencv_K[0,0] * 100
    fy_diff_percent = abs(zhang_K[1,1] - opencv_K[1,1]) / opencv_K[1,1] * 100
    
    if fx_diff_percent < 5 and fy_diff_percent < 5:
        print("Results are VERY SIMILAR (< 5% difference in focal lengths)")
        print("Zhang's method implementation is accurate!")
    elif fx_diff_percent < 10 and fy_diff_percent < 10:
        print("Results are SIMILAR (< 10% difference in focal lengths)")
        print("Minor differences may be due to:")
        print("  - OpenCV uses distortion model (Zhang's method doesn't)")
        print("  - Different optimization methods")
    else:
        print("Results show SIGNIFICANT DIFFERENCES")
        print("Possible reasons:")
        print("  - Implementation error in Zhang's method")
        print("  - Insufficient number of images")
        print("  - Poor corner detection quality")
    
    print("\nKey differences between methods:")
    print("  1. Zhang's method: No distortion correction")
    print("  2. OpenCV: Includes radial and tangential distortion")
    print("  3. OpenCV may use non-linear optimization (Levenberg-Marquardt)")
    print("-" * 80)


def main():
    """Main execution function"""
    
    image_paths = glob.glob('Data/*.jpg')
    
    if len(image_paths) == 0:
        print("Error: No images found in 'Data/' folder")
        print("Please add checkerboard images to the Data/ folder")
        return
    
    print("=" * 80)
    print(" " * 25 + "CAMERA CALIBRATION PROJECT")
    print("=" * 80)
    print(f"\nFound {len(image_paths)} images in Data/ folder")
    
    # Checkerboard configuration
    checkerboard_size = (12, 8)  
    square_size = 20              
    
    print(f"Checkerboard configuration: {checkerboard_size[0]}x{checkerboard_size[1]} corners")
    print(f"Square size: {square_size}mm\n")
    
    # ========== Zhang's Method ==========
    print("\n" + "=" * 80)
    print(" " * 30 + "ZHANG'S METHOD")
    print("=" * 80)
    
    zhang_calib = ZhangCalibration(
        checkerboard_size=checkerboard_size,
        square_size=square_size
    )
    
    zhang_K, zhang_extrinsics, zhang_error = zhang_calib.calibrate(image_paths)
    
    # ========== OpenCV Method ==========
    print("\n" + "=" * 80)
    print(" " * 30 + "OPENCV METHOD")
    print("=" * 80)
    
    opencv_calib = OpenCVCalibration(
        checkerboard_size=checkerboard_size,
        square_size=square_size
    )
    
    opencv_K, opencv_dist, opencv_rvecs, opencv_tvecs, opencv_error = opencv_calib.calibrate(image_paths)
    
    # ========== Comparison ==========
    if zhang_K is not None and opencv_K is not None:
        compare_results(zhang_K, zhang_error, opencv_K, opencv_error)
        
        # Final summary
        print("\n" + "=" * 80)
        print(" " * 30 + "CALIBRATION COMPLETE")
        print("=" * 80)
        print("\nResults saved in variables:")
        print("  - zhang_K: Intrinsic matrix (Zhang's method)")
        print("  - opencv_K: Intrinsic matrix (OpenCV)")
        print("  - opencv_dist: Distortion coefficients (OpenCV only)")
        print("\nFor report:")
        print("  - Include both intrinsic matrices")
        print("  - Compare reprojection errors")
        print("  - Discuss differences and reasons")
        print("=" * 80)
        
        return zhang_K, opencv_K, zhang_error, opencv_error
    else:
        print("\nCalibration failed! Check your images and try again.")
        return None, None, None, None


if __name__ == "__main__":
    zhang_K, opencv_K, zhang_error, opencv_error = main()