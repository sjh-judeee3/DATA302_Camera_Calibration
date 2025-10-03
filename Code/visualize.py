import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from zhang_calibration import ZhangCalibration
from opencv_calibration import OpenCVCalibration


def visualize_corner_detection(image_path, checkerboard_size):
    """
    Visualize detected corners on checkerboard
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        img_with_corners = img.copy()
        cv2.drawChessboardCorners(img_with_corners, checkerboard_size, corners, ret)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Detected Corners ({len(corners)} points)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('../Results/corner_detection.png', dpi=150, bbox_inches='tight')
        print("Saved: corner_detection.png")
        plt.show()
    else:
        print(f"Failed to detect corners in {image_path}")


def visualize_reprojection(calib, image_idx=0):
    """
    Visualize reprojection error for Zhang's method
    """
    if calib.K is None or len(calib.extrinsics) == 0:
        print("Calibration not done. Run calibrate() first.")
        return
    
    obj_pts = calib.points3d_list[image_idx]
    img_pts = calib.points2d_list[image_idx]
    R, t = calib.extrinsics[image_idx]
    
    # Reproject 3D points to 2D
    pts3d_hom = np.column_stack([obj_pts, np.ones(len(obj_pts))])
    RT = np.column_stack([R, t])
    projected_pts = (calib.K @ RT @ pts3d_hom.T).T
    projected_pts = projected_pts[:, :2] / projected_pts[:, 2:3]
    
    # Calculate errors
    errors = np.sqrt(np.sum((img_pts - projected_pts)**2, axis=1))
    mean_error = np.mean(errors)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(img_pts[:, 0], img_pts[:, 1], 
               c='red', s=50, alpha=0.6, label='Detected')
    plt.scatter(projected_pts[:, 0], projected_pts[:, 1], 
               c='blue', s=30, alpha=0.6, marker='x', label='Reprojected')
    
    # Draw lines between corresponding points
    for i in range(len(img_pts)):
        plt.plot([img_pts[i, 0], projected_pts[i, 0]], 
                [img_pts[i, 1], projected_pts[i, 1]], 
                'gray', alpha=0.3, linewidth=0.5)
    
    plt.title(f"Reprojection Visualization (Image {image_idx+1})\nMean Error: {mean_error:.4f} pixels")
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../Results/reprojection_error.png', dpi=150, bbox_inches='tight')
    print("Saved: reprojection_error.png")
    plt.show()


def visualize_undistortion(image_path, K, dist):
    """
    Visualize undistorted image using OpenCV results
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # Get optimal camera matrix
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    
    # Undistort
    undistorted = cv2.undistort(img, K, dist, None, new_K)
    
    # Crop if needed
    if roi != (0, 0, 0, 0):
        x, y, w_roi, h_roi = roi
        undistorted = undistorted[y:y+h_roi, x:x+w_roi]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original (Distorted)')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Undistorted (OpenCV)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('../Results/undistorted_image.png', dpi=150, bbox_inches='tight')
    print("Saved: undistorted_image.png")
    plt.show()


def visualize_comparison(zhang_K, opencv_K, zhang_error, opencv_error):
    """
    Visualize comparison of intrinsic parameters
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Intrinsic parameters comparison
    params = ['fx', 'fy', 'cx', 'cy']
    zhang_vals = [zhang_K[0,0], zhang_K[1,1], zhang_K[0,2], zhang_K[1,2]]
    opencv_vals = [opencv_K[0,0], opencv_K[1,1], opencv_K[0,2], opencv_K[1,2]]
    
    x = np.arange(len(params))
    width = 0.35
    
    axes[0].bar(x - width/2, zhang_vals, width, label="Zhang's Method", alpha=0.8)
    axes[0].bar(x + width/2, opencv_vals, width, label='OpenCV', alpha=0.8)
    axes[0].set_xlabel('Parameters')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Intrinsic Parameters Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(params)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reprojection error comparison
    methods = ["Zhang's\nMethod", 'OpenCV']
    errors = [zhang_error, opencv_error]
    colors = ['steelblue', 'coral']
    
    axes[1].bar(methods, errors, color=colors, alpha=0.8)
    axes[1].set_ylabel('Mean Reprojection Error (pixels)')
    axes[1].set_title('Reprojection Error Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(errors):
        axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../Results/calibration_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: calibration_comparison.png")
    plt.show()


def main():
    """
    Main visualization pipeline
    """
    print("=" * 60)
    print("Camera Calibration Visualization")
    print("=" * 60)
    
    # Load images
    image_paths = glob.glob('../Data/*.JPG')
    
    if len(image_paths) == 0:
        print("Error: No images found")
        return
    
    print(f"\nFound {len(image_paths)} images")
    
    checkerboard_size = (13, 9)
    square_size = 20
    
    # 1. Corner detection visualization
    print("\n[1] Visualizing corner detection...")
    visualize_corner_detection(image_paths[0], checkerboard_size)
    
    # 2. Run calibrations
    print("\n[2] Running Zhang's method...")
    zhang_calib = ZhangCalibration(checkerboard_size, square_size)
    zhang_K, zhang_extrinsics, zhang_error = zhang_calib.calibrate(image_paths)
    
    print("\n[3] Running OpenCV calibration...")
    opencv_calib = OpenCVCalibration(checkerboard_size, square_size)
    opencv_K, opencv_dist, opencv_rvecs, opencv_tvecs, opencv_error = opencv_calib.calibrate(image_paths)
    
    if zhang_K is None or opencv_K is None:
        print("\nCalibration failed!")
        return
    
    # 3. Reprojection visualization
    print("\n[4] Visualizing reprojection error...")
    visualize_reprojection(zhang_calib, image_idx=0)
    
    # 4. Undistortion visualization
    print("\n[5] Visualizing undistortion...")
    visualize_undistortion(image_paths[0], opencv_K, opencv_dist)
    
    # 5. Comparison visualization
    print("\n[6] Visualizing comparison...")
    visualize_comparison(zhang_K, opencv_K, zhang_error, opencv_error)
    
    print("\n" + "=" * 60)
    print("All visualizations saved!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - corner_detection.png")
    print("  - reprojection_error.png")
    print("  - undistorted_image.png")
    print("  - calibration_comparison.png")


if __name__ == "__main__":
    main()