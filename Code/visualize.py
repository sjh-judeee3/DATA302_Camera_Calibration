import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os

# Results 폴더 생성
os.makedirs('../Results', exist_ok=True)

# 이미지 로드
image_paths = glob.glob('../Data/*.JPG')
if len(image_paths) == 0:
    print("No images found!")
    exit()

img_path = image_paths[0]
checkerboard_size = (13, 9)

print("Generating visualizations...")

# ========== 1. Corner Detection Visualization ==========
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

if ret:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    img_with_corners = img.copy()
    cv2.drawChessboardCorners(img_with_corners, checkerboard_size, corners, ret)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Detected Corners ({len(corners)} points)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('../Results/1_corner_detection.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: 1_corner_detection.png")
    plt.close()

# ========== 2. Undistorted Image (OpenCV만 가능) ==========
from opencv_calibration import OpenCVCalibration

opencv_calib = OpenCVCalibration(checkerboard_size, 20)
print("\nRunning quick OpenCV calibration for undistortion...")
opencv_K, opencv_dist, _, _, _ = opencv_calib.calibrate(image_paths)

if opencv_K is not None:
    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(opencv_K, opencv_dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, opencv_K, opencv_dist, None, new_K)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original (with distortion)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Undistorted (OpenCV)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('../Results/2_undistorted_image.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: 2_undistorted_image.png")
    plt.close()

# ========== 3. Calibration Comparison Chart ==========
zhang_vals = [7560.22, 7680.26, 1493.25, 1924.49]
opencv_vals = [6877.11, 6940.59, 1543.04, 2023.87]
params = ['fx', 'fy', 'cx', 'cy']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Intrinsic parameters
x = np.arange(len(params))
width = 0.35
axes[0].bar(x - width/2, zhang_vals, width, label="Zhang's Method", alpha=0.8, color='steelblue')
axes[0].bar(x + width/2, opencv_vals, width, label='OpenCV', alpha=0.8, color='coral')
axes[0].set_xlabel('Parameters', fontsize=12)
axes[0].set_ylabel('Value (pixels)', fontsize=12)
axes[0].set_title('Intrinsic Parameters Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(params)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

# Reprojection error
methods = ["Zhang's\nMethod", 'OpenCV']
errors = [12.04, 0.37]
colors = ['steelblue', 'coral']
bars = axes[1].bar(methods, errors, color=colors, alpha=0.8)
axes[1].set_ylabel('Mean Reprojection Error (pixels)', fontsize=12)
axes[1].set_title('Reprojection Error Comparison', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

for i, v in enumerate(errors):
    axes[1].text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('../Results/3_comparison_chart.png', dpi=200, bbox_inches='tight')
print("✓ Saved: 3_comparison_chart.png")
plt.close()

print("\n" + "=" * 60)
print("All visualizations completed!")
print("=" * 60)
print("\nFiles saved in ../Results/:")
print("  1. 1_corner_detection.png")
print("  2. 2_undistorted_image.png")
print("  3. 3_comparison_chart.png")
print("\nThese images are ready for your report!")