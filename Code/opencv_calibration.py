import numpy as np
import cv2

class OpenCVCalibration:
    def __init__(self, checkerboard_size=(13, 9), square_size=20):
        """
        Camera Calibration using OpenCV's cv2.calibrateCamera() 
        
        Parameters:
        -----------
        checkerboard_size : tuple (width, height)
            number of checkerboard's corners
        square_size : float
            one square size in mm
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # 3D World Coordinates
        self.obj_points_3d = self._generate_3d_points()
        
        # Data storage
        self.points3d_list = []
        self.points2d_list = []
        
        # Calibration results
        self.K = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        
    def _generate_3d_points(self):
        """Generate 3D world coordinates for checkerboard corners"""
        num_corners = self.checkerboard_size[0] * self.checkerboard_size[1]
        objp = np.zeros((num_corners, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 
                               0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    def detect_corners(self, image_paths):
        """
        Detect checkerboard corners from images
        
        Parameters:
        -----------
        image_paths : list of str
            image file paths
            
        Returns:
        --------
        success_count : int
            number of successful corner detections
        """
        print("\n" + "=" * 60)
        print("OpenCV Corner Detection")
        print("=" * 60)
        
        success_count = 0
        
        for i, img_path in enumerate(image_paths, 1):
            img = cv2.imread(img_path)
            if img is None:
                print(f"[{i}] Cannot read image: {img_path}")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            if ret:
                # Refine corner locations
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                self.points3d_list.append(self.obj_points_3d)
                self.points2d_list.append(corners_refined)
                
                success_count += 1
                print(f"[{i}] Corner detection success: {img_path}")
            else:
                print(f"✗ [{i}] Failed to detect corners: {img_path}")
        
        print("-" * 60)
        print(f"Total {len(image_paths)}, Success {success_count}")
        print("-" * 60)
        
        return success_count
    
    def calibrate(self, image_paths):
        """
        Full OpenCV calibration pipeline
        
        Parameters:
        -----------
        image_paths : list of str
            image file paths
            
        Returns:
        --------
        K : ndarray (3x3)
            Intrinsic matrix
        dist : ndarray
            Distortion coefficients
        rvecs : list
            Rotation vectors for each image
        tvecs : list
            Translation vectors for each image
        mean_error : float
            Mean reprojection error
        """
        print("\n" + "=" * 60)
        print("OpenCV Camera Calibration - cv2.calibrateCamera()")
        print("=" * 60)
        
        # Detect corners
        success_count = self.detect_corners(image_paths)
        
        if success_count < 3:
            print("\nError: Need at least 3 images!")
            return None, None, None, None, None
        
        # Get image size from first image
        img = cv2.imread(image_paths[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]
        
        print("\nRunning cv2.calibrateCamera()...")
        
        # OpenCV calibration
        ret, self.K, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.points3d_list,
            self.points2d_list,
            img_size,
            None,
            None
        )
        
        print("Calibration successful!")
        
        # Compute reprojection error
        total_error = 0
        for i in range(len(self.points3d_list)):
            img_pts_reproj, _ = cv2.projectPoints(
                self.points3d_list[i],
                self.rvecs[i],
                self.tvecs[i],
                self.K,
                self.dist
            )
            error = cv2.norm(self.points2d_list[i], img_pts_reproj, cv2.NORM_L2) / len(img_pts_reproj)
            total_error += error
        
        mean_error = total_error / len(self.points3d_list)
        
        print("-" * 60)
        print("Intrinsic Matrix K:")
        print(self.K)
        print("\nDistortion Coefficients:")
        print(self.dist.ravel())
        print(f"\nMean Reprojection Error: {mean_error:.4f} pixels")
        print("-" * 60)
        
        print("\n" + "=" * 60)
        print("OpenCV Calibration Done!")
        print("=" * 60)
        
        return self.K, self.dist, self.rvecs, self.tvecs, mean_error


# Comparison function
def compare_calibrations(zhang_K, zhang_error, opencv_K, opencv_error):
    """
    Compare Zhang's method and OpenCV results
    
    Parameters:
    -----------
    zhang_K : ndarray (3x3)
        Intrinsic matrix from Zhang's method
    zhang_error : float
        Reprojection error from Zhang's method
    opencv_K : ndarray (3x3)
        Intrinsic matrix from OpenCV
    opencv_error : float
        Reprojection error from OpenCV
    """
    print("\n" + "=" * 60)
    print("COMPARISON: Zhang's Method vs OpenCV")
    print("=" * 60)
    
    print("\n[Intrinsic Parameters]")
    print("-" * 60)
    print(f"{'Parameter':<15} {'Zhang':<15} {'OpenCV':<15} {'Diff':<15} {'Diff %':<15}")
    print("-" * 60)
    
    params = [
        ('fx', zhang_K[0,0], opencv_K[0,0]),
        ('fy', zhang_K[1,1], opencv_K[1,1]),
        ('cx', zhang_K[0,2], opencv_K[0,2]),
        ('cy', zhang_K[1,2], opencv_K[1,2]),
        ('skew', zhang_K[0,1], opencv_K[0,1])
    ]
    
    for name, zhang_val, opencv_val in params:
        diff = zhang_val - opencv_val
        diff_percent = abs(diff) / opencv_val * 100 if opencv_val != 0 else 0
        print(f"{name:<15} {zhang_val:<15.2f} {opencv_val:<15.2f} {diff:<15.2f} {diff_percent:<15.2f}")
    
    print("\n[Reprojection Error]")
    print("-" * 60)
    print(f"Zhang's Method:  {zhang_error:.4f} pixels")
    print(f"OpenCV:          {opencv_error:.4f} pixels")
    print(f"Difference:      {abs(zhang_error - opencv_error):.4f} pixels")
    print("-" * 60)


if __name__ == "__main__":
    import glob
    
    image_paths = glob.glob('Data/*.jpg')
    
    if len(image_paths) == 0:
        print("Error: Cannot find images")
        exit()
    
    print(f"Found {len(image_paths)} images")
    
    # OpenCV Calibration
    opencv_calib = OpenCVCalibration(
        checkerboard_size=(13, 9),
        square_size=20
    )
    
    K, dist, rvecs, tvecs, mean_error = opencv_calib.calibrate(image_paths)
    
    if K is not None:
        print("\n[Final Results]")
        print(f"fx = {K[0,0]:.2f}")
        print(f"fy = {K[1,1]:.2f}")
        print(f"cx = {K[0,2]:.2f}")
        print(f"cy = {K[1,2]:.2f}")
        print(f"Mean Reprojection Error = {mean_error:.4f} pixels")