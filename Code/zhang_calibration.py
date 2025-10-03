import numpy as np
import cv2
import glob

class ZhangCalibration:
    # Initialize parameters
    def __init__(self, checkerboard_size=(12, 8), square_size=20):
        """
        Camera Calibration with Zhang's Method
        
        Parameters:
        -----------
        checkerboard_size : tuple (width, height)
            number of checkerboard's corner
            size 13x9 -> corners (12, 8)
        square_size : float
            one square size in mm
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # 3D World Coordinate (Z=0 plane, same for all images)
        self.obj_points_3d = self._generate_3d_points()
        
        # Data from each image
        self.points2d_list = []  # 2D Image Coordinates
        self.points3d_list = []  # 3D World Coordinates
        
        # Variables for calibration results
        self.homographies = []
        self.K = None
        self.extrinsics = []
    
    
    def _generate_3d_points(self):
        """
        Create 3D World Coordinates (Z=0 plane))
        
        Assume the checkerpord is flat on the Z=0 plane
        and arranged in a grid starting from the origin.
        
        Returns:
        --------
        objp : ndarray, shape (N, 3)
            3D coordinates(X, Y, Z=0) of N checkerboard corners
        """
    
        num_corners = self.checkerboard_size[0] * self.checkerboard_size[1]
        
        # Generate (X, Y, Z) coordinates for each corner
        objp = np.zeros((num_corners, 3), np.float32)
        
        # X, Y coordinates in grid form
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 
                               0:self.checkerboard_size[1]].T.reshape(-1, 2)
        
        objp *= self.square_size
        
        return objp
    
    ### Step 1: Corner Detection ###
    def detect_corners(self, image_paths):
        """
        Detect checkerboard corners in the provided images
        
        Parameters:
        -----------
        image_paths : list of str
            image file paths lists for calbiration
        
        Returns:
        --------
        success_count : int
            number of images where corners were successfully detected
        """
        print("=" * 60)
        print("STEP 1: checkerboard corner detection")
        print("=" * 60)
        
        success_count = 0
        
        for i, img_path in enumerate(image_paths, 1):
            img = cv2.imread(img_path)
            if img is None:
                print(f"[{i}] cannot read image: {img_path}")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect corners
            ret, corners = cv2.findChessboardCorners(
                gray, 
                self.checkerboard_size, 
                None
            )
            
            if ret: # if success
                # Sub-pixel accuracy
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,      
                    0.001   
                )
                corners_refined = cv2.cornerSubPix(
                    gray, 
                    corners, 
                    (11, 11), 
                    (-1, -1),
                    criteria
                )
                
                self.points2d_list.append(corners_refined.reshape(-1, 2))
                self.points3d_list.append(self.obj_points_3d)
                
                success_count += 1
                print(f"[{i}] corner detection success: {img_path}")
                print(f" {len(corners_refined)} corners found")
            else:
                print(f"✗ [{i}] failed to detect corner: {img_path}")
        
        print("-" * 60)
        print(f"total {len(image_paths)}, success {success_count}")
        print("-" * 60)
        
        return success_count
    
    def get_corner_info(self):
        """
        print out detected corner information (debugging)
        """
        if len(self.points2d_list) == 0:
            print("No corner data available. Please run detect_corners() first.")
            return
        
        print("\n[Detected Corners]")
        print(f"Number of images: {len(self.points2d_list)}")
        print(f"Number of corners per image: {len(self.points2d_list[0])}")
        print(f"\nFirst 5 corners of first image (2D):")
        print(self.points2d_list[0][:5])
        print(f"\nCorresponding 3D world coordinates:")
        print(self.points3d_list[0][:5])
        
    ### Step 2: Homography Estimation ###
    def estimate_homography(self, obj_pts, img_pts):
        """
        DLT(Direct Linear Transform) Homography estimation
        