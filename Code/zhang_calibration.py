import numpy as np
import cv2
import glob

class ZhangCalibration:
    # Initialize parameters
    def __init__(self, checkerboard_size=(13, 9), square_size=20):
        """
        Camera Calibration with Zhang's Method
        
        Parameters:
        -----------
        checkerboard_size : tuple (width, height)
            number of checkerboard's corner
            size 13x9 -> corners (13, 9)
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
        
        Parameters:
        -----------
        obj_pts : ndarray, shape (N, 2)
            3D world coordinates (X, Y) on Z=0 plane
        img_pts : ndarray, shape (N, 2)
            2D image coordinates (u, v)
        
        Returns:
        --------
        H : ndarray, shape (3, 3)
            Homography matrix
        """
        n = obj_pts.shape[0]  # number of corners
        A = [] 
        
        for i in range(n):
            X, Y = obj_pts[i, 0], obj_pts[i, 1]
            u, v = img_pts[i, 0], img_pts[i, 1]
            
            A.append([-X, -Y, -1, 0, 0, 0, u*X, u*Y, u])
            A.append([0, 0, 0, -X, -Y, -1, v*X, v*Y, v])
        
        A = np.array(A)
        
        # SVD to solve Ah = 0
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        
        # Normalize
        H = H / H[2, 2]
        
        return H
    
    def compute_homographies(self):
        print("\n" + "=" * 60)
        print("STEP 2: Homography calculation")
        print("=" * 60)
        
        if len(self.points2d_list) == 0:
            print("No corner data available. Please run detect_corners() first.")
            return
        
        for i, (obj_pts, img_pts) in enumerate(zip(self.points3d_list, self.points2d_list), 1):
            H = self.estimate_homography(obj_pts[:, :2], img_pts)  # Z=0이니까 X,Y만
            self.homographies.append(H)
            print(f"[{i}] Homography calculated.")
        
        print("-" * 60)
        print(f"total of {len(self.homographies)} Homography calculations done.")
        print("-" * 60)
        
        # First image's Homography matrix as an example
        if len(self.homographies) > 0:
            print(f"\nFirst image's Homography matrix:")
            print(self.homographies[0])
        
        return self.homographies
    
    
    ### Step 3: Intrinsic Parameter Calculation ###
    def compute_intrinsic_matrix(self):
        """
        Zhang's method to compute intrinsic matrix K
        
        Returns:
        --------
        K : ndarray, shape (3, 3)
            Intrinsic matrix 
        """
        print("\n" + "=" * 60)
        print("STEP 3: Intrinsic Matrix Calculation")
        print("=" * 60)
        
        if len(self.homographies) == 0:
            print("Homography not available. Please run compute_homographies() first.")
            return None
        
        def v_ij(H, i, j):
            """
            Compute the v_ij vector from Homography H
            
            Parameters:
            -----------
            H : ndarray, shape (3, 3)
                Homography matrix
            i, j : int
            
            Returns:
            --------
            v : ndarray, shape (6,)
                v_ij vector
            """
            return np.array([
                H[0, i] * H[0, j],
                H[0, i] * H[1, j] + H[1, i] * H[0, j],
                H[1, i] * H[1, j],
                H[2, i] * H[0, j] + H[0, i] * H[2, j],
                H[2, i] * H[1, j] + H[1, i] * H[2, j],
                H[2, i] * H[2, j]
            ])
        
        V = []
        for idx, H in enumerate(self.homographies, 1):
            # Constraint 1: v_12 (r1 ⊥ r2)
            V.append(v_ij(H, 0, 1))
            
            # Constraint 2: v_11 - v_22 (|r1| = |r2|)
            V.append(v_ij(H, 0, 0) - v_ij(H, 1, 1))
        
        V = np.array(V)
        print(f"\nV matrix Size: {V.shape}")
        
        # SVD to solve V*b = 0
        _, _, Vt = np.linalg.svd(V)
        b = Vt[-1]
        
        print(f"b vector: {b}")
        
        # Intrinsic parameters from b
        B11, B12, B22, B13, B23, B33 = b
        
        v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
        lambda_ = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
        alpha = np.sqrt(lambda_ / B11)
        beta = np.sqrt(lambda_ * B11 / (B11 * B22 - B12**2))
        gamma = -B12 * alpha**2 * beta / lambda_
        u0 = gamma * v0 / beta - B13 * alpha**2 / lambda_
        
        # Intrinsic matrix K
        self.K = np.array([
            [alpha, gamma, u0],
            [0,     beta,  v0],
            [0,     0,     1 ]
        ])
        
        print("\n" + "-" * 60)
        print("Intrinsic Matrix K:")
        print(self.K)
        print("-" * 60)
        print(f"\nCamera Parameters:")
        print(f"  fx (focal length X) = {alpha:.2f}")
        print(f"  fy (focal length Y) = {beta:.2f}")
        print(f"  cx (principal point X) = {u0:.2f}")
        print(f"  cy (principal point Y) = {v0:.2f}")
        print(f"  skew = {gamma:.6f}")
        print("-" * 60)
        
        return self.K

    ### Step 4: Extrinsic Parameter Calculation ###
    def compute_extrinsics(self):
        """
        Calculate Extrinsic parameters (R, t) for each image
        
        Returns:
        --------
        extrinsics : list of tuple
            (R, t) for each image
            R: Rotation matrix (3x3)
            t: Translation vector (3x1)
        """
        print("\n" + "=" * 60)
        print("STEP 4: Extrinsic Parameters Calculation")
        print("=" * 60)
        
        if self.K is None:
            print("Intrinsic matrix not available. Run compute_intrinsic_matrix() first")
            return None
        
        # inverse of K
        K_inv = np.linalg.inv(self.K)
        
        for idx, H in enumerate(self.homographies, 1):
            h1 = H[:, 0]
            h2 = H[:, 1]
            h3 = H[:, 2]
            
            # Compute scale lambda
            lambda_ = 1.0 / np.linalg.norm(K_inv @ h1)
            
            # r1, r2, t 
            r1 = lambda_ * (K_inv @ h1)
            r2 = lambda_ * (K_inv @ h2)
            r3 = np.cross(r1, r2)  
            t = lambda_ * (K_inv @ h3)
            
            # Approximate R
            R_approx = np.column_stack([r1, r2, r3])
            
            # SVD to orthogonalize R
            U, _, Vt = np.linalg.svd(R_approx)
            R = U @ Vt 
            self.extrinsics.append((R, t))
            
            print(f"[{idx}] Extrinsic parameters calculated.")
        
        print("-" * 60)
        print(f"Total of {len(self.extrinsics)} images' Extrinsic parameters calculated")
        print("-" * 60)
        
        if len(self.extrinsics) > 0:
            R, t = self.extrinsics[0]
            print(f"\nFirst image's Extrinsic parameters:")
            print(f"Rotation matrix R:")
            print(R)
            print(f"\nTranslation vector t:")
            print(t)
        
        return self.extrinsics
    
    ### Step 5: Reprojection Error Calculation ### 
    def compute_reprojection_errors(self):
        """
        Compute reprojection errors for all images
        
        Returns:
        --------
        mean_error : float
            Mean reprojection error across all images
        """
        print("\n" + "=" * 60)
        print("STEP 5: Reprojection Error Calculation")
        print("=" * 60)
        
        if self.K is None or len(self.extrinsics) == 0:
            print("Intrinsic or Extrinsic parameters not available. Please run previous steps first.")
            return None
        
        total_error = 0
        total_points = 0
        errors_per_image = []
        
        for i, (obj_pts, img_pts) in enumerate(zip(self.points3d_list, self.points2d_list), 1):
            R, t = self.extrinsics[i - 1]
            
            # 3D points -> homogeneous coordinates (X, Y, Z, 1)
            pts3d_hom = np.column_stack([obj_pts, np.ones(len(obj_pts))])
            
            RT = np.column_stack([R, t])
            
            projected_pts = (self.K @ RT @ pts3d_hom.T).T
            projected_pts = projected_pts[:, :2] / projected_pts[:, 2:3]
            
            # Error calculation
            errors = np.sqrt(np.sum((img_pts - projected_pts)**2, axis=1))     
            img_mean_error = np.mean(errors)
            img_max_error = np.max(errors)
            
            errors_per_image.append((img_mean_error))
            total_error += np.sum(errors)
            total_points += len(errors)
            
            print(f"[{i}] Mean: {img_mean_error:.4f}, Max: {img_max_error:.4f} px")
            
        mean_error = total_error / total_points
        
        print("-" * 60)
        print(f"Overall Mean Reprojection Error: {mean_error:.4f} pixels")
        print(f"Per-image errors: min={min(errors_per_image):.4f}, "
              f"max={max(errors_per_image):.4f}")
        print("-" * 60)
        
        return mean_error
    
    
    ### Full Calibration Pipeline ###
    def calibrate(self, image_paths):
        """
        Full Zhang's method calibration pipeline
        
        Parameters:
        -----------
        image_paths : list of str
            image file paths lists for calbiration
        
        Returns:
        --------
        K : ndarray, shape (3, 3)
            Intrinsic matrix 
        extrinsics : list of tuple
            (R, t) for each image
            R: Rotation matrix (3x3)
            t: Translation vector (3x1)
        mean_error : float
            Mean reprojection error across all images
        """
        print("=" * 60)
        print("Zhang's Method Camera Calibration - Full Pipeline")
        print("=" * 60)
        
        # STEP 1: Detect corners
        success_count = self.detect_corners(image_paths)
        
        if success_count < 3:
            print("\nError: Need at least 3 images!")
            return None, None, None
        
        # STEP 2: Homography calculation
        self.compute_homographies()
        
        # STEP 3: Compute Intrinsic Matrix
        self.compute_intrinsic_matrix()
        
        # STEP 4: Compute Extrinsic Parameters
        self.compute_extrinsics()
        
        # STEP 5: Reprojection Error calculation
        mean_error = self.compute_reprojection_errors()
        
        print("\n" + "=" * 60)
        print("Zhang's Method calibration done!")
        print("=" * 60)
        
        # 디버깅 정보 추가
        print("\n" + "=" * 60)
        print("DEBUG INFORMATION")
        print("=" * 60)
        print(f"\n3D points (first 3 corners):")
        print(self.points3d_list[0][:3])
        print(f"\n2D points (first 3 corners):")
        print(self.points2d_list[0][:3])
        print(f"\nImage size check - 2D points range:")
        all_2d = np.vstack(self.points2d_list)
        print(f"  X: [{np.min(all_2d[:, 0]):.1f}, {np.max(all_2d[:, 0]):.1f}]")
        print(f"  Y: [{np.min(all_2d[:, 1]):.1f}, {np.max(all_2d[:, 1]):.1f}]")
        print(f"\nFirst Homography H[0,0]: {self.homographies[0][0,0]:.6f}")
        print(f"K[0,0] (fx): {self.K[0,0]:.2f}")
        print(f"K[1,1] (fy): {self.K[1,1]:.2f}")
        
        return self.K, self.extrinsics, mean_error
    
