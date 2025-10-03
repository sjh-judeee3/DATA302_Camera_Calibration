from zhang_calibration import ZhangCalibration
import glob

if __name__ == "__main__":
    image_paths = glob.glob('Data/*.jpg')
    
    if len(image_paths) == 0:
        print("Error: Cannot find image")
        print("Save images in 'Data/'")
        exit()
    
    print(f"Found {len(image_paths)} images \n")
    
    calib = ZhangCalibration(
        checkerboard_size=(12, 8),  
        square_size=20               
    )
    
    K, extrinsics, mean_error = calib.calibrate(image_paths)