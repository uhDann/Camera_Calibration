import cv2, os, json
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_6X6_250  # Dictionary ID
SQUARES_VERTICALLY = 7               # Number of squares vertically
SQUARES_HORIZONTALLY = 5             # Number of squares horizontally
SQUARE_LENGTH = 35                   # Square side length (in pixels)
MARKER_LENGTH = 25                  # ArUco marker side length (in pixels)
MARGIN_PX = 20                       # Margins size (in pixels)

IMG_SIZE = tuple(i * SQUARE_LENGTH + 2 * MARGIN_PX for i in (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY))
OUTPUT_NAME = 'ChArUco_Marker.png'

def create_and_save_new_board():
    # Create and save a new ChArUco board
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = cv2.aruco.CharucoBoard.generateImage(board, IMG_SIZE, marginSize=MARGIN_PX)
    cv2.imwrite(OUTPUT_NAME, img)

def calibrate_perspective():
    # Calibrate the perspective transformation using known points in the image (Essential if the camera is angled)
    # (Work out the specific to your setup points)

    # All points are in format [cols, rows]
    pt_A = [371, 164]
    pt_B = [69, 866]
    pt_C = [1305, 874]
    pt_D = [1028, 166]

    # Work out the height and width of the final image using pythagoras theorem
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    # Define the 4 points of the image which will be transformed
    src_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    dst_pts = np.float32([[0, 0], [0, maxHeight - 1], [maxWidth - 1, maxHeight - 1], [maxWidth - 1, 0]])

    # Calculate Perspective transform matrix
    pmat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return pmat, maxWidth, maxHeight

def get_calibration_parameters(img_dir):
    # Define the aruco dictionary, charuco board and detector
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    
    # Load images from directory
    image_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpeg")]
    all_charuco_ids = []
    all_charuco_corners = []
    # print(image_files)
    total=0

    print(f'Intrinsic parameter calibration ==> Initiated')

    # Loop over images and extraction of corners
    for image_file in image_files:
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgSize = image.shape[:2]
        image_copy = image.copy()
        marker_corners, marker_ids, rejectedCandidates = detector.detectMarkers(image)
        
        # Check if marker_ids is None
        if marker_ids is None or len(marker_ids) == 0:  # Skip this image if no markers are detected
            print(f'No markers detected in {image_file}')
            continue

        if len(marker_ids) > 0: # If at least one marker is detected
            cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
            ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)

            if ret and len(charucoCorners) > 3:
                all_charuco_corners.append(charucoCorners)
                all_charuco_ids.append(charucoIds)
                total+=1
            # Display the image with identified ArUco markers (Optional)
            cv2.imshow('Identified ArUco Markers', image_copy)
            cv2.waitKey(10)
    cv2.destroyAllWindows()  # Close all OpenCV windows
    
    # Calibrate camera with extracted information
    _, mtx, dist, _, _ = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, imgSize, None, None)
    print(f'Intrinsic parameter calibration ==> Success (Calibrated with {total} images)')

    # Calculate the warping matrix
    print(f'Perspective callibration ==> Initiated')
    pmat, maxWidth, maxHeight = calibrate_perspective()
    print(f'Perspective callibration ==> Success')

    print(f'Callibration ==> Success')

    return mtx, dist, pmat, maxWidth, maxHeight

# Uncomment if a ChArUco board is needed
# create_and_save_new_board()

# Saving the calibration parameters to a JSON file
SENSOR = 'flir_blackfly_s_bfs-u3-16s2c'
LENS = 'cctv_lens_ir_6mm_F1.2_1/3"'
OUTPUT_JSON = 'calibration.json'

mtx, dist, pmat, maxWidth, maxHeight = get_calibration_parameters(img_dir="C:/Program Files/Teledyne/Spinnaker/src/Acquisition/Calibration_images/")
data = {"sensor": SENSOR, 
        "lens": LENS, 
        "mtx": mtx.tolist(), 
        "dist": dist.tolist(), 
        "pmat": pmat.tolist(), 
        "maxWidth": maxWidth, 
        "maxHeight": maxHeight,
        }

with open(OUTPUT_JSON, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f'Data has been saved to {OUTPUT_JSON}')