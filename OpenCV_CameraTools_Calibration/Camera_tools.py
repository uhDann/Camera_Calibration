import cv2, json
import numpy as np
import matplotlib.pyplot as plt

# Load the camera calibration results
json_file_path = './calibration.json'
image_path = "./Calibration_images/grid_new.jpeg"

def condition_image(image_path, json_file_path):

    # Change the image accounting for the camera calibration parameters
    # Camera must be callibrated before this function is called
    # Distortion correction and perspective correction are applied
    
    # Read the JSON file
    with open(json_file_path, 'r') as file: # Read the JSON file
        json_data = json.load(file)
    
    mtx = np.array(json_data['mtx'])
    dist = np.array(json_data['dist'])
    pmat = np.array(json_data['pmat'])
    maxWidth = json_data['maxWidth']
    maxHeight = json_data['maxHeight']

    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h,  w = image.shape[:2]
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    image = cv2.undistort(image, mtx, dist, None, newcameramtx)

    # Warp perspective
    image = cv2.warpPerspective(image, pmat, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

    return image

def transform_coordinates(json_file_path, coords, to_real=True):

    # Transform the coordinates
    # to_real=True: Transform from pixel to real-world coordinates
    # to_real=False: Transform from real-world to pixel coordinates

    # Read the JSON file
    with open(json_file_path, 'r') as file: # Read the JSON file
        json_data = json.load(file)
    
    trmat = np.array(json_data['trmat'])
    invtrmat = np.array(json_data['invtrmat'])
    
    coords = np.array([coords[0], coords[1], 1])

    if to_real:
        # Transform from pixel to real-world coordinates
        transformed_coords = np.dot(trmat, coords)
    else:
        # Transform from real-world to pixel coordinates
        transformed_coords = np.dot(invtrmat, coords)
    return transformed_coords[:2]

def coordinate_mapping_test(image_original, json_file_path):
    # Test the coordinate mapping function
    # image as an imput, means that condition_image function should be called before this function
    image=image_original.copy()
    # Read the JSON file
    with open(json_file_path, 'r') as file: # Read the JSON file
        json_data = json.load(file)
    
    invtrmat = np.array(json_data['invtrmat'])
    maxWidth = json_data['maxWidth']
    maxHeight = json_data['maxHeight']

    # Test the coordinate mapping function
    # Test the transformation from pixel to real-world coordinates and vice versa

    # Define the grid size in real-world coordinates
    grid_spacing_real = 0.01

    # Create a grid in real-world coordinates
    grid_real_world = []
    for x in np.arange(-0.06, 0.06, grid_spacing_real):  # Adjust the range according to your real-world coordinates
        for y in np.arange(-0.05, 0.06, grid_spacing_real):
            grid_real_world.append((x, y))

    grid_real_world = np.array(grid_real_world, dtype=np.float32)

    # Convert real-world grid coordinates to pixel coordinates using the inverse matrix
    grid_pixel_coordinates = cv2.transform(np.expand_dims(grid_real_world, axis=0), invtrmat)[0]

    # Draw the grid on the image
    for x, y in grid_pixel_coordinates:
        if 0 <= x < maxWidth and 0 <= y < maxHeight:
            cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), 2)  # Draw small circles at grid points

    return image

def display_image(image, title):
    # Displays the image
    # image: Image to display
    # title: Title of the image

    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    plt.axis('off')
    plt.show()

def grid_and_coordinate_config(image_original, json_file_path):
    # Test the grid display function and find the intersection points as well as record the configurations
    # image: Image to display
    # Returns:
    #   image_with_merged_lines: Image with the grid displayed
    #   intersections: Dictionary containing the intersection points, their IDs, and real-world coordinates, pixel coordinates
    image=image_original.copy()

    def merge_nearest_lines(image):
   
        # Merge the nearest lines
        # image: Image to be analyzed
        def merge_lines(line_positions, threshold):
            merged_lines = []
            current_line = line_positions[0]

            for line in line_positions[1:]:
                if line - current_line <= threshold:
                    continue
                else:
                    merged_lines.append(current_line)
                    current_line = line

            merged_lines.append(current_line)
            return merged_lines
        
        # Detect edges using the Canny edge detection algorithm
        # 20 and 40 are the threshold values for the Canny edge detection algorithm
        edges = cv2.Canny(image, 35, 60, apertureSize=3)
        # Detect lines using the Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=10)

        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 10:  # Horizontal line
                horizontal_lines.append(y1)
            elif abs(x1 - x2) < 10:  # Vertical line
                vertical_lines.append(x1)

        horizontal_lines = sorted(set(horizontal_lines))
        vertical_lines = sorted(set(vertical_lines))


        merged_horizontal_lines = merge_lines(horizontal_lines, 50)
        merged_vertical_lines = merge_lines(vertical_lines, 50)

        for y in merged_horizontal_lines:
            cv2.line(image, (0, y), (image.shape[1], y), (0, 255, 0), 2)

        for x in merged_vertical_lines:
            cv2.line(image, (x, 0), (x, image.shape[0]), (0, 255, 0), 2)

        return image, merged_horizontal_lines, merged_vertical_lines

    def find_intersections(horizontal_lines, vertical_lines):
        # Find the intersections of the grid lines
        # Returns a dictionary with the intersection ID as the key and the pixel and real-world coordinates as the values

        intersections = {}
        id_counter = 0

        def position(id):
            column = id % 9
            row = id // 9
            return column, row

        for y in horizontal_lines:
            for x in vertical_lines:
                column, row = position(id_counter)
                # Change -4 and 6 if ID 0 is not at the top-left corner of the grid
                intersections[id_counter] = (x, y, -4+column, 6-row)
                id_counter += 1

        return intersections
    
    def coordinate_mapping_mat(intersections, json_file_path):
        # Compute the affine transformation matrix for real world coordinate mapping to pixel coordinates 
        # (Work out the mapping between real-world and pixel coordinates specific to your setup)

        # Extract pixel and real-world coordinates
        pixels = np.array([(v[0], v[1]) for v in intersections.values()], dtype=np.float32)
        # /100 due to the world coordinate system being in meters but measured values are in centimeters
        real_world = np.array([(v[2]/100, v[3]/100) for v in intersections.values()], dtype=np.float32)    

        # Compute the affine transformation matrix
        trmat, _ = cv2.estimateAffine2D(pixels, real_world)

        # Compute the inverse transformation matrix
        invtrmat = np.linalg.inv(trmat[:, :2])
        invtrmat = np.hstack([invtrmat, -invtrmat @ trmat[:, 2].reshape(-1, 1)])

        data = {
            "trmat": trmat.tolist(),
            "invtrmat": invtrmat.tolist()
            }
        
        # Load the existing data from the JSON file
        with open(json_file_path, 'r') as json_file:
            data_full = json.load(json_file)
            
            # Replace or add the new data
            data_full["trmat"] =  data["trmat"]
            data_full["invtrmat"] = data["invtrmat"]

        with open(json_file_path, 'w') as json_file:
            json.dump(data_full, json_file, indent=4)

        print(f'Data has been saved to {json_file_path}')

        return
    
    # Utilizing the functions
    image_with_merged_lines, merged_horizontal_lines, merged_vertical_lines = merge_nearest_lines(image)
    intersections = find_intersections(merged_horizontal_lines, merged_vertical_lines)

    # Working out the mapping matrix and saving the data
    coordinate_mapping_mat(intersections, json_file_path)

    # Draw intersections on the image
    for id, (x, y, real_x, real_y) in intersections.items():
        # Draw the intersection point
        cv2.circle(image_with_merged_lines, (x, y), 5, (0, 0, 255), -1)
        # Draw the ID text
        cv2.putText(image_with_merged_lines, f'{id}', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image_with_merged_lines, f'({real_x:.2f}, {real_y:.2f})', (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return image_with_merged_lines

print(f"In Progress => Manually confirming the grid alignment with the image")

image = condition_image(image_path, json_file_path)
display_image(image, 'Conditioned Image')

print(f"Sucess 1/2 ==> Grid is identified ==> Working out the transformation matrix")
image_with_merged_lines= grid_and_coordinate_config(image, json_file_path)
display_image(image_with_merged_lines, 'Confirm the intersections and coordinates')

# Creating a coordinate grid to be confirmed
print(f"Sucess 2/2 ==> Transformation Matrix is Recorded ==> Manually confirm the grid alignment with the image")
image_matrix_test = coordinate_mapping_test(image, json_file_path)
display_image(image_matrix_test, 'Confirm the intersections and coordinates')

print(f"Process Completed ==> Restart the process if the grid is not aligned correctly")

# The grid alignment can be confirmed by checking the intersections and the coordinates

desired_coordinates = (-4/100, 1/100)  # Desired coordinates in real-world units
pixel_coordinates=transform_coordinates(json_file_path, desired_coordinates, to_real=False)  # Transform from real-world to pixel coordinates

cv2.circle(image, (int(pixel_coordinates[0]), int(pixel_coordinates[1])), 5, (255, 0, 0), 2)

display_image(image, 'Desired Coordinates')