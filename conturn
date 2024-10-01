import cv2
import numpy as np

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding to create a binary image
    binary = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    return binary

def remove_noise_and_fix_shapes(binary_image):
    # Define a kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)
    
    # Perform morphological operations: dilation followed by erosion
    dilated = cv2.dilate(binary_image, kernel, iterations=2)
    cleaned = cv2.erode(dilated, kernel, iterations=1)
    
    return cleaned

def find_and_draw_contours(cleaned_image, original_image):
    # Find contours
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image
    result_image = original_image.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)  # Draw contours in green
    
    return result_image

def main(image_path, output_path):
    # Preprocess the image
    binary_image = preprocess_image(image_path)
    
    # Remove noise and fix shapes
    cleaned_image = remove_noise_and_fix_shapes(binary_image)
    
    # Load original image for contour drawing
    original_image = cv2.imread(image_path)
    
    # Find and draw contours
    result_image = find_and_draw_contours(cleaned_image, original_image)
    
    # Save the result
    cv2.imwrite(output_path, result_image)
    
    # Display the result
    cv2.imshow("Processed Image", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    input_image_path = 'path_to_your_image.jpg'  # Change this to your image path
    output_image_path = 'output_image.jpg'       # Output image path
    main(input_image_path, output_image_path)
