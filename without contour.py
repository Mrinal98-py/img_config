import cv2
import numpy as np

def denoise_image(image):
    """ Apply Non-Local Means Denoising. """
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def preprocess_image(image):
    """ Convert to grayscale, blur, and threshold the image. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding to handle various lighting conditions
    binary = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return binary

def refine_image(binary_image):
    """ Apply morphological operations to clean up the image. """
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary_image, kernel, iterations=2)
    cleaned = cv2.erode(dilated, kernel, iterations=1)
    
    # Closing operation to fill small holes
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def sharpen_image(image):
    """ Sharpen the image using an unsharp mask. """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def main(image_path, output_path):
    # Load the original image
    original_image = cv2.imread(image_path)
    
    # Denoise the image
    denoised_image = denoise_image(original_image)
    
    # Preprocess the denoised image
    binary_image = preprocess_image(denoised_image)
    
    # Refine the binary image
    cleaned_image = refine_image(binary_image)
    
    # Optionally convert cleaned image back to color (for visualization)
    final_result = cv2.cvtColor(cleaned_image, cv2.COLOR_GRAY2BGR)
    
    # Sharpen the final image
    sharpened_result = sharpen_image(final_result)
    
    # Save the result
    cv2.imwrite(output_path, sharpened_result)
    
    # Display the result
    cv2.imshow("Processed Image", sharpened_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    input_image_path = 'path_to_your_image.jpg'  # Change this to your image path
    output_image_path = 'output_image.jpg'       # Output image path
    main(input_image_path, output_image_path)
