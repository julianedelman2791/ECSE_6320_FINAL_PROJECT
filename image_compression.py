import cv2
import numpy as np
import time

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image from path: {image_path}")
        return None, None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blurring
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return image, contours

def save_compressed_image(image, filename, quality):
    # Save the image with specific JPEG quality
    cv2.imwrite(filename, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

def test_compression(image_paths, quality_levels):
    results = {}
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Image cannot be loaded from {image_path}")
            continue
        
        results[image_path] = {}
        for quality in quality_levels:
            compressed_path = f'compressed_{quality}_{image_path}'
            save_compressed_image(image, compressed_path, quality)
            
            start_time = time.time()
            processed_image, contours = process_image(compressed_path)
            end_time = time.time()

            # Calculate processing time
            processing_time = end_time - start_time

            # Draw contours on the image for visualization
            if processed_image is not None:
                cv2.drawContours(processed_image, contours, -1, (0, 255, 0), 3)
                display_path = f'processed_{quality}_{image_path}'
                cv2.imwrite(display_path, processed_image)
            
            results[image_path][quality] = {
                'processing_time': processing_time,
                'contour_count': len(contours) if contours is not None else 0,
                'output_path': display_path if processed_image is not None else "Processing failed"
            }
    
    return results

def main():
    image_paths = ['rpi_86_field.png', 'rpi_empac.png', 'rpi_office_for_research.png']
    quality_levels = [90, 50, 1]  # High, medium, low quality
    results = test_compression(image_paths, quality_levels)
    
    for image_path, data in results.items():
        print(f"Results for {image_path}:")
        for quality, info in data.items():
            print(f"  Quality: {quality}%")
            print(f"  Processing Time: {info['processing_time']:.2f} seconds")
            print(f"  Number of Contours: {info['contour_count']}")
            print(f"  Processed Image Path: {info['output_path']}\n")

if __name__ == "__main__":
    main()