import numpy as np
import cv2
from queue import Queue
import matplotlib.pyplot as plt

def region_growing(image, seeds, threshold):
    #Dimensions of the input image
    height, width = image.shape

    #Create a queue for region growing
    queue = Queue()

    #Create a mask to keep track of visited pixels
    visited = np.zeros_like(image, dtype=np.uint8)

    #Create a segmented image
    segmented_image = np.zeros_like(image, dtype=np.uint8)

    #Add seed points to the queue and mark them as visited
    for seed in seeds:
        queue.put(seed)
        visited[seed] = 1

    #Define 4-connectivity neighbors
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    #Perform region growing
    while not queue.empty():
        current_pixel = queue.get()
        x, y = current_pixel

        #Add current pixel to segmented image
        segmented_image[x, y] = 255

        #Check neighbors
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy

            #Check if neighbor is within image bounds
            if 0 <= nx < height and 0 <= ny < width:
                #Check if neighbor is not visited
                if not visited[nx, ny]:
                    #Check pixel intensity difference
                    if abs(int(image[nx, ny]) - int(image[x, y])) < threshold:
                        queue.put((nx, ny))
                        visited[nx, ny] = 1

    return segmented_image

# Example usage
if __name__ == "__main__":
    #Read the image from the path
    image = cv2.imread("Images/Q2/Q2_test_image.jpg", cv2.IMREAD_GRAYSCALE)

    #Defining seed points
    seeds = [(50, 50), (100, 100), (120, 150)]
    #Set the threshold
    threshold = 20
    #Performing region growing
    segmented_image = region_growing(image, seeds, threshold)

    #Save the segmented image
    cv2.imwrite("Results/Q2/Q2_segmented_image.jpg", segmented_image)

    #Display the segmented image
    plt.figure("Segmented Image")
    plt.imshow(segmented_image, cmap='gray')
    plt.axis('off')
    plt.show()