import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_the_image(saving_path):
    """
    This function creates a image with 2 objects and a total of 3-pixel values
    """
    #Create a 200x200 NumPy array with balck backgorund
    img = np.zeros((200,200), dtype=np.uint8)

    #Draw the objects on that Numpy array
    cv2.circle(img, (120,60), 25, 255, -1) #This line draws a white circle on the 'img'
    cv2.rectangle(img, (50, 100), (100, 150), 200, -1)  #This line draws a gray rectangle on the 'img'

    #Save the image as a JPEG
    cv2.imwrite(saving_path, img)


def add_gaussian_noise(image, mean, std):
    """
    Add Gaussian noise to an image with specified mean and standard deviation
    """
    #Generate the noise
    noise = np.random.normal(mean, std, image.shape)

    #Adding the generated Gaussian noise to the image
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


def otsu_algorithm(image):
    #Calculate the histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    #Total number of pixels
    total_pixels = image.shape[0] * image.shape[1]

    #Initialize variables for equation
    sumT = 0
    sumB = 0
    wB = 0
    wF = 0
    varMax = 0
    threshold = 0

    #Iterating through all possible thresholds
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue

        wF = total_pixels - wB
        if wF == 0:
            break

        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sumT - sumB) / wF

        #Calculate between-class variance
        varBetween = wB * wF * (mB - mF) * (mB - mF)

        #Check if the variance is greater than current maximum
        if varBetween > varMax:
            varMax = varBetween
            threshold = t

        sumT += hist[t]

    #Apply the threshold to the image
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    return binary


if __name__ == "__main__":
    #Create the image with 2 objects and a total of 3-pixel values
    image_path = 'Images/Q1/Q1_test_image.jpg'
    create_the_image(image_path)

    #Load the created image
    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    #Add Gaussian noise to the image
    noisy_img = add_gaussian_noise(original_img, mean=50, std=50)

    #Apply Otsu's algorithm to the noisy image
    resulted_img = otsu_algorithm(noisy_img)

    #Save the resulted images
    cv2.imwrite("Results/Q1/Q1_noisy_image.jpg", noisy_img)
    cv2.imwrite("Results/Q1/Q1_otsu_image.jpg", resulted_img)

    #Display the results
    plt.figure("Otsu Thresholded Image", figsize=(10, 4))
    plt.subplot(131)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')
    plt.subplot(132)
    plt.imshow(noisy_img, cmap='gray')
    plt.title('Noisy Image with Gaussian Noise')
    plt.subplot(133)
    plt.imshow(resulted_img, cmap='gray')
    plt.title('Otsu Thresholded Image')
    plt.tight_layout()
    plt.show()
