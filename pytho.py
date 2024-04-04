# use kernel version 3.11.0
import numpy as np
from numpy import asarray
import os
from tqdm import tqdm
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout, Dense
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float, exposure
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.filters import threshold_otsu
from skimage.color import label2rgb
from skimage.morphology import closing, square
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from keras.preprocessing import image





def numpize(image_path: str, img_size=(128, 128), grayscale=False):
    """
    Convert a single image to a NumPy array.
    Parameters:
    - image_path (str): Path to the input image.
    - img_size (tuple): Tuple representing the desired size of the image after resizing.
    - grayscale (bool): If True, convert the image to grayscale.
    Returns:
    - numpy_array (numpy.ndarray): NumPy array containing the transformed image.
    """
    image = Image.open(image_path)

    # Ensure all images are in the same format
    if grayscale:
        image = image.convert('L')  # Convert to grayscale
    else:
        image = image.convert('RGB')  # Convert to RGB

    # Resize the image to the specified size without padding
    resized_image = image.resize(img_size)
    data = np.asarray(resized_image)

    return data


# img_path = r'C:\Users\prasa\Downloads\TB_Chest_Radiography_Database\Normal\Normal-999.png'

img_path = r'C:\Users\prasa\Desktop\FYP\Tuberculosis-71.png'


numpize(img_path,(128, 128),grayscale=False)

def img_and_hist(image_data, axes, bins=100):
    '''
    Plot an image along with its histogram and cumulative histogram.

    Parameters:
        - image_data (ndarray): Grayscale image data as a numpy array.
        - axes (list): List of axes for displaying the image, histogram, and cumulative histogram.
        - bins (int): Number of bins for the histogram.

    Returns:
        None

    This function displays an image along with its histogram and cumulative histogram. It takes the grayscale image data, a list of axes for plotting, and the number of bins for the histogram.
    '''
    image = img_as_float(image_data)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel Intensity')
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return

def plot_gray_scale_histogram(image, title, bins=100):
    '''
    Plot Gray Scale Histogram of an Image.

    Parameters:
        - image (numpy.ndarray): Grayscale image to plot histogram for.
        - title (str): Title for the histogram.
        - bins (int, optional): Number of bins for the histogram. Default is 100.

    Returns:
        None

    This function generates a histogram for a grayscale image and displays it along with the image.
    '''
    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    img_and_hist(image, axes, bins)

    mean_value = np.mean(image)
    std_value = np.std(image)
    min_value = np.min(image)
    max_value = np.max(image)

    axes[0].set_title('Image\nMean: {:.2f}, Std: {:.2f}, Min: {:.2f}, Max: {:.2f}'.format(
        mean_value, std_value, min_value, max_value), fontsize=12)
    axes[0].set_axis_off()

    y_min, y_max = axes[1].get_ylim()
    axes[1].set_title('Distribution of Pixel Intensities')
    axes[1].set_ylabel('Number of Pixels')
    axes[1].set_yticks(np.linspace(0, y_max, 5))

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig('images_exp/histogram_output.png', bbox_inches='tight')

    plt.show()

from skimage import io

# Load the image
image = io.imread(img_path, as_gray=True)

# Call the function to plot the grayscale histogram
plot_gray_scale_histogram(image, "Histogram of Normal-999.png")

# Ensure the plot is fully rendered before saving
plt.tight_layout()

# Save the plot as an image

# Display the plot
plt.show()


from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("trainedmodel.h5")


import numpy as np
from tensorflow.keras.preprocessing import image
# Load an example image for prediction

# img_path = "C:/Users/prasa/Downloads/TB_Chest_Radiography_Database/Tuberculosis/Tuberculosis-95.png"
# "C:\Users\prasa\Downloads\TB_Chest_Radiography_Database\Tuberculosis\Tuberculosis-77.png"
img = image.load_img(img_path, target_size=(128, 128))  # Adjust the target size based on your model's input shape
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

# Make a prediction
prediction = model.predict(img_array)
print(prediction)

# Print the prediction result
if prediction[0] >=0.8:
    print("Tuberculosis")
else:
    print("Normal")


import numpy as np
from tensorflow.keras.preprocessing import image
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# Load an example image for prediction
img_path = img_path
img = image.load_img(img_path, target_size=(128, 128))  # Adjust the target size based on your model's input shape
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

# Define LIME explainer for images
explainer = lime_image.LimeImageExplainer()

# Instantiate your CNN model (assuming you have already saved it)
# model = load_model("path_to_saved_model.h5")  # Load your saved model
# model.summary()

# Make predictions
prediction = model.predict(img_array)
print("Model Prediction:", prediction)

# Explain predictions for the image
explanation = explainer.explain_instance(img_array[0], model.predict, top_labels=1, hide_color=0, num_samples=1000)

# Show the top regions contributing to the prediction
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.title("LIME Explanation")
plt.savefig("images_exp/lime_exp")
plt.show()


import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define a function to compute Grad-CAM for a specific layer
def compute_gradcam_for_layer(model, img_array, layer_name):
    # Create a gradient model
    gradient_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    
    # Calculate gradients
    with tf.GradientTape() as tape:
        last_conv_output, model_output = gradient_model(img_array)
        tape.watch(last_conv_output)
        tape.watch(model_output)
        pred_index = tf.argmax(model_output[0])
        output = model_output[:, pred_index]

    # Get the gradients
    grads = tape.gradient(output, last_conv_output)[0]

    # Compute the guided gradients
    guided_grads = (last_conv_output[0] * grads)

    # Get the heatmap
    heatmap = tf.reduce_mean(guided_grads, axis=-1)

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    # Resize the heatmap to the original image size
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))

    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend the heatmap with the original image
    superimposed_img = cv2.addWeighted(np.uint8(img_array[0] * 255), 0.6, heatmap, 0.4, 0)

    return superimposed_img

# Load an example image for prediction
  # Path to your image
img = cv2.imread(img_path)
img = cv2.resize(img, (128, 128))  # Adjust the target size based on your model's input shape
img_array = np.expand_dims(img, axis=0) / 255.0  # Normalize the image

# Iterate over each layer and compute Grad-CAM
# model = your_model  # Replace 'your_model' with your actual model
layer_names = [layer.name for layer in model.layers]

for layer_name in layer_names:
    gradcam_img = compute_gradcam_for_layer(model, img_array, layer_name)
    plt.figure()
    plt.title(f'Grad-CAM for layer: {layer_name}')
    plt.imshow(gradcam_img)
    plt.axis('off')
    plt.savefig(f'images_exp/grad_cam_{layer_name}', bbox_inches='tight')


plt.show()


