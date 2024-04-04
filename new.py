# main.py
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from lime import lime_image
import cv2
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import io
import base64
import tensorflow as tf
from skimage import img_as_float, exposure

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model = load_model("trainedmodel.h5")

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

    return data / 255.0

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
     
    plt.savefig('static/histogram_output.png', bbox_inches='tight')

    # plt.show()
    



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
def generate_histogram(image_data, bins=100):
    hist, bins = np.histogram(image_data.flatten(), bins=bins)
    return hist, bins

def generate_lime_explanation(img_array, model):
    # Define LIME explainer for images
    explainer = lime_image.LimeImageExplainer()

    # Make predictions
    prediction = model.predict(img_array)
    print("Model Prediction:", prediction)

    # Explain predictions for the image
    explanation = explainer.explain_instance(img_array[0], model.predict, top_labels=1, hide_color=0, num_samples=1000)

    # Show the top regions contributing to the prediction
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    lime_exp_img = mark_boundaries(temp / 2 + 0.5, mask)
    return lime_exp_img

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return render_template('upload.html', error='No file part')
        
#         file = request.files['file']
        
#         if file.filename == '':
#             return render_template('upload.html', error='No selected file')
        
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(file_path)
#             return process_image(file_path, filename)
    
#     return render_template('upload.html')


# def process_image(file_path, filename):
#     # Load the uploaded image
#     img_array = numpize(file_path, img_size=(128, 128), grayscale=False)
    
#     # Prediction
#     prediction = model.predict(np.expand_dims(img_array, axis=0))
#     result = "Tuberculosis" if prediction[0] >= 0.8 else "Normal"

#     # Generate Histogram
#     hist, bins = generate_histogram(img_array)

#     # Generate LIME Explanation
#     lime_exp_img = generate_lime_explanation(np.expand_dims(img_array, axis=0), model)

# # Convert the lime_exp_img to base64 for HTML rendering
#     lime_exp_img_base64 = cv2_to_base64(lime_exp_img)

#     # Generate Grad-CAM for each layer
#     grad_cam_images = {}
#     layer_names = [layer.name for layer in model.layers]
#     for layer_name in layer_names:
#         grad_cam_img = compute_gradcam_for_layer(model, np.expand_dims(img_array, axis=0), layer_name)
#         grad_cam_images[layer_name] = grad_cam_img

#     # Convert images to base64 for HTML rendering
#     hist_img = plot_histogram(hist, bins)
#     # lime_exp_img = cv2_to_base64(lime_exp)
#     grad_cam_imgs = {layer: cv2_to_base64(grad_cam_images[layer]) for layer in grad_cam_images}

#     return render_template('result.html', filename=filename, result=result, hist_img=hist_img, lime_exp_img=lime_exp_img_base64, grad_cam_imgs=grad_cam_imgs)


import uuid

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error='No file part')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('upload.html', error='No selected file')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = process_image(file_path, filename)
            return result
    
    return render_template('upload.html')

def process_image(file_path, filename):
    # Load the uploaded image
    img_array = numpize(file_path, img_size=(128, 128), grayscale=False)
    
    # Prediction
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    result = "Tuberculosis" if prediction[0] >= 0.8 else "Normal"

    # Generate Histogram
    hist, bins = generate_histogram(img_array)
    plot_gray_scale_histogram(img_array,"histogram")
    
    # Generate LIME Explanation
    lime_exp_img = generate_lime_explanation(np.expand_dims(img_array, axis=0), model)

    # Save Lime explanation plot as a PNG file
    lime_exp_filename = f"lime_exp_{uuid.uuid4()}.png"
    lime_exp_path = os.path.join('static', lime_exp_filename)
    plt.imsave(lime_exp_path, lime_exp_img)

    # Generate Grad-CAM for each layer
    grad_cam_images = {}
    layer_names = [layer.name for layer in model.layers]
    for layer_name in layer_names:
        grad_cam_img = compute_gradcam_for_layer(model, np.expand_dims(img_array, axis=0), layer_name)
        grad_cam_images[layer_name] = grad_cam_img

    # Convert images to base64 for HTML rendering
    # hist_img = plot_histogram(hist, bins)
    grad_cam_imgs = {layer: cv2_to_base64(grad_cam_images[layer]) for layer in grad_cam_images}
    print(lime_exp_path)
    
    mfpp_exp_path=mfpp_exp(file_path)
    return render_template('result.html', filename=filename, result=result, hist_img="histogram_output.png", lime_exp_filename=lime_exp_filename, grad_cam_imgs=grad_cam_imgs, mfpp_exp=mfpp_exp_path)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def plot_histogram(hist, bins):
    plt.figure()
    plt.bar(bins[:-1], hist, width=np.diff(bins), edgecolor='k')
    plt.title('Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    hist_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return hist_img
import numpy as np
import matplotlib.pyplot as plt
import base64
import io

# def plot_gray_scale_histogram(image_data, title, bins=100):
#     '''
#     Plot Gray Scale Histogram of an Image.

#     Parameters:
#         - image_data (numpy.ndarray): Grayscale image data to plot histogram for.
#         - title (str): Title for the histogram.
#         - bins (int, optional): Number of bins for the histogram. Default is 100.

#     Returns:
#         str: Base64 encoded string representing the histogram image.

#     This function generates a histogram for a grayscale image and returns a base64 encoded string representing the image.
#     '''
#     fig, ax = plt.subplots(1, 2, figsize=(15, 6))

#     # Plot histogram
#     ax.hist(image_data.flatten(), bins=bins, color='gray', alpha=0.7)

#     img_and_hist(image, axes, bins)

#     mean_value = np.mean(image)
#     std_value = np.std(image)
#     min_value = np.min(image)
#     max_value = np.max(image)
#     # Customize plot
#     ax.set_title(title)
#     ax.set_xlabel('Pixel Intensity')
#     ax.set_ylabel('Frequency')

#     # Convert plot to base64
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     hist_img_base64 = base64.b64encode(buf.read()).decode('utf-8')

#     plt.close()

#     return hist_img_base64


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from sklearn.feature_extraction import image

# this is the supportive function for MFPP

def fragment_image(image, num_fragments):
    # Convert image to binary
    binary_image = image > np.mean(image)
    
    # Create a copy of the binary image
    fragmented_image = binary_image.copy()
    
    # Fragment the image by performing erosion iteratively
    for _ in range(num_fragments):
        fragmented_image = binary_erosion(fragmented_image)
    
    # Convert binary fragmented image back to float
    fragmented_image = fragmented_image.astype(float)
    
    return fragmented_image

def apply_mfpp(model, input_image, num_fragments):
    # Generate fragmented perturbations
    fragmented_images = [fragment_image(input_image, n) for n in range(1, num_fragments + 1)]
    
    # Perform prediction on perturbed images
    predictions = [model.predict(np.expand_dims(perturbed_img, axis=0))[0][0] for perturbed_img in fragmented_images]
    
    return predictions

# Example usage
# Load your trained model

def mfpp_exp(img_path):
    model = load_model("trained_model.h5")

    # Load and preprocess your input image
    # img_path = r'C:\Users\prasa\Desktop\FYP\Tuberculosis-71.png'
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
    input_image = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize the image

    # Specify the number of fragments for MFPP
    num_fragments = 5

    # Apply MFPP
    mfpp_predictions = apply_mfpp(model, input_image, num_fragments)

    # Visualize the MFPP predictions
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_fragments + 1), mfpp_predictions, marker='o')
    plt.xlabel("Number of Fragments")
    plt.ylabel("Model Prediction")
    plt.title("Effect of Morphological Fragmentation Perturbation Pyramid (MFPP)\n"
            "on Model Predictions for Input Image")
    plt.grid(True)

    # Detailed description below the plot
    description = """
    This graph illustrates the effect of Morphological Fragmentation Perturbation Pyramid (MFPP) 
    on the predictions of a trained Convolutional Neural Network (CNN) model for a given input image. 
    The x-axis represents the number of fragments in the MFPP, while the y-axis represents 
    the corresponding model predictions. Each point on the graph indicates the model prediction 
    when the input image is fragmented into a certain number of parts using erosion operations. 
    As the number of fragments increases, the model predictions may vary, reflecting the sensitivity 
    of the model to fragmented perturbations in the input image.
    """
    plt.text(0, -0.3, description, horizontalalignment='left', verticalalignment='center', fontsize=12, transform=plt.gca().transAxes)
    plt.savefig("static/mfpp_exp.png")
    
    return "mfpp_exp.png"
    
   
    # plt.show()


def cv2_to_base64(image):
    _, buf = cv2.imencode('.png', image)
    image_encoded = base64.b64encode(buf)
    image_base64 = image_encoded.decode('utf-8')
    return image_base64

if __name__ == '__main__':
    app.run(debug=True)
