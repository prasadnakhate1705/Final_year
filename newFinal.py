
import os
import uuid
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from lime import lime_image
import cv2
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import io
import tensorflow as tf
from skimage import img_as_float, exposure
import matplotlib.pyplot as plt
import base64
from scipy.ndimage import binary_erosion
from sklearn.feature_extraction import image


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from lime import lime_image
from sklearn.linear_model import Ridge
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.imagenet_utils import preprocess_input




app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
model = load_model("trainedmodel.h5")

global_data = {}

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

def compute_gradcam_for_layer(model, img_array, layer_name, output_folder='static'):
    print("inside gradcam")
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

    # Save the superimposed image
    output_path = os.path.join("templates/frontend/src/app/lib/",f"{layer_name}_gradcam.png")
    print("saving image for "+ f"{layer_name}_gradcam.png")
    cv2.imwrite(output_path, superimposed_img)
    output_path1 = os.path.join("static/",f"{layer_name}_gradcam.png")
    print("saving image for "+ f"{layer_name}_gradcam.png")
    cv2.imwrite(output_path1, superimposed_img)

    return f"{layer_name}_gradcam.png"

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
     
    plt.savefig('templates/frontend/src/app/lib/histogram_output.png', bbox_inches='tight') 
    plt.savefig('static/histogram_output.png', bbox_inches='tight') 

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


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from lime import lime_image
from sklearn.linear_model import Ridge
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

def predict_fn(images):
    return model.predict(images)

def perturb_image(image, segments, perturbation_mask):
        perturbed_image = image.copy()
        for segment_idx in np.unique(segments):
            if perturbation_mask[segment_idx] == 0:
                perturbed_image[segments == segment_idx] = [0, 0, 0]  # Hide superpixel (black it out)
        return perturbed_image
def generate_lime_EXP(model_path ,image_path):
    # Step 1: Load the CNN model
    model = load_model(model_path)
    

    # Step 2: Load and preprocess the image
    img_path = image_path
    img = Image.open(img_path)
    img = img.convert('RGB')  # Ensure the image is in RGB format
    img = img.resize((128, 128))  # Resize the image to match the model's input shape
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Step 3: Initialize LIME Image Explainer
    explainer = lime_image.LimeImageExplainer()

    # Step 4: Define the Prediction Function

    # Step 5: Generate Explanation
    explanation = explainer.explain_instance(img_array[0], predict_fn, top_labels=5, hide_color=0, num_samples=1000)

    # Step 6: Superpixel Segmentation
    segments = slic(img_array[0], n_segments=50, compactness=10)

    # Step 7: Perturb the Image


    num_samples = 1000
    perturbations = np.random.binomial(1, 0.5, size=(num_samples, np.max(segments) + 1))

    perturbed_images = []
    for perturbation in perturbations:
        perturbed_image = perturb_image(img_array[0], segments, perturbation)
        perturbed_images.append(perturbed_image)
    perturbed_images = np.array(perturbed_images)

    # Step 8: Model Predictions on Perturbed Images
    predictions = predict_fn(perturbed_images)

    # Step 9: Local Interpretable Model
    X = perturbations
    class_of_interest = np.argmax(model.predict(img_array))
    y = predictions[:, class_of_interest]
    interpretable_model = Ridge(alpha=1.0)
    interpretable_model.fit(X, y)

    # Step 10: Importance Scores
    importance_scores = interpretable_model.coef_

    # Step 11: Visualize the Explanation
    top_superpixels = np.argsort(importance_scores)[-10:]  # Adjust the number as needed
    mask = np.zeros_like(segments)
    for superpixel in top_superpixels:
        mask[segments == superpixel] = 1

    highlighted_image = mark_boundaries(img_array[0], mask, color=(1, 0, 0))

    plt.imshow(highlighted_image)
    plt.axis('off')  # Hide the axes for better visual quality
    print('saving in static')
    plt.savefig('static/lime_expd.png', bbox_inches='tight', pad_inches=0)
    print('savimg  in templates')
    plt.savefig('templates/frontend/src/app/lib/lime_expd.png', bbox_inches='tight', pad_inches=0)
    plt.close()
# plt.show()

# def generate_lime_explanation(img_array, model):
#     # Define LIME explainer for images
#     explainer = lime_image.LimeImageExplainer()

#     # Make predictions
#     prediction = model.predict(img_array)
#     # print("Model Prediction:", prediction)

#     # Explain predictions for the image
#     explanation = explainer.explain_instance(img_array[0], model.predict, top_labels=1, hide_color=0, num_samples=1000)

#     # Show the top regions contributing to the prediction
#     temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
#     lime_exp_img = mark_boundaries(temp / 2 + 0.5, mask)
#     return lime_exp_img





def process_image(file_path, filename):
    # Load the uploaded image
    img_array = numpize(file_path, img_size=(128, 128), grayscale=False)
    
    # Prediction
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    result = "Tuberculosis" if prediction[0] >= 0.7 else "Normal"

    # Generate Histogram
    hist, bins = generate_histogram(img_array)
    plot_gray_scale_histogram(img_array, "histogram")
    
    print('generating lime exp')
    generate_lime_EXP("trainedmodel.h5",file_path)
    
    
    LRP_exp(file_path)

    # Generate Grad-CAM for each layer
    grad_cam_images = {}
    layer_names = [layer.name for layer in model.layers]
    for layer_name in layer_names:
        if (layer_name not in [ 'dense_38' ,'dense_39','dropout_19','flatten_19']):
            grad_cam_path = compute_gradcam_for_layer(model, np.expand_dims(img_array, axis=0), layer_name)
            grad_cam_images[layer_name] = grad_cam_path

    # Convert images to base64 for HTML rendering
    grad_cam_imgs = {layer: grad_cam_images[layer] for layer in grad_cam_images}
    
    mfpp_exp_path = mfpp_exp(file_path)
    print(prediction[0][0])
    pred=str(prediction[0][0])
    print('------------------------')
    print("gracam" , grad_cam_imgs)

    main_seg_gen(file_path)
    
    # Assign the data to global variables
    global_data.update({
        
        'filename': filename,
        'result': result,
        'hist_img': "histogram_output.png",
        'lime_exp_filename': "lime_expd.png",
        'grad_cam_imgs': grad_cam_imgs,
        'mfpp_exp': mfpp_exp_path,
        'predictionBymodel': pred,
        'model_stats': {
            'Model': 'CNN',
            'accuracy_train': '0.7841071486473083',
            'accuracy_val': '0.8871428370475769',
            'Subset': 'Training',
            'Training time': "1 minutes 2 seconds",
            'Training in seconds': 62.358063,
            'TP': 2776,
            'FP': 24,
            'FN': 40,
            'TN': 2760,
            'Precision': 0.988587,
            'Recall': 0.988571,
            'F1': 0.988571,
            'AUC': 0.988571,
            'Accuracy': 0.988571,
            
        },
        'get_config': {
            'name': 'Adam',
            'weight_decay': None,
            'clipnorm': None,
            'global_clipnorm': None,
            'clipvalue': None,
            'use_ema': False,
            'ema_momentum': 0.99,
            'ema_overwrite_frequency': None,
            'jit_compile': False,
            'is_legacy_optimizer': False,
            'learning_rate': 0.001,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-07,
            'amsgrad': False
        },
        'lrp_image': 'conv2d_57_lrp.png',
        'lung_seg':"lung_seg_gen.png",
        
        
    })

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
    num_fragments = 15

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
    plt.savefig("templates/frontend/src/app/lib/mfpp_exp.png")
    plt.savefig("static/mfpp_exp.png")

    
    return "mfpp_exp.png"
    
   
    # plt.show()


def cv2_to_base64(image):
    _, buf = cv2.imencode('.png', image)
    image_encoded = base64.b64encode(buf)
    image_base64 = image_encoded.decode('utf-8')
    return image_base64


#LRP
def compute_lrp_for_layer(model, img_array, layer_name):
    # Define a sub-model that outputs the activations of the specified layer
    sub_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    
    # Get the layer output
    layer_output = sub_model(img_array)
    
    # Compute the relevance using LRP rule
    relevance = layer_output * model.predict(img_array)
    return relevance

def LRP_exp(file_path):
    

    # Load your trained model
    model = load_model("trainedmodel.h5")

    # Load and preprocess your input image
    img_path = file_path
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

    # Compute LRP for each layer
    lrp_images = {}
    layer_names = [layer.name for layer in model.layers]
    for layer_name in layer_names:
        lrp_images[layer_name] = compute_lrp_for_layer(model, img_array, layer_name)

    # Visualize LRP images for each layer
    for layer_name, lrp_image in lrp_images.items():
        # Convert the KerasTensor to a NumPy array
        lrp_array = lrp_image.numpy()
        
        # If lrp_array has more than 2 dimensions, remove the batch dimension
        if len(lrp_array.shape) == 4:
            lrp_array = np.squeeze(lrp_array, axis=0)
        
        # Plot the LRP array
        if len(lrp_array.shape) == 3:  # Check if it's a multi-channel image
            for i in range(lrp_array.shape[-1]):
                plt.figure()
                plt.imshow(lrp_array[:, :, i], cmap='jet')  
                plt.title(f"LRP for Layer: {layer_name}, Channel: {i}")
                plt.axis("off")
                plt.colorbar()
                if (layer_name=='conv2d_57'):
                    plt.savefig(f"templates/frontend/src/app/lib/{layer_name}_lrp.png")
                    plt.savefig(f"static/{layer_name}_lrp.png")

        else:
            plt.figure()
            plt.imshow(lrp_array, cmap='jet')  
            plt.title(f"LRP for Layer: {layer_name}")
            plt.axis("off")
            plt.colorbar()
            # plt.show()
            

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure
from skimage.color import label2rgb, rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches
from PIL import Image

class ImageSegmentation:
    def __init__(self, input_shape, clip_limit=0.01, sqr_value=1):
        self.input_shape = input_shape
        self.clip_limit = clip_limit
        self.sqr_value = sqr_value

    def segmentize(self, X, return_only_overlay=True):
        # Ensure the image is grayscale
        if X.ndim == 3:
            X = rgb2gray(X)
        X = exposure.equalize_adapthist(X, clip_limit=self.clip_limit, nbins=256)
        thresh = threshold_otsu(image=X, nbins=256)
        thresh = X > thresh
        closing_image = closing(X > thresh, square(self.sqr_value))
        cleared = clear_border(closing_image)
        label_image = label(cleared)
        image_label_overlay = label2rgb(label_image, image=X, bg_label=0, bg_color=(0, 0, 0))
        if return_only_overlay:
            return image_label_overlay, label_image
        else:
            return X, thresh, closing_image, cleared, image_label_overlay, label_image

    def create_rectangle_box(self, ax, label_image):
        region_sum = 0
        rectangles = []
        for region in regionprops(label_image):
            region_sum += region.area
        region_avg = region_sum / len(regionprops(label_image))
        for region in regionprops(label_image):
            if region.area >= region_avg:
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='purple', linewidth=1)
                ax.add_patch(rect)
                rectangles.append(rect)
        return len(rectangles)

    def display(self, image_path):
        image = io.imread(image_path)
        fig, ax = plt.subplots(figsize=(8, 8))
        image_label_overlay, label_image = self.segmentize(image)
        ax.imshow(image_label_overlay)
        num_rectangles = self.create_rectangle_box(ax, label_image)
        plt.title(f"Number of Segments: {num_rectangles}")
        plt.axis('off')
        # plt.savefig("")
        # plt.show()
        output_folder = "static"
        os.makedirs(output_folder, exist_ok=True)  # Ensure the static folder exists
        output_path = os.path.join(output_folder, "lung_seg_gen.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        output_folder = "templates/frontend/src/app/lib/"
        os.makedirs(output_folder, exist_ok=True)  # Ensure the static folder exists
        output_path = os.path.join(output_folder, "lung_seg_gen.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        # plt.close()

def main_seg_gen(image_path):
    input_image_path = image_path  # Replace with the path to your X-ray image
    segmentation = ImageSegmentation(input_shape=(128, 128))
    segmentation.display(input_image_path)






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
            return jsonify(result)
    
    return render_template('upload.html')

@app.route('/get_global_data', methods=['GET'])
def get_global_data():
    return jsonify(global_data)



if __name__ == '__main__':
    app.run(debug=True, port=8080)
