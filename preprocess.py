from tensorflow.keras.preprocessing.image import img_to_array, load_img

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocesses an image for feeding into the model.
    
    Args:
    - image_path (str): Path to the image file.
    - target_size (tuple): Target size for resizing the image.
    
    Returns:
    - numpy.ndarray: Preprocessed image as a NumPy array.
    """
    # Load the image
    img = load_img(image_path, target_size=target_size)
    
    # Convert image to array
    img_array = img_to_array(img)
    
    # Normalize pixel values
    img_array /= 255.0
    
    # Add batch dimension
    img_array = img_array[np.newaxis, ...]
    
    return img_array
