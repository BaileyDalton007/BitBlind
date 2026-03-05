import numpy as np
import cv2
import os
from google.cloud import vision

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Projects/bitblind/ocr/resources/black-nucleus-488500-r9-55236df87f19.json"

# If you are getting DefaultCredentialsError, you need to set an
#  enviorment variable for the path to your keyfile and restart your IDE.
client = vision.ImageAnnotatorClient()

def query_google_vision(image_np):
    """
    Requests OCR inference from the Google Cloud Vision OCR API.

    image_np: RGB uint8 numpy array (H, W, 3)

    Returns:
        {
            "text": str,
            "symbol_confs": list[float],
            "word_count": int
        }
    """

    # Encode numpy image to PNG bytes
    success, encoded = cv2.imencode(".png", image_np)
    if not success:
        raise RuntimeError("Failed to encode image")

    image = vision.Image(content=encoded.tobytes())
    response = client.document_text_detection(image=image)

    # Handle API error
    if response.error.message:
        raise Exception(response.error.message)

    # Handle no text detected
    if not response.full_text_annotation:
        return {
            "text": "",
            "symbol_confs": [],
            "word_count": 0
        }

    full_text = response.full_text_annotation.text.strip()
    symbol_confs = []
    word_count = 0

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_count += 1
                    for symbol in word.symbols:
                        if hasattr(symbol, "confidence"):
                            symbol_confs.append(symbol.confidence)

    return {
        "text": full_text,
        "symbol_confs": symbol_confs,
        "word_count": word_count
    }




def predict_and_annotate(img_data, x_offset=-100, y_offset=30, color=(0, 255, 0), font_size=0.8, line_thickness=2):
    """
    Annotates an input images with OCR inference from Google Cloud Vision API.

    image_np: RGB uint8 numpy array (H, W, 3)

    x_offset: horizontal pixel offset of text from corresponding bounding box
    y_offset: vertical pixel offset of text from corresponding bounding box
    color: RGB of line color
    font_size: size of labels
    line_thickness: stroke thickness of bounding boxes and labels

    Returns:
        RGB uint8 numpy array (H, W, 3) of image with annotated predictions
            from OCR API.
    """
    _, img_data_bytes = cv2.imencode(".png", img_data)

    image = vision.Image(content=img_data_bytes.tobytes())
    response = client.document_text_detection(image=image)

    annotated = img_data.copy()

    # Assume `response` is the Vision API response object
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    
                    # Get word text
                    word_text = "".join([symbol.text for symbol in word.symbols])

                    # Get bounding box vertices
                    vertices = [(v.x, v.y) for v in word.bounding_box.vertices]
                    pts = np.array(vertices, dtype=np.int32)

                    # Draw box
                    cv2.polylines(annotated, [pts], True, color, line_thickness)

                    # Put label
                    x, y = vertices[0]
                    cv2.putText(annotated, word_text, (x+x_offset, y+y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_size, color, line_thickness)

    return annotated