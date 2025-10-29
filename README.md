Introduction -:
This project detects and classifies plastic waste in images using OpenAI’s CLIP model combined with Selective Search for region proposals and Tesseract OCR for reading recyclable labels (e.g., PET, HDPE, etc.).
It performs open-vocabulary object detection — meaning it can identify new object categories using only text prompts, without retraining.

Features we have used in our project 
1) Open-vocabulary detection using CLIP (ViT-B/32 by default)
2) Selective Search region proposals for candidate object regions
3) OCR (Tesseract) to detect recycling codes and text on packaging
4) Dynamic thresholding based on detection confidence
5) Non-Max Suppression (NMS) to filter overlapping boxes
6) Annotated visualization showing detected objects and OCR text boxes
7) Recyclability classification based on recognized text (PET, HDPE, etc.)

Implementation of project -:
In this project, me and my group have implemented a zero-shot object detection system that can identify different types of plastic waste and recyclable items without requiring a custom-trained dataset.
Several libraries such as OpenCV (cv2), NumPy, Pytesseract, PIL, Torch, and CLIP (Contrastive Language–Image Pretraining) are being used 
The CLIP model helps in open-vocabulary object detection, where the system can recognize objects using text prompts instead of being limited to fixed labels.
The working process of the code is as follows:
1) Image Loading and Preprocessing:
The input image is read using PIL and converted to a NumPy array for further processing.
2) Region Proposals (Selective Search):
The program uses Selective Search from OpenCV to generate multiple candidate regions in the image where objects might exist. These regions act as potential bounding boxes.
3) CLIP Model Detection:
Each proposed region is passed into the CLIP model, which compares the image region with a set of text prompts such as “plastic bottle”, “plastic cup”, “plastic wrapper”, or “recycle bin”.
CLIP computes similarity scores between each region and the text prompts and based on these scores, the system identifies which regions correspond to the given prompts.
4) Dynamic Thresholding and Non-Max Suppression (NMS):
The model automatically adjusts its detection threshold depending on confidence scores and then Non-Max Suppression (NMS) is applied to remove overlapping detections and keep the most accurate bounding boxes.
5) Optical Character Recognition (OCR) using Tesseract:
After object detection, Pytesseract is used to detect and read any text printed on the image (like recycling codes or labels) , It then extracts tokens (words) with confidence scores and bounding boxes.
6) Recyclability Classification:
The text extracted from OCR is analyzed to check for recyclable material codes such as PET, HDPE, LDPE, PP, PS, PVC, or the word RECYCLABLE.
This helps in identifying whether the detected plastic object is recyclable or not.
7) Visualization and Output:
The final step combines all detections — both from CLIP and OCR — and draws colored bounding boxes around detected objects and text. It then saves the annotated image as result_final.jpg and displaying the following :-
a) Object detections with class labels and confidence scores
b) OCR text boxes
c) Recyclability information found through text recognition
