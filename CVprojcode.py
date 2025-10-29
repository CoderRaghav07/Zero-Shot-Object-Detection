import os
import cv2
import numpy as np
import pytesseract
import time
from PIL import Image, ImageDraw, ImageFont
import torch
import clip

# Paths
os.environ["XDG_CACHE_HOME"] = "C:/Users/Raghav Bhargava/Downloads/clip_cache"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------------- OCR FUNCTION ----------------------
def run_ocr(pil_img, min_confidence=60):
    np_img = np.array(pil_img)
    data = pytesseract.image_to_data(np_img, output_type=pytesseract.Output.DICT)

    tokens = []
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        text = data['text'][i].strip()
        if text == "":
            continue

        try:
            conf = int(data['conf'][i])
        except (ValueError, TypeError):
            conf = -1

        if conf > min_confidence:
            (x, y, w, h) = (data['left'][i], data['top'][i],
                            data['width'][i], data['height'][i])
            tokens.append({
                "text": text,
                "conf": conf,
                "bbox": (x, y, w, h)
            })
    return tokens


# ---------------------- NON-MAX SUPPRESSION ----------------------
def non_max_suppression(detections, iou_threshold=0.4):
    if not detections:
        return []

    boxes = np.array([det["box"] for det in detections])
    scores = np.array([det["score"] for det in detections])

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return [detections[i] for i in keep]


# ---------------------- ZERO SHOT DETECTION FUNCTION ----------------------
def detect_open_vocab(image_path, class_prompts, device="cpu",
                      max_props=200, clip_model_name="ViT-B/32", base_threshold=0.75):

    start_time = time.time()
    print(f"[INFO] Loading image: {image_path}")
    pil_img = Image.open(image_path).convert("RGB")
    np_img = np.array(pil_img)

    print(f"[INFO] Loading CLIP model: {clip_model_name}")
    model, preprocess = clip.load(clip_model_name, device=device)

    text_tokens = clip.tokenize(class_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    print("[INFO] Generating region proposals...")
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))
    ss.switchToSelectiveSearchFast()
    rects = ss.process()[:max_props]

    detections = []
    top_scores = []

    for (x, y, w, h) in rects:
        if w < 30 or h < 30:
            continue

        region = pil_img.crop((x, y, x + w, y + h))
        region_tensor = preprocess(region).unsqueeze(0).to(device)

        with torch.no_grad():
            image_feature = model.encode_image(region_tensor)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_feature @ text_features.T).softmax(dim=-1)
            score, class_idx = similarity[0].max(0)

        top_scores.append(score.item())
        detections.append({
            "label": class_prompts[class_idx],
            "score": float(score.item()),
            "box": (x, y, x + w, y + h)
        })

    max_score = max(top_scores) if top_scores else 0
    if max_score > 0.85:
        threshold = base_threshold - 0.1
    elif max_score < 0.65:
        threshold = base_threshold + 0.1
    else:
        threshold = base_threshold

    print(f"[INFO] Dynamic threshold set to: {threshold:.2f}")
    detections = [d for d in detections if d["score"] >= threshold]
    detections = non_max_suppression(detections, iou_threshold=0.4)

    elapsed = time.time() - start_time
    fps = 1 / elapsed if elapsed > 0 else 0
    print(f"[INFO] Processing time: {elapsed:.2f}s ({fps:.2f} FPS)")

    return np_img, detections, run_ocr(pil_img, min_confidence=60)


# ---------------------- RECYCLABILITY CHECK ----------------------
def classify_recyclability(ocr_tokens):
    recyclable = []
    for token in ocr_tokens:
        text = token["text"].upper()
        if any(code in text for code in ["PET", "HDPE", "LDPE", "PP", "PS", "PVC", "RECYCLABLE"]):
            recyclable.append(text)
    return recyclable


# ---------------------- MAIN DEMO FUNCTION ----------------------
def demo(args):
    class_prompts = ["Plastic bottle", "crumpled plastic wrapper", "plastic cup", "recycle bin"]

    img, detections, ocr_tokens = detect_open_vocab(
        args.image, class_prompts,
        device=args.device, max_props=args.max_props,
        clip_model_name=args.clip_model
    )

    recyclable_items = classify_recyclability(ocr_tokens)
    print(f"[INFO] Recyclable items detected (based on OCR): {recyclable_items}")

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()

    color_map = {
        "recycle bin": "green",
        "Plastic bottle": "red",
        "crumpled plastic wrapper": "orange",
        "plastic cup": "blue"
    }

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['label']} {det['score']:.2f}"
        color = color_map.get(det['label'], "red")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_bg = [x1, y1 - text_height - 4, x1 + text_width + 4, y1]
        draw.rectangle(text_bg, fill=color)
        draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)

    for token in ocr_tokens:
        x, y, w, h = token["bbox"]
        text = token["text"]
        draw.rectangle([x, y, x + w, y + h], outline="blue", width=2)
        draw.text((x, y - 12), text, fill="blue", font=font)


    out_path = "result_final.jpg"
    pil_img.save(out_path)

    print(f"[RESULT] Saved annotated image to {out_path}")
    print(f"[STATS] Total detections: {len(detections)} | OCR tokens: {len(ocr_tokens)} | Recyclable tags found: {len(recyclable_items)}")


# ---------------------- ARGPARSE ENTRY ----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu/cuda)")
    parser.add_argument("--max-props", type=int, default=200, help="Max proposals for selective search")
    parser.add_argument("--clip-model", default="ViT-B/32", help="CLIP model to use")
    args = parser.parse_args()

    demo(args)
