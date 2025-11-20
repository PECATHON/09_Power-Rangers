from ultralytics import YOLO
import cv2
import os
folder = r"E:\My folder\codes\pecfest\PEC-Hackathon\test-imgs"
img_list = [os.path.join(folder, file) for file in os.listdir(folder)]
# print(img_list)



model = YOLO('yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt')
results = model(source=img_list, save=False, show_labels=True, show_conf=True, show_boxes=True)
# print(results)

# Create output folder
os.makedirs("cropped_tables", exist_ok=True)

for i, result in enumerate(results):
    img_path = img_list[i]
    img = cv2.imread(img_path)

    boxes = result.boxes
    class_names = model.names

    for j, box in enumerate(boxes):
        cls_id = int(box.cls[0])               # class id
        label = class_names[cls_id].lower()    # class name
        conf = float(box.conf[0])              # confidence

        # Only crop "table" detections
        if "table" in label:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cropped = img[y1:y2, x1:x2]

            out_path = f"cropped_tables/img{i}_table{j}_{conf:.2f}.jpg"
            cv2.imwrite(out_path, cropped)

            print(f"✅ Saved: {out_path}")