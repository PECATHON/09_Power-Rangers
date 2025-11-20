import cv2
import easyocr
import numpy as np
import json
import os



# --------------------
# CONFIG
# --------------------
folder ="cropped_tables"
image_paths=[]

for root,dirs,files in os.walk(folder):
    for file in files:
        full_path=os.path.join(root,file)
        image_paths.append(full_path)
i=0
for IMAGE_PATH in (image_paths):

    OUTPUT_JSON = f"output_fast{i}.json"
    reader = easyocr.Reader(['en'], gpu=False)  # set True if GPU available
    i+=1
    # --------------------
    # STEP 1: LOAD IMAGE
    # --------------------
    img = cv2.imread(IMAGE_PATH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)[1]

    # --------------------
    # STEP 2: OCR WITH POSITION
    # --------------------
    results = reader.readtext(img)

    # results = [ (bbox, text, confidence), ... ]

    cells = []
    for (bbox, text, conf) in results:
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)

        cells.append({
            "text": text,
            "x": x1,
            "y": y1
        })

    # --------------------
    # STEP 3: SORT BY Y
    # --------------------
    cells = sorted(cells, key=lambda x: x["y"])

    # --------------------
    # STEP 4: GROUP INTO ROWS
    # --------------------
    rows = []
    current_row = [cells[0]]
    row_y = cells[0]["y"]
    threshold = 12

    for c in cells[1:]:
        if abs(c["y"] - row_y) < threshold:
            current_row.append(c)
        else:
            rows.append(sorted(current_row, key=lambda x: x["x"]))
            current_row = [c]
            row_y = c["y"]

    rows.append(sorted(current_row, key=lambda x: x["x"]))

    # --------------------
    # STEP 5: BUILD JSON
    # --------------------
    headers = [c["text"] for c in rows[0]]

    json_data = []
    for row in rows[1:]:
        row_values = [c["text"] for c in row]
        row_obj = {}
        for i in range(len(headers)):
            key = headers[i]
            val = row_values[i] if i < len(row_values) else ""
            row_obj[key] = val
        json_data.append(row_obj)

    # --------------------
    # SAVE
    # --------------------
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)

    print("✅ Fast table extraction complete!")




