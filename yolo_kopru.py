import sys
from ultralytics import YOLO
def ana_fonksiyon():
    if len(sys.argv) < 2:
        return
    img_path = sys.argv[1]
    try:
        model = YOLO('best.pt')
        results = model.predict(img_path, verbose=False)
        boxes = results[0].boxes
        if len(boxes) == 0:
            turkce_sinif = "SAĞLAM"
            guven_skoru = 100.00
            bbox_str = "0" 
        else:
            best_box_idx = boxes.conf.argmax()
            guven_skoru = float(boxes.conf[best_box_idx]) * 100
            cls_idx = int(boxes.cls[best_box_idx])
            ingilizce_sinif = results[0].names[cls_idx]
            if ingilizce_sinif == "damaged-cables" or str(cls_idx) == "0":
                turkce_sinif = "HASARLI"
            else:
                turkce_sinif = ingilizce_sinif.upper()
            num_boxes = len(boxes)
            bbox_coords = []
            for i in range(num_boxes):
                xyxy = boxes.xyxy[i].tolist()
                bbox_coords.append(f"{xyxy[0]:.1f},{xyxy[1]:.1f},{xyxy[2]:.1f},{xyxy[3]:.1f}")
            bbox_str = f"{num_boxes}," + ",".join(bbox_coords)
        with open('yolo_sonuc.txt', 'w', encoding='utf-8') as f:
            f.write(f"{turkce_sinif},{guven_skoru:.2f},{bbox_str}")
    except Exception as e:
        with open('yolo_sonuc.txt', 'w', encoding='utf-8') as f:
            f.write(f"HATA,{str(e)},0")
if __name__ == "__main__":
    ana_fonksiyon()