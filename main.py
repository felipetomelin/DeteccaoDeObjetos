import os
import numpy as np
from tqdm import tqdm
import shutil
import cv2
from ultralytics import YOLO, solutions
from collections import Counter

# Anotações
#No Cityscapes:
#person: ID 24
#car: ID 26

#No YOLO (0-indexado para classe escolhida):
#car: 0
#pedestrian: 1

CITYSCAPES_BASE_DIR = 'cityscapes'  # Diretório base do Cityscapes
YOLO_DATASET_DIR = 'cityscapes_yolo'  # Diretório base do dataset formatado para YOLO

# IDs completos do Cityscapes: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
CITYSCAPES_CLASS_IDS_TO_NAMES = {
    24: 'person',
    25: 'rider',
    26: 'car',
    27: 'truck',
    28: 'bus',
    31: 'train',
    32: 'motorcycle',
    33: 'bicycle'
}

TARGET_CITYSCAPES_CLASSES_MAP = {
    26: {'name': 'car', 'yolo_id': 0},
    24: {'name': 'pedestrian', 'yolo_id': 1}
    # Adicione/modifique se quiser outras classes
}

# ------- IMPLEMENTAÇÃO --------

def get_yolo_bbox_from_instance_mask(mask, img_width, img_height):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    all_points = np.concatenate(contours, axis=0)
    x, y, w, h = cv2.boundingRect(all_points)

    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width_norm = w / img_width
    height_norm = h / img_height

    return x_center, y_center, width_norm, height_norm


# TRAIN e VAL
def process_cityscapes_split(split_name):
    print(f"Processando split: {split_name}...")

    img_input_dir = os.path.join(CITYSCAPES_BASE_DIR, 'leftImg8bit', split_name)
    ann_input_dir = os.path.join(CITYSCAPES_BASE_DIR, 'gtFine', split_name)

    img_output_dir = os.path.join(YOLO_DATASET_DIR, 'images', split_name)
    lbl_output_dir = os.path.join(YOLO_DATASET_DIR, 'labels', split_name)

    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(lbl_output_dir, exist_ok=True)

    city_folders = [f for f in os.listdir(img_input_dir) if os.path.isdir(os.path.join(img_input_dir, f))]

    for city in tqdm(city_folders, desc=f"Cidades em {split_name}"):
        city_img_path = os.path.join(img_input_dir, city)
        city_ann_path = os.path.join(ann_input_dir, city)

        for ann_file in os.listdir(city_ann_path):
            if ann_file.endswith('_instanceIds.png'):
                base_name = ann_file.replace('_gtFine_instanceIds.png', '')
                img_file_name = f"{base_name}_leftImg8bit.png"

                img_path_orig = os.path.join(city_img_path, img_file_name)
                ann_path_orig = os.path.join(city_ann_path, ann_file)

                if not os.path.exists(img_path_orig):
                    print(f"Imagem correspondente não encontrada para {ann_file}, pulando.")
                    continue

                shutil.copy(img_path_orig, os.path.join(img_output_dir, img_file_name))

                instance_img = cv2.imread(ann_path_orig, cv2.IMREAD_UNCHANGED)  # Lê como está (16-bit)
                img_height, img_width = instance_img.shape[:2]

                yolo_annotations = []
                unique_instance_ids = np.unique(instance_img)

                for inst_id in unique_instance_ids:
                    if inst_id < 1000:
                        continue

                    # Convenção do Cityscapes
                    class_id_cs = inst_id // 1000

                    if class_id_cs in TARGET_CITYSCAPES_CLASSES_MAP:
                        yolo_class_id = TARGET_CITYSCAPES_CLASSES_MAP[class_id_cs]['yolo_id']
                        instance_mask = (instance_img == inst_id)
                        bbox = get_yolo_bbox_from_instance_mask(instance_mask, img_width, img_height)
                        if bbox:
                            x_c, y_c, w_n, h_n = bbox
                            yolo_annotations.append(f"{yolo_class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

                if yolo_annotations:
                    lbl_file_path = os.path.join(lbl_output_dir, f"{base_name}_leftImg8bit.txt")
                    with open(lbl_file_path, 'w') as f:
                        f.write("\n".join(yolo_annotations))

def classes_count():
    # --- CONFIGURAÇÕES ---
    YOUTUBE_VIDEO_URL = 'https://www.youtube.com/watch?v=RDySyTODfQc'
    MODEL_PATH = 'runs/train/yolov8n_cityscapes_2class/weights/best.pt'
    OUTPUT_VIDEO_PATH = "youtube_object_counting_output.avi"
    temp_model = YOLO(MODEL_PATH)
    class_names = temp_model.model.names
    print(f"Classes no modelo: {class_names}")

    target_class_names_to_count = ['car', 'pedestrian']
    classes_to_count_ids = [k for k, v in class_names.items() if v in target_class_names_to_count]
    if not classes_to_count_ids:
        exit()

    print(f"Contando as seguintes classes (IDs): {target_class_names_to_count} ({classes_to_count_ids})")

    try:
        cap = cv2.VideoCapture(YOUTUBE_VIDEO_URL)
        assert cap.isOpened(), "Erro ao abrir o vídeo do YouTube. Verifique a URL"
    except Exception as e:
        exit()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    obj_counter = solutions.ObjectCounter(
        model=MODEL_PATH,
        classes=classes_to_count_ids,
    )
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        processed_frame_results = obj_counter(frame, persist=True)
        cv2.imshow("Contagem de Objetos YOLO - Video do YouTube", processed_frame_results)
        video_writer.write(processed_frame_results)

    print(f"Processamento do vídeo concluído. Vídeo salvo em: {OUTPUT_VIDEO_PATH}")

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # if os.path.exists(YOLO_DATASET_DIR):
    #     print(f"Diretório {YOLO_DATASET_DIR} já existe. Removendo para recriar...")
    #     shutil.rmtree(YOLO_DATASET_DIR)
    #
    # process_cityscapes_split('train')
    # process_cityscapes_split('val')

    classes_count()
