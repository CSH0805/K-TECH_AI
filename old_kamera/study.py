import os
import torch
from ultralytics import YOLO
import shutil
import yaml
from pathlib import Path

def setup_training_environment():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.backends.cudnn.benchmark = True
    print("✅ 훈련 환경 설정 완료")

def validate_and_fix_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    paths_to_check = ['train', 'val', 'test']
    for path_key in paths_to_check:
        if path_key in data:
            path = Path(data[path_key])
            if not path.exists():
                relative_path = Path(yaml_path).parent / path.name
                if relative_path.exists():
                    data[path_key] = str(relative_path)
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return data

def train_model():
    setup_training_environment()

    yaml_file = 'data.yaml'
    if not os.path.exists(yaml_file):
        print(f"❌ YAML 파일 없음: {yaml_file}")
        return False

    validate_and_fix_yaml(yaml_file)

    device = 0 if torch.cuda.is_available() else 'cpu'

    model = YOLO('yolov8s.pt')

    results = model.train(
        data=yaml_file,
        epochs=120,
        batch=32,
        imgsz=640,
        project='.',
        name='single_training',
        device=device,
        workers=4,
        verbose=True,
        save=True,
        plots=True,
        exist_ok=True,
        cache=False,
        amp=True,
    )

    best_model = './single_training/weights/best.pt'
    final_model = './trained_yolov8s_model.pt'

    if os.path.exists(best_model):
        shutil.copy(best_model, final_model)
        print(f"🎉 훈련 완료: {final_model}")
        return True
    else:
        print("❌ 모델 훈련 실패")
        return False

if __name__ == '__main__':
    train_model()
