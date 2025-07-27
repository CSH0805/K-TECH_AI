import os
import torch
from ultralytics import YOLO
import shutil

def setup_training_environment():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.backends.cudnn.benchmark = True
    print("✅ 훈련 환경 설정 완료")

def train_model():
    setup_training_environment()

    yaml_file = 'thermal_imaging/data.yaml'

    if not os.path.exists(yaml_file):
        print(f"❌ YAML 파일 없음: {yaml_file}")
        return False

    device = 0 if torch.cuda.is_available() else 'cpu'

    model = YOLO('yolov8s.pt')

    results = model.train(
        data=yaml_file,
        epochs=120,
        batch=32,
        imgsz=640,
        project='thermal_imaging',  # 변경된 부분
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

    best_model = './thermal_imaging/single_training/weights/best.pt'  # 변경된 부분
    final_model = './thermal_imaging/thermal_yolov8s_model.pt'  # 변경된 부분

    if os.path.exists(best_model):
        shutil.copy(best_model, final_model)
        print(f"🎉 훈련 완료: {final_model}")
        return True
    else:
        print("❌ 모델 훈련 실패")
        return False

if __name__ == '__main__':
    train_model()
