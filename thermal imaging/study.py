import os
import torch
from ultralytics import YOLO
import shutil
import yaml
from pathlib import Path

def setup_advanced_training_environment():
    """고급 훈련을 위한 환경 설정"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # 고급 YOLO 설정
    os.environ['YOLO_VERBOSE'] = 'True'
    
    print("✅ 고급 훈련 환경 설정 완료")

def validate_and_fix_yaml(yaml_path):
    """YAML 파일 검증 및 경로 수정"""
    print("🔍 YAML 파일 검증 중...")
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 경로 검증 및 수정
        paths_to_check = ['train', 'val', 'test']
        for path_key in paths_to_check:
            if path_key in data:
                path = Path(data[path_key])
                if not path.exists():
                    print(f"❌ 경로가 존재하지 않습니다: {path}")
                    # 상대 경로로 변경 시도
                    relative_path = Path(yaml_path).parent / path.name
                    if relative_path.exists():
                        data[path_key] = str(relative_path)
                        print(f"✅ 경로 수정: {data[path_key]}")
                    else:
                        print(f"⚠️  경로를 찾을 수 없습니다: {path_key}")
                else:
                    print(f"✅ 경로 확인: {path_key} -> {path}")
        
        # 수정된 YAML 저장
        backup_path = yaml_path.replace('.yaml', '_backup.yaml')
        shutil.copy(yaml_path, backup_path)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        return data
        
    except Exception as e:
        print(f"❌ YAML 파일 검증 실패: {e}")
        return None

def analyze_dataset(data_yaml):
    """데이터셋 분석 및 개선 제안"""
    print("📊 데이터셋 분석 중...")
    
    try:
        # YAML 파일 검증 및 수정
        data = validate_and_fix_yaml(data_yaml)
        if data is None:
            return 0, 0
        
        # 각 경로별 이미지 수 확인
        train_path = Path(data.get('train', 'train/images'))
        val_path = Path(data.get('val', 'valid/images'))
        test_path = Path(data.get('test', 'test/images')) if 'test' in data else None
        
        # 이미지 파일 확장자
        img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        
        train_count = 0
        val_count = 0
        test_count = 0
        
        if train_path.exists():
            train_count = len([f for f in train_path.iterdir() if f.suffix.lower() in img_extensions])
        
        if val_path.exists():
            val_count = len([f for f in val_path.iterdir() if f.suffix.lower() in img_extensions])
            
        if test_path and test_path.exists():
            test_count = len([f for f in test_path.iterdir() if f.suffix.lower() in img_extensions])
        
        print(f"📁 훈련 이미지: {train_count}개")
        print(f"📁 검증 이미지: {val_count}개")
        if test_count > 0:
            print(f"📁 테스트 이미지: {test_count}개")
        
        # 클래스 정보
        nc = data.get('nc', 0)
        names = data.get('names', [])
        print(f"🏷️  클래스 수: {nc}개")
        print(f"🏷️  클래스 명: {names}")
        
        # 개선 제안
        if train_count < 500:
            print("⚠️  경고: 훈련 데이터가 부족합니다. 최소 500개 이상 권장")
        
        if val_count < 100:
            print("⚠️  경고: 검증 데이터가 부족합니다. 최소 100개 이상 권장")
            
        # 라벨 파일 확인
        train_label_path = train_path.parent / 'labels'
        val_label_path = val_path.parent / 'labels'
        
        if not train_label_path.exists():
            print("❌ 훈련 라벨 폴더를 찾을 수 없습니다")
        if not val_label_path.exists():
            print("❌ 검증 라벨 폴더를 찾을 수 없습니다")
            
        return train_count, val_count
        
    except Exception as e:
        print(f"❌ 데이터셋 분석 실패: {e}")
        return 0, 0

def optimize_training_speed():
    """훈련 속도 최적화"""
    print("⚡ 훈련 속도 최적화 중...")
    
    # PyTorch 최적화 설정
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 메모리 최적화
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # RTX 4070 12GB를 위한 메모리 설정 (80% 사용)
        torch.cuda.memory.set_per_process_memory_fraction(0.8)
    
    print("✅ 속도 최적화 완료")

def get_optimal_batch_size():
    """RTX 4070 12GB에 최적화된 배치 크기"""
    if not torch.cuda.is_available():
        return 4
    
    # GPU 메모리 확인
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    print(f"🔍 GPU 메모리: {gpu_memory:.1f}GB")
    
    # RTX 4070 12GB 기준 안전한 배치 크기
    batch_size = 8  # 메모리 에러 방지를 위해 보수적으로 설정
    
    print(f"📊 권장 배치 크기: {batch_size}")
    return batch_size

def train_fast_model():
    """빠른 훈련을 위한 최적화된 설정"""
    print("🏃‍♂️ 빠른 훈련 모드 시작...")
    
    setup_advanced_training_environment()
    optimize_training_speed()
    
    yaml_file = 'data.yaml'
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"YAML 파일을 찾을 수 없습니다: {os.path.abspath(yaml_file)}")

    # 데이터셋 분석
    train_count, val_count = analyze_dataset(yaml_file)
    
    if train_count == 0 or val_count == 0:
        print("❌ 데이터셋 경로 문제로 훈련을 중단합니다.")
        return False
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    batch_size = get_optimal_batch_size()
    
    # 속도 우선 설정
    epochs = 50  # 빠른 테스트를 위해 적은 에포크
    
    try:
        # 작은 모델 사용 (속도 우선)
        model = YOLO('yolo11n.pt')  # nano 모델로 빠른 훈련
        
        print(f"⚡ 설정: 배치크기={batch_size}, 에포크={epochs}, 모델=yolo11n")
        
        import time
        start_time = time.time()
        
        results = model.train(
            data=yaml_file,
            epochs=epochs,
            batch=batch_size,              # 메모리 안전 배치 크기
            imgsz=640,                     
            project='.',
            name='fast_model',
        
            lr0=0.01,
            lrf=0.01,
            optimizer='SGD',
        
            augment=True,
            hsv_h=0.01,
            hsv_s=0.3,
            hsv_v=0.2,
            degrees=5.0,
            translate=0.1,
            scale=0.2,
            flipud=0.0,
            fliplr=0.3,
        
            dropout=0.0,
            weight_decay=0.0001,
            patience=30,

            workers=1,                     # 메모리 에러 방지를 위해 1로 설정
            cache=False,                   # 캐시 비활성화로 메모리 절약
            amp=True,
            device=0,
            verbose=True,
            save=True,
            plots=False,
            exist_ok=True,
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        time_per_epoch = total_time / epochs
        
        print(f"⏱️  총 훈련 시간: {total_time/60:.1f}분")
        print(f"⏱️  에포크당 시간: {time_per_epoch:.1f}초")
        
        # 모델 저장
        best_model = './fast_model/weights/best.pt'
        if os.path.exists(best_model):
            shutil.copy(best_model, './Trash_model_fast.pt')
            print("🎉 빠른 훈련 완료!")
            return True
        
    except Exception as e:
        print(f"❌ 빠른 훈련 실패: {e}")
        return False

def train_advanced_model():
    """성능 개선을 위한 고급 훈련 (메모리 최적화)"""
    setup_advanced_training_environment()
    optimize_training_speed()
    
    yaml_file = 'data.yaml'
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"YAML 파일을 찾을 수 없습니다: {os.path.abspath(yaml_file)}")

    # 데이터셋 분석
    train_count, val_count = analyze_dataset(yaml_file)
    
    if train_count == 0 or val_count == 0:
        print("❌ 데이터셋 경로 문제로 훈련을 중단합니다.")
        return False
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = 0
        torch.cuda.set_device(0)
        print(f"사용 GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        device = 'cpu'
        print("CPU 모드로 실행")
    
    # 기존 모델 정리
    if os.path.exists('./advanced_model'):
        shutil.rmtree('./advanced_model')
    
    print("🚀 고급 훈련 시작...")
    
    # RTX 4070에 최적화된 배치 크기
    batch_size = 8  # 메모리 안전을 위해 8로 설정
    
    # 데이터셋 크기에 따른 동적 설정
    if train_count < 1000:
        epochs = 100  
        patience = 30
    else:
        epochs = 80
        patience = 20
    
    try:
        # YOLOv11s 모델 사용 (메모리 절약)
        model = YOLO('yolo11s.pt')  # m 대신 s 모델로 메모리 절약
        
        print(f"⚡ 설정: 배치크기={batch_size}, 에포크={epochs}, 모델=yolo11s")
        
        # 메모리 최적화된 훈련 파라미터
        results = model.train(
            data=yaml_file,
            epochs=epochs,
            batch=batch_size,           # 안전한 배치 크기
            imgsz=640,              
            project='.',
            name='advanced_model',
            
            # 학습률 설정
            lr0=0.01,               
            lrf=0.01,               
            
            # 옵티마이저 설정
            optimizer='SGD',        
            
            # 데이터 증강 설정
            augment=True,
            hsv_h=0.01,             
            hsv_s=0.3,              
            hsv_v=0.2,              
            degrees=5.0,            
            translate=0.1,          
            scale=0.2,              
            flipud=0.0,             
            fliplr=0.3,             
            
            # 정규화 설정
            dropout=0.0,            
            weight_decay=0.0001,    
            
            # 조기 종료 설정
            patience=patience,
            
            # 메모리 최적화 설정
            workers=1,                     # 워커 수를 1로 설정 (메모리 에러 방지)
            device=device,
            verbose=True,
            save=True,
            plots=False,               
            exist_ok=True,
            
            # 캐시 비활성화 (메모리 절약)
            cache=False,               # RAM 캐시 비활성화
            
            # 혼합 정밀도
            amp=True,                  
        )
        
        # 모델 저장
        trained_model_path = './advanced_model/weights/best.pt'
        new_model_path = './Trash_model_advanced_yolo11s.pt'
        
        if os.path.exists(trained_model_path):
            shutil.copy(trained_model_path, new_model_path)
            print(f"🎉 고급 훈련 완료! 새로운 모델: {new_model_path}")
            
            # 모델 크기 정보
            model_size = os.path.getsize(new_model_path) / (1024 * 1024)
            print(f"📁 모델 크기: {model_size:.1f}MB")
            
            # 성능 정보 출력
            if hasattr(results, 'results_dict'):
                print(f"📈 최고 mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
                print(f"📈 최고 mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        else:
            print("❌ 훈련된 모델을 찾을 수 없습니다.")
            return False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        print(f"❌ 훈련 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_high_performance_model():
    """고성능 훈련 (메모리 최적화)"""
    print("🎯 고성능 훈련 시작...")
    
    setup_advanced_training_environment()
    
    yaml_file = 'data.yaml'
    train_count, val_count = analyze_dataset(yaml_file)
    
    if train_count == 0 or val_count == 0:
        return False
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    try:
        model = YOLO('yolo11m.pt')  # m 모델 사용
        
        results = model.train(
            data=yaml_file,
            epochs=100,
            batch=6,                       # 메모리 안전을 위해 더 작은 배치
            imgsz=640,
            project='.',
            name='high_performance_model',
            
            # 원래 고성능 설정
            lr0=0.001,
            optimizer='AdamW',
            
            # 강한 데이터 증강
            augment=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.2,
            scale=0.9,
            shear=2.0,
            perspective=0.0002,
            flipud=0.5,
            fliplr=0.5,
            mixup=0.1,
            copy_paste=0.1,
            
            dropout=0.1,
            weight_decay=0.0005,
            patience=50,
            workers=1,                     # 메모리 안전
            device=device,
            verbose=True,
            save=True,
            plots=True,
            exist_ok=True,
            cache=False,                   # 캐시 비활성화
        )
        
        best_model = './high_performance_model/weights/best.pt'
        if os.path.exists(best_model):
            shutil.copy(best_model, './Trash_model_high_performance.pt')
            return True
            
    except Exception as e:
        print(f"❌ 고성능 훈련 실패: {e}")
        return False

def train_with_pretrained_model():
    """사전 훈련된 모델로 전이 학습"""
    print("🔄 전이 학습 시작...")
    
    # 기존에 학습한 모델이 있다면 그것부터 시작
    if os.path.exists('Trash_model_advanced_yolo11s.pt'):
        print("📦 기존 고급 모델에서 추가 학습 시작...")
        model = YOLO('Trash_model_advanced_yolo11s.pt')
    elif os.path.exists('Trash_model.pt'):
        print("📦 기존 모델에서 추가 학습 시작...")
        model = YOLO('Trash_model.pt')
    else:
        print("📦 YOLOv11s 모델로 시작...")
        model = YOLO('yolo11s.pt')  # 메모리 절약을 위해 s 모델
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    try:
        results = model.train(
            data='data.yaml',
            epochs=120,
            batch=8,                       # 안전한 배치 크기
            imgsz=640,
            project='.',
            name='transfer_model',
            
            # 전이학습 최적화 설정
            lr0=0.0001,            
            warmup_epochs=5,       
            
            # 강화된 증강
            augment=True,
            mosaic=1.0,            
            mixup=0.15,            
            copy_paste=0.3,        
            
            # 정규화
            dropout=0.2,           
            
            patience=50,
            device=device,
            workers=1,                     # 메모리 안전
            verbose=True,
            exist_ok=True,
            cache=False,                   # 캐시 비활성화
        )
        
        # 결과 저장
        best_model = './transfer_model/weights/best.pt'
        final_model = './Trash_model_transfer_yolo11s.pt'
        
        if os.path.exists(best_model):
            shutil.copy(best_model, final_model)
            print(f"🎉 전이학습 완료: {final_model}")
            return True
            
    except Exception as e:
        print(f"❌ 전이학습 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def ensemble_training():
    """앙상블을 위한 다중 모델 훈련 (메모리 최적화)"""
    print("🎭 앙상블 모델 훈련...")
    
    # 메모리 절약을 위해 작은 모델들만 사용
    models = ['yolo11n.pt', 'yolo11s.pt']
    trained_models = []
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    for i, model_name in enumerate(models):
        print(f"📦 모델 {i+1}/{len(models)} 훈련 중: {model_name}")
        
        model = YOLO(model_name)
        
        # 모델 크기에 따른 배치 크기 조정 (보수적)
        if 'n' in model_name:
            batch_size = 12
        else:  # s
            batch_size = 8
            
        if not torch.cuda.is_available():
            batch_size = max(2, batch_size // 4)
        
        try:
            results = model.train(
                data='data.yaml',
                epochs=80,
                batch=batch_size,
                imgsz=640,
                project='.',
                name=f'ensemble_model_{i}',
                
                # 각 모델마다 다른 증강 설정
                augment=True,
                hsv_h=0.015 * (i + 1),
                degrees=5.0 * (i + 1),
                mixup=0.1 + 0.05 * i,
                
                patience=30,
                device=device,
                verbose=False,
                plots=False,
                exist_ok=True,
                workers=1,                 # 메모리 안전
                cache=False,               # 캐시 비활성화
            )
            
            # 모델 저장
            best_path = f'./ensemble_model_{i}/weights/best.pt'
            if os.path.exists(best_path):
                final_path = f'./Trash_model_ensemble_{i}_yolo11.pt'
                shutil.copy(best_path, final_path)
                trained_models.append(final_path)
                print(f"✅ 모델 {i+1} 완료: {final_path}")
            
        except Exception as e:
            print(f"❌ 모델 {i+1} 훈련 실패: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"🎊 앙상블 훈련 완료! {len(trained_models)}개 모델 생성")
    return trained_models

def create_sample_yaml():
    """샘플 YAML 파일 생성 (경로가 잘못된 경우)"""
    sample_yaml = """
# 현재 디렉토리 기준 상대 경로로 수정해주세요
train: train/images
val: valid/images
test: test/images

nc: 5
names: ['Contamination', 'Glass', 'Metal', 'Paper', 'Plastic']

# Roboflow 정보 (선택사항)
roboflow:
  workspace: mvidimension-measurement
  project: plastic-segregation-od
  version: 8
  license: CC BY 4.0
  url: https://universe.roboflow.com/mvidimension-measurement/plastic-segregation-od/dataset/8
"""
    
    with open('data_sample.yaml', 'w', encoding='utf-8') as f:
        f.write(sample_yaml.strip())
    
    print("📄 샘플 YAML 파일 생성: data_sample.yaml")
    print("💡 경로를 수정한 후 data.yaml로 이름을 바꿔주세요")

def main():
    """메인 실행 함수"""
    print("🚀 YOLOv11 메모리 최적화 훈련 (RTX 4070 12GB 최적화)")
    print("=" * 60)
    
    # YAML 파일 존재 확인
    if not os.path.exists('data.yaml'):
        print("❌ data.yaml 파일을 찾을 수 없습니다.")
        create_sample_yaml()
        return
    
    choice = input("""
어떤 방법으로 훈련하시겠습니까?
1. 🏃‍♂️ 빠른 훈련 (nano 모델, 배치8, workers=1)
2. ⚖️  균형 훈련 (small 모델, 배치8, workers=1)  
3. 🎯 고성능 훈련 (medium 모델, 배치6, workers=1)
4. 🔄 전이학습 (기존 모델 기반, 배치8)
5. 🎭 앙상블 모델 훈련 (nano+small 모델)
6. 📊 데이터셋 분석만 실행
선택 (1-6): """)
    
    if choice == '1':
        success = train_fast_model()
        if success:
            print("🎊 빠른 훈련이 성공적으로 완료되었습니다!")
    elif choice == '2':
        success = train_advanced_model()
        if success:
            print("🎊 균형 훈련이 성공적으로 완료되었습니다!")
    elif choice == '3':
        success = train_high_performance_model()
        if success:
            print("🎊 고성능 훈련이 성공적으로 완료되었습니다!")
    elif choice == '4':
        success = train_with_pretrained_model()
        if success:
            print("🎊 전이학습이 성공적으로 완료되었습니다!")
    elif choice == '5':
        models = ensemble_training()
        if models:
            print(f"🎊 앙상블 훈련 완료! {len(models)}개 모델 생성")
    elif choice == '6':
        analyze_dataset('data.yaml')
    else:
        print("❌ 잘못된 선택입니다.")

if __name__ == '__main__':
    main()