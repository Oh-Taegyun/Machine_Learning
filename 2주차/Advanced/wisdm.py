import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tqdm

class WISDMDataset(Dataset):
    """WISDM 데이터를 위한 커스텀 데이터셋 클래스"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_preprocess_data(file_path):
    """
    WISDM 원시 데이터를 로드하고 전처리하는 함수
    
    Args:
        file_path: WISDM 원시 데이터 파일 경로
        
    Returns:
        X: 시계열 가속도 데이터 (samples, sequence_length, features)
        y: 활동 레이블
        label_encoder: 레이블 인코더
    """
    print("데이터를 로드하는 중...")
    
    # 데이터 로드 (세미콜론으로 끝나는 형태를 고려)
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # 빈 줄 제거
                # 세미콜론 제거하고 분리
                parts = line.strip().rstrip(';').split(',')
                if len(parts) == 6:  # 정상적인 데이터만
                    data.append(parts)
    
    df = pd.DataFrame(data, columns=['user', 'activity', 'timestamp', 'x', 'y', 'z'])
    
    # 빈 문자열이나 잘못된 데이터 제거
    df = df.replace('', np.nan)  # 빈 문자열을 NaN으로 변환
    df = df.dropna()  # NaN 값이 있는 행 제거
    
    # 데이터 타입 변환 (에러 처리 포함)
    try:
        df['user'] = pd.to_numeric(df['user'], errors='coerce').astype(int)
        df['x'] = pd.to_numeric(df['x'], errors='coerce').astype(float)
        df['y'] = pd.to_numeric(df['y'], errors='coerce').astype(float) 
        df['z'] = pd.to_numeric(df['z'], errors='coerce').astype(float)
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').astype(int)
        
        # 변환 후에도 NaN이 있는 행 제거
        df = df.dropna()
        
    except Exception as e:
        print(f"데이터 타입 변환 중 오류: {e}")
        # 문제가 있는 행들을 확인
        print("문제가 있는 행들:")
        for idx, row in df.iterrows():
            try:
                float(row['x'])
                float(row['y'])
                float(row['z'])
            except:
                print(f"행 {idx}: {row}")
                break
    
    print(f"총 {len(df)} 개의 데이터 포인트 로드됨")
    print("활동별 분포:")
    print(df['activity'].value_counts())
    
    # 레이블 인코딩
    label_encoder = LabelEncoder()
    df['activity_encoded'] = label_encoder.fit_transform(df['activity'])
    
    # 시계열 윈도우 생성 (200개 샘플 = 10초)
    window_size = 200
    step_size = 100  # 50% 오버랩
    
    X, y = create_time_windows(df, window_size, step_size)
    
    print(f"생성된 윈도우: {X.shape}")
    
    return X, y, label_encoder

def create_time_windows(df, window_size, step_size):
    """
    시계열 데이터를 고정 크기 윈도우로 분할하는 함수
    
    Args:
        df: 전체 데이터프레임
        window_size: 윈도우 크기 (샘플 개수)
        step_size: 슬라이딩 윈도우 스텝 크기
        
    Returns:
        X: 윈도우 데이터 (num_windows, window_size, 3)
        각각, 생성된 윈도우 갯수, 시간 스템 갯수, 가속도계 x, y, z 갯수
        y: 윈도우별 레이블
    """
    X, y = [], []
    
    # 사용자별, 활동별로 윈도우 생성
    for user in df['user'].unique():
        for activity in df['activity'].unique():
            # 해당 사용자의 해당 활동 데이터만 필터링
            subset = df[(df['user'] == user) & (df['activity'] == activity)].copy()
            
            if len(subset) < window_size:
                continue
                
            # 시간순 정렬
            subset = subset.sort_values('timestamp')
            
            # 가속도 데이터만 추출
            accel_data = subset[['x', 'y', 'z']].values
            
            # 윈도우 생성
            for i in range(0, len(accel_data) - window_size + 1, step_size):
                window_data = accel_data[i:i + window_size]
                X.append(window_data)
                y.append(subset['activity_encoded'].iloc[i])
    
    return np.array(X), np.array(y)

class ActivityRecognitionCNN(nn.Module):
    """
    1D CNN을 사용한 활동 인식 신경망
    
    시계열 가속도 데이터의 패턴을 학습하여 활동을 분류
    """
    def __init__(self, input_channels=3, num_classes=6, sequence_length=200):
        super(ActivityRecognitionCNN, self).__init__()
        
        # 1D Convolutional layers - 시간축을 따라 패턴 추출
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 200 -> 100
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 100 -> 50
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # 50 -> 25
        
        # Global Average Pooling으로 특성 벡터 생성
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 완전연결층 - 최종 분류
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # 입력: (batch_size, sequence_length, channels)
        # Conv1d를 위해 (batch_size, channels, sequence_length)로 변환
        x = x.transpose(1, 2)
        
        # 첫 번째 컨볼루션 블록
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 두 번째 컨볼루션 블록
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 세 번째 컨볼루션 블록
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # flatten
        
        # 완전연결층
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class ActivityRecognitionLSTM(nn.Module):
    """
    LSTM을 사용한 활동 인식 신경망
    
    시계열 데이터의 장기 의존성을 학습하여 활동 패턴 인식
    """
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, num_classes=6):
        super(ActivityRecognitionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어 - 시계열 패턴 학습
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        
        # 양방향 LSTM이므로 hidden_size * 2
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # LSTM에 입력: (batch_size, sequence_length, input_size)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # 마지막 시간 스텝의 출력 사용
        last_output = lstm_out[:, -1, :]
        
        # 완전연결층
        x = self.dropout(last_output)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, model_name='model'):
    """
    모델 훈련 함수
    
    Args:
        model: 훈련할 신경망 모델
        train_loader: 훈련 데이터 로더
        val_loader: 검증 데이터 로더
        num_epochs: 훈련 에포크 수
        learning_rate: 학습률
        
    Returns:
        train_losses: 훈련 손실 히스토리
        val_accuracies: 검증 정확도 히스토리
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                    patience=5, factor=0.5)
    
    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    for epoch in tqdm.tqdm(range(num_epochs)):
        # 훈련 단계
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 검증 단계
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        
        # 학습률 스케줄러 업데이트
        scheduler.step(val_acc)
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'./best_{model_name}_model.pth')
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Accuracy: {val_acc:.2f}%')
    
    print(f'최고 검증 정확도: {best_val_acc:.2f}%')
    return train_losses, val_accuracies

def evaluate_model(model, test_loader, label_encoder):
    """
    모델 평가 및 결과 분석
    
    Args:
        model: 평가할 모델
        test_loader: 테스트 데이터 로더
        label_encoder: 레이블 인코더
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 정확도 계산
    accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
    print(f'테스트 정확도: {accuracy * 100:.2f}%')
    
    # 활동별 정확도 분석
    from collections import Counter
    activity_names = label_encoder.classes_
    
    print("\n활동별 예측 결과:")
    for i, activity in enumerate(activity_names):
        mask = np.array(all_targets) == i
        if np.sum(mask) > 0:
            activity_acc = np.mean(np.array(all_predictions)[mask] == i)
            print(f"{activity}: {activity_acc * 100:.2f}%")

def plot_training_history(train_losses, val_accuracies, model_name='Model'):
    """훈련 과정 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 훈련 손실
    ax1.plot(train_losses)
    ax1.set_title(f'{model_name} 훈련 손실')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # 검증 정확도
    ax2.plot(val_accuracies)
    ax2.set_title(f'{model_name} 검증 정확도')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_training_history.png')
    plt.show()

def main():
    """메인 실행 함수"""
    # 데이터 경로 설정
    data_path = 'dataset/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'
    
    print("=== WISDM 활동 인식 신경망 ===")
    
    # 1. 데이터 로드 및 전처리
    X, y, label_encoder = load_and_preprocess_data(data_path)
    
    # 2. 데이터 정규화
    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(X.shape)
    
    # 3. 훈련/검증/테스트 분할
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"훈련 데이터: {X_train.shape}")
    print(f"검증 데이터: {X_val.shape}")
    print(f"테스트 데이터: {X_test.shape}")
    
    # 4. 데이터 로더 생성
    train_dataset = WISDMDataset(X_train, y_train)
    val_dataset = WISDMDataset(X_val, y_val)
    test_dataset = WISDMDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 5. 모델 생성 및 훈련
    print("\n=== CNN 모델 훈련 ===")
    cnn_model = ActivityRecognitionCNN(num_classes=len(label_encoder.classes_))
    cnn_losses, cnn_accuracies = train_model(cnn_model, train_loader, val_loader, model_name='cnn')
    
    print("\n=== LSTM 모델 훈련 ===")
    lstm_model = ActivityRecognitionLSTM(num_classes=len(label_encoder.classes_))
    lstm_losses, lstm_accuracies = train_model(lstm_model, train_loader, val_loader, model_name='lstm')
    
    # 6. 모델 평가
    print("\n=== CNN 모델 평가 ===")
    cnn_model.load_state_dict(torch.load('best_cnn_model.pth'))
    evaluate_model(cnn_model, test_loader, label_encoder)
    
    print("\n=== LSTM 모델 평가 ===")
    lstm_model.load_state_dict(torch.load('best_lstm_model.pth'))
    evaluate_model(lstm_model, test_loader, label_encoder)
    
    # 7. 결과 시각화
    plot_training_history(cnn_losses, cnn_accuracies, 'CNN')
    plot_training_history(lstm_losses, lstm_accuracies, 'LSTM')
    
    print("\n훈련 완료! 활동 인식 모델이 준비되었습니다.")

if __name__ == "__main__":
    main()