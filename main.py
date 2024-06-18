import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sqlite3

# Função para carregar dados do dataset local
def load_local_dataset(directory):
    data = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.dat'):
                filepath = os.path.join(subdir, file)
                with open(filepath, 'r') as f:
                    raw_data = f.read().replace('\n', ' ')
                    data.append(np.fromstring(raw_data, sep=' '))
    return np.concatenate(data) if data else None

# Funções de Filtragem
def apply_filters(data, fs=1000):
    # Filtro passa-alta para remover baixa frequência de ruído
    high_cutoff = 10  # frequência de corte do filtro passa-alta
    b, a = butter(6, high_cutoff / (0.5 * fs), btype='high')
    high_passed = filtfilt(b, a, data)

    # Filtro passa-baixa para remover alta frequência de ruído
    low_cutoff = 450  # frequência de corte do filtro passa-baixa
    b, a = butter(6, low_cutoff / (0.5 * fs), btype='low')
    low_passed = filtfilt(b, a, high_passed)
    
    # Filtro Notch para remover a frequência da rede elétrica (60 Hz no Brasil)
    notch_freq = 60  # frequência central do filtro notch
    quality_factor = 30  # fator de qualidade que define a largura da banda de rejeição
    b, a = iirnotch(notch_freq / (0.5 * fs), quality_factor)
    notch_filtered = filtfilt(b, a, low_passed)
    
    return notch_filtered

# Extração de características para várias amostras
def extract_features(data, window_size=100):
    features = []
    for i in range(0, len(data), window_size):
        window = data[i:i+window_size]
        if len(window) == window_size:
            # Valor absoluto médio (MAV)
            mav = np.mean(np.abs(window))
            # Valor quadrático médio (RMS)
            rms = np.sqrt(np.mean(np.square(window)))
            features.append([mav, rms])
    return np.array(features)

# Conectar ao banco de dados
def connect_db(db_name='emg_data.db'):
    conn = sqlite3.connect(db_name)
    return conn

# Criar tabela no banco de dados
def create_table(conn):
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS emg_data
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       mav REAL,
                       rms REAL,
                       label INTEGER)''')
    conn.commit()

# Inserir dados no banco de dados
def insert_data(conn, features, labels):
    cursor = conn.cursor()
    for feature, label in zip(features, labels):
        cursor.execute("INSERT INTO emg_data (mav, rms, label) VALUES (?, ?, ?)", 
                       (feature[0], feature[1], label))
    conn.commit()

# Treinar modelo com dados do banco de dados
def train_model(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT mav, rms, label FROM emg_data")
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['mav', 'rms', 'label'])
    X = df[['mav', 'rms']].values
    y = df['label'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)
    return model, scaler

# Plotar gráfico dos dados
def plot_data(raw_data, filtered_data):
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    plt.plot(raw_data[:2000])
    plt.title('Sinal EMG Original')
    plt.subplot(3, 1, 2)
    plt.plot(filtered_data[:2000])
    plt.title('Sinal EMG Filtrado')
    plt.tight_layout()
    plt.show()

# Função para simular dados EMG
def simulate_emg_data(movement_type, length=2000):
    # Simula dados EMG para diferentes tipos de movimento
    if movement_type == 'flexion':
        data = np.random.randn(length) * 0.5 + 1  # Exemplo de simulação
    elif movement_type == 'extension':
        data = np.random.randn(length) * 0.5 - 1  # Exemplo de simulação
    elif movement_type == 'grasp':
        data = np.random.randn(length) * 0.5  # Exemplo de simulação
    else:
        data = np.random.randn(length)  # Dados aleatórios
    return data

# Simular controle de prótese mioelétrica
def control_prosthesis(model, scaler, new_data):
    filtered_data = apply_filters(new_data)
    features = extract_features(filtered_data)
    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)
    # Implementar lógica de controle da prótese baseada nas previsões
    print("Predicted movements:", predictions)

# Caminho para a pasta local onde os arquivos do dataset estão armazenados
local_dataset_directory = 'path_to_local_dataset/1dof_dataset'  # Altere para o caminho real

# Carregar dados do dataset local
public_data = load_local_dataset(local_dataset_directory)
if public_data is not None:
    public_filtered_data = apply_filters(public_data)
    public_features = extract_features(public_filtered_data)
    public_labels = np.ones(len(public_features))  # Rótulos fictícios, substitua conforme necessário

    # Conectar ao banco de dados
    conn = connect_db()
    create_table(conn)
    insert_data(conn, public_features, public_labels)  # Inserir dados do dataset público

    # Simular a leitura de dados EMG "sujos"
    movement_type = 'flexion'  # Escolha: 'flexion', 'extension', 'grasp'
    simulated_data = simulate_emg_data(movement_type)

    # Processar dados simulados
    raw_data = simulated_data  # Dados brutos simulados
    filtered_data = apply_filters(raw_data)

    # Extração de características
    features = extract_features(filtered_data)
    labels = np.zeros(len(features))  # Rótulos fictícios, substitua conforme necessário

    insert_data(conn, features, labels)

    # Treinar modelo com dados do banco de dados
    model, scaler = train_model(conn)

    # Plotar dados
    plot_data(raw_data, filtered_data)

    # Simular controle da prótese com novos dados (exemplo fictício)
    new_data = simulate_emg_data('extension')  # Novo movimento para controle da prótese
    control_prosthesis(model, scaler, new_data)

    # Fechar conexão com o banco de dados
    conn.close()
else:
    print("Public dataset could not be loaded.")
