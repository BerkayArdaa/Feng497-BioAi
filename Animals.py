import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt


def preprocess_data(data_path):

    data = pd.read_csv(data_path)


    label_encoder = LabelEncoder()
    data['Anomaly_Flag_Encoded'] = label_encoder.fit_transform(data['Anomaly_Flag'])


    features = [
        'Heart_Rate_BPM_Stress', 'Respiration_Rate_BPM_Stress',
        'Oxygen_Level_Percentage_Stress', 'Movement_Speed_kmh_Stress',
        'Vocalization_Level_dB_Stress'
    ]
    target = 'Anomaly_Flag_Encoded'

    X = data[features]
    y = data[target]

    return X, y, label_encoder, data


def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return model


def predict_fire_risk_and_visualize(data, features, model, label_encoder):

    X = data[features]
    probabilities = model.predict_proba(X)
    predictions = model.predict(X)
    data['Predicted_Anomaly'] = label_encoder.inverse_transform(predictions)


    high_stress_index = np.where(label_encoder.classes_ == 'High Stress')[0][0]
    data['Fire_Risk_Probability'] = probabilities[:, high_stress_index] * 100


    print("Veri sütunları:", data.columns)


    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data['Location_Longitude'], data['Location_Latitude'],
                          c=data['Fire_Risk_Probability'], cmap='Reds', s=50, alpha=0.7)
    plt.colorbar(scatter, label='Yangın Riski (%)')
    plt.title('Enlem ve Boylama Göre Yangın Riski')
    plt.xlabel('Boylam')
    plt.ylabel('Enlem')
    plt.show()

    return data[['Location_Latitude', 'Location_Longitude', 'Fire_Risk_Probability']]

# Ana Akış
if __name__ == "__main__":
    data_path = 'data/Updated_Wildlife_Dataset_for_Turkey.csv'  # Yeni dosya yolu


    # Veriyi yükle ve işle
    X, y, label_encoder, data = preprocess_data(data_path)

    # Modeli eğit
    model = train_random_forest(X, y)

    # Yangın riski tahminlerini hesapla ve görselleştir
    fire_risk_data = predict_fire_risk_and_visualize(data, [
        'Heart_Rate_BPM_Stress', 'Respiration_Rate_BPM_Stress',
        'Oxygen_Level_Percentage_Stress', 'Movement_Speed_kmh_Stress',
        'Vocalization_Level_dB_Stress'
    ], model, label_encoder)

    # Sonuçları kaydet
    fire_risk_data.to_csv('AnimalsResults', index=False)
    print("Yangın risk yüzdeleri 'fire_risk_results_turkey.csv' dosyasına kaydedildi.")
