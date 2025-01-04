import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt


# 1. Veri Ön İşleme
def preprocess_data(data_path):
    data = pd.read_csv(data_path)

    # Sütun isimlerini kontrol et
    print("Veri setindeki sütun isimleri:")
    print(data.columns)

    if 'Suppression Difficulty' not in data.columns:
        raise KeyError("'Suppression Difficulty' sütunu veri setinde bulunamadı!")

    # Hedef değişkeni encode et
    label_encoder = LabelEncoder()
    data['Suppression_Difficulty_Encoded'] = label_encoder.fit_transform(data['Suppression Difficulty'])

    # Özellikler ve hedef değişken
    features = [
        'BRIGHTNESS', 'Wind Speed (km/h)', 'Fire Spread Speed (km/h)',
        'Affected Area (km²)', 'FRP'
    ]
    target = 'Suppression_Difficulty_Encoded'

    X = data[features]
    y = data[target]

    return X, y, label_encoder, data


# 2. Model Eğitimi
def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Test sonuçları
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return model


# 3. Olasılıkların Çıkartılması ve Görselleştirme
def predict_fire_risk_and_visualize(data, features, model, label_encoder):
    # Tahmin yap
    X = data[features]
    probabilities = model.predict_proba(X)
    predictions = model.predict(X)
    data['Predicted_Difficulty'] = label_encoder.inverse_transform(predictions)

    # Olasılık sütunlarını ekle
    critical_index = np.where(label_encoder.classes_ == 'Critical')[0][0]
    data['Critical_Risk_Probability'] = probabilities[:, critical_index] * 100

    # Görselleştirme
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data['LONGITUDE'], data['LATITUDE'],
                          c=data['Critical_Risk_Probability'], cmap='Reds', s=50, alpha=0.7)
    plt.colorbar(scatter, label='Kritik Yangın Riski (%)')
    plt.title('Enlem ve Boylama Göre Kritik Yangın Riski')
    plt.xlabel('Boylam')
    plt.ylabel('Enlem')

    # Görseli kaydet ve göster
    plt.savefig('fire_risk_map.png')
    plt.show()

    return data[['LATITUDE', 'LONGITUDE', 'Critical_Risk_Probability']]


# Ana Akış
if __name__ == "__main__":
    # Veri yolu
    data_path = 'data/NewAllData.csv'

    try:
        # Veriyi yükle ve işle
        X, y, label_encoder, data = preprocess_data(data_path)

        # Modeli eğit
        model = train_random_forest(X, y)

        # Yangın riski tahminlerini hesapla ve görselleştir
        fire_ris        k_data = predict_fire_risk_and_visualize(data, [
            'BRIGHTNESS', 'Wind Speed (km/h)', 'Fire Spread Speed (km/h)',
            'Affected Area (km²)', 'FRP'
        ], model, label_encoder)

        # Sonuçları kaydet
        fire_risk_data.to_csv('fire_risk_results.csv', index=False)
        print("Kritik yangın risk yüzdeleri 'fire_risk_results.csv' dosyasına kaydedildi.")

    except Exception as e:
        print(f"Hata oluştu: {e}")
