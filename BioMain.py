import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

# 1. Veriyi Yükleme
file_path = 'data/generated_fire_risk_dataset.xlsx'  # Dosya yolu
data = pd.read_excel(file_path)

# 2. Risk Sınıflandırma Fonksiyonu
def calculate_fire_risk(row):
    if row['fv/fm'] > 0.75 and row['temp_c'] < 30 and row['soil_moisture_%'] > 20 and row['humidity_%'] > 40:
        return "Low"
    elif 0.70 <= row['fv/fm'] <= 0.75 and 30 <= row['temp_c'] <= 35 and 15 <= row['soil_moisture_%'] <= 20 and 30 <= row['humidity_%'] <= 40:
        return "Moderate"
    elif row['fv/fm'] < 0.70 and row['temp_c'] > 35 and row['soil_moisture_%'] < 15 and row['humidity_%'] < 30:
        return "High"
    else:
        return "Unknown"

# 3. Risk Sınıflarını Ekleme
data['fire_risk'] = data.apply(calculate_fire_risk, axis=1)
print("\nRisk Durumları:\n", data['fire_risk'].value_counts())

# 4. Model için Hazırlık
# Hedef ve özellik sütunları
X = data[['fv/fm', 'temp_c', 'soil_moisture_%', 'humidity_%']]
y = data['fire_risk']

# 'Unknown' sınıfını dışarıda bırakma
X = X[y != "Unknown"]
y = y[y != "Unknown"]

# 5. Veri Setini Dengeleme
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# 6. Veriyi Train-Test Ayırma
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 7. Model Oluşturma ve Eğitme
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("\nSınıflandırma Raporu:\n", classification_report(y_test, predictions, zero_division=0))

# 9. Tüm Saatler için Risk Tahmini
data['predicted_risk'] = model.predict(data[['fv/fm', 'temp_c', 'soil_moisture_%', 'humidity_%']])

# 10. Saat Bazında Riskleri Görselleştirme
plt.figure(figsize=(12, 6))
plt.plot(data['hour'], data['predicted_risk'], marker='o', linestyle='-', label='Predicted Fire Risk')
plt.title('Saat Bazında Yangın Risk Tahmini')
plt.xlabel('Saat')
plt.ylabel('Yangın Riski')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 11. Risk tahminlerini dosyaya kaydetme
output_path = 'data/BioResults.xlsx'
data.to_excel(output_path, index=False)
print(f"\nTahmin edilen yangın riskleri '{output_path}' olarak kaydedildi.")
