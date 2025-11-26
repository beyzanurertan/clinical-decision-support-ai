"""
9000 adet hastanın verileri incelenecek

age: 10-80
gender: K(1)-E(2)-Ç(0)

septom1:ates + oksuruk
septom2:bas agrısı + mide bulantısı

ates: 0(yok), 1(hafif), 2(siddetli)
oksuruk: 0(yok), 1(hafif), 2(siddetli)
bas agrısı: 0(yok), 1(hafif), 2(siddetli)
mide bulantısı: 0(yok), 1(hafif), 2(siddetli)

kan basıncı: 80 den az, 80-120 arası, 120 den fazla :60-150
kan sekeri: 100 den az, 100-140 arası, 140 dan fazla:80-160
CRP:10 dan az, 10 ile 100 arası, 100 den fazla: 0-111

0- tedavi 1: çocuk, kan basıncı <80, kan sekeri<100, septom1>2, septom2<2, CPR<10
1- tedavi 2: çocuk, 80<kan basıncı<120, 100<kan sekeri<140, septom1<2, septom2>2, 10<CPR<100
2- tedavi 3: cocuk, kan basıncı> 120, kan sekeri>140, septom1>2, septom2>2, CPR>100
3- tedavi 4: kadın, kan basıncı <80, kan sekeri<100, septom1>2, septom2<2, CPR<10
4- tedavi 5: kadın, 80<kan basıncı<120, 100<kan sekeri<140, septom1<2, septom2>2, 10<CPR<100
5- tedavi 6: kadın, kan basıncı> 120, kan sekeri>140, septom1>2, septom2>2, CPR>100
6- tedavi 7: erkek, kan basıncı <80, kan sekeri<100, septom1>2, septom2<2, CPR<10
7- tedavi 8: erkek, 80<kan basıncı<120, 100<kan sekeri<140, septom1<2, septom2>2, 10<CPR<100
8- tedavi 9: erkek, kan basıncı> 120, kan sekeri>140, septom1>2, septom2>2, CPR>100

veri setinde belli oranda bir gürültü olmali ki gercege yakin bir veri seti olsun
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical

#Gurultulu veri uretme
def generate_soft_val(n, main_range, noise_range, prob=0.80):
    
    """
    Belirli bir olasılıkla ana aralıktan, geri kalan olasılıkla gürültü aralığından sayı üretir.
    
    """
    main_vals = np.random.randint(main_range[0], main_range[1], n)
    noise_vals = np.random.randint(noise_range[0], noise_range[1], n)
    mask = np.random.choice([1, 0], size=n, p=[prob, 1-prob])
    return np.where(mask == 1, main_vals, noise_vals)

def generate_balanced_data(n_per_class=1000):
    data_list = []

#Toplam skordan tekil semptomlara dönüştürme fonksiyonu
    def get_symptoms_for_sum(target_sum_array):
        v1 = np.zeros_like(target_sum_array)
        v2 = np.zeros_like(target_sum_array)
        for i, s_sum in enumerate(target_sum_array):
            #Toplamı s_sum eden olası (0,1,2) çiftlerini bulur
            pairs = [(x, s_sum - x) for x in range(3) if 0 <= s_sum - x <= 2]
            if not pairs: # Eğer gürültü yüzünden 4'ten büyük veya 0'dan küçük bir toplam geldiyse
                 pairs = [(0,0)] if s_sum < 0 else [(2,2)]
            pair = pairs[np.random.choice(len(pairs))]
            v1[i], v2[i] = pair
        return v1, v2

    n = n_per_class

    #grup 0 (Tedavi 0, 3, 6 için temel)
    df0 = pd.DataFrame({
        'age': np.random.randint(11, 18, n),
        'gender': 0,
        #Tansiyon: Genelde 60-80, bazen 80-95
        'blood_pressure': generate_soft_val(n, (60, 81), (81, 96), prob=0.70),
        #Şeker: Genelde 80-100, bazen 100-120
        'blood_sugar': generate_soft_val(n, (80, 101), (101, 121), prob=0.90),
        #CRP: Genelde 0-10, bazen 10-25
        'crp': generate_soft_val(n, (0, 11), (11, 26), prob=0.85),
        'treatment_plan': 0
    })
    #Semptom 1: Genelde Yüksek (3-4), bazen Düşük (0-2)
    df0['symptom_one'] = generate_soft_val(n, (3, 5), (0, 3), prob=0.75)
    #Semptom 2: Genelde Düşük (0-2), bazen Yüksek (3-4)
    df0['symptom_two'] = generate_soft_val(n, (0, 3), (3, 5), prob=0.80)
    data_list.append(df0)

    #grup 1 (Tedavi 1, 4, 7 için temel)
    df1 = pd.DataFrame({
        'age': np.random.randint(11, 18, n),
        'gender': 0,
         #Tansiyon: Genelde 81-120, bazen 70-80
        'blood_pressure': generate_soft_val(n, (81, 121), (70, 81), prob=0.82),
         #Şeker: Genelde 101-140, bazen 140-150)
        'blood_sugar': generate_soft_val(n, (101, 141), (141, 151), prob=0.70),
         #CRP: Genelde 11-100, bazen 0-10
        'crp': generate_soft_val(n, (11, 101), (0, 11), prob=0.80),
        'treatment_plan': 1
    })
    #Semptom 1: Genelde Düşük (0-2), bazen Yüksek (3-4)
    df1['symptom_one'] = generate_soft_val(n, (0, 3), (3, 5), prob=0.75)
    #Semptom 2: Genelde Yüksek (4), bazen Düşük (1-2)
    df1['symptom_two'] = generate_soft_val(n, (4, 5), (1, 3), prob=0.85)
    data_list.append(df1)

    #grup 2 (Tedavi 2, 5, 8 için temel)
    df2 = pd.DataFrame({
        'age': np.random.randint(11, 18, n),
        'gender': 0,
        #Tansiyon: Genelde 121-150, bazen 110-120
        'blood_pressure': generate_soft_val(n, (121, 151), (110, 121), prob=0.60),
        #Şeker: Genelde 141-160, bazen 130-140
        'blood_sugar': generate_soft_val(n, (141, 161), (130, 141), prob=0.80),
        #CRP: Genelde 101-150, bazen 90-100
        'crp': generate_soft_val(n, (101, 151), (90, 101), prob=0.70),
        'treatment_plan': 2
    })
    
    df2['symptom_one'] = generate_soft_val(n, (3, 5), (0, 3), prob=0.85)
    df2['symptom_two'] = generate_soft_val(n, (3, 5), (0, 3), prob=0.80)
    data_list.append(df2)

    
    
    # Kadınlar (3, 4, 5)
    df3 = df0.copy(); df3['age'] = np.random.randint(19, 81, n); df3['gender'] = 1; df3['treatment_plan'] = 3
    data_list.append(df3)
    df4 = df1.copy(); df4['age'] = np.random.randint(19, 81, n); df4['gender'] = 1; df4['treatment_plan'] = 4
    data_list.append(df4)
    df5 = df2.copy(); df5['age'] = np.random.randint(19, 81, n); df5['gender'] = 1; df5['treatment_plan'] = 5
    data_list.append(df5)

    # Erkekler (6, 7, 8)
    df6 = df0.copy(); df6['age'] = np.random.randint(19, 81, n); df6['gender'] = 2; df6['treatment_plan'] = 6
    data_list.append(df6)
    df7 = df1.copy(); df7['age'] = np.random.randint(19, 81, n); df7['gender'] = 2; df7['treatment_plan'] = 7
    data_list.append(df7)
    df8 = df2.copy(); df8['age'] = np.random.randint(19, 81, n); df8['gender'] = 2; df8['treatment_plan'] = 8
    data_list.append(df8)

    #Birleştirme ve işleme
    all_data = pd.concat(data_list, ignore_index=True)
    all_data['symptom_fever'], all_data['symptom_cough'] = get_symptoms_for_sum(all_data['symptom_one'].values)
    all_data['symptom_headache'], all_data['symptom_sickness'] = get_symptoms_for_sum(all_data['symptom_two'].values)

    return all_data[['age', 'gender', 'symptom_fever', 'symptom_cough',
                     'symptom_headache', 'symptom_sickness', 'blood_pressure',
                     'blood_sugar', 'crp', 'symptom_one', 'symptom_two', 'treatment_plan']]

#patient data
balanced_data = generate_balanced_data(n_per_class=1000)
dosya_adi = "patient_data.csv"
balanced_data.to_csv(dosya_adi, index=False)


x = balanced_data.drop(["treatment_plan", "symptom_one", "symptom_two"], axis=1).values
y = to_categorical(balanced_data['treatment_plan'], num_classes=9)

# Ölçeklendirme 
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#Train/Test Ayırma
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)


model = Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(9, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", 
              metrics=["accuracy", tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])


history = model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test), batch_size=32)


#Shap analizi

#Özellik İsimlerini Tanımlama
feature_names = ["Age", "Gender", "Fever", "Cough", "Headache",
                 "Sickness", "Blood Pressure", "Blood Sugar", "CRP"]

#Hızlı Analiz Için Veri Alt Kumesi Secme
#Arka plan için eğitim setinden 50 rastgele ornek
background = x_train[np.random.choice(x_train.shape[0], 50, replace=False)]
#Test edilecek örnekler (ilk 20 tane)
test_subset = x_test[:20]

#Modern Explainer kullanımı
explainer = shap.Explainer(model.predict, background)
shap_values_obj = explainer(test_subset)

# SHAP nesnesine özellik isimlerini atama
shap_values_obj.feature_names = feature_names

#gorsel 1: genel ozet
print("\nGrafik 1 çiziliyor: Tedavi 0 İçin Genel Özet...")
plt.figure(figsize=(10, 6)) 
# Tedavi 0 (sınıf indeksi 0) için özet grafik
shap.summary_plot(shap_values_obj[:, :, 0], test_subset, feature_names=feature_names, show=False)
plt.title("General Feature Importance for Treatment 0", fontsize=14)
plt.tight_layout()
plt.show()

#gorsel 2: tekil hasta analizi
idx = 0  # İncelemek istediğimiz hastanın test setindeki sırası
#Bu hasta için modelin tahmini nedir?
tahmin_olasiliklari = model.predict(test_subset[idx].reshape(1, -1), verbose=0)
tahmin_sinif = np.argmax(tahmin_olasiliklari)

print(f"\nGrafik 2 çiziliyor: {idx}. İndeksli Hasta Analizi...")
print(f"Bu hasta için modelin önerdiği tedavi: {tahmin_sinif}")

plt.figure(figsize=(10, 6))
# Sadece o hasta (idx) ve tahmin edilen sınıf (tahmin_sinif) için bar grafiği
shap.plots.bar(shap_values_obj[idx, :, tahmin_sinif], show=False)
plt.title(f"Reasons for Treatment {tahmin_sinif} Decision for Patient {idx}", fontsize=14)
plt.tight_layout()
plt.show()

#sklearn modelleri icin veri hazırlıgı
y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

#modellerin tanımlanması
models = {
    #olusturdugum derin ogrenme modeli
    'Mevcut YSA (MLP)': model,

    #Klasik ML modelleri
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (Support Vector Machine)': SVC(random_state=42),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

#egitim ve test
results = []

for name, clf in models.items():
    print(f"-> {name} işleniyor...")

    #Eğitim
    if 'YSA' in name:
        #Keras modelinden tahmin alma
        y_pred_prob = clf.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
    else:
        #Klasik ML modelleri eğitimi
        clf.fit(x_train, y_train_labels)
        y_pred = clf.predict(x_test)

    #Metrikleri Hesaplama
    acc = accuracy_score(y_test_labels, y_pred)
    prec = precision_score(y_test_labels, y_pred, average='weighted')
    rec = recall_score(y_test_labels, y_pred, average='weighted')
    f1 = f1_score(y_test_labels, y_pred, average='weighted')

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })

#sonucları gorsellestirme
results_df = pd.DataFrame(results)

print("\n KARŞILAŞTIRMALI SONUÇ TABLOSU")
print(results_df.sort_values(by='F1-Score', ascending=False))

#Grafik çizimi
plt.figure(figsize=(12, 6))
# Veriyi uzun formata çevirme
results_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

sns.barplot(data=results_melted, x="Model", y="Score", hue="Metric", palette="viridis")
plt.title("Performance Comparison of Different Models")
plt.ylim(0.8, 1.01) # Skoru daha iyi görmek için ekseni daraltma
plt.xticks(rotation=15)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()