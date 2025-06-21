# 🍽️ GachaFood - Aplikasi Rekomendasi Tempat Makan

Aplikasi web berbasis Flask yang menggunakan machine learning clustering (K-Means) untuk memberikan rekomendasi tempat makan berdasarkan harga, jarak, rating, dan jenis tempat.

## ✨ Fitur Utama

- **Clustering Machine Learning**: Menggunakan algoritma K-Means untuk mengelompokkan tempat makan berdasarkan karakteristik serupa
- **Filter Pencarian**: Filter berdasarkan harga maksimal, jarak maksimal, rating minimal, dan jenis tempat
- **Rekomendasi Cerdas**: Memberikan rekomendasi tempat makan yang sesuai dengan preferensi pengguna
- **Analisis Cluster**: Melihat informasi detail setiap cluster dengan statistik rata-rata
- **Interface Modern**: Antarmuka web yang responsif dan user-friendly
- **API RESTful**: Endpoint API untuk integrasi dengan aplikasi lain

## 🚀 Cara Menjalankan Aplikasi

### 1. Persiapan Environment

```bash
# Clone atau download project
cd GachaFood

# Buat virtual environment (opsional tapi direkomendasikan)
python -m venv venv

# Aktifkan virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi

```bash
python app.py
```

Aplikasi akan berjalan di `http://localhost:5000`

## 📊 Dataset

Aplikasi menggunakan dataset `data/tempat_makan_cleaned.csv` yang berisi informasi:

- **Nama Tempat**: Nama restoran/warung
- **Tipe Tempat**: Indoor, Outdoor, Warung, Cafe, dll.
- **Estimasi Harga**: Harga rata-rata dalam Rupiah
- **Rating**: Rating tempat makan (1-5)
- **Lokasi**: Link Google Maps atau alamat
- **Jarak dari Kampus**: Jarak dalam kilometer

## 🔧 API Endpoints

### 1. GET `/api/recommendations`
Mendapatkan rekomendasi tempat makan berdasarkan filter

**Query Parameters:**
- `max_harga` (int): Harga maksimal
- `max_jarak` (float): Jarak maksimal dalam km
- `min_rating` (float): Rating minimal
- `tipe_tempat` (string): Jenis tempat (Warung, Indoor, Outdoor, Cafe)
- `cluster_id` (int): ID cluster tertentu
- `limit` (int): Jumlah hasil maksimal (default: 10)

**Contoh:**
```
GET /api/recommendations?max_harga=20000&min_rating=4&limit=5
```

### 2. GET `/api/clusters`
Mendapatkan informasi semua cluster

**Response:**
```json
{
  "clusters": [
    {
      "cluster_id": 0,
      "count": 15,
      "avg_harga": 18500.0,
      "avg_rating": 4.2,
      "avg_jarak": 1.3,
      "tipe_tempat_common": "Warung"
    }
  ],
  "silhouette_score": 0.65,
  "total_restaurants": 95
}
```

### 3. GET `/api/stats`
Mendapatkan statistik umum dataset

### 4. GET `/api/restaurant/<int:restaurant_id>`
Mendapatkan detail restoran dan rekomendasi serupa

## 🤖 Machine Learning

### Algoritma Clustering
- **K-Means Clustering** dengan 5 cluster
- **Features**: Harga, Rating, Jarak, Tipe Tempat (encoded)
- **Preprocessing**: StandardScaler untuk normalisasi data
- **Evaluasi**: Silhouette Score untuk mengukur kualitas clustering

### Proses Clustering
1. **Data Preprocessing**: Cleaning data dan encoding kategorikal
2. **Feature Scaling**: Normalisasi menggunakan StandardScaler
3. **K-Means**: Clustering dengan 5 cluster
4. **Evaluasi**: Perhitungan Silhouette Score

## 🎯 Cara Menggunakan

### 1. Pencarian Rekomendasi
1. Buka aplikasi di browser
2. Atur filter sesuai preferensi:
   - Harga maksimal yang diinginkan
   - Jarak maksimal dari kampus
   - Rating minimal
   - Jenis tempat
3. Klik "Cari Rekomendasi"
4. Lihat hasil rekomendasi yang diurutkan berdasarkan rating dan jarak

### 2. Analisis Cluster
1. Klik "Info Cluster" untuk melihat karakteristik setiap cluster
2. Setiap cluster menunjukkan:
   - Jumlah tempat makan
   - Rata-rata harga, rating, dan jarak
   - Jenis tempat yang dominan

### 3. Filter Berdasarkan Cluster
1. Pilih cluster tertentu dari dropdown
2. Aplikasi akan menampilkan tempat makan hanya dari cluster tersebut

## 📱 Fitur Interface

- **Responsive Design**: Tampilan optimal di desktop dan mobile
- **Real-time Search**: Pencarian langsung tanpa reload halaman
- **Interactive Cards**: Kartu tempat makan dengan informasi lengkap
- **Statistics Dashboard**: Dashboard statistik dataset
- **Modern UI**: Desain modern dengan gradient dan animasi

## 🔍 Contoh Penggunaan

### Skenario 1: Mahasiswa dengan Budget Terbatas
- Set harga maksimal: Rp 15.000
- Set jarak maksimal: 1 km
- Rating minimal: 4
- Hasil: Tempat makan murah, dekat, dan berkualitas

### Skenario 2: Mencari Cafe untuk Nongkrong
- Pilih tipe tempat: "Cafe"
- Rating minimal: 4
- Hasil: Cafe-cafe dengan rating tinggi

### Skenario 3: Eksplorasi Cluster
- Klik "Info Cluster" untuk melihat karakteristik setiap cluster
- Pilih cluster yang menarik untuk eksplorasi lebih lanjut

## 📁 Project Structure

```
GachaFood/
├── app.py                          # Main Flask application (entry point)
├── config/
│   ├── __init__.py
│   └── config.py                   # Application configuration
├── models/
│   ├── __init__.py
│   ├── clustering.py               # ML clustering logic
│   ├── data_loader.py              # Data loading and preprocessing
│   └── recommendation_engine.py    # Recommendation algorithms
├── routes/
│   ├── __init__.py
│   ├── api_routes.py               # API endpoints
│   └── web_routes.py               # Web page routes
├── utils/
│   ├── __init__.py
│   ├── helpers.py                  # Utility functions
│   └── logger.py                   # Logging configuration
├── templates/
│   └── index.html                  # Web interface
├── data/
│   └── tempat_makan_cleaned.csv    # Restaurant dataset
├── logs/                           # Application logs (auto-created)
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## 🛠️ Teknologi yang Digunakan

- **Backend**: Flask (Python)
- **Architecture**: Modular design with blueprints and factory pattern
- **Machine Learning**: Scikit-learn (K-Means, StandardScaler)
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Styling**: CSS Grid, Flexbox, Gradient
- **Configuration**: Environment-based configuration management
- **Logging**: Structured logging with file rotation

## 🏗️ Architecture Benefits

### Modular Design
- **Separation of Concerns**: Each module has a specific responsibility
- **Maintainability**: Easy to modify and extend individual components
- **Testability**: Components can be tested independently
- **Scalability**: Easy to add new features without affecting existing code

### Configuration Management
- **Environment-based**: Different configs for development, testing, production
- **Centralized**: All settings in one place
- **Flexible**: Easy to modify without code changes

### Logging System
- **Structured**: Consistent logging across all modules
- **Configurable**: Different log levels for different environments
- **File Rotation**: Automatic log file management

## 📈 Pengembangan Selanjutnya

- [ ] Tambah algoritma clustering lain (DBSCAN, Hierarchical)
- [ ] Implementasi sistem rating pengguna
- [ ] Integrasi dengan Google Maps API
- [ ] Fitur bookmark tempat makan favorit
- [ ] Sistem rekomendasi berbasis collaborative filtering
- [ ] Mobile app dengan React Native
- [ ] Dashboard admin untuk manajemen data
- [ ] Testing: Comprehensive unit and integration tests
- [ ] Deployment: Docker containerization and CI/CD pipeline

## 🤝 Kontribusi

Silakan buat pull request atau issue untuk kontribusi dan saran perbaikan.

## 📄 Lisensi

MIT License - Silakan gunakan untuk keperluan edukasi dan pengembangan.

---

**Dibuat dengan ❤️ untuk membantu mahasiswa menemukan tempat makan terbaik!**