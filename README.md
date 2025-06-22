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
# Clone project dari GitHub
git clone https://github.com/Bhimonw/GachaFood.git
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
      "cluster_name": "Ekonomis",
      "count": 28,
      "avg_harga": 15000.0,
      "avg_rating": 4.1,
      "avg_jarak": 1.2,
      "tipe_tempat_common": "Warung"
    },
    {
      "cluster_id": 1,
      "cluster_name": "Menengah",
      "count": 32,
      "avg_harga": 22000.0,
      "avg_rating": 4.3,
      "avg_jarak": 1.5,
      "tipe_tempat_common": "Indoor"
    },
    {
      "cluster_id": 2,
      "cluster_name": "Premium",
      "count": 22,
      "avg_harga": 35000.0,
      "avg_rating": 4.6,
      "avg_jarak": 2.1,
      "tipe_tempat_common": "Cafe"
    }
  ],
  "silhouette_score": 0.271,
  "calinski_harabasz_score": 27.1,
  "davies_bouldin_score": 1.365,
  "cluster_balance": 0.553,
  "total_restaurants": 82
}
```

### 3. GET `/api/stats`
Mendapatkan statistik umum dataset

### 4. GET `/api/restaurant/<int:restaurant_id>`
Mendapatkan detail restoran dan rekomendasi serupa

## 🤖 Machine Learning

### Algoritma Clustering
- **K-Means Clustering** dengan 3 cluster yang dioptimalkan
- **Features**: Harga (50%), Rating (30%), Jarak (15%), Tipe Tempat (5%)
- **Preprocessing**: StandardScaler untuk normalisasi data optimal
- **Evaluasi**: Multiple metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)

### Kategori Cluster
1. **Ekonomis** - Tempat makan dengan harga terjangkau
2. **Menengah** - Gabungan kategori sedang dan menengah untuk fleksibilitas
3. **Premium** - Tempat makan dengan harga tinggi dan kualitas premium

### Proses Clustering
1. **Data Preprocessing**: Advanced cleaning dengan duplicate detection
2. **Feature Weighting**: Optimized weights untuk separasi cluster yang lebih baik
3. **Feature Scaling**: StandardScaler dengan parameter yang dioptimalkan
4. **K-Means**: Clustering dengan parameter advanced (n_init=100, max_iter=2000)
5. **Evaluasi**: Comprehensive evaluation dengan multiple metrics

### Optimasi Clustering
- **Advanced K-Means Parameters**: 
  - n_init: 100 (multiple initializations)
  - max_iter: 2000 (better convergence)
  - algorithm: 'elkan' (efficient for small datasets)
  - tol: 1e-10 (strict tolerance)
- **Feature Engineering**: Weighted features untuk hasil clustering yang lebih baik
- **Data Quality**: Advanced outlier detection dan data cleaning

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

### Backend & Architecture
- **Framework**: Flask (Python) dengan modular blueprint design
- **Architecture**: Clean architecture dengan separation of concerns
- **Configuration**: Environment-based configuration management
- **Logging**: Structured logging dengan file rotation

### Machine Learning & Data
- **ML Framework**: Scikit-learn dengan optimized parameters
- **Clustering**: Advanced K-Means dengan multiple evaluation metrics
- **Data Processing**: Pandas, NumPy untuk data manipulation
- **Feature Engineering**: Weighted features dan advanced preprocessing
- **Data Quality**: Duplicate detection dan outlier handling

### Frontend & UI
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Styling**: Modern CSS dengan Grid, Flexbox, dan Gradient
- **Responsive**: Mobile-first responsive design
- **UX**: Interactive components dengan real-time feedback

### Development & Quality
- **Code Quality**: Modular design dengan clear separation
- **Error Handling**: Comprehensive error handling dan validation
- **Performance**: Optimized algorithms dan efficient data processing
- **Maintainability**: Clean code principles dan documentation

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

### Immediate Improvements
- [ ] **Clustering Enhancement**: Implementasi algoritma clustering alternatif (DBSCAN, Gaussian Mixture)
- [ ] **Feature Engineering**: Tambah fitur baru untuk meningkatkan Silhouette Score
- [ ] **Data Augmentation**: Ekspansi dataset untuk clustering yang lebih robust

### Feature Enhancements
- [ ] **User System**: Implementasi sistem rating dan review pengguna
- [ ] **Geolocation**: Integrasi dengan Google Maps API untuk lokasi real-time
- [ ] **Personalization**: Sistem rekomendasi berbasis collaborative filtering
- [ ] **Favorites**: Fitur bookmark tempat makan favorit
- [ ] **Social Features**: Sharing dan review antar pengguna

### Technical Improvements
- [ ] **Mobile App**: React Native atau Flutter mobile application
- [ ] **Admin Dashboard**: Interface untuk manajemen data dan monitoring
- [ ] **API Enhancement**: GraphQL API untuk query yang lebih fleksibel
- [ ] **Real-time Updates**: WebSocket untuk update real-time

### Infrastructure & Quality
- [ ] **Testing**: Comprehensive unit, integration, dan end-to-end tests
- [ ] **Deployment**: Docker containerization dan CI/CD pipeline
- [ ] **Monitoring**: Application performance monitoring dan logging
- [ ] **Security**: Authentication, authorization, dan data protection
- [ ] **Scalability**: Database optimization dan caching strategies

## 🤝 Kontribusi

Silakan buat pull request atau issue untuk kontribusi dan saran perbaikan.

## 📄 Lisensi

MIT License - Silakan gunakan untuk keperluan edukasi dan pengembangan.

---

**Dibuat dengan ❤️ untuk membantu mahasiswa menemukan tempat makan terbaik!**