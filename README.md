# 🧠 AI-Based Memory Usage Forecaster

A professional machine learning–powered system that predicts future memory usage and enables **proactive memory management** strategies. This dashboard provides a full end-to-end visualization of the telemetry pipeline.

---

## 🚀 Key Features

- **End-to-End Pipeline**: From raw data collection to model evaluation.
- **Hybrid AI Core**: Utilizes both **Random Forest** and **LSTM (Long Short-Term Memory)** networks.
- **Glassmorphism UI**: High-fidelity dashboard with Lucide vector icons and smooth animations.
- **Proactive Engine**: Simulates memory management decisions to prevent OOM (Out-Of-Memory) events.

---

## 🛠️ Cross-Platform Deployment

This project is built to run on **Windows, macOS, and Linux**.

### Option 1: Native Execution (Recommended for Local Dev)

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**:
   - **Windows**: Double-click `memory-forecaster/start_prod.bat`
   - **Mac/Linux**: Run `bash memory-forecaster/start_prod.sh`

### Option 2: Docker (Recommended for Universal Compatibility)

If you have Docker installed, you can run the app without installing Python or any libraries locally:

```bash
# Build and start the container
docker-compose up --build
```
Access the dashboard at `http://localhost:5000`.

---

## 🌐 Web Deployment

This repository is optimized for one-click cloud deployment:

- **Hugging Face Spaces**: Create a "Docker" space and link this repo. Excellent for ML demonstrations.
- **Render / Railway**: Link your GitHub repo and use the `Dockerfile`. Use `0.0.0.0` as the host.
- **AWS/GCP/Azure**: Deploy via the provided `docker-compose.yml` to any container service.

---

## 📊 Tech Stack

- **Core**: Python 3.10+
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism), JavaScript, [Lucide Icons](https://lucide.dev)
- **ML Frameworks**: PyTorch (LSTM), Scikit-Learn (Random Forest)
- **Data Handling**: Pandas, NumPy
- **System Monitoring**: Psutil
- **Production Server**: Waitress WSGI

---

## 🏗️ Project Structure

- `/memory-forecaster`: Main application source code.
- `/documents`: PRD and implementation documentation.
- `Dockerfile` & `docker-compose.yml`: Containerization manifests.
- `requirements.txt`: Unified dependency list.

---

**Developed for the Advanced Operating Systems Course.**  
*Syncing to: [GitHub Official Repository](https://github.com/Mukta01/AI_Based_Memoryforecaster)*
