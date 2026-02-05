# KrishiShield

KrishiShield is an integrated crop risk advisory and market prediction platform designed to help farmers and stakeholders make informed decisions based on weather, market, and risk analytics. The system consists of a Python-based backend (FastAPI/Streamlit) and a modern React frontend, supporting both online and offline usage.

---

## Features

- **Crop Risk Analysis:** Predicts crop risk levels using weather and market data.
- **Market Price Forecasting:** Provides market-specific price predictions for major crops.
- **Automated Alerts:** Monitors for sudden price changes and sends notifications.
- **Offline Support:** Caches data for offline access and analysis.
- **Multi-language UI:** Supports English and Marathi.
- **Interactive Charts:** Visualizes trends and predictions.
- **User-friendly Dashboard:** Fixed sidebar, responsive design, and notification system.

---

## Project Structure

```
backend/
  ├── app.py                # Streamlit app & core logic
  ├── main.py               # FastAPI backend & scheduler
  ├── requirements.txt      # Python dependencies
  ├── Dockerfile            # Backend containerization
  └── ...                   # Data files, models, utilities

frontend/
  ├── src/
  │   ├── App.jsx           # Main React app
  │   ├── App.css           # Styling
  │   ├── db.js             # IndexedDB caching
  │   └── ...               # Components, i18n, assets
  ├── package.json          # Frontend dependencies & scripts
  └── ...
```

---

## Getting Started

### Prerequisites

- **Backend:** Python 3.9+, pip
- **Frontend:** Node.js 18+, npm

---

### Backend Setup

1. **Install dependencies:**
   ```sh
   cd backend
   pip install -r requirements.txt
   ```

2. **Run FastAPI server:**
   ```sh
   uvicorn main:app --reload --port 8000
   ```

3. *(Optional)* **Run Streamlit app for local testing:**
   ```sh
   streamlit run app.py
   ```

4. **Docker (optional):**
   ```sh
   docker build -t krishishield-backend .
   docker run -p 80:80 krishishield-backend
   ```

---

### Frontend Setup

1. **Install dependencies:**
   ```sh
   cd frontend
   npm install
   ```

2. **Start development server:**
   ```sh
   npm run dev
   ```

3. **Build for production:**
   ```sh
   npm run build
   ```

---

## Usage

- Access the frontend at [http://localhost:5173](http://localhost:5173) (default Vite port).
- The frontend communicates with the backend API at [http://127.0.0.1:8000](http://127.0.0.1:8000).
- Analyze crop risk, view market predictions, and receive alerts.

---

## Team

This project was developed by a team of three members:

- **Member 1:** Atharv Lalage
- **Member 2:** Vivek More
- **Member 3:** Swayam Korde

---

## License

This project is for academic/demo purposes.

---

## Acknowledgements

- [React](https://react.dev/)
- [Vite](https://vitejs.dev/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [Recharts](https://recharts.org/)
- [Lucide Icons](https://lucide.dev/)
