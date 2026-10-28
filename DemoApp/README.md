# E-Waste Classification Web App

This repository contains the Two-Tier Architecture (FastAPI Backend + React Frontend) for the E-Waste Classification model (Optimized DenseNet-121).

## How to Start the Application

You will need two separate terminal windows open: one for the backend and one for the frontend.

### 1. Start the Backend

The backend is a FastAPI server that handles the AI model predictions.

1. Open a terminal and navigate to the `backend` directory:
   ```bash
   cd backend
   ```

2. Activate the Python virtual environment:
   - **Mac/Linux:**
     ```bash
     source venv/bin/activate
     ```
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```

3. Run the Uvicorn server:
   ```bash
   uvicorn main:app --reload
   ```
   *(Note: Remove the `--reload` flag during the actual presentation to prevent unexpected restarts).*

   **The backend will be running at: `http://localhost:8000`**

---

### 2. Start the Frontend

The frontend is a React application built with Vite.

1. Open a **new, separate** terminal window and navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```

2. Install the necessary node modules (only required the first time, or if packages are added):
   ```bash
   npm install
   ```

3. Start the Vite development server:
   ```bash
   npm run dev
   ```

   **The frontend will be running at: `http://localhost:5173`**
