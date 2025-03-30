# React + FastAPI Web App Setup  

This guide will help you set up a web application with a **React** frontend and a **FastAPI** backend. The project structure is as follows:  

```
/deepfake-detection
│── /client    # React frontend  
│── /server    # FastAPI backend  
│── README.md  # This file  
```

## Prerequisites  

Ensure you have the following installed:  
- **Node.js** (for React) – [Download Here](https://nodejs.org/)  
- **Python 3.11.11** (for FastAPI, recommended via pyenv) – [Download Here](https://www.python.org/)  
- **npm** (comes with Node.js)  
- **pip** (comes with Python)  
- **virtualenv** (recommended for Python dependencies)  
- **pyenv** (for managing Python versions) – [Installation Guide](https://github.com/pyenv/pyenv)

---

## Setup Instructions  

### 1. Clone the Repository  
```sh
git clone https://github.com/Shawnehh/deepfake-detection
cd deepfake-detection

git checkout dev # For latest branch
```

---

### 2. Setting Up the Frontend (React)  
```sh
cd client
npm install  # Install dependencies
npm run dev  # Start React development server
```

The frontend should now be running at **http://localhost:3000**.

---

### 3. Setting Up the Backend (FastAPI)  

#### a) Set Up Python with pyenv  
```sh
pyenv install 3.11.11  # Install the required Python version
pyenv local 3.11.11    # Set it for this project
```

#### b) Create a Virtual Environment  
```sh
cd ../server
python -m venv .venv  
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate    # On Windows
```

#### c) Install Dependencies  
```sh
pip install -r requirements.txt
```

#### d) Start the FastAPI Server  
```sh
uvicorn main:app --reload
```

The backend should now be running at **http://127.0.0.1:8000**.

---

### 4. Model Setup  

> (Date: 30/03/25)

The model is currently **gitignored**, so you need to manually place it in the following path:

```
deepfake-detection/server/app/model/
```

Ensure the model files are in the correct format and accessible by the FastAPI app.

---

## API Documentation  

Once the FastAPI server is running, you can access auto-generated API documentation at:  
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)  

