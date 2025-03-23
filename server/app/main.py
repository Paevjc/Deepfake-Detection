from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import detection

app = FastAPI()

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Deepfake Detection API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)