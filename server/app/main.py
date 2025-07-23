from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.routers import detection

app = FastAPI()

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000/",
        "https://deepfake-detector-frontend.fly.dev/"
    ],
    allow_credentials=True,
    allow_methods=[""],
    allow_headers=[""],
    expose_headers=["Content-Disposition"]
)

# Add security headers middleware for SharedArrayBuffer support
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    # These headers are needed for SharedArrayBuffer and cross-origin resource sharing
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response

app.include_router(detection.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Deepfake Detection API is running"}

if __name__ == "main":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
