from openclip_encoder import OpenClipEncoder
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

encoder = OpenClipEncoder()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to ["http://127.0.0.1:5500"] etc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/encode_image")
def encode_image(image_url: str):

    if not encoder:
        return {"error": "Encoder not initialized. Set PRELOAD_MODEL=1 to initialize."}
    
    try:
        # Encode the image URL to get the vector
        vector = encoder.encode_image(image_url)
     
        return {"data": vector}
    except Exception as e:
        return {"error": str(e)}
    
