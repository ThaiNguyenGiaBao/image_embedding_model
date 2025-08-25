import torch
import open_clip
from PIL import Image
import requests
import os
from concurrent.futures import ProcessPoolExecutor, as_completed



class OpenClipEncoder:
    def __init__(self,model_name='hf-hub:Marqo/marqo-ecommerce-embeddings-L', device=None):
        self.model_name = model_name
        device = device or (
            "cuda" if torch.cuda.is_available()
            #else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
            #"cpu"  
        )
        self.device = torch.device(device)
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(self.model_name)
        self.model = self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        print(f"Model {self.model_name} loaded on {self.device}")
                
        if self.device.type == "cuda":
            self.model.half()  # FP16 weights
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        if self.device.type == "cpu":
            self.model = torch.compile(self.model, backend="inductor")
            # Tune threads for CPU inference
            torch.set_num_threads(os.cpu_count() or 4)
            torch.set_num_interop_threads(2)
    
    def encode_image(self, image_url):
        img = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        images = self.preprocess_val(img).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type=="cuda")):
            feats = self.model.encode_image(images, normalize=True)   # (1, D)
        return feats.squeeze(0).detach().to(torch.float32).cpu().numpy().tolist()

    def encode_text(self, text):
        texts = self.tokenizer([text]).to(self.device)
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type=="cuda")):
            feats = self.model.encode_text(texts, normalize=True)
        return feats.squeeze(0).detach().to(torch.float32).cpu().numpy().tolist()
    
    
    def encode_content(self, image_url=None, text=None):
        if not image_url and not text:
            raise ValueError("Either image_url or text must be provided.")

        with ProcessPoolExecutor(max_workers=2) as executor:
            fut_img = executor.submit(self.encode_image, image_url) 
            fut_text = executor.submit(self.encode_text, text)
            
            vector_img = fut_img.result() 
            vector_text = fut_text.result()
        
        # img*0.9 + text*0.1
        vector = [(v_img*8 + v_text*2)/10 for v_img, v_text in zip(vector_img, vector_text)]
        return vector
        
       
            
        
