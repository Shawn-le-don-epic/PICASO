import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from model import JND_LIC_Lite_Autoencoder
from jnd import calculate_jnd_map

class PICASO_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PICASO - User-Guided Image Compression")
        
        self.image_path = None
        self.original_image = None
        self.tk_image = None
        self.selection_rect = None
        self.start_x, self.start_y, self.end_x, self.end_y = None, None, None, None
        self.model = self.load_model()

        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill="both", expand=True)
        
        button_frame = tk.Frame(root)
        button_frame.pack(fill="x", side="bottom", pady=5)

        self.btn_load = tk.Button(button_frame, text="Load Image", command=self.load_image)
        self.btn_load.pack(side="left", padx=10)
        self.btn_compress_guided = tk.Button(button_frame, text="Compress with Selection", command=self.run_guided_compression)
        self.btn_compress_guided.pack(side="right", padx=10)

        # --- FIX: Re-add the missing mouse event bindings ---
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def load_model(self):
        try:
            model = JND_LIC_Lite_Autoencoder()
            model.load_state_dict(torch.load('best_autoencoder.pth', map_location=torch.device('cpu')))
            model.eval()
            print("Successfully loaded trained model: best_autoencoder.pth")
            return model
        except FileNotFoundError:
            print("Error: best_autoencoder.pth not found. Please train the model and place the file here.")
            return None

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not self.image_path: return
        self.original_image = Image.open(self.image_path).convert("RGB")
        w, h = self.original_image.size
        max_size = 600
        if w > max_size or h > max_size: self.original_image.thumbnail((max_size, max_size))
        self.tk_image = ImageTk.PhotoImage(self.original_image)
        self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def on_button_press(self, event):
        self.start_x, self.start_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        if self.selection_rect: self.canvas.delete(self.selection_rect)
        self.selection_rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)

    def on_mouse_drag(self, event):
        cur_x, cur_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.canvas.coords(self.selection_rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        self.end_x, self.end_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

    # In app.py, replace the entire run_guided_compression method

    def run_guided_compression(self):
        if not self.original_image or self.model is None: return
        if self.start_x is None or self.end_x is None:
            print("Error: Please select a region first.")
            return

        # Prepare image and jnd map
        open_cv_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
        jnd_map_np = calculate_jnd_map(open_cv_image)

        # Prepare transforms
        img_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        jnd_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128), antialias=True)])

        input_tensor = img_transform(self.original_image).unsqueeze(0)
        jnd_tensor = jnd_transform(jnd_map_np.astype(np.float32)).unsqueeze(0)

        with torch.no_grad():
            bottleneck, jnd_f1, jnd_f2 = self.model.encode(input_tensor, jnd_tensor)
            
            # Create priority mask from user's selection
            priority_mask_full = np.zeros((open_cv_image.shape[0], open_cv_image.shape[1]), dtype=np.uint8)
            x1, y1 = int(min(self.start_x, self.end_x)), int(min(self.start_y, self.end_y))
            x2, y2 = int(max(self.start_x, self.end_x)), int(max(self.start_y, self.end_y))
            priority_mask_full[y1:y2, x1:x2] = 1

            latent_h, latent_w = bottleneck.shape[2], bottleneck.shape[3]
            priority_mask_latent = cv2.resize(priority_mask_full, (latent_w, latent_h), interpolation=cv2.INTER_NEAREST)
            priority_mask_tensor = torch.from_numpy(priority_mask_latent).float().unsqueeze(0).unsqueeze(0)

            # --- THE FIX IS HERE ---
            # Expand the mask to match the number of channels in the bottleneck
            # This changes its shape from [1, 1, 16, 16] to [1, 128, 16, 16]
            priority_mask_tensor = priority_mask_tensor.expand_as(bottleneck)
            
            # Create a simple two-level quantization map
            quant_map = torch.full_like(bottleneck, 10.0) # Aggressive background compression
            
            # Now the shapes match, so this operation will succeed
            quant_map[priority_mask_tensor == 1] = 0.1 # High quality for selected region
            
            quantized_data = torch.round(bottleneck / quant_map)
            dequantized_data = quantized_data * quant_map
            
            reconstructed_tensor = self.model.decode(dequantized_data, jnd_f1, jnd_f2)

        # Save Output
        output_image = transforms.ToPILImage()(reconstructed_tensor.squeeze(0))
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            output_image.save(save_path)
            print(f"Compressed image saved to: {save_path}")

if __name__ == '__main__':
    root = tk.Tk()
    app = PICASO_GUI(root)
    root.mainloop()