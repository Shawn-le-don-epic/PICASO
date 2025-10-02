import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2  # OpenCV for image processing

# --- NEW: Import our Autoencoder model ---
from model import Autoencoder, AutoencoderWithSkip

class PICASO_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PICASO - User-Guided Image Compression")
        
        # --- Member variables ---
        self.image_path = None
        self.original_image = None
        self.tk_image = None
        self.selection_rect = None
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None

        # --- NEW: Load the trained model ---
        self.model = self.load_model()

        # --- UI Widgets (Same as before) ---
        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill="both", expand=True)
        
        button_frame = tk.Frame(root)
        button_frame.pack(fill="x", side="bottom", pady=5)

        self.btn_load = tk.Button(button_frame, text="Load Image", command=self.load_image)
        self.btn_load.pack(side="left", padx=10)

        self.btn_compress = tk.Button(button_frame, text="Compress Image", command=self.compress_image)
        self.btn_compress.pack(side="right", padx=10)
        
        # --- Bind mouse events for drawing (Same as before) ---
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    # --- NEW: Method to load our saved autoencoder.pth ---
    def load_model(self):
        try:
            model = AutoencoderWithSkip()
            model.load_state_dict(torch.load('autoencoder.pth'))
            model.eval()  # Set the model to evaluation mode
            print("Successfully loaded trained model: autoencoder.pth")
            return model
        except FileNotFoundError:
            print("Error: autoencoder.pth not found. Please train the model first.")
            return None

    # --- This method is the same as before ---
    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not self.image_path:
            return
        self.original_image = Image.open(self.image_path).convert("RGB")
        w, h = self.original_image.size
        max_size = 600
        if w > max_size or h > max_size:
            self.original_image.thumbnail((max_size, max_size))
        self.tk_image = ImageTk.PhotoImage(self.original_image)
        self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    # --- These drawing methods are the same as before ---
    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.selection_rect:
            self.canvas.delete(self.selection_rect)
        self.selection_rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)

    def on_mouse_drag(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.selection_rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        self.end_x = self.canvas.canvasx(event.x)
        self.end_y = self.canvas.canvasy(event.y)

    # --- MODIFIED: This is where we implement the core PICASO logic ---
    '''
    def compress_image(self):
        if not self.original_image or self.model is None:
            print("Error: Please load an image and ensure the model is loaded.")
            return
        if self.start_x is None or self.end_x is None:
            print("Error: Please select a region on the image first.")
            return

        # 1. CREATE THE SIMPLIFIED PERCEPTUAL MAP (EDGE MAP)
        # Convert PIL image to OpenCV format (NumPy array)
        open_cv_image = np.array(self.original_image)
        gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        
        # Use Sobel filter to detect edges
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
        edge_map = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize the edge map to be used as a "JND" proxy
        # We'll set a base quantization step and make it larger in complex areas
        base_quant_step = 1.0
        perceptual_map = base_quant_step + (edge_map / np.max(edge_map)) * 4.0 # Values will range approx from 2 to 12

        # 2. CREATE THE USER'S PRIORITY MASK
        priority_mask = np.zeros_like(gray_image, dtype=np.uint8)
        x1 = int(min(self.start_x, self.end_x))
        y1 = int(min(self.start_y, self.end_y))
        x2 = int(max(self.start_x, self.end_x))
        y2 = int(max(self.start_y, self.end_y))
        priority_mask[y1:y2, x1:x2] = 1 # Set selected region to 1

        # 3. COMBINE MAPS TO CREATE THE GUIDED MAP
        guided_map = perceptual_map.copy()
        # In the user's selected region, set a very low quantization step for high quality
        guided_map[priority_mask == 1] = 0.1 
        
        # 4. PERFORM GUIDED COMPRESSION
        # Prepare the image tensor for the model
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        input_tensor = transform(self.original_image).unsqueeze(0) # Add batch dimension

        with torch.no_grad(): # We don't need to calculate gradients
            # Encode the image
            encoded_data = self.model.encoder(input_tensor)
            
            # Prepare the guided map for quantization
            # Resize map to match the latent space dimensions
            latent_h, latent_w = encoded_data.shape[2], encoded_data.shape[3]
            guided_map_resized = cv2.resize(guided_map, (latent_w, latent_h))
            guided_map_tensor = torch.from_numpy(guided_map_resized).float().unsqueeze(0).unsqueeze(0)

            # Quantize the encoded data using our guided map
            quantized_data = torch.round(encoded_data / guided_map_tensor)
            
            # De-quantize
            dequantized_data = quantized_data * guided_map_tensor
            
            # Decode to get the final image
            reconstructed_tensor = self.model.decoder(dequantized_data)

        # 5. SAVE THE OUTPUT
        # Convert the output tensor back to a PIL Image
        output_image = transforms.ToPILImage()(reconstructed_tensor.squeeze(0))
        
        # Ask user where to save the file
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if save_path:
            output_image.save(save_path)
            print(f"Compressed image saved to: {save_path}")
    '''

    # In app.py, replace the whole compress_image method

    def compress_image(self):
        if not self.original_image or self.model is None:
            print("Error: Please load an image and ensure the model is loaded.")
            return
        if self.start_x is None or self.end_x is None:
            print("Error: Please select a region on the image first.")
            return

        # Steps 1, 2, and 3 are the same (creating the guided_map)
        open_cv_image = np.array(self.original_image)
        gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
        edge_map = np.sqrt(sobelx**2 + sobely**2)
        base_quant_step = 1.0
        perceptual_map = base_quant_step + (edge_map / np.max(edge_map)) * 4.0
        priority_mask = np.zeros_like(gray_image, dtype=np.uint8)
        x1 = int(min(self.start_x, self.end_x))
        y1 = int(min(self.start_y, self.end_y))
        x2 = int(max(self.start_x, self.end_x))
        y2 = int(max(self.start_y, self.end_y))
        priority_mask[y1:y2, x1:x2] = 1
        guided_map = perceptual_map.copy()
        guided_map[priority_mask == 1] = 0.1
        
        # 4. PERFORM GUIDED COMPRESSION (Updated Logic)
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        input_tensor = transform(self.original_image).unsqueeze(0)

        with torch.no_grad():
            # --- THE FIX IS HERE ---
            # Call the new encode method, which returns the bottleneck and skips
            bottleneck, skip2, skip1 = self.model.encode(input_tensor)
            
            # Prepare the guided map (same as before)
            latent_h, latent_w = bottleneck.shape[2], bottleneck.shape[3]
            guided_map_resized = cv2.resize(guided_map, (latent_w, latent_h))
            guided_map_tensor = torch.from_numpy(guided_map_resized).float().unsqueeze(0).unsqueeze(0)

            # Quantize the bottleneck data
            quantized_data = torch.round(bottleneck / guided_map_tensor)
            
            # De-quantize
            dequantized_data = quantized_data * guided_map_tensor
            
            # --- AND THE FIX IS HERE ---
            # Call the new decode method, passing in the dequantized data and the skips
            reconstructed_tensor = self.model.decode(dequantized_data, skip2, skip1)

        # 5. SAVE THE OUTPUT (Same as before)
        output_image = transforms.ToPILImage()(reconstructed_tensor.squeeze(0))
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if save_path:
            output_image.save(save_path)
            print(f"Compressed image saved to: {save_path}")

if __name__ == '__main__':
    root = tk.Tk()
    app = PICASO_GUI(root)
    root.mainloop()