import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from face_feature import FacialFeatureDetector
import os
import cv2

class FacialFeatureDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Feature Detector")
        self.root.geometry("1200x700")
        
        self.detector = FacialFeatureDetector()

        self.create_widgets()
        
        self.image_path = None
        self.result_image = None
        self.original_image = None
        
    def create_widgets(self):
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(pady=10)
        
        self.select_btn = tk.Button(
            self.top_frame, 
            text="Select Image", 
            command=self.select_image,
            width=15,
            height=2
        )
        self.select_btn.pack(side=tk.LEFT, padx=10)
        
        self.process_btn = tk.Button(
            self.top_frame, 
            text="Process Image", 
            command=self.process_image,
            width=15,
            height=2,
            state=tk.DISABLED
        )
        self.process_btn.pack(side=tk.LEFT, padx=10)
        
        self.save_btn = tk.Button(
            self.top_frame, 
            text="Save Result", 
            command=self.save_result,
            width=15,
            height=2,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=10)
        
        self.help_btn = tk.Button(
            self.top_frame, 
            text="Help", 
            command=self.show_help,
            width=15,
            height=2
        )
        self.help_btn.pack(side=tk.LEFT, padx=10)

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # original image
        self.original_canvas = tk.Canvas(self.canvas_frame, width=500, height=500)
        self.original_canvas.grid(row=0, column=0, padx=10, pady=10)
        self.original_scroll_y = tk.Scrollbar(self.canvas_frame, orient="vertical", command=self.original_canvas.yview)
        self.original_scroll_y.grid(row=0, column=1, sticky="ns")
        self.original_scroll_x = tk.Scrollbar(self.canvas_frame, orient="horizontal", command=self.original_canvas.xview)
        self.original_scroll_x.grid(row=1, column=0, sticky="ew")
        self.original_canvas.configure(yscrollcommand=self.original_scroll_y.set, xscrollcommand=self.original_scroll_x.set)
        
        # result image
        self.result_canvas = tk.Canvas(self.canvas_frame, width=500, height=500)
        self.result_canvas.grid(row=0, column=2, padx=10, pady=10)
        self.result_scroll_y = tk.Scrollbar(self.canvas_frame, orient="vertical", command=self.result_canvas.yview)
        self.result_scroll_y.grid(row=0, column=3, sticky="ns")
        self.result_scroll_x = tk.Scrollbar(self.canvas_frame, orient="horizontal", command=self.result_canvas.xview)
        self.result_scroll_x.grid(row=1, column=2, sticky="ew")
        self.result_canvas.configure(yscrollcommand=self.result_scroll_y.set, xscrollcommand=self.result_scroll_x.set)
        
        tk.Label(self.canvas_frame, text="Original Image").grid(row=2, column=0)
        tk.Label(self.canvas_frame, text="Result Image").grid(row=2, column=2)
    
    def select_image(self):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        test_image_dir = os.path.join(root_dir,'test_images')
        
        self.image_path = filedialog.askopenfilename(
            title="Select Image",
            initialdir=test_image_dir,
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if self.image_path:
            self.display_original_image()
            self.process_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.DISABLED)
    
    def display_original_image(self):
        try:
            self.original_image = Image.open(self.image_path)
            self.original_photo = ImageTk.PhotoImage(self.original_image)
            
            self.original_canvas.delete("all")
            self.original_canvas.create_image(0, 0, anchor="nw", image=self.original_photo)
            self.original_canvas.config(scrollregion=self.original_canvas.bbox("all"))
            self.original_canvas.image = self.original_photo
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")
    
    def process_image(self):
        if not self.image_path:
            return
            
        try:
            self.result_image = self.detector.identify_facial_features(self.image_path)
            self.display_result_image()
            self.save_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")

    def show_help(self):
        help_text = """Facial Feature Detector Help:

        1. Click 'Select Image' to choose a photo
        2. Click 'Process Image' to analyze facial features
        3. Click 'Save Result' to save the analysis

        Features Detected:
        - Face outline (Green)
        - Left/Right Eyebrows (Orange)
        - Left/Right Eyes (Blue/Red)
        - Nose (Purple)
        - Mouth (Yellow)
        - Jawline (Cyan)

        For best results:
        - Use clear, front-facing photos
        - Ensure good lighting
        - Avoid obstructions (glasses, hair, etc.)"""
        messagebox.showinfo("Help", help_text)
    
    def display_result_image(self):
        # Convert OpenCV image to PIL format
        try:
            image = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB)
            self.result_pil_image = Image.fromarray(image)
            self.result_photo = ImageTk.PhotoImage(self.result_pil_image)
            
            self.result_canvas.delete("all")
            self.result_canvas.create_image(0, 0, anchor="nw", image=self.result_photo)
            self.result_canvas.config(scrollregion=self.result_canvas.bbox("all"))
            self.result_canvas.image = self.result_photo
        except Exception as e:
            messagebox.showerror("Error", f"Could not display result: {str(e)}")
    
    def save_result(self):
        try:
            if self.result_image is None:
                messagebox.showwarning("Warning", "No processed image to save")
                return

            root_dir = os.path.dirname(os.path.abspath(__file__))

            results_dir = os.path.join(root_dir, 'results')

            os.makedirs(results_dir, exist_ok=True)

            save_path = filedialog.asksaveasfilename(
                title="Save Result Image",
                defaultextension=".jpg",
                initialdir=results_dir,
                filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
            )

            if save_path:
                cv2.imwrite(save_path, self.result_image)
                messagebox.showinfo("Success", f"Image saved successfully to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save image: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FacialFeatureDetectorApp(root)
    root.mainloop()