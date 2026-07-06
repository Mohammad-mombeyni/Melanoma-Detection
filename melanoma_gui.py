import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import os
import threading
from ultralytics import YOLO

# ------------------------------------------------------------
# Appearance settings (Dark/Light theme)
ctk.set_appearance_mode("dark")       # "dark" or "light"
ctk.set_default_color_theme("blue")   # "blue", "green", "dark-blue"

MODEL_PATH = "best.pt"  # or full path
# ------------------------------------------------------------

class MelanomaApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🩺 Melanoma Detection with AI")
        self.geometry("850x900")
        self.minsize(700, 750)
        self.resizable(True, True)

        # Load model
        self.model = None
        self.class_names = ["Melanoma", "NotMelanoma"]
        self.load_model()

        # Variables
        self.image_path = None
        self.ctk_image = None

        # Build widgets
        self.create_widgets()

    # --------------------------------------------------------
    # Load YOLO model
    # --------------------------------------------------------
    def load_model(self):
        try:
            self.model = YOLO(MODEL_PATH)
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = list(self.model.names.values())
        except Exception as e:
            messagebox.showerror("Error", f"Model not found!\n{str(e)}")
            self.destroy()

    # --------------------------------------------------------
    # Build GUI elements
    # --------------------------------------------------------
    def create_widgets(self):
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)  # for image

        # ----- Header with title and theme toggle button -----
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, padx=30, pady=(20, 10), sticky="ew")
        header_frame.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(header_frame, text="Melanoma Skin Cancer Detection",
                             font=ctk.CTkFont(size=30, weight="bold"))
        title.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        theme_btn = ctk.CTkButton(header_frame, text="🌓", width=45, height=45,
                                  corner_radius=25, command=self.toggle_theme)
        theme_btn.grid(row=0, column=1, padx=10, pady=5, sticky="e")

        # Subtitle
        sub = ctk.CTkLabel(self, text="Upload an image and get diagnosis with YOLOv8",
                           font=ctk.CTkFont(size=14))
        sub.grid(row=1, column=0, padx=30, pady=(0, 15), sticky="w")

        # ----- File selection section -----
        select_frame = ctk.CTkFrame(self, fg_color="transparent")
        select_frame.grid(row=2, column=0, padx=30, pady=10, sticky="ew")
        select_frame.grid_columnconfigure(1, weight=1)

        btn_select = ctk.CTkButton(select_frame, text="📁 Select Image",
                                   command=self.select_image,
                                   height=45, corner_radius=12,
                                   font=ctk.CTkFont(size=14))
        btn_select.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.file_label = ctk.CTkLabel(select_frame, text="No file selected",
                                       font=ctk.CTkFont(size=13))
        self.file_label.grid(row=0, column=1, padx=15, pady=5, sticky="w")

        # ----- Image display (with rounded frame and border) -----
        image_frame = ctk.CTkFrame(self, corner_radius=20, border_width=3,
                                   border_color="#2b6a9e")
        image_frame.grid(row=3, column=0, padx=30, pady=15, sticky="nsew")
        image_frame.grid_rowconfigure(0, weight=1)
        image_frame.grid_columnconfigure(0, weight=1)

        self.image_label = ctk.CTkLabel(image_frame, text="Image will be displayed here",
                                        font=ctk.CTkFont(size=18))
        self.image_label.grid(row=0, column=0, padx=15, pady=15)

        # ----- Detect button -----
        btn_detect = ctk.CTkButton(self, text="🔍 Detect",
                                   command=self.detect_image,
                                   height=55, corner_radius=15,
                                   font=ctk.CTkFont(size=18, weight="bold"))
        btn_detect.grid(row=4, column=0, padx=30, pady=20, sticky="ew")

        # ----- Result display -----
        result_frame = ctk.CTkFrame(self, fg_color="transparent")
        result_frame.grid(row=5, column=0, padx=30, pady=10, sticky="ew")
        result_frame.grid_columnconfigure(0, weight=1)

        self.result_label = ctk.CTkLabel(result_frame, text="",
                                         font=ctk.CTkFont(size=26, weight="bold"))
        self.result_label.grid(row=0, column=0, padx=10, pady=5)

        self.prob_label = ctk.CTkLabel(result_frame, text="",
                                       font=ctk.CTkFont(size=18))
        self.prob_label.grid(row=1, column=0, padx=10, pady=5)

        # ----- Progress bar (indeterminate) -----
        self.progress = ctk.CTkProgressBar(self, width=400, height=25,
                                           corner_radius=12, mode='indeterminate')
        self.progress.grid(row=6, column=0, padx=30, pady=10)
        self.progress.grid_remove()  # hidden initially

        # ----- Exit button -----
        btn_exit = ctk.CTkButton(self, text="🚪 Exit", command=self.quit,
                                 height=45, corner_radius=12,
                                 fg_color="#b83b3b", hover_color="#8b2b2b",
                                 font=ctk.CTkFont(size=14))
        btn_exit.grid(row=7, column=0, padx=30, pady=(10, 25), sticky="ew")

    # --------------------------------------------------------
    # Core functions
    # --------------------------------------------------------
    def toggle_theme(self):
        """Switch between dark and light themes"""
        current = ctk.get_appearance_mode()
        ctk.set_appearance_mode("dark" if current == "Light" else "light")

    def select_image(self):
        """Pick an image from the system"""
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp *.tif")]
        path = filedialog.askopenfilename(title="Select Image", filetypes=filetypes)
        if path:
            self.image_path = path
            self.file_label.configure(text=os.path.basename(path))
            self.display_image(path)
            # Clear previous results
            self.result_label.configure(text="")
            self.prob_label.configure(text="")

    def display_image(self, path):
        """Show the image in the frame"""
        try:
            img = Image.open(path)
            max_size = (650, 550)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            self.ctk_image = ctk.CTkImage(light_image=img, dark_image=img,
                                          size=(img.width, img.height))
            self.image_label.configure(image=self.ctk_image, text="")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot display image:\n{str(e)}")

    def detect_image(self):
        """Start detection in a separate thread"""
        if self.image_path is None:
            messagebox.showwarning("Warning", "Please select an image first.")
            return
        if self.model is None:
            messagebox.showerror("Error", "Model is not loaded.")
            return

        # Show progress bar
        self.progress.grid()
        self.progress.start()
        self.update()

        # Run detection in background
        threading.Thread(target=self.run_detection, daemon=True).start()

    def run_detection(self):
        """Perform inference using the model"""
        try:
            results = self.model(self.image_path)
            res = results[0]
            probs = res.probs
            if probs is None:
                raise ValueError("Model output has no probability information.")

            top1_idx = probs.top1
            top1_conf = probs.top1conf.item() * 100
            predicted_class = self.class_names[top1_idx] if top1_idx < len(self.class_names) else "Unknown"

            # Update UI in main thread
            self.after(0, self.show_result, predicted_class, top1_conf)

        except Exception as e:
            self.after(0, self.show_error, str(e))
        finally:
            self.after(0, self.finish_loading)

    def show_result(self, predicted_class, confidence):
        """Update result labels"""
        if predicted_class.lower() == "melanoma":
            text = "⚠️ Diagnosis: Melanoma (Cancer risk)"
            color = "#e74c3c"
        else:
            text = "✅ Diagnosis: Not Melanoma (No cancer)"
            color = "#27ae60"

        self.result_label.configure(text=text, text_color=color)
        self.prob_label.configure(text=f"Confidence: {confidence:.2f}%", text_color="#2980b9")

    def show_error(self, error_msg):
        """Display error message"""
        messagebox.showerror("Error", f"Detection failed:\n{error_msg}")

    def finish_loading(self):
        """Hide progress bar"""
        self.progress.stop()
        self.progress.grid_remove()


# ------------------------------------------------------------
# Run the application
if __name__ == "__main__":
    app = MelanomaApp()
    app.mainloop()