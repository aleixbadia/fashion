import os
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

program_directory_name = os.path.dirname(os.path.abspath(__file__))
os.chdir(program_directory_name)

# Load the model (assuming it's already trained)
model = tf.keras.models.load_model('fashion_mnist_model.h5')

# Preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale

    # Get the original dimensions
    width, height = img.size
    max_dim = max(width, height)

    # Resize the original image to fit within 28x28 while maintaining aspect ratio
    if width > height:
        new_width = 28
        new_height = int(height * (28 / width))
    else:
        new_height = 28
        new_width = int(width * (28 / height))

    img = img.resize((new_width, new_height))

    # Create a new white (255) image with 28x28 dimensions
    new_img = Image.new('L', (28, 28), color=255)

    # Paste the resized original image at the center of the new image
    new_img.paste(img, ((28 - new_width) // 2, (28 - new_height) // 2))

    img_array = np.array(new_img).astype('float32') / 255.0  # Normalize pixel values
    img_array = 1.0 - img_array  # Invert colors to match Fashion MNIST format
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    processed_img = Image.fromarray((img_array.squeeze() * 255).astype(np.uint8))
    return img_array, processed_img

# Predict the label of the image
def predict_image(image_path):
    img_array, _ = preprocess_image(image_path)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

# Update the model incrementally
def update_model(image_array, label):
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    label_array = np.array([label])
    
    # Recreate and compile the optimizer
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Fit the model with the new data
    model.fit(image_array, label_array, epochs=1, verbose=0)
    
    # Save the updated model
    model.save('fashion_mnist_model.h5')
    
    messagebox.showinfo("Update", "Model updated successfully!")

# Tkinter interface
class App(tk.Tk):
    def __init__(self, class_names):
        super().__init__()
        self.title("Clothing Image Classifier")
        self.geometry("500x600")

        self.class_names = class_names

        # Create a frame for the canvas and scrollbar
        frame = tk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=1)

        # Add a canvas in that frame
        self.canvas = tk.Canvas(frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        # Add a scrollbar to the canvas
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure the canvas
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Create another frame inside the canvas
        self.scrollable_frame = tk.Frame(self.canvas)

        # Add that new frame to a window in the canvas
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.upload_btn = tk.Button(self.scrollable_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=20)

        self.original_img_label = tk.Label(self.scrollable_frame)
        self.original_img_label.pack(pady=20)

        self.processed_img_label = tk.Label(self.scrollable_frame)
        self.processed_img_label.pack(pady=20)

        self.prediction_label = tk.Label(self.scrollable_frame, text="", font=("Helvetica", 16))
        self.prediction_label.pack(pady=20)

        self.correct_btn = tk.Button(self.scrollable_frame, text="Correct", command=self.correct_label)
        self.correct_btn.pack(pady=10)
        self.correct_btn.config(state=tk.DISABLED)

        self.incorrect_btn = tk.Button(self.scrollable_frame, text="Incorrect", command=self.incorrect_label)
        self.incorrect_btn.pack(pady=10)
        self.incorrect_btn.config(state=tk.DISABLED)

        self.label_dropdown = ttk.Combobox(self.scrollable_frame, values=class_names)
        self.label_dropdown.pack(pady=10)
        self.label_dropdown.config(state=tk.DISABLED)

        self.submit_btn = tk.Button(self.scrollable_frame, text="Submit Correction", command=self.submit_correction)
        self.submit_btn.pack(pady=10)
        self.submit_btn.config(state=tk.DISABLED)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            original_img = Image.open(file_path)
            original_img.thumbnail((200, 200))
            original_img_tk = ImageTk.PhotoImage(original_img)
            self.original_img_label.config(image=original_img_tk)
            self.original_img_label.image = original_img_tk

            _, processed_img = preprocess_image(file_path)
            processed_img_tk = ImageTk.PhotoImage(processed_img.resize((200, 200)))
            self.processed_img_label.config(image=processed_img_tk)
            self.processed_img_label.image = processed_img_tk

            predicted_class = predict_image(file_path)
            self.predicted_label = self.class_names[predicted_class]
            self.prediction_label.config(text=f"Predicted: {self.predicted_label}")

            self.correct_btn.config(state=tk.NORMAL)
            self.incorrect_btn.config(state=tk.NORMAL)

    def correct_label(self):
        img_array, _ = preprocess_image(self.image_path)
        update_model(img_array, self.class_names.index(self.predicted_label))
        self.reset_interface()

    def incorrect_label(self):
        self.label_dropdown.config(state=tk.NORMAL)
        self.submit_btn.config(state=tk.NORMAL)

    def submit_correction(self):
        corrected_label = self.label_dropdown.get()
        if corrected_label:
            img_array, _ = preprocess_image(self.image_path)
            update_model(img_array, self.class_names.index(corrected_label))
            self.reset_interface()

    def reset_interface(self):
        self.original_img_label.config(image='')
        self.processed_img_label.config(image='')
        self.prediction_label.config(text='')
        self.correct_btn.config(state=tk.DISABLED)
        self.incorrect_btn.config(state=tk.DISABLED)
        self.label_dropdown.config(state=tk.DISABLED)
        self.submit_btn.config(state=tk.DISABLED)

# Load Fashion MNIST class names
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Initialize and run the app
app = App(class_names)
app.mainloop()