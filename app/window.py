#IMPORTS
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QFileDialog , QMessageBox
from app.sidebar import Sidebar,LimeWindow,ModelWindow
from app.image_viewer import ImageViewer
from app.image_utils import generate_explanations,generate_lime_exp,get_model,get_named_conv_layers,get_all_models_dict,compute_overlap_with_mask
from app.utils import load_stylesheet,resource_path
from app.bb_window import BoundingBoxDrawer
import os
import shutil
import numpy as np 

#--------------------------------------------------------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """
    Main application window for PeliCAM.
    Provides UI components to load images, select models, generate visual explanations (CAM/LIME),
    and manage custom model files.

    Methods:
        - load_image(): Opens a dialog to load an image and display it in the viewer.
        - load_py_file(): Loads and saves a Python model definition file into the models directory.
        - load_pth_file(): Loads and saves model weights file into the models directory.
        - refresh_data(): Clears the viewer, resets UI fields and closes auxiliary windows.
        - save_data(): Saves all images from the image viewer to a selected directory.
        - generate_exp(): Generates and displays model explanation visualizations (CAM/LIME).
        - update_model_dropdown(): Refreshes the model dropdown with available models.
        - update_layer_dropdown(model_name): Updates layer dropdown based on selected model.
        - open_lime_window(): Opens the LIME settings window.
        - open_model_window(): Opens the model loading window for custom models.
        - closeEvent(event): On window close, cleans up temporary files in the models directory.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PeliCAM")
        self.setMinimumSize(1200, 800)
        self.setMaximumSize(1200,800)
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Sidebar and viewer
        self.sidebar = Sidebar()
        self.image_viewer = ImageViewer()
        

        layout.addWidget(self.sidebar)
        layout.addWidget(self.image_viewer)
        self.sidebar.setStyleSheet(load_stylesheet(resource_path("app/style/sidebar.qss")))
        self.image_viewer.setStyleSheet(load_stylesheet(resource_path("app/style/image_viewer.qss")))
       
        # Connect sidebar button
        self.sidebar.load_button.clicked.connect(self.load_image)
        self.sidebar.load_custom_model_btn.clicked.connect(self.open_model_window)
        self.sidebar.generate_button.clicked.connect(self.generate_exp)
        self.sidebar.refresh_button.clicked.connect(self.refresh_data)
        self.sidebar.save_button.clicked.connect(self.save_data)
        self.sidebar.lime_btn.clicked.connect(self.open_lime_window)
        self.sidebar.bb_box_btn.clicked.connect(self.open_bbox_window)
        self.sidebar.model_dropdown.currentTextChanged.connect(self.update_layer_dropdown)
        self.lime_window = None
        self.model_window= None
        self.bb_window= None
        self.update_layer_dropdown(list(get_all_models_dict().keys())[0])
        self.loaded_image_path = None
        
        

    def load_image(self):
        """
        Opens a file dialog to select an image file, and displays it in a new tab.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            image_name = os.path.basename(path)  
            self.image_viewer.add_image_tab(image_name, path)
            self.loaded_image_path = path

    
    def load_py_file(self):
        """
        Opens a dialog to select a Python file, and copies it to the models directory.
        Updates the model window with the saved file path.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Select .py File", "", "Python Files (*.py)")
        if not path:
            return

        model_dir=resource_path('app/models')
        dest_path = os.path.join(model_dir, os.path.basename(path))
        
        try:
            shutil.copy(path, dest_path)
            self.model_window.py_path = dest_path
            self.model_window.py_label.setText(f"Saved: {dest_path}")
            QMessageBox.information(self, "Python File Loaded", f".py file saved to: {dest_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save .py file:\n{str(e)}")

    def load_pth_file(self):
        """
        Opens a dialog to select a model weight file, and copies it to the models directory.
        Updates the model window with the saved file path.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Select .pth File", "", "Model Weights (*.pth *.pt *.pkl)")
        if not path:
            return

        model_dir=resource_path('app/models')
        dest_path = os.path.join(model_dir, os.path.basename(path))

        try:
            shutil.copy(path, dest_path)
            self.model_window.pth_path = dest_path
            self.model_window.pth_label.setText(f"Saved: {dest_path}")
            QMessageBox.information(self, "Weights Loaded", f"Model weights saved to: {dest_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save model weights:\n{str(e)}")

    def refresh_data(self):
        """
        Clears all loaded images, resets the UI, and closes auxiliary windows (LIME, Model).
        """
        self.image_viewer.clear()
        self.loaded_image_path = None
        for cb in self.sidebar.checkboxes :
            cb.setChecked(False)
        self.sidebar.exp_button.setChecked(False)
        self.sidebar.threshold_box.clear()
        self.sidebar.threshold_box.setPlaceholderText("0.5")
        if self.lime_window:
            self.lime_window.close()
        if self.model_window:
            self.model_window.close()
        if self.bb_window:
            self.bb_window.close()
    
    def save_data(self):
        """
        Opens a folder dialog and saves all loaded image tabs into the selected folder.
        """
        if self.image_viewer.count() < 1:
            QMessageBox.warning(self, "No Image Loaded", "Please load at least one image before saving.")
            QMessageBox.setStyleSheet('background-color:#f5f5f5;')
            return
        else:
            folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
            if folder:
                self.image_viewer.save_all_images(folder)

    def generate_exp(self, bbox_mask_path=None):
        """
        Generates explanations using selected methods (CAM/LIME), model, and layer.
        Displays the resulting heatmaps or overlays in new tabs.
        """
        if not self.loaded_image_path:
            QMessageBox.warning(self, "No Image Loaded", "Please load at least one image before generating.")
            return

        model_name = self.sidebar.model_dropdown.currentText()
        selected_methods = [cb.text() for cb in self.sidebar.checkboxes if cb.isChecked()]
        threshold_str = self.sidebar.threshold_box.text()
        threshold_str = threshold_str.strip()  # Remove extra spaces

        if threshold_str == "":
            threshold = 0.5  # Default if empty
        else:
            try:
                threshold = float(threshold_str)
                if not (0 <= threshold <= 1):
                    raise ValueError("Threshold must be between 0 and 1.")
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter a number between 0 and 1.\nSet default value 0.5.")
                QMessageBox.setStyleSheet('background-color:#f5f5f5;')
                threshold = 0.5 

        explain_on = self.sidebar.exp_button.isChecked()
        layer_name = self.sidebar.layer_dropdown.currentText()
        if self.model_window:
            python_file_path=self.model_window.py_path 
            class_name=self.model_window.class_input.text() 
            weight_path=self.model_window.pth_path
        else:
            python_file_path="" 
            class_name="" 
            weight_path=""
        if self.bb_window:
            color=self.bb_window.get_current_color()
            r = color.red()
            g = color.green()
            b = color.blue()
            rgb_tuple = (r, g, b)
        debug_image=None
        results,cams = generate_explanations(self.loaded_image_path, model_name,layer_name, selected_methods, threshold,python_file_path, class_name, weight_path, explain_on)
        for title, cam_img in cams.items():
            if bbox_mask_path:
                    iou, recall, precision,debug_image = compute_overlap_with_mask(
                        cam_img, bbox_mask_path,
                        threshold=threshold,  # same as sidebar threshold
                        bbox_color=rgb_tuple,
                        t=title
                    )
                    score_text = f"IoU: {iou:.2f},  Recall: {recall:.2f},  Precision: {precision:.2f}"
            else:
                score_text = None
            self.image_viewer.add_image_tab(title, results[title],score_text,debug_image_input=debug_image)
        
        for title,img in results.items():
            if title.endswith('exp'):
                self.image_viewer.add_image_tab(title, img,score_text=None,debug_image_input=None)


        lime_methods = []
        if self.lime_window and self.lime_window.isVisible():
            if self.lime_window.cb1.isChecked():
                lime_methods.append("heatmap")
            if self.lime_window.cb2.isChecked():
                lime_methods.append("positive")
            if self.lime_window.cb3.isChecked():
                lime_methods.append("both")

            try:
                num_features = int(self.lime_window.feature_input.text())
            except ValueError:
                num_features = 5  # fallback default
                QMessageBox.warning(self, "Invalid Input", "Please enter a integer > 1 .\nSet default value 5.")

            if self.model_window:
                python_file_path=self.model_window.py_path 
                class_name=self.model_window.class_input.text() 
                weight_path=self.model_window.pth_path
            else:
                python_file_path="" 
                class_name="" 
                weight_path=""
            lime_res = generate_lime_exp(self.loaded_image_path, model_name, lime_methods, num_features,self.sidebar.progress_bar,python_file_path, class_name, weight_path)
            for title, img in lime_res.items():
                self.image_viewer.add_image_tab(title, img)

    def update_model_dropdown(self):
        """
        Refreshes the model dropdown with all available model names.
        """
        self.sidebar.model_dropdown.clear()
        self.sidebar.model_dropdown.addItems(get_all_models_dict())

    def update_layer_dropdown(self, model_name):
        """
        Loads the given model and updates the layer dropdown with its convolutional layers.
        """
        if self.model_window:
            python_file_path=self.model_window.py_path 
            class_name=self.model_window.class_input.text() 
            weight_path=self.model_window.pth_path
        else:
            python_file_path="" 
            class_name="" 
            weight_path=""
        if model_name:
            model = get_model(model_name,python_file_path, class_name, weight_path)
            conv_dict = get_named_conv_layers(model)

            self.current_model = model
            self.layer_dict = conv_dict

            self.sidebar.layer_dropdown.clear()
            self.sidebar.layer_dropdown.addItems(conv_dict.keys())

    def open_lime_window(self):
        """
        Opens the LIME configuration window.
        """
        if not self.lime_window:
            self.lime_window = LimeWindow()
        self.lime_window.show()
        self.lime_window.raise_()

    def open_bbox_window(self):
        """
        Opens the bounding box drawer window with the currently loaded image.
        """
        if not self.loaded_image_path:
            QMessageBox.warning(self, "No Image Loaded", "Please load at least one image before drawing.")
            return

        imgpath = self.loaded_image_path

        # Always create a fresh instance
        self.bb_window = BoundingBoxDrawer(image_path=imgpath, parent=self)
        self.bb_window.setWindowFlags(Qt.Window)
        self.bb_window.show()
        self.bb_window.raise_()

    def generate_exp_from_bbox(self, bbox_mask_path):
        """
        Triggers generate_exp with bbox mask scoring after 'Send' is clicked.
        """
        self.generate_exp(bbox_mask_path=bbox_mask_path)


    def open_model_window(self):
        """
        Opens the custom model loader window and connects button signals.
        """
        if not self.model_window:
            self.model_window = ModelWindow()
            self.model_window.py_button.clicked.connect(self.load_py_file)
            self.model_window.pth_button.clicked.connect(self.load_pth_file)
            self.model_window.ok_btn.clicked.connect(self.update_model_dropdown)
        self.model_window.show()
        self.model_window.raise_()

    def closeEvent(self, event):
        """
        Triggered when the main window is closed.
        Deletes temporary model files except defaults in the models folder.
        """
        models_dir = os.path.join("app", "models")
        models_dir=resource_path(models_dir)
        try:
            # Delete only files inside the folder, not the folder itself
            for filename in os.listdir(models_dir):

                if filename!="resnet50_imagenet.pth" and filename!="vgg16_imagenet.pth" and filename!="vgg19_imagenet.pth":
                    file_path = os.path.join(models_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

            # print("Model files cleared from models folder.")
        except Exception as e:
            print(f"Failed to clear model files: {e}")

        # Call base class implementation to proceed with closing
        super().closeEvent(event)