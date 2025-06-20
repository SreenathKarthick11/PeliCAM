#IMPORTS
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,QPushButton, QComboBox, QCheckBox, QFrame , QLineEdit,QLabel,QProgressBar
)
from PyQt5.QtGui import QIcon
from app.image_utils import get_all_models_dict
from app.utils import resource_path,load_stylesheet
from PyQt5.QtCore import Qt

#------------------------------------------------------------------------------------------------------------------

class ModelWindow(QWidget):
    """
    A QWidget window for selecting a custom PyTorch model and its class.

    Features:
        - Load a .py file containing the model definition.
        - Load a .pth file with model weights.
        - Input for specifying the model class name.
        - "OK" button to confirm selection.
    """
     
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Loader")
        self.setMinimumSize(300, 250)
        self.setMaximumSize(300, 250)
        self.setWindowIcon(QIcon(resource_path("assets/icon.png"))) 
        self.setStyleSheet(load_stylesheet(resource_path('app/style/model_window.qss')))
        # Widgets
        self.py_button = QPushButton("Load .py File")
        self.pth_button = QPushButton("Load .pth File")
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("Enter class name (e.g., CustomCNN)")

        self.py_label = QLabel("No .py file loaded")
        self.pth_label = QLabel("No .pth file loaded")
        self.ok_btn=QPushButton("OK")

        # Layouts
        layout = QVBoxLayout()

        layout.addWidget(self.py_button)
        layout.addWidget(self.py_label)

        layout.addWidget(self.pth_button)
        layout.addWidget(self.pth_label)

        layout.addWidget(QLabel("Class Name:"))
        layout.addWidget(self.class_input)
        layout.addWidget(self.ok_btn)
        self.setLayout(layout)

        # State
        self.py_path = ""
        self.pth_path = ""
    
#-----------------------------------------------------------------------------------------------------------------

class LimeWindow(QWidget):
    """
    A QWidget for configuring LIME (Local Interpretable Model-agnostic Explanations) parameters.

    Features:
        - Allows users to select different LIME explanation modes:
            • Heatmap
            • Positive-only
            • Both positive and negative contributions
        - Allows setting the number of features for explanation.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LIME-EXP")
        self.setMinimumSize(250, 150)
        self.setMaximumSize(250,150)
        self.setWindowIcon(QIcon(resource_path("assets/icon.png"))) 
        self.setStyleSheet(load_stylesheet(resource_path('app/style/lime_window.qss')))
        layout = QVBoxLayout()

        # Checkboxes
        self.cb1 = QCheckBox("Heat-Map")
        self.cb2 = QCheckBox("Postive-only")
        self.cb3 = QCheckBox("Both(+/-)")

        layout.addWidget(self.cb1)
        layout.addWidget(self.cb2)
        layout.addWidget(self.cb3)

        self.f_widget=QWidget()
        flayout=QHBoxLayout()
        # Label and LineEdit
        flayout.addWidget(QLabel("Num Features"))
        self.feature_input = QLineEdit()
        self.feature_input.setAlignment(Qt.AlignCenter)
        self.feature_input.setPlaceholderText("5")
        flayout.addWidget(self.feature_input)
        self.f_widget.setLayout(flayout)
        layout.addWidget(self.f_widget)
        self.setLayout(layout)

    def closeEvent(self, event):
        """Resets checkboxes and clears feature input when the window is closed."""
        self.cb1.setChecked(False)
        self.cb2.setChecked(False)
        self.cb3.setChecked(False)
        self.feature_input.clear()  # Clears the text in QLineEdit
        event.accept()
#---------------------------------------------------------------------------------------------------------------------

class Sidebar(QFrame):
    """
    Sidebar widget for the CAM/LIME GUI.
    
    Provides:
        - Image and model loading options.
        - Model and layer selection dropdowns.
        - CAM technique selection.
        - Threshold input for explanation.
        - Buttons to trigger explanations and LIME.
        - Progress bar and utility actions.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(250)

        self.layout = QVBoxLayout(self)

        self.load_button = QPushButton("Load Image")
        self.load_custom_model_btn = QPushButton("Load Custom Model")
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(get_all_models_dict())
        
        self.layer_dropdown = QComboBox()

        self.checkboxes = [QCheckBox("Grad-CAM"),QCheckBox("Grad-CAM++"),QCheckBox("Layer-CAM")]

        self.exp_widget = QWidget()
        self.exp_widget.setStyleSheet('background-color:rgba(20,20,20,200);')
        self.exp_layout = QHBoxLayout(self.exp_widget)  
        self.exp_button = QPushButton("Explanation")
        self.exp_button.setStyleSheet("""
                                        QPushButton {
                                            background-color: rgba(90,180,170,255);
                                            padding: 8px;
                                            border-radius: 6px;
                                            font-size: 16px;
                                            color: white;
                                        }

                                        QPushButton:hover {
                                            background-color: rgba(90,180,170,150);
                                        }

                                        QPushButton:pressed {
                                            background-color: #f39c12;
                                            padding-top: 12px;
                                        }
                                        QPushButton:checked {
                                            background-color: #f39c12;
                                        }
                                        """)
        self.exp_button.setCheckable(True)
        self.threshold_box = QLineEdit()
        self.threshold_box.setAlignment(Qt.AlignCenter)
        self.threshold_box.setPlaceholderText("0.5")
        self.threshold_box.setStyleSheet('background-color:#f5f5f5;padding:6px;border-radius:10px')
        self.exp_layout.addWidget(self.exp_button)
        self.exp_layout.addWidget(self.threshold_box)
        self.lime_btn=QPushButton("Lime-Exp")
        self.bb_box_btn=QPushButton("BB_Box")
        self.generate_button = QPushButton("Generate")
        self.save_button = QPushButton("Save")
        self.refresh_button = QPushButton("Refresh")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        self.layout.addWidget(self.load_button)
        self.layout.addWidget(self.load_custom_model_btn)
        self.layout.addWidget(QLabel("Select Model:"))
        self.layout.addWidget(self.model_dropdown)
        self.layout.addWidget(QLabel("Select Layer:"))
        self.layout.addWidget(self.layer_dropdown)
        self.layout.addWidget(QLabel("CAM Techniques:"))
        for cb in self.checkboxes:
            self.layout.addWidget(cb)
        self.layout.addWidget(self.exp_widget)
        self.layout.addWidget(self.bb_box_btn)
        self.layout.addStretch()
        self.layout.addWidget(self.lime_btn)
        self.layout.addWidget(self.progress_bar)
        self.layout.addStretch()
        self.layout.addWidget(self.generate_button)
        self.layout.addWidget(self.save_button)
        self.layout.addWidget(self.refresh_button)
        self.progress_bar.show()

    
    
        
    
    