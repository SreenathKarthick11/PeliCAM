import sys
from PyQt5.QtWidgets import (
    QWidget, QLabel, QApplication, QFileDialog,
    QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QMessageBox
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QBrush, QColor, QImage,QIcon
from PyQt5.QtCore import Qt, QRect, QPoint
import os 
from app.utils import resource_path,load_stylesheet
from app.image_utils import compute_overlap_with_mask
import numpy as np

class ImageCanvas(QLabel):
    """
    A QLabel subclass for displaying an image and allowing users to draw colored bounding boxes.

    Methods:
        - mousePressEvent(event): Start drawing a rectangle.
        - mouseMoveEvent(event): Update rectangle dimensions as mouse moves.
        - mouseReleaseEvent(event): Finalize rectangle on mouse release.
        - undo_last(): Remove the last drawn rectangle.
        - get_boxes(): Return all drawn rectangles with their colors.
        - paintEvent(event): Draw the image and all bounding boxes.
        - get_final_image_with_boxes(): Return a QPixmap of image with all rectangles drawn.
    """
    def __init__(self, image_path, get_current_color):
        """
        Initialize the canvas with the given image and setup drawing state.
        """
        super().__init__()
        self.image = QPixmap(image_path)
        self.setPixmap(self.image)
        self.get_current_color = get_current_color

        self.rectangles = []  # List of tuples (QRect, QColor)
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()

        self.setFixedSize(self.image.size())

    def mousePressEvent(self, event):
        """
        Begin drawing a rectangle on mouse press.
        """
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        """
        Update the current rectangle while the mouse is moving.
        """
        if self.drawing:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        """
        Finalize the rectangle on mouse release.
        """
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            rect = QRect(self.start_point, self.end_point).normalized()
            color = self.get_current_color()
            self.rectangles.append((rect, color))
            self.update()

    def undo_last(self):
        """
        Undo the last drawn bounding box.
        """
        if self.rectangles:
            self.rectangles.pop()
            self.update()

    def get_boxes(self):
        """
        Return the list of drawn rectangles and their associated colors.
        """
        return self.rectangles
    
    def paintEvent(self, event):
        """
        Custom paint event to draw rectangles over the image.
        """
        super().paintEvent(event)
        painter = QPainter(self)
        
        # Draw finalized rectangles
        for rect, color in self.rectangles:
            painter.setPen(QPen(Qt.black, 2))
            painter.setBrush(QBrush(color, Qt.SolidPattern))
            painter.drawRect(rect)

        # Draw the currently dragged rectangle (not finalized yet)
        if self.drawing:
            rect = QRect(self.start_point, self.end_point).normalized()
            color = self.get_current_color()
            painter.setPen(QPen(Qt.black, 2, Qt.DashLine))
            painter.setBrush(QBrush(color, Qt.Dense4Pattern))  # hatch fill for visibility
            painter.drawRect(rect)
    

    def get_final_image_with_boxes(self):
        """
        Returns a new QPixmap with rectangles drawn on top of the original image.
        """
        result = QPixmap(self.image)
        painter = QPainter(result)
        for rect, color in self.rectangles:
            painter.setBrush(QBrush(color, Qt.SolidPattern))
            painter.drawRect(rect)
        painter.end()
        return result


class BoundingBoxDrawer(QWidget):
    """
    A QWidget window for displaying an image and allowing users to draw, undo, and save bounding boxes.

    Components:
        - Color dropdown: select bounding box color.
        - Undo button: remove last drawn box.
        - Send button: save the image with drawn boxes.
    """
    def __init__(self, image_path,parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bounding Box Drawer")

        self.image_path = image_path

        main_layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()
        self.setWindowFlags(Qt.Window)
        self.color_dropdown = QComboBox()
        self.color_dropdown.addItems(['blue', 'green', 'red', 'yellow'])
        undo_btn = QPushButton("Undo")
        send_btn = QPushButton("Send")
        self.setWindowIcon(QIcon(resource_path("assets/icon.png"))) 
        self.setStyleSheet(load_stylesheet(resource_path('app/style/bb_window.qss')))
        top_bar.addWidget(self.color_dropdown)
        top_bar.addWidget(undo_btn)
        top_bar.addWidget(send_btn)
        top_bar.addStretch()

        main_layout.addLayout(top_bar)

        self.canvas = ImageCanvas(image_path, self.get_current_color)
        main_layout.addWidget(self.canvas)

        undo_btn.clicked.connect(self.canvas.undo_last)
        send_btn.clicked.connect(self.send_boxes)

    def get_current_color(self):
        """
        Return the currently selected color from the dropdown.
        """
        return QColor(self.color_dropdown.currentText())

    def send_boxes(self):
        """
        Save the image with bounding boxes drawn to the bb_images directory.
        Also logs used colors.
        """
        result_pixmap = self.canvas.get_final_image_with_boxes()

        # Ensure the output directory exists
        output_dir = resource_path("app/bb_images")
        os.makedirs(output_dir, exist_ok=True)

        # Create unique filename with timestamp
        output_path = os.path.join(output_dir, f"bbox_.png")

        # Save the image
        result_pixmap.save(output_path, "PNG")

        # Now call generate_exp from parent (main window)
        if hasattr(self.parent(), 'generate_exp_from_bbox'):
            self.parent().generate_exp_from_bbox(bbox_mask_path=output_path)
            self.close()



    def closeEvent(self, event):
        """
        Clear rectangles when the window is closed to avoid retaining state on next open.
        """
        self.canvas.rectangles.clear()
        event.accept()

 

