#IMPORTS
from PyQt5.QtWidgets import QTabWidget, QLabel,QVBoxLayout,QWidget,QFileDialog,QMenu,QApplication
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt,QPoint
import os
import numpy as np

#-----------------------------------------------------------------------------------------
class ClickableLabel(QLabel):
    """
    QLabel subclass that shows a context menu on click with actions like 'Save Image'.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def show_context_menu(self, pos: QPoint):
        menu = QMenu(self)

        save_action = menu.addAction("Save Image")
        copy_action = menu.addAction("Copy Image")
        # zoom_action = menu.addAction("Zoom")

        action = menu.exec_(self.mapToGlobal(pos))

        if action == save_action:
            self.save_image()
        elif action == copy_action:
            self.copy_image()

    def save_image(self):
        pixmap = self.pixmap()
        if pixmap and not pixmap.isNull():
            filename, _ = QFileDialog.getSaveFileName(
                self.window(),
                "Save Image As",
                "image.png",
                "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
            )
            if filename:
                success = pixmap.save(filename)
    
    def copy_image(self):
        pixmap = self.pixmap()
        if pixmap and not pixmap.isNull():
            QApplication.clipboard().setPixmap(pixmap)

#----------------------------------------------------------------------------------------------------------------

class ImageViewer(QTabWidget):
    """
    A QWidget subclass to display multiple images in tabs using PyQt.
    Supports adding images as tabs, closing tabs, and saving all images.

    Methods:
        - add_image_tab(title, image_input): Adds an image (file path or NumPy array) as a new tab.
        - close_tab(index): Closes the tab at the given index.
        - save_all_images(directory_path): Saves all images from tabs to the specified directory.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the tabbed image viewer.

        Parameters:
            parent (QWidget): Optional parent widget.
        """
        super().__init__(parent)
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(self.close_tab)

    def add_image_tab(self, title, image_input, score_text=None, debug_image_input=None):
        """
        Adds a new image tab with optional score text and optional debug image below the main image.

        Parameters:
            title (str): Title of the tab.
            image_input (str, np.ndarray, or QPixmap): Main image.
            score_text (str, optional): Score text like IoU/Recall.
            debug_image_input (str, np.ndarray, or QPixmap, optional): Optional debug image to display below.
        """
        layout = QVBoxLayout()

        def to_pixmap(img_input):
            if isinstance(img_input, str):
                pix = QPixmap(img_input)
                return pix if not pix.isNull() else None

            elif isinstance(img_input, np.ndarray):
                if img_input.dtype != np.uint8:
                    img_input = (img_input * 255).clip(0, 255).astype(np.uint8)

                if img_input.ndim == 2:
                    h, w = img_input.shape
                    qimage = QImage(img_input.data, w, h, w, QImage.Format_Grayscale8)
                elif img_input.ndim == 3 and img_input.shape[2] == 3:
                    h, w, ch = img_input.shape
                    qimage = QImage(img_input.data, w, h, ch * w, QImage.Format_RGB888)
                else:
                    print(f"[!] Unsupported NumPy image shape: {img_input.shape}")
                    return None
                return QPixmap.fromImage(qimage)

            elif isinstance(img_input, QPixmap):
                return img_input

            else:
                print(f"[!] Unsupported image input type: {type(img_input)}")
                return None

        # Main image
        main_pixmap = to_pixmap(image_input)
        if main_pixmap is None:
            print(f"[!] Failed to load main image.")
            return

        image_label = ClickableLabel()
        image_label.setToolTip("Click to save this image")
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setPixmap(main_pixmap.scaled(600, 400, Qt.KeepAspectRatio))
        layout.addWidget(image_label)
        
        # Debug image (if provided)
        if debug_image_input is not None:
            debug_pixmap = to_pixmap(debug_image_input)
            if debug_pixmap:
                debug_label = ClickableLabel()
                debug_label.setToolTip("Click to save this image")
                debug_label.setAlignment(Qt.AlignCenter)
                debug_label.setPixmap(debug_pixmap.scaled(600, 400, Qt.KeepAspectRatio))
                layout.addWidget(debug_label)

        # Score text
        if score_text:
            score_label = QLabel(score_text)
            score_label.setObjectName("scoreLabel")
            score_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(score_label)

        # Final tab widget
        tab_widget = QWidget()
        tab_widget.setLayout(layout)
        self.addTab(tab_widget, title)

    def close_tab(self,index):
        """
        Closes the tab at the given index.

        Parameters:
            index (int): Index of the tab to close.
        """
        self.removeTab(index)
    
    def save_all_images(self, directory_path):
        """
        Saves all images (main and debug) displayed in the tabs to the specified directory.

        Parameters:
            directory_path (str): Folder path where the images will be saved.

        Function:
            Iterates over all tabs, extracts QPixmaps from QLabel widgets (excluding score labels),
            and saves them as PNG images using the tab title as part of the filename.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        for i in range(self.count()):
            tab_widget = self.widget(i)
            tab_name = self.tabText(i).replace(" ", "_").replace("/", "_")
            
            layout = tab_widget.layout()
            if not isinstance(layout, QVBoxLayout):
                continue

            image_counter = 1  # To distinguish main and debug images
            for j in range(layout.count()):
                item = layout.itemAt(j)
                if item is None:
                    continue

                child_widget = item.widget()
                if isinstance(child_widget, QLabel):
                    pixmap = child_widget.pixmap()
                    if pixmap and not pixmap.isNull():
                        if child_widget.objectName() == "scoreLabel":
                            continue  # Skip saving score text labels

                        image_name = f"{tab_name}_image{image_counter}.png"
                        save_path = os.path.join(directory_path, image_name)
                        if not pixmap.save(save_path):
                            print(f"[!] Failed to save image: {save_path}")
                        else:
                            print(f"[+] Saved: {save_path}")
                        image_counter += 1

#------------------------------------------------------------------------------------------------------------------