# IMPORTS  
import sys
from PyQt5.QtWidgets import QApplication
from app.window import MainWindow
from PyQt5.QtGui import QFont,QIcon
from app.utils import resource_path
import matplotlib
matplotlib.use("Qt5Agg")
# ------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Entry point of the application.

    This function initializes the QApplication, sets the application-wide font and stylesheet,
    creates the main window, applies the window icon, and starts the event loop.
    """
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI"))
    window = MainWindow()
    window.setWindowIcon(QIcon(resource_path("assets/icon.png"))) 
    window.setStyleSheet("""background-color:rgba(40,40,40,255);color:#f5f5f5""")
    window.show()
    sys.exit(app.exec_())
