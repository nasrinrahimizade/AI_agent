##main.py###
import sys
import os
import asyncio
from qasync import QEventLoop
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication
from gui.main_window import MainWindow

def main():
    # Fix Qt plugin path
    if hasattr(QCoreApplication, 'addLibraryPath'):
        plugin_path = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'PySide6', 'plugins')
        if os.path.exists(plugin_path):
            QCoreApplication.addLibraryPath(plugin_path)
    
    app = QApplication(sys.argv)
    
    # wrap Qt in an asyncio event loop
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    window = MainWindow()
    window.show()
    
    with loop:
        loop.run_forever()

if __name__ == "__main__":
    main()