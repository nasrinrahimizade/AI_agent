from PySide6.QtWidgets import (
    QMainWindow, QStatusBar, QFileDialog,
    QStackedWidget, QToolBar, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from gui.views.chat_view import ChatView
from gui.views.help_view import HelpView
from core.data_loader import load_dataset
from core.ml_interface import ml_interface

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Dataset Analyzer")
        self.resize(1200, 800)

        self._createMenuBar()
        self._createToolBar()
        self._createStatusBar()
        self._createCentralWidget()

    def _createMenuBar(self):
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu("File")

        openAction = QAction("Open Dataset", self)
        openAction.triggered.connect(self.openDataset)
        fileMenu.addAction(openAction)

        # Help menu
        helpMenu = menuBar.addMenu("Help")
        openHelpAction = QAction("Open Help", self)
        openHelpAction.triggered.connect(lambda: self.stack.setCurrentIndex(1))
        helpMenu.addAction(openHelpAction)

    def _createToolBar(self):
        toolBar = QToolBar("Main Toolbar")
        toolBar.setMovable(False)  # Prevent toolbar from being moved
        toolBar.setFloatable(False)  # Prevent toolbar from being floated
        self.addToolBar(Qt.TopToolBarArea, toolBar)
        
        
        
        # Add view switching actions
        chatAction = QAction("Chat + Plots", self)
        chatAction.triggered.connect(lambda: self.stack.setCurrentIndex(0))
        toolBar.addAction(chatAction)

        helpAction = QAction("Help", self)
        helpAction.triggered.connect(lambda: self.stack.setCurrentIndex(1))
        toolBar.addAction(helpAction)

    def _createStatusBar(self):
        statusBar = QStatusBar()
        self.setStatusBar(statusBar)

    def _createCentralWidget(self):
        self.stack = QStackedWidget()
        # Pass the openDataset method to ChatView
        self.chatView = ChatView(open_dataset_callback=self.openDataset)
        self.helpView = HelpView()

        # add to stack
        self.stack.addWidget(self.chatView)
        self.stack.addWidget(self.helpView)

        self.setCentralWidget(self.stack)
        
        # Set chat view as the default first screen (plots are now integrated into chat)
        self.stack.setCurrentIndex(0)  # Chat view is now at index 0



    def openDataset(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open Dataset", "", "CSV Files (*.csv);;All Files (*)"
        )
        if fileName:
            df = load_dataset(fileName)
            
            # Update ML interface with the new data path
            ml_interface.feature_matrix_path = fileName
            ml_interface.mock_data = df  # Set the dataframe directly
            
            # Distribute to views
            self.chatView.set_dataframe(df)
