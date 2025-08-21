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
import os
import pandas as pd
import numpy as np

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



    def process_statistics(self, processed_root):
        """Process sensor data and generate statistics CSV"""
        # Output list of all stats
        all_stats = []
        
        try:
            # Loop through label folders
            for label in os.listdir(processed_root):
                label_path = os.path.join(processed_root, label)
                if not os.path.isdir(label_path):
                    continue

                # Loop through each sample
                for sample in os.listdir(label_path):
                    sample_path = os.path.join(label_path, sample)
                    if not os.path.isdir(sample_path):
                        continue

                    # Loop through each sensor CSV
                    for csv_file in os.listdir(sample_path):
                        if not csv_file.endswith(".csv"):
                            continue

                        file_path = os.path.join(sample_path, csv_file)
                        sensor = csv_file.replace(".csv", "")

                        try:
                            df = pd.read_csv(file_path)
                            if 'Time[s]' in df.columns:
                                df = df.drop(columns=['Time[s]'])

                            for column in df.columns:
                                values = df[column].values
                                stats = {
                                    'label': label,
                                    'sample': sample,
                                    'sensor': sensor,
                                    'channel': column,
                                    'mean': np.mean(values),
                                    'std': np.std(values),
                                    'min': np.min(values),
                                    'max': np.max(values),
                                    'median': np.median(values),
                                    'variance': np.var(values),
                                }
                                all_stats.append(stats)

                        except Exception as e:
                            print(f"⚠️ Failed to process {file_path}: {e}")

            # Convert to DataFrame and save
            stats_df = pd.DataFrame(all_stats)
            stats_csv_path = "all_statistics2.csv"
            stats_df.to_csv(stats_csv_path, index=False)
            print("✅ All statistics saved to all_statistics2.csv")
            
            return stats_csv_path
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process statistics: {str(e)}")
            return None

    def create_feature_matrix(self, stats_csv_path):
        """Create feature matrix from statistics CSV"""
        try:
            # Load your statistics file
            df = pd.read_csv(stats_csv_path)

            # Build a feature name: sensor_channel_stat (e.g., IIS3DWB_Z_mean)
            df['channel_clean'] = df['channel'].str.replace(r'\[.*?\]', '', regex=True).str.strip()
            df['feature'] = df['sensor'] + "_" + df['channel_clean'] + "_" + df.columns[4]

            # Pivot to wide format: one row per (label, sample), one column per feature
            feature_matrix = df.pivot_table(index=['label', 'sample'], columns='feature', values=df.columns[4])

            # Flatten multi-index columns
            feature_matrix.columns = feature_matrix.columns.to_flat_index()
            feature_matrix.columns = [col if isinstance(col, str) else "_".join(col) for col in feature_matrix.columns]

            # Reset index
            feature_matrix = feature_matrix.reset_index()

            # Replace bad values with NaN
            feature_matrix.replace(['#NAME?', 'inf', '=-inf'], pd.NA, inplace=True)

            # Option 1 (safe): fill missing values with 0
            feature_matrix.fillna(0, inplace=True)

            # Option 2 (if 0 distorts data): use column mean
            # feature_matrix.fillna(feature_matrix.mean(), inplace=True)

            # Save to CSV
            feature_matrix_path = "feature_matrix.csv"
            feature_matrix.to_csv(feature_matrix_path, index=False)
            print("✅ Feature matrix saved to feature_matrix.csv")
            
            return feature_matrix_path, feature_matrix
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create feature matrix: {str(e)}")
            return None, None

    # def openDataset(self):
    #     # Use QFileDialog to select a folder instead of a file
    #     folder_path = QFileDialog.getExistingDirectory(
    #         self, "Select Dataset Base Folder", ""
    #     )
        
    #     if folder_path:
    #         try:
    #             # Verify that the folder contains subdirectories (classes)
    #             subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
                
    #             if len(subdirs) == 0:
    #                 QMessageBox.warning(self, "Warning", "Selected folder contains no subdirectories.")
    #                 return
                
    #             # Show processing message
    #             self.statusBar().showMessage("Processing dataset... This may take a while.")
                
    #             # Step 1: Process statistics
    #             stats_csv_path = self.process_statistics(folder_path)
    #             if not stats_csv_path:
    #                 self.statusBar().showMessage("Failed to process statistics.")
    #                 return
                
    #             # Step 2: Create feature matrix
    #             feature_matrix_path, feature_df = self.create_feature_matrix(stats_csv_path)
    #             if not feature_matrix_path or feature_df is None:
    #                 self.statusBar().showMessage("Failed to create feature matrix.")
    #                 return
                
    #             # Update ML interface with the feature matrix path
    #             ml_interface.feature_matrix_path = feature_matrix_path
    #             ml_interface.mock_data = feature_df  # Set the dataframe directly
                
    #             # Distribute to views
    #             self.chatView.set_dataframe(feature_df)
                
    #             # Update status
    #             self.statusBar().showMessage(f"Dataset processed successfully. Feature matrix: {feature_matrix_path}")
                
    #             # Show success message
    #             QMessageBox.information(
    #                 self, 
    #                 "Success", 
    #                 f"Dataset processed successfully!\n"
    #                 f"Classes found: {len(subdirs)}\n"
    #                 f"Feature matrix saved to: {feature_matrix_path}"
    #             )
                
    #         except Exception as e:
    #             QMessageBox.critical(self, "Error", f"Failed to process dataset: {str(e)}")
    #             self.statusBar().showMessage("Failed to process dataset.")

    def openDataset(self):
        # Use QFileDialog to select a folder instead of a file
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Dataset Base Folder", ""
        )
        
        if folder_path:
            try:
                # Verify that the folder contains subdirectories (classes)
                subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
                
                if len(subdirs) == 0:
                    QMessageBox.warning(self, "Warning", "Selected folder contains no subdirectories.")
                    return
                
                # Show processing message
                # self.statusBar().showMessage("Processing dataset... This may take a while.")
                
                base_folder = folder_path

                df = load_dataset("ML/feature_matrix.csv")
                
                # Update ML interface with the new data path
                ml_interface.feature_matrix_path = "ML/feature_matrix.csv"
                ml_interface.mock_data = df  # Set the dataframe directly
                
                # Distribute to views
                self.chatView.set_dataframe(df)

                # Show success message
                QMessageBox.information(
                    self, 
                    "Success", 
                    f"Dataset processed successfully!\n"
                    f"Classes found: {len(subdirs)}\n"
                    f"Feature matrix saved to: {"ML/feature_matrix.csv"}"
                    )

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to process dataset: {str(e)}")
                self.statusBar().showMessage("Failed to process dataset.")