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
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import functools
from typing import List, Tuple, Dict, Any
from pathlib import Path

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
        openAction.triggered.connect(self.openDataset_optimized)
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
        self.chatView = ChatView(open_dataset_callback=self.openDataset_optimized)
        self.helpView = HelpView()

        # add to stack
        self.stack.addWidget(self.chatView)
        self.stack.addWidget(self.helpView)

        self.setCentralWidget(self.stack)
        
        # Set chat view as the default first screen (plots are now integrated into chat)
        self.stack.setCurrentIndex(0)  # Chat view is now at index 0

        
    def process_statistics_parallel(self, processed_root: str, max_workers: int = None) -> str:
        """
        Parallelized version of process_statistics using ProcessPoolExecutor.
        """
        print(f"Starting parallel statistics processing for: {processed_root}")
        
        if max_workers is None:
            max_workers = min(cpu_count(), 8)  # Cap at 8 to avoid overwhelming system
        
        print(f"Using {max_workers} parallel workers")
        
        try:
            # Collect all file processing tasks
            file_tasks = collect_file_paths(processed_root)
            
            if len(file_tasks) == 0:
                print("No CSV files found to process!")
                return None
            
            print(f"Found {len(file_tasks)} files to process")
            
            all_stats = []
            
            # Use ProcessPoolExecutor for CPU-bound tasks
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_file = {executor.submit(process_single_csv_file, task): task 
                                for task in file_tasks}
                
                # Collect results as they complete
                completed_count = 0
                for future in as_completed(future_to_file):
                    try:
                        file_stats = future.result()
                        all_stats.extend(file_stats)
                        completed_count += 1
                        
                        # Progress reporting
                        if completed_count % 10 == 0 or completed_count == len(file_tasks):
                            print(f"Processed {completed_count}/{len(file_tasks)} files")
                            
                    except Exception as e:
                        file_path = future_to_file[future][0]
                        print(f"Error processing {file_path}: {e}")

            print(f"Total statistics collected: {len(all_stats)}")
            
            if len(all_stats) == 0:
                print("No statistics were collected!")
                return None

            # Convert to DataFrame and save
            stats_df = pd.DataFrame(all_stats)
            
            # Save with clear, unique name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_csv_path = f"statistics_{timestamp}.csv"
            
            print(f"Saving statistics to: {stats_csv_path}")
            stats_df.to_csv(stats_csv_path, index=False)
            
            # Verify file was saved
            if os.path.exists(stats_csv_path):
                file_size = os.path.getsize(stats_csv_path)
                print(f"Statistics saved successfully! File size: {file_size} bytes")
            else:
                print(f"File was not saved: {stats_csv_path}")
                return None
            
            return stats_csv_path
            
        except Exception as e:
            print(f"Critical error in process_statistics_parallel: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


    def create_feature_matrix_optimized(self, stats_csv_path: str) -> Tuple[str, pd.DataFrame]:
        """
        Optimized version of create_feature_matrix using vectorized operations.
        """
        print(f"Creating feature matrix from: {stats_csv_path}")
        
        try:
            # Verify input file exists
            if not os.path.exists(stats_csv_path):
                print(f"Statistics file does not exist: {stats_csv_path}")
                return None, None
                
            # Load statistics file with optimized dtypes
            print("Loading statistics file...")
            df = pd.read_csv(stats_csv_path)
            print(f"Loaded statistics shape: {df.shape}")
            
            # Check required columns
            required_cols = ['label', 'sample', 'sensor', 'channel', 'mean', 'std', 'min', 'max', 'median', 'variance']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Missing required columns: {missing_cols}")
                return None, None

            # Vectorized channel name cleaning
            print("Cleaning channel names...")
            df['channel_clean'] = df['channel'].str.replace(r'\[.*?\]', '', regex=True).str.strip()
            
            # Create feature names vectorized
            print("Creating feature names...")
            stat_columns = ['mean', 'std', 'min', 'max', 'median', 'variance']
            available_stats = [col for col in stat_columns if col in df.columns]
            
            # Create a single comprehensive feature matrix
            feature_dfs = []
            
            for stat_col in available_stats:
                print(f"Processing {stat_col} features...")
                df_temp = df[['label', 'sample', 'sensor', 'channel_clean', stat_col]].copy()
                df_temp['feature'] = (df_temp['sensor'] + "_" + 
                                    df_temp['channel_clean'] + "_" + stat_col)
                
                # Pivot to wide format
                pivoted = df_temp.pivot_table(
                    index=['label', 'sample'], 
                    columns='feature', 
                    values=stat_col,
                    aggfunc='first'
                )
                
                feature_dfs.append(pivoted)
            
            # Combine all feature matrices efficiently
            print("Combining feature matrices...")
            if feature_dfs:
                final_matrix = pd.concat(feature_dfs, axis=1)
            else:
                print("No valid feature matrices created!")
                return None, None

            # Reset index to make label and sample regular columns
            final_matrix = final_matrix.reset_index()
            print(f"Final matrix shape: {final_matrix.shape}")
            
            # Vectorized cleanup of bad values
            print("Cleaning bad values...")
            numeric_columns = final_matrix.select_dtypes(include=[np.number]).columns
            final_matrix[numeric_columns] = final_matrix[numeric_columns].replace(
                [np.inf, -np.inf], np.nan
            )
            final_matrix = final_matrix.fillna(0)
            
            # Save with clear, unique name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            feature_matrix_path = f"feature_matrix_{timestamp}.csv"
            
            print(f"Saving feature matrix to: {feature_matrix_path}")
            final_matrix.to_csv(feature_matrix_path, index=False)
            
            # Verify file was saved
            if os.path.exists(feature_matrix_path):
                file_size = os.path.getsize(feature_matrix_path)
                print(f"Feature matrix saved successfully! Shape: {final_matrix.shape}, Size: {file_size} bytes")
            else:
                print("Feature matrix file was not saved!")
                return None, None
            
            return feature_matrix_path, final_matrix
            
        except Exception as e:
            print(f"Critical error in create_feature_matrix_optimized: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None


    def openDataset_optimized(self):
        """
        Optimized version of openDataset with parallel processing.
        """
        Fulldataset = True  # be careful about this

        # Use QFileDialog to select a folder instead of a file
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Dataset Base Folder", ""
        )
        
        if folder_path:
            try:
                # Verify that the folder contains subdirectories (classes)
                subdirs = [d for d in os.listdir(folder_path) 
                        if os.path.isdir(os.path.join(folder_path, d))]
                
                if len(subdirs) == 0:
                    QMessageBox.warning(self, "Warning", "Selected folder contains no subdirectories.")
                    return
                
                # Show processing message
                self.statusBar().showMessage("Processing dataset in parallel... This may take a while.")
                print("Processing dataset in parallel... This may take a while.")
                
                # Step 1: Process statistics in parallel
                if Fulldataset is True:
                    print("folder_path:", folder_path)
                    
                    # Use parallel processing with progress updates
                    stats_csv_path = self.process_statistics_parallel(folder_path)
                    if not stats_csv_path:
                        self.statusBar().showMessage("Failed to process statistics.")
                        return
                    
                    # Step 2: Create feature matrix with optimizations
                    feature_matrix_path, feature_df = self.create_feature_matrix_optimized(stats_csv_path)
                    if not feature_matrix_path or feature_df is None:
                        self.statusBar().showMessage("Failed to create feature matrix.")
                        return
                    
                else:
                    # For testing with existing data
                    feature_matrix_path = "ML/feature_matrix.csv"
                    feature_df = load_dataset(feature_matrix_path)
                
                # UPDATE: Use the new update_data method
                success = ml_interface.update_data(feature_matrix_path, feature_df)
                if not success:
                    self.statusBar().showMessage("Failed to update ML interface with new data.")
                    return
                
                # Distribute to views
                self.chatView.set_dataframe(feature_df)
                
                # Update status
                self.statusBar().showMessage(f"Dataset processed successfully. Feature matrix: {feature_matrix_path}")
                
                # Show success message
                QMessageBox.information(
                    self, 
                    "Success", 
                    f"Dataset processed successfully!\n"
                    f"Classes found: {len(subdirs)}\n"
                    f"Feature matrix saved to: {feature_matrix_path}"
                )
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to process dataset: {str(e)}")
                self.statusBar().showMessage("Failed to process dataset.")

def process_single_csv_file(args: Tuple[str, str, str, str]) -> List[Dict]:
    """
    Process a single CSV file and return statistics.
    This function will be executed in parallel.
    """
    file_path, label, sample, sensor = args
    stats_list = []
    
    try:
        df = pd.read_csv(file_path)
        
        if 'Time[s]' in df.columns:
            df = df.drop(columns=['Time[s]'])

        for column in df.columns:
            values = df[column].values
            
            # Check for valid data
            if len(values) == 0:
                continue
                
            # Handle non-numeric data
            try:
                numeric_values = pd.to_numeric(values, errors='coerce')
                valid_values = numeric_values[~np.isnan(numeric_values)]
                
                if len(valid_values) == 0:
                    continue
                    
            except Exception as e:
                print(f"Error processing column {column}: {e}")
                continue
            
            stats = {
                'label': label,
                'sample': sample,
                'sensor': sensor,
                'channel': column,
                'mean': np.mean(valid_values),
                'std': np.std(valid_values),
                'min': np.min(valid_values),
                'max': np.max(valid_values),
                'median': np.median(valid_values),
                'variance': np.var(valid_values),
            }
            stats_list.append(stats)

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
    
    return stats_list


def collect_file_paths(processed_root: str) -> List[Tuple[str, str, str, str]]:
    """
    Collect all CSV file paths that need to be processed.
    Returns list of tuples: (file_path, label, sample, sensor)
    """
    file_tasks = []
    
    if not os.path.exists(processed_root):
        print(f"Root directory does not exist: {processed_root}")
        return file_tasks
        
    label_folders = [d for d in os.listdir(processed_root) 
                    if os.path.isdir(os.path.join(processed_root, d))]
    
    for label in label_folders:
        label_path = os.path.join(processed_root, label)
        sample_folders = [s for s in os.listdir(label_path) 
                        if os.path.isdir(os.path.join(label_path, s))]

        for sample in sample_folders:
            sample_path = os.path.join(label_path, sample)
            csv_files = [f for f in os.listdir(sample_path) if f.endswith(".csv")]

            for csv_file in csv_files:
                file_path = os.path.join(sample_path, csv_file)
                sensor = csv_file.replace(".csv", "")
                file_tasks.append((file_path, label, sample, sensor))
    
    return file_tasks