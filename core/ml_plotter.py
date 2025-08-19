import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
from scipy import stats
from scipy.fft import fft, fftfreq
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class PlottingEngine:
    """
    Advanced Plotting Engine for Statistical AI Agent
    Handles natural language requests and generates appropriate visualizations
    """
    
    def __init__(self, feature_matrix_path: str = None):
        """Initialize with your feature matrix data"""
        # Fix path resolution - use absolute path from project root
        if feature_matrix_path is None:
            import os
            # Get the project root directory (parent of 'core' directory)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            feature_matrix_path = os.path.join(project_root, "ML", "feature_matrix2.csv")
        
        print(f"🔍 Initializing plotting engine with: {feature_matrix_path}")
        try:
            self.df = pd.read_csv(feature_matrix_path)
            print(f"✅ Successfully loaded data: {self.df.shape}")
            print(f"📊 Columns: {list(self.df.columns[:5])}...")  # Show first 5 columns
            self.prepare_data()
        except FileNotFoundError:
            print(f"⚠️ Warning: {feature_matrix_path} not found. Using fallback data.")
            self.df = self._create_fallback_data()
            self.prepare_data()
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self.df = self._create_fallback_data()
            self.prepare_data()
        
    def _create_fallback_data(self):
        """Create fallback data if feature_matrix.csv is not available"""
        # Create sample data with similar structure
        np.random.seed(42)
        n_samples = 50
        
        data = {
            'label': np.random.choice(['OK', 'KO_HIGH_2mm', 'KO_LOW_2mm', 'KO_LOW_4mm'], n_samples),
            'sample': [f'Sample_{i}' for i in range(n_samples)]
        }
        
        # Add some sample features
        sensors = ['HTS221_HUM', 'HTS221_TEMP', 'IIS2DH_ACC', 'IIS2MDC_MAG']
        for sensor in sensors:
            for stat in ['mean', 'std', 'max', 'min']:
                data[f'{sensor}_{stat}'] = np.random.normal(0, 1, n_samples)
        
        return pd.DataFrame(data)
        
    def prepare_data(self):
        """Prepare data for plotting - use 4-class structure (OK, KO_HIGH_2mm, KO_LOW_2mm, KO_LOW_4mm)"""
        # Maintain binary classification for legacy callers
        self.df['binary_class'] = self.df['label'].apply(
            lambda x: 'OK' if x == 'OK' else 'KO'
        )
        
        # Multi-class setup
        self.classes = sorted(self.df['label'].unique())
        self.feature_columns = [col for col in self.df.columns if col not in ['label', 'sample', 'binary_class']]
        self.class_data = {cls: self.df[self.df['label'] == cls] for cls in self.classes}
        
        # Colors for classes
        self.class_colors = {
            'OK': '#2E8B57',            # Sea Green
            'KO_HIGH_2mm': '#FF8C00',   # Dark Orange
            'KO_LOW_2mm': '#8B008B',    # Dark Magenta
            'KO_LOW_4mm': '#4169E1',    # Royal Blue
            'KO': '#CD5C5C'             # Fallback if aggregated label exists
        }
        
        # Convenience binary subsets
        self.ok_data = self.df[self.df['binary_class'] == 'OK']
        self.ko_data = self.df[self.df['binary_class'] == 'KO']
        
        print(f"✅ Data prepared: classes={self.classes}")
        print(f"📊 Features available: {len(self.feature_columns)} features")
        print(f"🔍 Sample features: {self.feature_columns[:5]}")
        temp_features = [col for col in self.feature_columns if 'temp' in col.lower()]
        print(f"🌡️ Temperature features found: {temp_features}")
    
    def _is_feature_for_sensor(self, feature: str, sensor_type: str) -> bool:
        """Check if a feature belongs to a specific sensor type"""
        feature_lower = feature.lower()
        sensor_type_lower = sensor_type.lower()
        
        print(f"🔍 _is_feature_for_sensor: feature='{feature}', sensor_type='{sensor_type}'")
        
        # Precise sensor mapping
        sensor_mapping = {
            'temperature': ['temp'],
            'humidity': ['hum'],
            'pressure': ['press'],
            'accelerometer': ['acc'],
            'gyroscope': ['gyro'],
            'magnetometer': ['mag'],
            'microphone': ['mic']
        }
        
        if sensor_type_lower in sensor_mapping:
            keywords = sensor_mapping[sensor_type_lower]
            print(f"🔍 Checking keywords {keywords} for sensor type '{sensor_type_lower}'")
            for keyword in keywords:
                if keyword in feature_lower:
                    print(f"🔍 Keyword '{keyword}' found in feature '{feature_lower}' - MATCH!")
                    return True
                else:
                    print(f"🔍 Keyword '{keyword}' NOT found in feature '{feature_lower}'")
            print(f"🔍 No keywords matched for sensor type '{sensor_type_lower}'")
            return False
        
        print(f"🔍 Sensor type '{sensor_type_lower}' not in mapping, checking direct substring")
        result = sensor_type_lower in feature_lower
        print(f"🔍 Direct substring check: '{sensor_type_lower}' in '{feature_lower}' = {result}")
        return result
    
    def get_sensor_features(self, sensor_name: str) -> List[str]:
        """Get all features for a specific sensor with precise matching"""
        sensor_name_lower = sensor_name.lower()
        print(f"🔍 get_sensor_features called with: '{sensor_name}' (lowercase: '{sensor_name_lower}')")
        print(f"🔍 Available feature columns: {len(self.feature_columns)} features")
        print(f"🔍 Sample features: {self.feature_columns[:5]}")
        
        # Precise sensor mapping to actual column names
        sensor_mapping = {
            'temperature': ['HTS221_TEMP_TEMP_mean', 'LPS22HH_TEMP_TEMP_mean', 'STTS751_TEMP_TEMP_mean'],
            'humidity': ['HTS221_HUM_HUM_mean'],
            'pressure': ['LPS22HH_PRESS_PRESS_mean'],
            'accelerometer': ['IIS2DH_ACC_A_x_mean', 'IIS2DH_ACC_A_y_mean', 'IIS2DH_ACC_A_z_mean',
                             'IIS3DWB_ACC_A_x_mean', 'IIS3DWB_ACC_A_y_mean', 'IIS3DWB_ACC_A_z_mean',
                             'ISM330DHCX_ACC_A_x_mean', 'ISM330DHCX_ACC_A_y_mean', 'ISM330DHCX_ACC_A_z_mean'],
            'gyroscope': ['ISM330DHCX_GYRO_G_x_mean', 'ISM330DHCX_GYRO_G_y_mean', 'ISM330DHCX_GYRO_G_z_mean'],
            'magnetometer': ['IIS2MDC_MAG_M_x_mean', 'IIS2MDC_MAG_M_y_mean', 'IIS2MDC_MAG_M_z_mean'],
            'microphone': ['IMP23ABSU_MIC_MIC_mean', 'IMP34DT05_MIC_MIC_mean']
        }
        
        print(f"🔍 Sensor mapping keys: {list(sensor_mapping.keys())}")
        
        # Return exact features for the requested sensor
        if sensor_name_lower in sensor_mapping:
            print(f"🔍 Found exact match for '{sensor_name_lower}' in sensor_mapping")
            available_features = [col for col in sensor_mapping[sensor_name_lower] 
                                if col in self.feature_columns]
            print(f"🔍 Found {len(available_features)} features for {sensor_name}: {available_features}")
            return available_features
        
        # Fallback: search for partial matches
        print(f"🔍 No exact match, trying partial match for '{sensor_name_lower}'")
        sensor_features = [col for col in self.feature_columns 
                          if sensor_name_lower in col.lower()]
        print(f"🔍 Fallback: Found {len(sensor_features)} features for {sensor_name}: {sensor_features}")
        return sensor_features
    
    def get_top_discriminative_features(self, n: int = 5) -> List[str]:
        """Get top discriminative features between OK and KO classes"""
        if len(self.ok_data) == 0 or len(self.ko_data) == 0:
            return self.feature_columns[:n]
        
        feature_scores = {}
        
        for feature in self.feature_columns:
            try:
                ok_values = self.ok_data[feature].dropna()
                ko_values = self.ko_data[feature].dropna()
                
                if len(ok_values) > 0 and len(ko_values) > 0:
                    # Calculate t-statistic
                    t_stat, p_value = stats.ttest_ind(ok_values, ko_values)
                    feature_scores[feature] = abs(t_stat)
            except:
                continue
        
        # Sort by absolute t-statistic
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        return [feature for feature, score in sorted_features[:n]]
    
    def parse_plot_request(self, request: str) -> Dict:
        """
        Parse natural language plot requests
        Returns dict with plot_type, features, and parameters
        """
        request = request.lower()
        print(f"🔍 Parsing plot request: '{request}'")
        
        # Initialize result
        result = {
            'plot_type': 'timeseries',  # Changed default to timeseries
            'features': [],
            'sensor': None,
            'statistic': None,
            'comparison': True,  # OK vs KO comparison by default
            'class_filter': None,  # NEW: Add class filtering
            'file_paths': []       # NEW: Add custom file paths
        }
        
        # Detect plot types - prioritize line graphs (time series) and frequency analysis
        if any(word in request for word in ['line graph', 'line', 'time series', 'time', 'temporal', 'trend', 'over time']):
            result['plot_type'] = 'timeseries'
        elif any(word in request for word in ['frequency', 'fft', 'spectrum', 'oscillation', 'vibration']):
            result['plot_type'] = 'frequency'
        elif any(word in request for word in ['histogram', 'hist', 'distribution']):
            result['plot_type'] = 'histogram'
        elif any(word in request for word in ['correlation', 'corr', 'relationship']):
            result['plot_type'] = 'correlation'
        elif any(word in request for word in ['scatter', 'scatter plot']):
            result['plot_type'] = 'scatter'
        

        # NEW: Detect class filtering
        if any(word in request for word in ['only ok', 'just ok', 'ok only', 'ok class']):
            result['class_filter'] = ['OK']
        elif any(word in request for word in ['only ko', 'just ko', 'ko only', 'ko class']):
            result['class_filter'] = ['KO_HIGH_2mm', 'KO_LOW_2mm', 'KO_LOW_4mm']
        elif any(word in request for word in ['only ko_high', 'ko high', 'high ko']):
            result['class_filter'] = ['KO_HIGH_2mm']
        elif any(word in request for word in ['only ko_low_2mm', 'ko low 2mm']):
            result['class_filter'] = ['KO_LOW_2mm']
        elif any(word in request for word in ['only ko_low_4mm', 'ko low 4mm']):
            result['class_filter'] = ['KO_LOW_4mm']

        # Detect sensors with precise matching
        sensor_keywords = {
            'temperature': ['temperature', 'temp', 'thermal'],
            'humidity': ['humidity', 'hum', 'moisture'],
            'pressure': ['pressure', 'press', 'barometric'],
            'accelerometer': ['accelerometer', 'acc', 'acceleration', 'motion'],
            'gyroscope': ['gyroscope', 'gyro', 'rotation', 'angular'],
            'magnetometer': ['magnetometer', 'mag', 'magnetic', 'compass'],
            'microphone': ['microphone', 'mic', 'audio', 'sound']
        }
        
        print(f"🔍 Checking sensor keywords: {sensor_keywords}")
        
        # Find the most specific sensor match
        best_match = None
        best_score = 0
        
        for sensor_type, keywords in sensor_keywords.items():
            for keyword in keywords:
                if keyword in request:
                    # Score based on keyword length (more specific = higher score)
                    score = len(keyword)
                    print(f"🔍 Found keyword '{keyword}' for sensor '{sensor_type}' with score {score}")
                    if score > best_score:
                        best_score = score
                        best_match = sensor_type
                        print(f"🔍 New best match: '{sensor_type}' with score {score}")
        
        if best_match:
            result['sensor'] = best_match
            print(f"🎯 Final sensor match: {best_match}")
            result['features'] = self.get_sensor_features(best_match)
            print(f"🎯 Detected sensor: {best_match}")
        else:
            print(f"⚠️ No sensor detected in request")
        
        # If no sensor detected, try to find specific sensor names in the request
        if not result['features']:
            for col in self.feature_columns:
                if any(sensor_type in col.lower() for sensor_type in ['temp', 'hum', 'press', 'acc', 'gyro', 'mag', 'mic']):
                    if any(word in col.lower() for word in request.split()):
                        result['features'].append(col)
                        if not result['sensor']:
                            # Determine sensor type from feature name
                            if 'temp' in col.lower():
                                result['sensor'] = 'temperature'
                            elif 'hum' in col.lower():
                                result['sensor'] = 'humidity'
                            elif 'press' in col.lower():
                                result['sensor'] = 'pressure'
                            elif 'acc' in col.lower():
                                result['sensor'] = 'accelerometer'
                            elif 'gyro' in col.lower():
                                result['sensor'] = 'gyroscope'
                            elif 'mag' in col.lower():
                                result['sensor'] = 'magnetometer'
                            elif 'mic' in col.lower():
                                result['sensor'] = 'microphone'
        
        # Detect statistics
        stats = ['mean', 'median', 'max', 'min', 'std', 'variance', 'var']
        for stat in stats:
            if stat in request:
                result['statistic'] = stat
                break
        
        # If specific feature mentioned, try to find it
        if not result['features'] and result['sensor'] is None:
            # Look for specific feature names in the request
            words = request.split()
            for word in words:
                matching_features = [col for col in self.feature_columns 
                                   if word in col.lower()]
                if matching_features:
                    result['features'] = matching_features[:5]  # Limit to 5 features
                    break
        
        # Final fallback: if still no features and temperature was requested, use temperature features
        if not result['features'] and ('temperature' in request.lower() or 'temp' in request.lower()):
            # Look for any temperature-related features in the dataset
            temp_features = [col for col in self.feature_columns if 'temp' in col.lower()]
            if temp_features:
                result['features'] = temp_features[:3]  # Use up to 3 temperature features
                result['sensor'] = 'temperature'
        
        # Final fallback: if still no features, use top discriminative features
        if not result['features']:
            result['features'] = self.get_top_discriminative_features(3)
        
        # Ensure we only show features for the requested sensor type
        if result['sensor'] and result['features']:
            print(f"🔍 Filtering features for sensor '{result['sensor']}'")
            print(f"🔍 Features before filtering: {result['features']}")
            
            # Filter features to only include the requested sensor type
            filtered_features = []
            for feature in result['features']:
                if self._is_feature_for_sensor(feature, result['sensor']):
                    filtered_features.append(feature)
                    print(f"🔍 Feature '{feature}' matches sensor '{result['sensor']}'")
                else:
                    print(f"🔍 Feature '{feature}' does NOT match sensor '{result['sensor']}'")
            
            if filtered_features:
                result['features'] = filtered_features
                print(f"🎯 Filtered to {len(filtered_features)} {result['sensor']} features: {filtered_features}")
            else:
                print(f"⚠️ No {result['sensor']} features found, using all features")
        else:
            print(f"⚠️ No sensor or features to filter")
        
        print(f"🎯 Final parsed result: {result}")
        return result
    
    def plot_feature_comparison(self, features: List[str], plot_type: str = 'histogram') -> plt.Figure:
        """Generate histogram plots by class for specific features (4-class)"""
        
        if not features:
            # If no specific features, use top discriminative features
            features = self.get_top_discriminative_features(n=3)
        
        # Limit features to avoid overcrowded plots
        features = features[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(features):
            if i >= 6:  # Safety check
                break
                
            ax = axes[i]
            
            # Only histogram plots are supported now
            if plot_type == 'histogram':
                # Overlay per class; ensure every class appears in the legend even if no data
                plotted_classes = []
                for class_name in self.classes:
                    class_values = self.class_data[class_name][feature].dropna()
                    if len(class_values) > 0:
                        ax.hist(
                            class_values,
                            alpha=0.6,
                            label=class_name,
                            bins=20,
                            color=self.class_colors.get(class_name, 'gray'),
                            density=True
                        )
                        plotted_classes.append(class_name)
                # Add legend entries for classes with no data so they still show up
                missing_classes = [c for c in self.classes if c not in plotted_classes]
                for class_name in missing_classes:
                    ax.plot([], [], color=self.class_colors.get(class_name, 'lightgray'),
                            label=f"{class_name} (no data)")
                ax.legend(title='Class')
            
            # Clean up feature name for title
            clean_name = feature.replace('_', ' ').title()
            ax.set_title(f'{clean_name}', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(features), 6):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Histogram Comparison by Class', fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, features: Optional[List[str]] = None) -> plt.Figure:
        """Generate correlation matrix for features"""
        
        if features is None or len(features) == 0:
            # Use top discriminative features
            features = self.get_top_discriminative_features(n=10)
        
        # Limit to reasonable number for readability
        features = features[:15]
        
        # Calculate correlation matrix
        corr_data = self.df[features].corr()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Generate heatmap
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, fmt='.2f', cbar_kws={'shrink': 0.8})
        
        ax.set_title('Feature Correlation Matrix', fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_time_series(self, features: Optional[List[str]] = None) -> plt.Figure:
        """
        Plot time series data (simulated since we don't have actual time stamps)
        Uses sample order as time proxy
        """
        
        if features is None or len(features) == 0:
            features = self.get_top_discriminative_features(n=3)
        
        features = features[:4]  # Limit to 4 features
        
        fig, axes = plt.subplots(len(features), 1, figsize=(14, 3*len(features)))
        if len(features) == 1:
            axes = [axes]
        
        for i, feature in enumerate(features):
            ax = axes[i]
            
            # Plot each class
            for class_name in self.classes:
                class_df = self.class_data[class_name]
                if len(class_df) == 0:
                    continue
                indices = class_df.index
                ax.plot(
                    indices,
                    class_df[feature],
                    'o-',
                    label=class_name,
                    alpha=0.7,
                    color=self.class_colors.get(class_name, None)
                )
            
            ax.set_title(f'{feature.replace("_", " ").title()}')
            ax.set_xlabel('Sample Index')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Time Series Analysis (Sample Order)', fontsize=16)
        plt.tight_layout()
        return fig
    
    def get_sensor_file_config(self) -> Dict:
        """Get sensor file configuration - make this configurable later"""
        # Detect dataset root dynamically
        base_path = self._detect_dataset_root()

        # Detect class folders and condition folders dynamically if possible
        class_folders: List[str] = ['OK', 'KO_HIGH_2mm', 'KO_LOW_2mm', 'KO_LOW_4mm']
        condition_folders: List[str] = ['PMI_50rpm']

        if base_path and os.path.isdir(base_path):
            try:
                # Classes are the top-level directories
                detected_classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                if detected_classes:
                    class_folders = detected_classes
                # Conditions are subdirectories under the first existing class
                class_to_scan = None
                for cand in ['OK'] + detected_classes:
                    cand_path = os.path.join(base_path, cand)
                    if os.path.isdir(cand_path):
                        class_to_scan = cand_path
                        break
                if class_to_scan:
                    detected_conditions = [d for d in os.listdir(class_to_scan) if os.path.isdir(os.path.join(class_to_scan, d))]
                    if detected_conditions:
                        condition_folders = detected_conditions
            except Exception:
                pass

        return {
            'base_path': base_path,
            'class_folders': class_folders,
            'condition_folders': condition_folders,  # Can add more conditions
            'sensors': {
                'accelerometer': {
                    'file_patterns': ['IIS2DH_ACC.csv', 'IIS3DWB_ACC.csv', 'ISM330DHCX_ACC.csv'],
                    'columns': ['A_x [g]', 'A_y [g]', 'A_z [g]']
                },
                'microphone': {
                    'file_patterns': ['IMP23ABSU_MIC.csv', 'IMP34DT05_MIC.csv'],
                    'columns': ['MIC [dB]', 'MIC']
                },
                'gyroscope': {
                    'file_patterns': ['ISM330DHCX_GYRO.csv'],
                    'columns': ['G_x [dps]', 'G_y [dps]', 'G_z [dps]']
                },
                'magnetometer': {
                    'file_patterns': ['IIS2MDC_MAG.csv'],
                    'columns': ['M_x [mgauss]', 'M_y [mgauss]', 'M_z [mgauss]']
                },
                'temperature': {
                    'file_patterns': ['HTS221_TEMP.csv', 'LPS22HH_TEMP.csv', 'STTS751_TEMP.csv'],
                    'columns': ['TEMP [°C]', 'Temperature [°C]']
                },
                'humidity': {
                    'file_patterns': ['HTS221_HUM.csv'],
                    'columns': ['HUM [%]', 'Humidity [%]']
                },
                'pressure': {
                    'file_patterns': ['LPS22HH_PRESS.csv'],
                    'columns': ['PRESS [hPa]', 'Pressure [hPa]']
                }
            }
        }


    def plot_frequency_domain(self, features: Optional[List[str]] = None, sensor_type: str = None, class_filter: List[str] = None) -> plt.Figure:
        """Generate frequency domain analysis using FFT from raw sensor data files"""
        
        if features is None or len(features) == 0:
            features = self.get_top_discriminative_features(n=4)
        
        config = self.get_sensor_file_config()
        # print(f"#################classes: {class_filter}")
        
        # Determine which classes to plot
        if class_filter:
            classes_to_plot = class_filter
            print(f"🎯 Filtering to classes: {classes_to_plot}")
        else:
            classes_to_plot = config['class_folders']
        
        # If we have a specific sensor type, try to load raw data
        if sensor_type and sensor_type in config['sensors']:
            sensor_config = config['sensors'][sensor_type]

            # If dataset root missing, fallback
            if not config['base_path'] or not os.path.isdir(config['base_path']):
                print("⚠️ Dataset root not found for frequency plot. Falling back.")
                return self._fallback_frequency_plot(features, class_filter)
            
            # Calculate total combinations: classes × sensor files
            n_classes = len(classes_to_plot)
            n_sensors = len(sensor_config['file_patterns'])
            total_combinations = n_classes * n_sensors
            
            print(f"📊 Planning layout: {n_classes} classes × {n_sensors} sensors = {total_combinations} combinations")
            
            # Create subplots: each sensor file gets one row with 2 columns (linear + log)
            rows = total_combinations
            cols = 2  # Always linear + logarithmic
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
            
            # Ensure axes is always 2D
            if rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            
            current_row = 0
            plots_created = 0
            
            for class_name in classes_to_plot:
                print(f"📊 Processing class: {class_name}")
                
                for sensor_idx, file_pattern in enumerate(sensor_config['file_patterns']):
                    sensor_name = file_pattern.replace('.csv', '').replace('_', ' ')
                    data_loaded = False
                    
                    print(f"  🔍 Processing sensor file: {file_pattern}")
                    
                    for condition in config['condition_folders']:
                        file_path = os.path.join(config['base_path'], class_name, condition, file_pattern)
                        
                        try:
                            df = pd.read_csv(file_path)
                            print(f"    ✅ Loaded: {file_path}")
                            
                            # Find time column
                            time_cols = [col for col in df.columns if 'time' in col.lower()]
                            if not time_cols:
                                print(f"    ⚠️ No time column in {file_path}")
                                continue
                            time_col = time_cols[0]
                            
                            # Find signal column - try multiple column patterns
                            signal_col = None
                            for col_pattern in sensor_config['columns']:
                                if col_pattern in df.columns:
                                    signal_col = col_pattern
                                    break

                            # Fallback: heuristic based on sensor type and column keywords
                            if not signal_col:
                                candidate_cols = [c for c in df.columns if c != time_col]
                                keyword_map = {
                                    'temperature': ['temp'],
                                    'humidity': ['hum'],
                                    'pressure': ['press'],
                                    'accelerometer': ['a_x', 'a y', 'a_z', 'a_'],
                                    'gyroscope': ['g_x', 'g y', 'g_z', 'g_'],
                                    'magnetometer': ['m_x', 'm y', 'm_z', 'm_'],
                                    'microphone': ['mic']
                                }
                                keys = keyword_map.get(sensor_type, [])
                                for col in candidate_cols:
                                    lname = col.lower()
                                    if any(k in lname for k in keys):
                                        signal_col = col
                                        break
                                # Last resort: first numeric non-time column
                                if not signal_col:
                                    for col in candidate_cols:
                                        if pd.api.types.is_numeric_dtype(df[col]):
                                            signal_col = col
                                            break

                            if not signal_col:
                                print(f"    ⚠️ No signal column found in {file_path}. Available: {df.columns.tolist()}")
                                continue
                                
                            time = df[time_col].values
                            signal = df[signal_col].values
                            n = len(signal)
                            
                            print(f"    📈 Processing {n} data points from column '{signal_col}'")
                            
                            # Calculate sampling parameters
                            if len(time) > 1:
                                dt = np.mean(np.diff(time))
                            else:
                                dt = 1.0  # Default sampling interval
                            
                            # Perform FFT
                            yf = fft(signal)
                            xf = fftfreq(n, d=dt)
                            
                            # Plot positive frequencies only
                            positive_freqs = xf[:n//2]
                            positive_fft = np.abs(yf[:n//2])
                            
                            # Linear scale plot (left column)
                            ax_linear = axes[current_row, 0]
                            ax_linear.plot(positive_freqs, positive_fft, 
                                        color=self.class_colors.get(class_name, 'blue'),
                                        label=f'{class_name}', linewidth=1.5)
                            ax_linear.set_title(f'{sensor_name} - {class_name}\n(Linear Scale)', fontsize=11, pad=10)
                            ax_linear.set_xlabel('Frequency [Hz]')
                            ax_linear.set_ylabel('Amplitude')
                            ax_linear.grid(True, alpha=0.3)
                            ax_linear.legend(fontsize=9)
                            
                            # Logarithmic scale plot (right column)
                            ax_log = axes[current_row, 1]
                            positive_fft_db = 20 * np.log10(positive_fft + 1e-12)  # Avoid log(0)
                            ax_log.plot(positive_freqs, positive_fft_db, 
                                    color=self.class_colors.get(class_name, 'blue'),
                                    label=f'{class_name}', linewidth=1.5)
                            ax_log.set_title(f'{sensor_name} - {class_name}\n(Log Scale dB)', fontsize=11, pad=10)
                            ax_log.set_xlabel('Frequency [Hz]')
                            ax_log.set_ylabel('Amplitude [dB]')
                            ax_log.grid(True, alpha=0.3)
                            ax_log.legend(fontsize=9)
                            
                            plots_created += 1
                            data_loaded = True
                            print(f"    ✅ Plotted row {current_row}: {sensor_name} - {class_name}")
                            break  # Break condition loop once file is loaded
                            
                        except Exception as e:
                            print(f"    ❌ Could not load {file_path}: {e}")
                            continue
                    
                    if not data_loaded:
                        print(f"    ❌ No data loaded for {class_name} - {file_pattern}")
                        # Hide empty row
                        axes[current_row, 0].set_visible(False)
                        axes[current_row, 1].set_visible(False)
                    
                    current_row += 1  # Always move to next row, whether data was loaded or not
            
            # Create title based on filtering
            if class_filter:
                title_suffix = f" - {', '.join(class_filter)} Classes"
            else:
                title_suffix = " - All Classes"
            
            plt.suptitle(f'{sensor_type.title()} Frequency Analysis{title_suffix}', fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0.02, 1, 0.96])  # Leave space for suptitle
            
            print(f"🎯 Final result: Created {plots_created} plots in {current_row} rows")
            return fig
        
        # Fallback to feature matrix method
        else:
            print("🔄 Falling back to feature matrix data")
            return self._fallback_frequency_plot(features, class_filter)

    def _fallback_frequency_plot(self, features: Optional[List[str]] = None, class_filter: List[str] = None) -> plt.Figure:
        """Fallback frequency plot using feature matrix data when raw data fails"""
        
        if features is None or len(features) == 0:
            features = self.get_top_discriminative_features(n=4)
        
        # Remove the 4-feature limit - use all available features
        print(f"🔄 Fallback: Using {len(features)} features")
        
        # Filter data by class if specified
        if class_filter:
            filtered_df = self.df[self.df['label'].isin(class_filter)]
            print(f"🔄 Fallback: Using {len(filtered_df)} samples from classes {class_filter}")
        else:
            filtered_df = self.df
        
        # Calculate subplot layout for all features
        n_features = len(features)
        cols = 2  # Linear + Log
        rows = n_features
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
        
        # Ensure axes is always 2D
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, feature in enumerate(features):
            # Get data for FFT from filtered dataset
            all_data = filtered_df[feature].dropna().values
            
            if len(all_data) > 1:
                # Perform FFT
                fft_result = fft(all_data)
                freqs = fftfreq(len(all_data))
                
                # Plot magnitude spectrum
                magnitude = np.abs(fft_result)
                
                # Linear plot
                ax_linear = axes[i, 0]
                ax_linear.plot(freqs[:len(freqs)//2], magnitude[:len(freqs)//2])
                
                # Add class info to title if filtering
                if class_filter:
                    title_suffix = f" - {', '.join(class_filter)}"
                else:
                    title_suffix = ""
                    
                ax_linear.set_title(f'{feature.replace("_", " ").title()}\nFrequency Spectrum (Linear){title_suffix}', fontsize=10)
                ax_linear.set_xlabel('Frequency')
                ax_linear.set_ylabel('Amplitude')
                ax_linear.grid(True, alpha=0.3)
                
                # Logarithmic plot
                ax_log = axes[i, 1]
                magnitude_db = 20 * np.log10(magnitude[:len(freqs)//2] + 1e-12)
                ax_log.plot(freqs[:len(freqs)//2], magnitude_db)
                ax_log.set_title(f'{feature.replace("_", " ").title()}\nFrequency Spectrum (Log dB){title_suffix}', fontsize=10)
                ax_log.set_xlabel('Frequency')
                ax_log.set_ylabel('Amplitude [dB]')
                ax_log.grid(True, alpha=0.3)
            else:
                # Hide empty plots
                axes[i, 0].set_visible(False)
                axes[i, 1].set_visible(False)
        
        # Create title based on filtering
        if class_filter:
            title_suffix = f" - {', '.join(class_filter)} Only"
        else:
            title_suffix = " - All Classes"
        
        plt.suptitle(f'Frequency Domain Analysis (Feature Matrix){title_suffix}', fontsize=16)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        return fig
    def plot_scatter(self, features: Optional[List[str]] = None) -> plt.Figure:
        """Generate scatter plots showing feature relationships"""
        
        if features is None or len(features) == 0:
            features = self.get_top_discriminative_features(n=4)
        
        features = features[:4]  # Limit to 4 features
        
        if len(features) < 2:
            # If only one feature, plot per class against sample index
            fig, ax = plt.subplots(figsize=(10, 6))
            for class_name in self.classes:
                class_df = self.class_data[class_name]
                if len(class_df) == 0:
                    continue
                ax.scatter(
                    class_df.index,
                    class_df[features[0]],
                    label=class_name,
                    alpha=0.7,
                    c=self.class_colors.get(class_name, 'gray')
                )
            ax.set_title(f'{features[0].replace("_", " ").title()} vs Sample Index')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel(features[0].replace('_', ' ').title())
            ax.legend()
            return fig
        
        # Create scatter matrix
        fig, axes = plt.subplots(len(features), len(features), figsize=(12, 12))
        
        for i, feat1 in enumerate(features):
            for j, feat2 in enumerate(features):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: histogram
                    ax.hist(self.df[feat1].dropna(), bins=20, alpha=0.7)
                    ax.set_title(f'{feat1.replace("_", " ").title()}')
                else:
                    # Off-diagonal: scatter plot
                    # Plot each class with its color
                    for class_name in self.classes:
                        class_df = self.class_data[class_name]
                        if len(class_df) == 0:
                            continue
                        ax.scatter(
                            class_df[feat1],
                            class_df[feat2],
                            alpha=0.6,
                            label=class_name,
                            c=self.class_colors.get(class_name, 'gray')
                        )
                    if i == 0 and j == 1:
                        ax.legend(fontsize=7)
                    ax.set_xlabel(feat1.replace('_', ' ').title())
                    ax.set_ylabel(feat2.replace('_', ' ').title())
                
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Feature Relationship Matrix', fontsize=16)
        plt.tight_layout()
        return fig
    
    def handle_plot_request(self, request: str) -> plt.Figure:
        """Main method to handle plot requests"""
        
        # Parse the request
        parsed = self.parse_plot_request(request)
        print(f"🎯 Parsed request: {parsed}")
        
        # Generate appropriate plot - all plots are now line graphs or specialized visualizations
        if parsed['plot_type'] == 'timeseries':
            print(f"📈 Generating line graph (time series)")
            # Try dataset-driven time plot first (dynamic paths)
            fig = self._plot_time_series_from_dataset(request)
            if fig is not None:
                return fig
            # Fallback to feature_matrix-based time series
            return self.plot_time_series(parsed['features'])
        # elif parsed['plot_type'] == 'frequency':
        #     print(f"📡 Generating frequency domain plot")
        #     return self.plot_frequency_domain(parsed['features'])
        elif parsed['plot_type'] == 'frequency':
            print(f"📡 Generating frequency domain plot")
            return self.plot_frequency_domain(parsed['features'], parsed['sensor'], parsed['class_filter'])
        elif parsed['plot_type'] == 'histogram':
            print(f"📊 Generating histogram plot")
            return self.plot_feature_comparison(parsed['features'], 'histogram')
        elif parsed['plot_type'] == 'correlation':
            print(f"🔗 Generating correlation matrix")
            return self.plot_correlation_matrix(parsed['features'])
        elif parsed['plot_type'] == 'scatter':
            print(f"💫 Generating scatter plot")
            return self.plot_scatter(parsed['features'])
        else:
            # Default to line graph (time series)
            print(f"🔄 Defaulting to line graph (time series)")
            fig = self._plot_time_series_from_dataset(request)
            if fig is not None:
                return fig
            return self.plot_time_series(parsed['features'])

    def _detect_dataset_root(self) -> str:
        """Detect dataset root directory dynamically."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            candidates = [
                os.path.join(project_root, 'dataset', 'vel-fissa')
            ]
            for path in candidates:
                if os.path.isdir(path):
                    return path
        except Exception:
            pass
        return ''

    def _detect_labels_from_request(self, request: str) -> list:
        """Extract label filters (e.g., OK, KO variants) from the natural language request."""
        text = (request or '').lower()
        labels = []
        mapping = {
            'ok': 'OK',
            'ko_high_2mm': 'KO_HIGH_2mm',
            'ko low 2mm': 'KO_LOW_2mm',
            'ko_low_2mm': 'KO_LOW_2mm',
            'ko low 4mm': 'KO_LOW_4mm',
            'ko_low_4mm': 'KO_LOW_4mm',
            'ko': 'KO'
        }
        for key, val in mapping.items():
            if key in text:
                labels.append(val)
        # Deduplicate preserving order
        seen = set()
        filtered = []
        for l in labels:
            if l not in seen:
                seen.add(l)
                filtered.append(l)
        return filtered

    def _sensor_keywords(self, request: str) -> list:
        text = (request or '').lower()
        if any(k in text for k in ['temperature', 'temp']):
            return ['temp']
        if any(k in text for k in ['humidity', 'hum']):
            return ['hum']
        if any(k in text for k in ['pressure', 'press']):
            return ['press']
        if any(k in text for k in ['accelerometer', 'acc', 'acceleration']):
            return ['acc']
        if any(k in text for k in ['gyroscope', 'gyro']):
            return ['gyro']
        if any(k in text for k in ['magnetometer', 'mag']):
            return ['mag']
        if any(k in text for k in ['microphone', 'mic', 'audio']):
            return ['mic']
        return []

    def _plot_time_series_from_dataset(self, request: str):
        """Create a time plot directly from dataset CSVs (dynamic paths). Return Figure or None."""
        dataset_root = self._detect_dataset_root()
        if not dataset_root:
            return None

        labels = self._detect_labels_from_request(request)
        if labels:
            expanded = []
            for l in labels:
                if l == 'KO':
                    expanded.extend(['KO_HIGH_2mm', 'KO_LOW_2mm', 'KO_LOW_4mm'])
                else:
                    expanded.append(l)
            label_dirs = expanded
        else:
            label_dirs = ['OK', 'KO_HIGH_2mm', 'KO_LOW_2mm', 'KO_LOW_4mm']

        sensor_kws = self._sensor_keywords(request)
        if not sensor_kws:
            return None

        matched_files = []
        try:
            for label in label_dirs:
                label_path = os.path.join(dataset_root, label)
                if not os.path.isdir(label_path):
                    continue
                for csv_path in glob.iglob(os.path.join(label_path, '**', '*.csv'), recursive=True):
                    name = os.path.basename(csv_path).lower()
                    if any(kw in name for kw in sensor_kws):
                        matched_files.append(csv_path)
        except Exception:
            return None

        if not matched_files:
            return None

        # Group files by class to ensure representation from each class
        class_to_files = {}
        for path in matched_files:
            try:
                rel_path = os.path.relpath(path, dataset_root)
                class_label = rel_path.split(os.sep)[0] if os.sep in rel_path else rel_path.split('/')[0]
            except Exception:
                class_label = 'Unknown'
            class_to_files.setdefault(class_label, []).append(path)

        # Prefer specific temperature files for comparability if temperature requested
        preferred_temp_files = ['STTS751_TEMP.csv', 'LPS22HH_TEMP.csv', 'HTS221_TEMP.csv'] if 'temp' in sensor_kws else []

        # Build a balanced selection across classes
        selected_files = []
        # Step 1: pick one preferred file per class if available
        for class_label in label_dirs:
            files = class_to_files.get(class_label, [])
            if not files:
                continue
            picked = None
            if preferred_temp_files:
                for pref in preferred_temp_files:
                    for f in files:
                        if os.path.basename(f).lower() == pref.lower():
                            picked = f
                            break
                    if picked:
                        break
            if not picked:
                picked = files[0]
            selected_files.append(picked)

        # Step 2: if we still have room, round-robin remaining files per class
        remaining_per_class = {c: [f for f in class_to_files.get(c, []) if f not in selected_files] for c in label_dirs}

        max_plots = min(6, max(1, len(selected_files)))
        # Try to fill up to max_plots by interleaving classes
        while len(selected_files) < max_plots:
            added = False
            for c in label_dirs:
                rem = remaining_per_class.get(c, [])
                if rem:
                    selected_files.append(rem.pop(0))
                    added = True
                    if len(selected_files) >= max_plots:
                        break
            if not added:
                break

        fig, axes = plt.subplots(len(selected_files), 1, figsize=(14, 3 * len(selected_files)))
        if len(selected_files) == 1:
            axes = [axes]

        for i, path in enumerate(selected_files):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue

            lower_cols = [c.lower() for c in df.columns]
            time_col = None
            for idx, c in enumerate(lower_cols):
                if 'time' in c:
                    time_col = df.columns[idx]
                    break
            value_cols = [c for c in df.columns if c != time_col]

            ax = axes[i]
            # Extract class label from path and annotate on the plot
            try:
                rel_path = os.path.relpath(path, dataset_root)
                class_label = rel_path.split(os.sep)[0] if os.sep in rel_path else rel_path.split('/')[0]
            except Exception:
                class_label = 'Unknown'
            for col in value_cols:
                try:
                    if time_col is not None:
                        ax.plot(df[time_col], df[col], label=col)
                    else:
                        ax.plot(df[col], label=col)
                except Exception:
                    continue
            # Title includes filename and class label
            ax.set_title(f"{os.path.basename(path)} — {class_label}")
            ax.set_xlabel(time_col if time_col else 'Sample Index')
            ax.set_ylabel('Sensor Value')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            # Add a visible class tag inside the axes
            try:
                ax.text(
                    0.98,
                    0.02,
                    class_label,
                    transform=ax.transAxes,
                    ha='right',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(
                        facecolor=self.class_colors.get(class_label, 'lightgray'),
                        alpha=0.25,
                        edgecolor='none'
                    )
                )
            except Exception:
                pass

        plt.suptitle('Time Series (dataset)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def get_available_sensors(self) -> List[str]:
        """Get list of available sensors"""
        sensors = set()
        for feature in self.feature_columns:
            # Extract sensor name from feature name
            parts = feature.split('_')
            if len(parts) >= 2:
                sensors.add(parts[0])
        return list(sensors)
    
    def get_feature_summary(self) -> Dict:
        """Get summary of available features"""
        return {
            'total_features': len(self.feature_columns),
            'total_samples': len(self.df),
            'ok_samples': len(self.ok_data),
            'ko_samples': len(self.ko_data),
            'available_sensors': self.get_available_sensors(),
            'top_features': self.get_top_discriminative_features(n=5)
        }

# Global plotting engine instance
_plotting_engine = None

def get_plotting_engine():
    """Get or create the global plotting engine instance"""
    global _plotting_engine
    if _plotting_engine is None:
        _plotting_engine = PlottingEngine()
    return _plotting_engine

# Legacy functions for backward compatibility
def prepare_example_plot():
    """Create and return a matplotlib figure for the example plot."""
    engine = get_plotting_engine()
    return engine.plot_time_series(engine.get_top_discriminative_features(n=3))

def prepare_line_graph_accelerometer():
    """Create line graph (time series) for accelerometer data"""
    engine = get_plotting_engine()
    acc_features = engine.get_sensor_features('acc')
    if not acc_features:
        acc_features = engine.get_top_discriminative_features(n=3)
    return engine.plot_time_series(acc_features)

def prepare_temperature_histogram():
    """Create histogram comparing temperature distributions"""
    engine = get_plotting_engine()
    temp_features = engine.get_sensor_features('temp')
    if not temp_features:
        temp_features = engine.get_top_discriminative_features(n=3)
    return engine.plot_feature_comparison(temp_features, 'histogram')

def prepare_correlation_matrix():
    """Create correlation matrix of numerical features"""
    engine = get_plotting_engine()
    return engine.plot_correlation_matrix()

def prepare_time_series_analysis():
    """Create time series analysis plots"""
    engine = get_plotting_engine()
    return engine.plot_time_series()

def prepare_frequency_domain_plot():
    """Create frequency domain analysis using FFT"""
    engine = get_plotting_engine()
    return engine.plot_frequency_domain()

def prepare_scatter_plot_features():
    """Create scatter plots showing feature relationships"""
    engine = get_plotting_engine()
    return engine.plot_scatter()

def prepare_sensor_comparison_plot():
    """Create a comparison plot showing different sensors and their statistics"""
    engine = get_plotting_engine()
    return engine.plot_time_series(engine.get_top_discriminative_features(n=6))

def prepare_statistics_summary_plot():
    """Create a summary plot showing overall statistics"""
    engine = get_plotting_engine()
    return engine.plot_time_series(engine.get_top_discriminative_features(n=4))

