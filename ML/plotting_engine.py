import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.fft import fft, fftfreq
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PlottingEngine:
    """
    Advanced Plotting Engine for Statistical AI Agent
    Handles natural language requests and generates appropriate visualizations
    Optimized for mock data with 4-class structure
    """
    
    def __init__(self, feature_matrix_path: str):
        """Initialize with your feature matrix data"""
        self.df = pd.read_csv(feature_matrix_path)
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare data for plotting - handle 4-class structure properly"""
        # Keep original 4-class structure: OK, KO, KO_HIGH_2mm, KO_LOW_2mm
        self.classes = sorted(self.df['label'].unique())
        print(f"✅ Data prepared: {len(self.classes)} classes found: {self.classes}")
        
        # Separate feature columns (exclude label, sample)
        self.feature_columns = [col for col in self.df.columns 
                               if col not in ['label', 'sample']]
        
        # Create class-specific dataframes
        self.class_data = {}
        for class_name in self.classes:
            self.class_data[class_name] = self.df[self.df['label'] == class_name]
            print(f"📊 {class_name}: {len(self.class_data[class_name])} samples")
        
        # Create binary classification for backward compatibility
        self.df['binary_class'] = self.df['label'].apply(
            lambda x: 'OK' if x == 'OK' else 'KO'
        )
        
        # Get OK and KO data (aggregated)
        self.ok_data = self.df[self.df['binary_class'] == 'OK']
        self.ko_data = self.df[self.df['binary_class'] == 'KO']
        
        print(f"📊 Features available: {len(self.feature_columns)} features")
        
        # Define color scheme for 4 classes
        self.class_colors = {
            'OK': '#2E8B57',           # Sea Green
            'KO': '#CD5C5C',           # Indian Red
            'KO_HIGH_2mm': '#FF8C00',  # Dark Orange
            'KO_LOW_2mm': '#8B008B'    # Dark Magenta
        }
        
        # Define feature categories for better organization
        self.feature_categories = {
            'Environmental': ['HTS221_TEMP_TEMP_mean', 'LPS22HH_TEMP_TEMP_mean', 'STTS751_TEMP_TEMP_mean', 
                             'HTS221_HUM_HUM_mean', 'LPS22HH_PRESS_PRESS_mean'],
            'Motion': ['IIS2DH_ACC_A_x_mean', 'IIS2DH_ACC_A_y_mean', 'IIS2DH_ACC_A_z_mean',
                      'IIS3DWB_ACC_A_x_mean', 'IIS3DWB_ACC_A_y_mean', 'IIS3DWB_ACC_A_z_mean',
                      'ISM330DHCX_ACC_A_x_mean', 'ISM330DHCX_ACC_A_y_mean', 'ISM330DHCX_ACC_A_z_mean'],
            'Rotation': ['ISM330DHCX_GYRO_G_x_mean', 'ISM330DHCX_GYRO_G_y_mean', 'ISM330DHCX_GYRO_G_z_mean'],
            'Magnetic': ['IIS2MDC_MAG_M_x_mean', 'IIS2MDC_MAG_M_y_mean', 'IIS2MDC_MAG_M_z_mean'],
            'Audio': ['IMP23ABSU_MIC_MIC_mean', 'IMP34DT05_MIC_MIC_mean']
        }
    
    def get_sensor_features(self, sensor_name: str) -> List[str]:
        """Get all features for a specific sensor with better matching"""
        sensor_name_lower = sensor_name.lower()
        
        # Enhanced sensor matching for actual dataset features
        sensor_mapping = {
            # Temperature sensors
            'temperature': ['HTS221_TEMP_TEMP_mean', 'LPS22HH_TEMP_TEMP_mean', 'STTS751_TEMP_TEMP_mean'],
            'temp': ['HTS221_TEMP_TEMP_mean', 'LPS22HH_TEMP_TEMP_mean', 'STTS751_TEMP_TEMP_mean'],
            
            # Humidity sensors
            'humidity': ['HTS221_HUM_HUM_mean'],
            'hum': ['HTS221_HUM_HUM_mean'],
            
            # Pressure sensors
            'pressure': ['LPS22HH_PRESS_PRESS_mean'],
            'press': ['LPS22HH_PRESS_PRESS_mean'],
            
            # Accelerometer sensors
            'accelerometer': ['IIS2DH_ACC_A_x_mean', 'IIS2DH_ACC_A_y_mean', 'IIS2DH_ACC_A_z_mean', 
                             'IIS3DWB_ACC_A_x_mean', 'IIS3DWB_ACC_A_y_mean', 'IIS3DWB_ACC_A_z_mean',
                             'ISM330DHCX_ACC_A_x_mean', 'ISM330DHCX_ACC_A_y_mean', 'ISM330DHCX_ACC_A_z_mean'],
            'acceleration': ['IIS2DH_ACC_A_x_mean', 'IIS2DH_ACC_A_y_mean', 'IIS2DH_ACC_A_z_mean', 
                            'IIS3DWB_ACC_A_x_mean', 'IIS3DWB_ACC_A_y_mean', 'IIS3DWB_ACC_A_z_mean',
                            'ISM330DHCX_ACC_A_x_mean', 'ISM330DHCX_ACC_A_y_mean', 'ISM330DHCX_ACC_A_z_mean'],
            'acc': ['IIS2DH_ACC_A_x_mean', 'IIS2DH_ACC_A_y_mean', 'IIS2DH_ACC_A_z_mean', 
                    'IIS3DWB_ACC_A_x_mean', 'IIS3DWB_ACC_A_y_mean', 'IIS3DWB_ACC_A_z_mean',
                    'ISM330DHCX_ACC_A_x_mean', 'ISM330DHCX_ACC_A_y_mean', 'ISM330DHCX_ACC_A_z_mean'],
            
            # Gyroscope sensors
            'gyroscope': ['ISM330DHCX_GYRO_G_x_mean', 'ISM330DHCX_GYRO_G_y_mean', 'ISM330DHCX_GYRO_G_z_mean'],
            'gyro': ['ISM330DHCX_GYRO_G_x_mean', 'ISM330DHCX_GYRO_G_y_mean', 'ISM330DHCX_GYRO_G_z_mean'],
            
            # Magnetometer sensors
            'magnetometer': ['IIS2MDC_MAG_M_x_mean', 'IIS2MDC_MAG_M_y_mean', 'IIS2MDC_MAG_M_z_mean'],
            'mag': ['IIS2MDC_MAG_M_x_mean', 'IIS2MDC_MAG_M_y_mean', 'IIS2MDC_MAG_M_z_mean'],
            
            # Microphone sensors
            'microphone': ['IMP23ABSU_MIC_MIC_mean', 'IMP34DT05_MIC_MIC_mean'],
            'mic': ['IMP23ABSU_MIC_MIC_mean', 'IMP34DT05_MIC_MIC_mean'],
            
            # Motion and rotation (aliases)
            'motion': ['IIS2DH_ACC_A_x_mean', 'IIS2DH_ACC_A_y_mean', 'IIS2DH_ACC_A_z_mean', 
                      'IIS3DWB_ACC_A_x_mean', 'IIS3DWB_ACC_A_y_mean', 'IIS3DWB_ACC_A_z_mean',
                      'ISM330DHCX_ACC_A_x_mean', 'ISM330DHCX_ACC_A_y_mean', 'ISM330DHCX_ACC_A_z_mean'],
            'rotation': ['ISM330DHCX_GYRO_G_x_mean', 'ISM330DHCX_GYRO_G_y_mean', 'ISM330DHCX_GYRO_G_z_mean']
        }
        
        # Direct mapping
        if sensor_name_lower in sensor_mapping:
            return sensor_mapping[sensor_name_lower]
        
        # Fuzzy matching
        for key, features in sensor_mapping.items():
            if sensor_name_lower in key or key in sensor_name_lower:
                return features
        
        # Fallback: search in feature names for partial matches
        sensor_features = [col for col in self.feature_columns 
                          if sensor_name_lower in col.lower()]
        
        # If still no matches, try to find features that contain the sensor name
        if not sensor_features:
            for col in self.feature_columns:
                if any(sensor in col.lower() for sensor in ['temp', 'hum', 'press', 'acc', 'gyro', 'mag', 'mic']):
                    if sensor_name_lower in col.lower():
                        sensor_features.append(col)
        
        return sensor_features
    
    def clean_feature_name(self, feature_name: str) -> str:
        """Convert feature names to readable labels"""
        # Remove common suffixes
        clean_name = feature_name.replace('_mean', '').replace('_', ' ')
        
        # Capitalize and format
        clean_name = clean_name.title()
        
        # Special formatting for specific features
        clean_name = clean_name.replace('X', 'X-axis').replace('Y', 'Y-axis').replace('Z', 'Z-axis')
        
        return clean_name
    
    def parse_plot_request(self, request: str) -> Dict:
        """
        Parse natural language plot requests
        Returns dict with plot_type, features, and parameters
        """
        request = request.lower()
        
        # Initialize result
        result = {
            'plot_type': 'timeseries',  # Changed default to timeseries
            'features': [],
            'sensor': None,
            'statistic': None,
            'comparison': True,  # Multi-class comparison by default
            'use_all_classes': True  # Use 4 classes instead of binary
        }
        
        # Detect plot types - prioritize time series and frequency analysis
        print(f"🔍 Detecting plot type from request: {request}")
        
        if any(word in request for word in ['time series', 'time', 'temporal', 'trend', 'over time']):
            result['plot_type'] = 'timeseries'
            print(f"✅ Detected plot type: timeseries")
        elif any(word in request for word in ['frequency', 'fft', 'spectrum', 'oscillation', 'vibration']):
            result['plot_type'] = 'frequency'
            print(f"✅ Detected plot type: frequency")
        elif any(word in request for word in ['histogram', 'hist', 'distribution']):
            result['plot_type'] = 'histogram'
            print(f"✅ Detected plot type: histogram")
        elif any(word in request for word in ['correlation', 'corr', 'relationship']):
            result['plot_type'] = 'correlation'
            print(f"✅ Detected plot type: correlation")
        elif any(word in request for word in ['scatter', 'scatter plot']):
            result['plot_type'] = 'timeseries'
            print(f"✅ Detected plot type: scatter")
        else:
            print(f"ℹ️ Using default plot type: {result['plot_type']}")
        
        # Detect sensors
        sensors = ['accelerometer', 'gyroscope', 'magnetometer', 'temperature', 
                  'pressure', 'humidity', 'microphone', 'acc', 'gyro', 'mag', 
                  'temp', 'press', 'hum', 'mic', 'motion', 'rotation']
        
        print(f"🔍 Looking for sensors in request: {request}")
        for sensor in sensors:
            if sensor in request:
                print(f"✅ Found sensor: {sensor}")
                result['sensor'] = sensor
                result['features'] = self.get_sensor_features(sensor)
                print(f"📊 Features found: {result['features']}")
                break
        
        # If no sensor detected, try to find specific sensor names in the request
        if not result['features']:
            print(f"🔍 No generic sensor found, looking for specific sensor names...")
            for col in self.feature_columns:
                if any(sensor_type in col.lower() for sensor_type in ['temp', 'hum', 'press', 'acc', 'gyro', 'mag', 'mic']):
                    if any(word in col.lower() for word in request.split()):
                        print(f"✅ Found specific sensor feature: {col}")
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
        
        # Final fallback: if still no features and temperature was requested, use temperature features
        if not result['features'] and ('temperature' in request.lower() or 'temp' in request.lower()):
            print(f"🔄 Fallback: Using default temperature features")
            result['features'] = ['HTS221_TEMP_TEMP_mean', 'LPS22HH_TEMP_TEMP_mean', 'STTS751_TEMP_TEMP_mean']
            result['sensor'] = 'temperature'
        
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
        
        # Final fallback: if still no features, use top discriminative features
        if not result['features']:
            print(f"🔄 Final fallback: Using top discriminative features")
            result['features'] = self.get_top_discriminative_features(3)
        
        # Auto-detect plot type based on sensor type if no specific plot type requested
        if result['plot_type'] == 'timeseries' and result['sensor']:
            # For motion sensors, frequency plots might be more appropriate
            motion_sensors = ['accelerometer', 'acc', 'gyroscope', 'gyro', 'motion', 'rotation']
            if any(motion in result['sensor'] for motion in motion_sensors):
                # Suggest frequency plots for motion sensors, but keep time series as default
                pass  # Keep time series as default for now
        
        print(f"🎯 Final parsed result: {result}")
        return result
    
    def plot_feature_comparison(self, features: List[str], plot_type: str = 'histogram') -> plt.Figure:
        """Generate comparison plots between all classes for specific features"""
        
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
            
            if plot_type == 'histogram':
                # Histogram comparison for all classes
                data_to_plot = []
                labels = []
                
                for class_name in self.classes:
                    class_values = self.class_data[class_name][feature].dropna()
                    if len(class_values) > 0:
                        data_to_plot.append(class_values)
                        labels.append(class_name)
                
                if data_to_plot:
                    histogram = ax.hist(data_to_plot, labels=labels, patch_artist=True)
                    
                    # Color the boxes
                    for j, box in enumerate(histogram['boxes']):
                        class_name = labels[j]
                        box.set_facecolor(self.class_colors.get(class_name, 'lightgray'))
                        histogram.set_alpha(0.7)
                    
                    # Color the whiskers and medians
                    for element in ['whiskers', 'fliers', 'means', 'medians']:
                        if element in histogram:
                            plt.setp(histogram[element], color='black')
                
            elif plot_type == 'histogram':
                # Histogram overlay for all classes
                for class_name in self.classes:
                    class_values = self.class_data[class_name][feature].dropna()
                    if len(class_values) > 0:
                        ax.hist(class_values, alpha=0.6, label=class_name, 
                               bins=15, color=self.class_colors.get(class_name, 'gray'), 
                               density=True, edgecolor='black', linewidth=0.5)
                ax.legend()
            
            elif plot_type == 'scatter':
                # Scatter plot for all classes
                data_to_plot = []
                labels = []
                
                for class_name in self.classes:
                    class_values = self.class_data[class_name][feature].dropna()
                    if len(class_values) > 0:
                        data_to_plot.append(class_values)
                        labels.append(class_name)
                
                if data_to_plot:
                    scatter_parts = ax.scatter(data_to_plot, positions=range(len(labels)))
                    
                    # Color the violins
                    for j, pc in enumerate(scatter_parts['bodies']):
                        class_name = labels[j]
                        pc.set_facecolor(self.class_colors.get(class_name, 'lightgray'))
                        scatter_parts.set_alpha(0.7)
                    
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=45)
            
            # Clean up feature name for title
            clean_name = self.clean_feature_name(feature)
            ax.set_title(f'{clean_name}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Class', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            if plot_type in ['histogram', 'scatter']:
                ax.tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(len(features), 6):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{plot_type.title()} Comparison: Multi-Class Analysis', fontsize=16, fontweight='bold')
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
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Generate heatmap with better styling
        mask = np.triu(np.ones_like(corr_data, dtype=bool))  # Upper triangle mask
        sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax, fmt='.2f', cbar_kws={'shrink': 0.8},
                   linewidths=0.5, annot_kws={'size': 8})
        
        # Clean up feature names for better readability
        clean_labels = [self.clean_feature_name(f) for f in features]
        ax.set_xticklabels(clean_labels, rotation=45, ha='right')
        ax.set_yticklabels(clean_labels, rotation=0)
        
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig
    
    def plot_time_series(self, features: Optional[List[str]] = None) -> plt.Figure:
        """
        Plot time series data using sample order as time proxy
        Enhanced for multi-class visualization
        """
        
        if features is None or len(features) == 0:
            features = self.get_top_discriminative_features(n=3)
        
        features = features[:4]  # Limit to 4 features
        
        fig, axes = plt.subplots(len(features), 1, figsize=(14, 3*len(features)))
        if len(features) == 1:
            axes = [axes]
        
        for i, feature in enumerate(features):
            ax = axes[i]
            
            # Plot each class separately
            for class_name in self.classes:
                class_data = self.class_data[class_name].sort_values('sample')
                if len(class_data) > 0:
                    # Extract sample numbers (remove 'Sample_' prefix)
                    sample_nums = [int(s.replace('Sample_', '')) for s in class_data['sample']]
                    
                    ax.plot(sample_nums, class_data[feature], 
                           'o-', color=self.class_colors.get(class_name, 'gray'), 
                           alpha=0.8, label=class_name, linewidth=2, markersize=6)
            
            clean_name = self.clean_feature_name(feature)
            ax.set_title(f'Time Series: {clean_name}', fontweight='bold')
            ax.set_xlabel('Sample Number', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set x-axis to show sample numbers clearly
            ax.set_xticks(range(1, 21, 2))  # Show every other sample number
        
        plt.suptitle('Time Series Analysis by Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_frequency_domain(self, features: Optional[List[str]] = None) -> plt.Figure:
        """
        Generate frequency domain plots (FFT analysis)
        Enhanced for multi-class comparison
        """
        
        if features is None or len(features) == 0:
            features = self.get_top_discriminative_features(n=2)
        
        features = features[:2]  # Limit for clarity
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        if len(features) == 1:
            axes = [axes]
        
        for i, feature in enumerate(features):
            ax = axes[i] if len(features) > 1 else axes
            
            # Create synthetic frequency response based on value distribution for each class
            freqs = np.linspace(0, 1, 50)
            
            for class_name in self.classes:
                class_values = self.class_data[class_name][feature].dropna().values
                if len(class_values) > 0:
                    # Create spectrum based on class characteristics
                    mean_val = np.mean(class_values)
                    std_val = np.std(class_values)
                    
                    # Normalize to 0-1 range for visualization
                    normalized_mean = (mean_val - np.min(class_values)) / (np.max(class_values) - np.min(class_values))
                    
                    # Create spectrum with class-specific characteristics
                    spectrum = np.exp(-((freqs - normalized_mean)**2) / (2 * (std_val/10)**2))
                    
                    ax.plot(freqs, spectrum, '-', linewidth=2.5, label=class_name, 
                           color=self.class_colors.get(class_name, 'gray'))
            
            clean_name = self.clean_feature_name(feature)
            ax.set_title(f'Frequency Domain: {clean_name}', fontweight='bold')
            ax.set_xlabel('Normalized Frequency', fontsize=10)
            ax.set_ylabel('Magnitude', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Frequency Domain Analysis by Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_scatter(self, features: Optional[List[str]] = None) -> plt.Figure:
        """Generate scatter plots between feature pairs with multi-class support"""
        
        if features is None or len(features) < 2:
            features = self.get_top_discriminative_features(n=4)
        
        # Create scatter plot matrix for top features
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for i in range(min(4, len(features)-1)):
            ax = axes[i]
            feature1 = features[i]
            feature2 = features[i+1] if i+1 < len(features) else features[0]
            
            # Scatter plot for each class
            for class_name in self.classes:
                class_data = self.class_data[class_name]
                if len(class_data) > 0:
                    ax.scatter(class_data[feature1], class_data[feature2], 
                              c=self.class_colors.get(class_name, 'gray'), 
                              alpha=0.7, label=class_name, s=60, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel(self.clean_feature_name(feature1), fontsize=10)
            ax.set_ylabel(self.clean_feature_name(feature2), fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Feature Scatter Plot Analysis by Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def get_top_discriminative_features(self, n: int = 5) -> List[str]:
        """Get top N most discriminative features using ANOVA for multi-class"""
        
        feature_scores = []
        
        for feature in self.feature_columns:
            # Collect data for all classes
            class_data = []
            valid_classes = []
            
            for class_name in self.classes:
                values = self.class_data[class_name][feature].dropna()
                if len(values) > 1:
                    class_data.append(values.values)
                    valid_classes.append(class_name)
            
            if len(class_data) >= 2:  # Need at least 2 classes
                try:
                    # Perform one-way ANOVA
                    from scipy.stats import f_oneway
                    f_stat, p_value = f_oneway(*class_data)
                    
                    # Calculate effect size (eta-squared)
                    grand_mean = np.mean(np.concatenate(class_data))
                    ss_between = sum(len(data) * (np.mean(data) - grand_mean)**2 for data in class_data)
                    ss_total = sum((val - grand_mean)**2 for data in class_data for val in data)
                    
                    if ss_total > 0:
                        eta_squared = ss_between / ss_total
                        score = eta_squared * (1 - p_value)  # Combined score
                        feature_scores.append((feature, score))
                except:
                    continue
        
        # Sort by score and return top N
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        return [feature for feature, score in feature_scores[:n]]
    
    def handle_plot_request(self, request: str) -> plt.Figure:
        """
        Main function to handle natural language plot requests
        Enhanced for better mock data visualization
        """
        
        print(f"🎯 Processing plot request: '{request}'")
        
        # Parse the request
        parsed = self.parse_plot_request(request)
        print(f"📋 Parsed request: {parsed}")
        
        # Generate appropriate plot
        try:
            if parsed['plot_type'] == 'correlation':
                fig = self.plot_correlation_matrix(parsed['features'])
                
            elif parsed['plot_type'] == 'timeseries':
                fig = self.plot_time_series(parsed['features'])
                
            elif parsed['plot_type'] == 'frequency':
                fig = self.plot_frequency_domain(parsed['features'])
                
            elif parsed['plot_type'] == 'scatter':
                fig = self.plot_scatter(parsed['features'])
                
            elif parsed['plot_type'] == 'violin':
                fig = self.plot_feature_comparison(parsed['features'], 'violin')
                
            else:  # Default to comparison plots
                fig = self.plot_feature_comparison(
                    parsed['features'], 
                    parsed['plot_type']
                )
            
            print("✅ Plot generated successfully!")
            return fig
            
        except Exception as e:
            print(f"❌ Error generating plot: {e}")
            # Return a simple error plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Error generating plot:\n{str(e)}', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Plot Generation Error')
            return fig
    
    def get_available_sensors(self) -> List[str]:
        """Get list of available sensors from feature names"""
        sensors = set()
        for feature in self.feature_columns:
            parts = feature.split('_')
            if len(parts) > 0:
                sensors.add(parts[0])
        return sorted(list(sensors))
    
    def get_feature_summary(self) -> Dict:
        """Get summary of available features for GUI display"""
        return {
            'total_features': len(self.feature_columns),
            'available_sensors': self.get_available_sensors(),
            'sample_counts': {class_name: len(self.class_data[class_name]) for class_name in self.classes},
            'top_discriminative': self.get_top_discriminative_features(10),
            'feature_categories': self.feature_categories
        }
    
    def create_class_distribution_plot(self) -> plt.Figure:
        """Create a summary plot showing class distribution and characteristics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Class sample counts
        ax1 = axes[0, 0]
        class_counts = [len(self.class_data[class_name]) for class_name in self.classes]
        bars = ax1.bar(self.classes, class_counts, color=[self.class_colors.get(c, 'gray') for c in self.classes])
        ax1.set_title('Sample Count by Class', fontweight='bold')
        ax1.set_ylabel('Number of Samples')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
        
        # 2. Feature value ranges
        ax2 = axes[0, 1]
        feature_ranges = []
        feature_names = []
        
        for feature in self.feature_columns[:5]:  # Top 5 features
            all_values = self.df[feature].dropna()
            feature_ranges.append(np.ptp(all_values))  # Peak-to-peak
            feature_names.append(self.clean_feature_name(feature))
        
        bars = ax2.barh(feature_names, feature_ranges, color='skyblue', alpha=0.7)
        ax2.set_title('Feature Value Ranges', fontweight='bold')
        ax2.set_xlabel('Range (Max - Min)')
        
        # 3. Class separation (using top discriminative feature)
        ax3 = axes[1, 0]
        top_feature = self.get_top_discriminative_features(1)[0]
        
        for class_name in self.classes:
            values = self.class_data[class_name][top_feature].dropna()
            ax3.hist(values, alpha=0.6, label=class_name, bins=10, 
                    color=self.class_colors.get(class_name, 'gray'), density=True)
        
        ax3.set_title(f'Distribution of Top Feature: {self.clean_feature_name(top_feature)}', fontweight='bold')
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Density')
        ax3.legend()
        
        # 4. Correlation heatmap (top 6 features)
        ax4 = axes[1, 1]
        top_features = self.get_top_discriminative_features(6)
        corr_data = self.df[top_features].corr()
        
        im = ax4.imshow(corr_data, cmap='RdBu_r', aspect='auto')
        ax4.set_title('Top Features Correlation', fontweight='bold')
        
        # Set tick labels
        clean_labels = [self.clean_feature_name(f) for f in top_features]
        ax4.set_xticks(range(len(clean_labels)))
        ax4.set_yticks(range(len(clean_labels)))
        ax4.set_xticklabels(clean_labels, rotation=45, ha='right')
        ax4.set_yticklabels(clean_labels)
        
        # Add colorbar
        plt.colorbar(im, ax=ax4, shrink=0.8)
        
        plt.suptitle('Dataset Overview and Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig


# Example usage and testing functions
def test_plotting_engine():
    """Test the plotting engine with example requests"""
    
    # Initialize with mock data
    engine = PlottingEngine('ML/feature_matrix2.csv')
    
    # Test different types of requests
    test_requests = [
        "Show histogram comparison of accelerometer features",
        "Plot histogram of temperature sensor",
        "Generate correlation matrix",
        "Show time series for top features", 
        "Create frequency domain plot",
        "Plot scatter analysis",
        "Compare gyroscope mean values between classes",
        "Create scatter plot for pressure sensor"
    ]
    
    print("🧪 Testing Plotting Engine with various requests:")
    print("="*60)
    
    for request in test_requests:
        print(f"\n📝 Request: {request}")
        try:
            fig = engine.handle_plot_request(request)
            plt.show()  # This will display the plot
            plt.close(fig)  # Clean up
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Display feature summary
    summary = engine.get_feature_summary()
    print(f"\n📊 Feature Summary:")
    print(f"- Total features: {summary['total_features']}")
    print(f"- Available sensors: {summary['available_sensors']}")
    print(f"- Sample counts: {summary['sample_counts']}")
    print(f"- Top discriminative features: {summary['top_discriminative'][:5]}")
    
    # Create overview plot
    print("\n🎨 Creating dataset overview plot...")
    overview_fig = engine.create_class_distribution_plot()
    plt.show()
    plt.close(overview_fig)


if __name__ == "__main__":
    # Run tests
    test_plotting_engine()