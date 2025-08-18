import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
            'comparison': True  # OK vs KO comparison by default
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
                # Overlay per class
                for class_name in self.classes:
                    class_values = self.class_data[class_name][feature].dropna()
                    if len(class_values) == 0:
                        continue
                    ax.hist(
                        class_values,
                        alpha=0.6,
                        label=class_name,
                        bins=20,
                        color=self.class_colors.get(class_name, 'gray'),
                        density=True
                    )
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
    
    def plot_frequency_domain(self, features: Optional[List[str]] = None) -> plt.Figure:
        """Generate frequency domain analysis using FFT"""
        
        if features is None or len(features) == 0:
            features = self.get_top_discriminative_features(n=4)
        
        features = features[:4]  # Limit to 4 features
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(features):
            if i >= 4:
                break
                
            ax = axes[i]
            
            # Get data for FFT
            all_data = self.df[feature].dropna().values
            
            if len(all_data) > 1:
                # Perform FFT
                fft_result = fft(all_data)
                freqs = fftfreq(len(all_data))
                
                # Plot magnitude spectrum
                magnitude = np.abs(fft_result)
                ax.plot(freqs[:len(freqs)//2], magnitude[:len(freqs)//2])
                ax.set_title(f'{feature.replace("_", " ").title()} Frequency Spectrum')
                ax.set_xlabel('Frequency')
                ax.set_ylabel('Magnitude')
                ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(features), 4):
            axes[i].set_visible(False)
        
        plt.suptitle('Frequency Domain Analysis', fontsize=16)
        plt.tight_layout()
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
            return self.plot_time_series(parsed['features'])
        elif parsed['plot_type'] == 'frequency':
            print(f"📡 Generating frequency domain plot")
            return self.plot_frequency_domain(parsed['features'])
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
            return self.plot_time_series(parsed['features'])
    
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
