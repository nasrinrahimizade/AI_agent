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
    
    def __init__(self, feature_matrix_path: str = "ML/feature_matrix.csv"):
        """Initialize with your feature matrix data"""
        try:
            self.df = pd.read_csv(feature_matrix_path)
            self.prepare_data()
        except FileNotFoundError:
            print(f"Warning: {feature_matrix_path} not found. Using fallback data.")
            self.df = self._create_fallback_data()
            self.prepare_data()
        
    def _create_fallback_data(self):
        """Create fallback data if feature_matrix.csv is not available"""
        # Create sample data with similar structure
        np.random.seed(42)
        n_samples = 50
        
        data = {
            'label': np.random.choice(['OK', 'KO_HIGH_2mm', 'KO_LOW_2mm'], n_samples),
            'sample': [f'Sample_{i}' for i in range(n_samples)]
        }
        
        # Add some sample features
        sensors = ['HTS221_HUM', 'HTS221_TEMP', 'IIS2DH_ACC', 'IIS2MDC_MAG']
        for sensor in sensors:
            for stat in ['mean', 'std', 'max', 'min']:
                data[f'{sensor}_{stat}'] = np.random.normal(0, 1, n_samples)
        
        return pd.DataFrame(data)
        
    def prepare_data(self):
        """Prepare data for plotting - separate OK vs KO classes"""
        # Create binary classification: OK vs all KO types
        self.df['binary_class'] = self.df['label'].apply(
            lambda x: 'OK' if x == 'OK' else 'KO'
        )
        
        # Separate feature columns (exclude label, sample, binary_class)
        self.feature_columns = [col for col in self.df.columns 
                               if col not in ['label', 'sample', 'binary_class']]
        
        # Get OK and KO data
        self.ok_data = self.df[self.df['binary_class'] == 'OK']
        self.ko_data = self.df[self.df['binary_class'] == 'KO']
        
        print(f"✅ Data prepared: {len(self.ok_data)} OK samples, {len(self.ko_data)} KO samples")
        print(f"📊 Features available: {len(self.feature_columns)} features")
    
    def get_sensor_features(self, sensor_name: str) -> List[str]:
        """Get all features for a specific sensor"""
        sensor_features = [col for col in self.feature_columns 
                          if sensor_name.lower() in col.lower()]
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
        
        # Initialize result
        result = {
            'plot_type': 'boxplot',  # default
            'features': [],
            'sensor': None,
            'statistic': None,
            'comparison': True  # OK vs KO comparison by default
        }
        
        # Detect plot types
        if any(word in request for word in ['histogram', 'hist', 'distribution']):
            result['plot_type'] = 'histogram'
        elif any(word in request for word in ['boxplot', 'box', 'compare']):
            result['plot_type'] = 'boxplot'
        elif any(word in request for word in ['correlation', 'corr', 'relationship']):
            result['plot_type'] = 'correlation'
        elif any(word in request for word in ['time series', 'time', 'temporal']):
            result['plot_type'] = 'timeseries'
        elif any(word in request for word in ['frequency', 'fft', 'spectrum']):
            result['plot_type'] = 'frequency'
        elif any(word in request for word in ['scatter', 'scatter plot']):
            result['plot_type'] = 'scatter'
        
        # Detect sensors
        sensors = ['accelerometer', 'gyroscope', 'magnetometer', 'temperature', 
                  'pressure', 'humidity', 'microphone', 'acc', 'gyro', 'mag', 
                  'temp', 'press', 'hum', 'mic']
        
        for sensor in sensors:
            if sensor in request:
                result['sensor'] = sensor
                result['features'] = self.get_sensor_features(sensor)
                break
        
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
        
        return result
    
    def plot_feature_comparison(self, features: List[str], plot_type: str = 'boxplot') -> plt.Figure:
        """Generate comparison plots between OK and KO for specific features"""
        
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
            
            if plot_type == 'boxplot':
                # Boxplot comparison
                data_to_plot = [
                    self.ok_data[feature].dropna(),
                    self.ko_data[feature].dropna()
                ]
                box_plot = ax.boxplot(data_to_plot, labels=['OK', 'KO'], patch_artist=True)
                box_plot['boxes'][0].set_facecolor('lightblue')
                box_plot['boxes'][1].set_facecolor('lightcoral')
                
            elif plot_type == 'histogram':
                # Histogram overlay
                ax.hist(self.ok_data[feature].dropna(), alpha=0.7, label='OK', 
                       bins=20, color='lightblue', density=True)
                ax.hist(self.ko_data[feature].dropna(), alpha=0.7, label='KO', 
                       bins=20, color='lightcoral', density=True)
                ax.legend()
            
            # Clean up feature name for title
            clean_name = feature.replace('_', ' ').title()
            ax.set_title(f'{clean_name}', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(features), 6):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{plot_type.title()} Comparison: OK vs KO Classes', fontsize=16)
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
            
            # Plot OK samples
            ok_indices = self.ok_data.index
            ax.plot(ok_indices, self.ok_data[feature], 'o-', label='OK', alpha=0.7)
            
            # Plot KO samples
            ko_indices = self.ko_data.index
            ax.plot(ko_indices, self.ko_data[feature], 's-', label='KO', alpha=0.7)
            
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
            # If only one feature, use it against sample index
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(range(len(self.df)), self.df[features[0]], 
                      c=self.df['binary_class'].map({'OK': 'blue', 'KO': 'red'}))
            ax.set_title(f'{features[0].replace("_", " ").title()} vs Sample Index')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel(features[0].replace('_', ' ').title())
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
                    ax.scatter(self.df[feat1], self.df[feat2], 
                             c=self.df['binary_class'].map({'OK': 'blue', 'KO': 'red'}),
                             alpha=0.6)
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
        
        # Generate appropriate plot
        if parsed['plot_type'] == 'boxplot':
            return self.plot_feature_comparison(parsed['features'], 'boxplot')
        elif parsed['plot_type'] == 'histogram':
            return self.plot_feature_comparison(parsed['features'], 'histogram')
        elif parsed['plot_type'] == 'correlation':
            return self.plot_correlation_matrix(parsed['features'])
        elif parsed['plot_type'] == 'timeseries':
            return self.plot_time_series(parsed['features'])
        elif parsed['plot_type'] == 'frequency':
            return self.plot_frequency_domain(parsed['features'])
        elif parsed['plot_type'] == 'scatter':
            return self.plot_scatter(parsed['features'])
        else:
            # Default to boxplot
            return self.plot_feature_comparison(parsed['features'], 'boxplot')
    
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
    return engine.plot_feature_comparison(engine.get_top_discriminative_features(n=3), 'boxplot')

def prepare_boxplot_accelerometer():
    """Create boxplot comparing OK vs KO conditions for accelerometer data"""
    engine = get_plotting_engine()
    acc_features = engine.get_sensor_features('acc')
    if not acc_features:
        acc_features = engine.get_top_discriminative_features(n=3)
    return engine.plot_feature_comparison(acc_features, 'boxplot')

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
    return engine.plot_feature_comparison(engine.get_top_discriminative_features(n=6), 'boxplot')

def prepare_statistics_summary_plot():
    """Create a summary plot showing overall statistics"""
    engine = get_plotting_engine()
    return engine.plot_feature_comparison(engine.get_top_discriminative_features(n=4), 'histogram')
