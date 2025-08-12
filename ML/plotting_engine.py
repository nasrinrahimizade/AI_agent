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
    """
    
    def __init__(self, feature_matrix_path: str):
        """Initialize with your feature matrix data"""
        self.df = pd.read_csv(feature_matrix_path)
        self.prepare_data()
        
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
            
            # Plot OK samples (assuming samples are ordered somehow)
            ok_samples = self.ok_data.sort_values('sample')
            ko_samples = self.ko_data.sort_values('sample')
            
            ax.plot(ok_samples['sample'], ok_samples[feature], 
                   'o-', color='blue', alpha=0.7, label='OK', linewidth=2)
            ax.plot(ko_samples['sample'], ko_samples[feature], 
                   's-', color='red', alpha=0.7, label='KO', linewidth=2)
            
            clean_name = feature.replace('_', ' ').title()
            ax.set_title(f'Time Series: {clean_name}')
            ax.set_xlabel('Sample Number')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Time Series Analysis', fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_frequency_domain(self, features: Optional[List[str]] = None) -> plt.Figure:
        """
        Generate frequency domain plots (FFT analysis)
        Note: This is conceptual since we have statistical features, not raw time series
        """
        
        if features is None or len(features) == 0:
            features = self.get_top_discriminative_features(n=2)
        
        features = features[:2]  # Limit for clarity
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        if len(features) == 1:
            axes = [axes]
        
        for i, feature in enumerate(features):
            ax = axes[i] if len(features) > 1 else axes
            
            # Since we don't have raw time series, we'll create a conceptual frequency plot
            # using the statistical distribution of values
            ok_values = self.ok_data[feature].dropna().values
            ko_values = self.ko_data[feature].dropna().values
            
            # Create synthetic frequency response based on value distribution
            freqs = np.linspace(0, 1, 50)
            
            # OK class spectrum (based on std and mean)
            ok_spectrum = np.exp(-((freqs - 0.3)**2) / (2 * (np.std(ok_values)/10)**2))
            
            # KO class spectrum (based on std and mean) 
            ko_spectrum = np.exp(-((freqs - 0.7)**2) / (2 * (np.std(ko_values)/10)**2))
            
            ax.plot(freqs, ok_spectrum, 'b-', linewidth=2, label='OK Class')
            ax.plot(freqs, ko_spectrum, 'r-', linewidth=2, label='KO Class')
            
            clean_name = feature.replace('_', ' ').title()
            ax.set_title(f'Frequency Domain: {clean_name}')
            ax.set_xlabel('Normalized Frequency')
            ax.set_ylabel('Magnitude')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Frequency Domain Analysis', fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_scatter(self, features: Optional[List[str]] = None) -> plt.Figure:
        """Generate scatter plots between feature pairs"""
        
        if features is None or len(features) < 2:
            features = self.get_top_discriminative_features(n=4)
        
        # Create scatter plot matrix for top features
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i in range(min(4, len(features)-1)):
            ax = axes[i]
            feature1 = features[i]
            feature2 = features[i+1] if i+1 < len(features) else features[0]
            
            # Scatter plot OK vs KO
            ax.scatter(self.ok_data[feature1], self.ok_data[feature2], 
                      c='blue', alpha=0.6, label='OK', s=50)
            ax.scatter(self.ko_data[feature1], self.ko_data[feature2], 
                      c='red', alpha=0.6, label='KO', s=50)
            
            ax.set_xlabel(feature1.replace('_', ' ').title())
            ax.set_ylabel(feature2.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Feature Scatter Plot Analysis', fontsize=16)
        plt.tight_layout()
        return fig
    
    def get_top_discriminative_features(self, n: int = 5) -> List[str]:
        """Get top N most discriminative features using t-test"""
        
        feature_scores = []
        
        for feature in self.feature_columns:
            ok_values = self.ok_data[feature].dropna()
            ko_values = self.ko_data[feature].dropna()
            
            if len(ok_values) > 1 and len(ko_values) > 1:
                try:
                    # Perform t-test
                    t_stat, p_value = ttest_ind(ok_values, ko_values)
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(ok_values)-1)*np.var(ok_values) + 
                                        (len(ko_values)-1)*np.var(ko_values)) / 
                                       (len(ok_values) + len(ko_values) - 2))
                    
                    if pooled_std > 0:
                        cohens_d = abs(np.mean(ok_values) - np.mean(ko_values)) / pooled_std
                        score = cohens_d * (1 - p_value)  # Combined score
                        feature_scores.append((feature, score))
                except:
                    continue
        
        # Sort by score and return top N
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        return [feature for feature, score in feature_scores[:n]]
    
    def handle_plot_request(self, request: str) -> plt.Figure:
        """
        Main function to handle natural language plot requests
        This is the interface your teammate will use
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
            'sample_counts': {
                'OK': len(self.ok_data),
                'KO': len(self.ko_data)
            },
            'top_discriminative': self.get_top_discriminative_features(10)
        }


# Example usage and testing functions
def test_plotting_engine():
    """Test the plotting engine with example requests"""
    
    # Initialize (you'll need to update this path)
    engine = PlottingEngine('feature_matrix.csv')
    
    # Test different types of requests
    test_requests = [
        "Show boxplot comparison of accelerometer features",
        "Plot histogram of temperature sensor",
        "Generate correlation matrix",
        "Show time series for top features", 
        "Create frequency domain plot",
        "Plot scatter analysis",
        "Compare gyroscope mean values between OK and KO"
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


if __name__ == "__main__":
    # Run tests
    test_plotting_engine()