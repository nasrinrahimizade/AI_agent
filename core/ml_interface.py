"""
ML Interface Wrapper Module

This module provides a clean API interface to the ML layer, abstracting away
the complexity of the underlying statistical and plotting engines.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

try:
    from ML.ai_agent_backend import StatisticalAIAgent
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML layer not available, using stub functions")

class MLInterface:
    def __init__(self, feature_matrix_path: str = "core/mock_data.csv"):
        self.feature_matrix_path = feature_matrix_path
        self.ml_agent = None
        self.initialized = False
        self.mock_data = None
        
        if ML_AVAILABLE:
            try:
                self.ml_agent = StatisticalAIAgent(feature_matrix_path)
                self.initialized = True
                logging.info("ML Interface initialized with StatisticalAIAgent")
            except Exception as e:
                logging.error(f"Failed to initialize ML agent: {e}")
                self.initialized = False
        
        # Load mock data for stub functions
        try:
            self.mock_data = pd.read_csv(self.feature_matrix_path)
            logging.info(f"Mock data loaded from {self.feature_matrix_path}")
        except Exception as e:
            logging.error(f"Failed to load mock data: {e}")
            self.mock_data = None
        
        self.available_sensors = ['temperature', 'humidity', 'acceleration_x', 'acceleration_y', 
                                'acceleration_z', 'pressure', 'gyroscope_x', 'gyroscope_y', 
                                'gyroscope_z', 'microphone']
        self.available_classes = ['OK', 'KO', 'KO_HIGH_2mm', 'KO_LOW_2mm']
        self.statistical_measures = ['mean', 'median', 'mode', 'std', 'variance', 'min', 'max', 
                                   'range', 'iqr', 'skewness', 'kurtosis', 'count', 'sum']

    def get_statistic(self, stat: str, column: str, filters: Optional[Dict] = None,
                       group_by: Optional[str] = None) -> Dict[str, Any]:
        """Get statistical measure for a specific column with optional filtering and grouping"""
        try:
            # Map sensor names to actual column names
            target_column = self._map_sensor_to_column(column)
            
            if target_column not in self.mock_data.columns:
                return {
                    'status': 'error',
                    'message': f'Column {column} not found in data',
                    'data_source': 'Mock data',
                    'data_quality': 'invalid_column'
                }
            
            # Apply filters if specified
            data = self.mock_data.copy()
            if filters and 'class' in filters:
                data = data[data['label'].isin(filters['class'])]
            
            # Validate data quality before analysis
            validation = self._validate_data_quality(data, f"statistic calculation for {stat}")
            if not validation['is_valid']:
                return {
                    'status': 'error',
                    'message': f'Data quality issues prevent accurate {stat} calculation: {", ".join(validation["errors"])}',
                    'data_source': 'Mock data',
                    'data_quality': 'poor',
                    'validation_details': validation
                }
            
            # Check if we have enough data for the requested statistic
            if len(data) < 2:
                return {
                    'status': 'error',
                    'message': f'Insufficient data for {stat} calculation. Need at least 2 samples, got {len(data)}',
                    'data_source': 'Mock data',
                    'data_quality': 'insufficient_samples'
                }
            
            # Get the target column data
            column_data = data[target_column]
            
            # Calculate the requested statistic
            if stat.lower() in ['mean', 'average', 'avg']:
                result = column_data.mean()
            elif stat.lower() in ['median', 'med']:
                result = column_data.median()
            elif stat.lower() in ['std', 'standard deviation', 'standard_deviation']:
                result = column_data.std()
            elif stat.lower() in ['variance', 'var']:
                result = column_data.var()
            elif stat.lower() in ['min', 'minimum']:
                result = column_data.min()
            elif stat.lower() in ['max', 'maximum']:
                result = column_data.max()
            elif stat.lower() in ['count', 'n']:
                result = len(column_data)
            else:
                return {
                    'status': 'error',
                    'message': f'Unsupported statistic: {stat}',
                    'data_source': 'Mock data',
                    'data_quality': 'unsupported_statistic'
                }
            
            # Handle grouping if requested
            if group_by == 'class':
                grouped_result = {}
                for class_name in data['label'].unique():
                    class_data = data[data['label'] == class_name][target_column]
                    if stat.lower() in ['mean', 'average', 'avg']:
                        grouped_result[class_name] = class_data.mean()
                    elif stat.lower() in ['median', 'med']:
                        grouped_result[class_name] = class_data.median()
                    elif stat.lower() in ['std', 'standard deviation', 'standard_deviation']:
                        grouped_result[class_name] = class_data.std()
                    elif stat.lower() in ['variance', 'var']:
                        grouped_result[class_name] = class_data.var()
                    elif stat.lower() in ['min', 'minimum']:
                        grouped_result[class_name] = class_data.min()
                    elif stat.lower() in ['max', 'maximum']:
                        grouped_result[class_name] = class_data.max()
                    elif stat.lower() in ['count', 'n']:
                        grouped_result[class_name] = len(class_data)
                
                result = grouped_result
            
            # Add data quality information to response
            response = {
                'status': 'success',
                'result': result,
                'statistic': stat,
                'column': target_column,
                'filters': filters,
                'grouping': group_by,
                'sample_count': len(data),
                'data_source': 'Mock data analysis',
                'data_quality': 'good',
                'validation_details': validation,
                'confidence': 'high' if len(data) >= 10 else 'medium'
            }
            
            # Add warnings if any
            if validation['warnings']:
                response['warnings'] = validation['warnings']
            
            return response
            
        except Exception as e:
            logging.error(f"Error calculating {stat} for {column}: {e}")
            return {
                'status': 'error',
                'message': f'Error calculating {stat}: {str(e)}',
                'data_source': 'Mock data',
                'data_quality': 'calculation_error'
            }
    
    def get_top_features(self, n: int = 3, classes: List[str] = None) -> Dict[str, Any]:
        """Get the top N most discriminative features between classes"""
        try:
            if classes is None:
                classes = ['OK', 'KO']
            
            # Filter data for the specified classes
            filtered_data = self.mock_data[self.mock_data['label'].isin(classes)]
            
            # Validate data quality before analysis
            validation = self._validate_data_quality(filtered_data, f"top {n} features analysis")
            if not validation['is_valid']:
                return {
                    'status': 'error',
                    'message': f'Data quality issues prevent accurate feature analysis: {", ".join(validation["errors"])}',
                    'data_source': 'Mock data',
                    'data_quality': 'poor',
                    'validation_details': validation
                }
            
            if len(filtered_data) == 0:
                return {
                    'status': 'error',
                    'message': f'No data found for classes: {classes}',
                    'data_source': 'Mock data',
                    'data_quality': 'no_data'
                }
            
            # Check if we have enough samples for reliable analysis
            if len(filtered_data) < 5:
                return {
                    'status': 'error',
                    'message': f'Insufficient data for reliable feature analysis. Need at least 5 samples, got {len(filtered_data)}',
                    'data_source': 'Mock data',
                    'data_quality': 'insufficient_samples'
                }
            
            # Get feature columns (exclude label and sample)
            feature_columns = [col for col in self.mock_data.columns if col not in ['label', 'sample']]
            
            if len(feature_columns) == 0:
                return {
                    'status': 'error',
                    'message': 'No feature columns found for analysis',
                    'data_source': 'Mock data',
                    'data_quality': 'no_features'
                }
            
            # Calculate F-statistics for each feature
            feature_scores = {}
            feature_details = {}
            
            for feature in feature_columns:
                try:
                    # Get data for this feature
                    feature_data = filtered_data[feature].dropna()
                    
                    # Skip if not enough data
                    if len(feature_data) < 3:
                        continue
                    
                    # Calculate F-statistic using ANOVA
                    from scipy import stats
                    
                    class_data = []
                    for class_name in classes:
                        class_values = filtered_data[filtered_data['label'] == class_name][feature].dropna()
                        if len(class_values) > 0:
                            class_data.append(class_values)
                    
                    if len(class_data) < 2:
                        continue
                    
                    # Perform ANOVA
                    f_stat, p_value = stats.f_oneway(*class_data)
                    
                    # Calculate class means for context
                    class_means = {}
                    for i, class_name in enumerate(classes):
                        if i < len(class_data):
                            class_means[class_name] = class_data[i].mean()
                    
                    feature_scores[feature] = f_stat
                    feature_details[feature] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'class_means': class_means,
                        'sample_count': len(feature_data)
                    }
                    
                except Exception as e:
                    logging.warning(f"Error calculating F-statistic for {feature}: {e}")
                    continue
            
            if not feature_scores:
                return {
                    'status': 'error',
                    'message': 'Unable to calculate feature scores due to data issues',
                    'data_source': 'Mock data',
                    'data_quality': 'calculation_failed'
                }
            
            # Sort features by F-statistic (higher = more discriminative)
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get top N features
            top_features = [feature for feature, score in sorted_features[:n]]
            
            # Add data quality information to response
            response = {
                'status': 'success',
                'top_features': top_features,
                'feature_details': feature_details,
                'classes': classes,
                'sample_count': len(filtered_data),
                'data_source': 'Mock data analysis',
                'data_quality': 'good',
                'validation_details': validation,
                'confidence': 'high' if len(filtered_data) >= 10 else 'medium',
                'analysis_method': 'ANOVA F-test'
            }
            
            # Add warnings if any
            if validation['warnings']:
                response['warnings'] = validation['warnings']
            
            return response
            
        except Exception as e:
            logging.error(f"Error in top features analysis: {e}")
            return {
                'status': 'error',
                'message': f'Error in feature analysis: {str(e)}',
                'data_source': 'Mock data',
                'data_quality': 'analysis_error'
            }
    
    def _map_sensor_to_column(self, sensor_name: str) -> str:
        """Map sensor names to actual column names in the dataset."""
        sensor_mapping = {
            'HTS221_TEMP': 'temperature_mean',
            'HTS221_HUM': 'humidity_mean',
            'LPS22HH_PRESS': 'pressure_mean',
            'IIS3DWB_ACC': 'acceleration_x_mean',  # Default to X-axis
            'IIS3DWB_ACC_X': 'acceleration_x_mean',
            'IIS3DWB_ACC_Y': 'acceleration_y_mean',
            'IIS3DWB_ACC_Z': 'acceleration_z_mean'
        }
        
        return sensor_mapping.get(sensor_name, sensor_name)
    
    def _get_stub_statistic(self, stat: str, column: str, filters: Optional[Dict] = None, 
                           group_by: Optional[str] = None) -> Dict[str, Any]:
        """Stub function that returns realistic mock data based on the CSV."""
        if self.mock_data is None:
            return self._get_fallback_stub_statistic(stat, column, filters, group_by)
        
        try:
            # Map sensor names to actual columns
            actual_column = self._map_sensor_to_column(column)
            
            # Apply filters if specified
            data = self.mock_data.copy()
            if filters and 'class' in filters:
                data = data[data['label'].isin(filters['class'])]
            
            if actual_column not in data.columns:
                return {
                    'status': 'error',
                    'message': f'Column {column} (mapped to {actual_column}) not found in dataset',
                    'available_columns': list(data.columns)
                }
            
            # Calculate statistic
            if stat == 'mean':
                result = data[actual_column].mean()
            elif stat == 'median':
                result = data[actual_column].median()
            elif stat == 'std':
                result = data[actual_column].std()
            elif stat == 'variance':
                result = data[actual_column].var()
            elif stat == 'min':
                result = data[actual_column].min()
            elif stat == 'max':
                result = data[actual_column].max()
            elif stat == 'count':
                result = data[actual_column].count()
            else:
                result = data[actual_column].mean()  # fallback
            
            # Group by class if requested
            if group_by == 'class':
                grouped_result = {}
                for class_label in self.available_classes:
                    class_data = data[data['label'] == class_label]
                    if not class_data.empty:
                        if stat == 'mean':
                            grouped_result[class_label] = class_data[actual_column].mean()
                        elif stat == 'median':
                            grouped_result[class_label] = class_data[actual_column].median()
                        elif stat == 'std':
                            grouped_result[class_label] = class_data[actual_column].std()
                        elif stat == 'variance':
                            grouped_result[class_label] = class_data[actual_column].var()
                        elif stat == 'count':
                            grouped_result[class_label] = len(class_data)
                        else:
                            grouped_result[class_label] = class_data[actual_column].mean()
                result = grouped_result
            
            return {
                'status': 'success',
                'statistic': stat,
                'column': column,
                'mapped_column': actual_column,
                'result': result,
                'filters': filters,
                'group_by': group_by,
                'sample_count': len(data),
                'note': 'Mock data from local CSV simulation'
            }
            
        except Exception as e:
            logging.error(f"Error in stub statistic calculation: {e}")
            return self._get_fallback_stub_statistic(stat, column, filters, group_by)

    def _get_fallback_stub_statistic(self, stat: str, column: str, filters: Optional[Dict] = None, 
                                   group_by: Optional[str] = None) -> Dict[str, Any]:
        """Fallback stub function with hardcoded values."""
        if group_by == 'class':
            if stat == 'mean':
                return {
                    'status': 'success', 'statistic': stat, 'column': column,
                    'result': {"OK": 24.6, "KO": 22.3, "KO_HIGH_2mm": 25.1, "KO_LOW_2mm": 21.8},
                    'filters': filters, 'group_by': group_by, 'note': 'Fallback mock data'
                }
            elif stat == 'std':
                return {
                    'status': 'success', 'statistic': stat, 'column': column,
                    'result': {"OK": 0.8, "KO": 1.2, "KO_HIGH_2mm": 1.5, "KO_LOW_2mm": 0.9},
                    'filters': filters, 'group_by': group_by, 'note': 'Fallback mock data'
                }
        
        return {
            'status': 'success', 'statistic': stat, 'column': column,
            'result': {"OK": 24.6, "KO": 22.3},
            'filters': filters, 'group_by': group_by, 'note': 'Fallback mock data'
        }

    def get_dataset_overview(self) -> Dict[str, Any]:
        """Get overview of the dataset."""
        if self.mock_data is not None:
            try:
                overview = {
                    'status': 'success',
                    'total_samples': len(self.mock_data),
                    'classes': self.mock_data['label'].value_counts().to_dict(),
                    'features': list(self.mock_data.columns[2:]),  # Exclude label and sample
                    'data_source': 'Mock data simulation'
                }
                return overview
            except Exception as e:
                logging.error(f"Error getting dataset overview: {e}")
        
        return {
            'status': 'success',
            'total_samples': 20,
            'classes': {'OK': 5, 'KO': 5, 'KO_HIGH_2mm': 5, 'KO_LOW_2mm': 5},
            'features': self.available_sensors,
            'data_source': 'Fallback mock data'
        }

    def get_feature_analysis(self, feature: str) -> Dict[str, Any]:
        """Analyze a specific feature across all classes."""
        if self.mock_data is not None:
            try:
                if feature not in self.mock_data.columns:
                    return {'status': 'error', 'message': f'Feature {feature} not found'}
                
                analysis = {}
                for class_label in self.available_classes:
                    class_data = self.mock_data[self.mock_data['label'] == class_label][feature]
                    if not class_data.empty:
                        analysis[class_label] = {
                            'mean': class_data.mean(),
                            'std': class_data.std(),
                            'min': class_data.min(),
                            'max': class_data.max(),
                            'count': class_data.count()
                        }
                
                return {
                    'status': 'success',
                    'feature': feature,
                    'analysis': analysis,
                    'data_source': 'Mock data simulation'
                }
            except Exception as e:
                logging.error(f"Error in feature analysis: {e}")
        
        return {
            'status': 'success',
            'feature': feature,
            'analysis': {
                'OK': {'mean': 24.6, 'std': 0.8, 'min': 23.5, 'max': 25.5, 'count': 5},
                'KO': {'mean': 22.3, 'std': 1.2, 'min': 21.0, 'max': 23.5, 'count': 5}
            },
            'data_source': 'Fallback mock data'
        }

    def get_class_comparison(self, feature: str, classes: List[str] = None) -> Dict[str, Any]:
        """Compare feature values between different classes."""
        if classes is None:
            classes = self.available_classes
        
        if self.mock_data is not None:
            try:
                comparison = {}
                for class_label in classes:
                    if class_label in self.mock_data['label'].values:
                        class_data = self.mock_data[self.mock_data['label'] == class_label][feature]
                        if not class_data.empty:
                            comparison[class_label] = {
                                'mean': class_data.mean(),
                                'std': class_data.std(),
                                'samples': len(class_data)
                            }
                
                return {
                    'status': 'success',
                    'feature': feature,
                    'comparison': comparison,
                    'data_source': 'Mock data simulation'
                }
            except Exception as e:
                logging.error(f"Error in class comparison: {e}")
        
        return {
            'status': 'success',
            'feature': feature,
            'comparison': {
                'OK': {'mean': 24.6, 'std': 0.8, 'samples': 5},
                'KO': {'mean': 22.3, 'std': 1.2, 'samples': 5}
            },
            'data_source': 'Fallback mock data'
        }

    def get_plot_data(self, plot_type: str, features: List[str] = None,
                       filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Get data for plotting with optional filtering"""
        try:
            # Apply filters if specified
            data = self.mock_data.copy()
            if filters and 'class' in filters:
                data = data[data['label'].isin(filters['class'])]
            
            if len(data) == 0:
                return {
                    'status': 'error',
                    'message': f'No data found for the specified filters',
                    'data_source': 'Mock data'
                }
            
            # Map features to actual column names
            target_features = []
            if features:
                for feature in features:
                    mapped_feature = self._map_sensor_to_column(feature)
                    if mapped_feature in data.columns:
                        target_features.append(mapped_feature)
            
            if not target_features:
                # Default to all numeric features
                target_features = [col for col in data.columns if col not in ['label', 'sample']]
            
            # Prepare plot data based on type
            if plot_type.lower() in ['histogram', 'hist']:
                plot_data = {}
                for feature in target_features:
                    plot_data[feature] = {
                        'values': data[feature].tolist(),
                        'bins': 10,
                        'title': f'{feature} Distribution',
                        'xlabel': feature,
                        'ylabel': 'Frequency'
                    }
                    
            elif plot_type.lower() in ['boxplot', 'box']:
                plot_data = {}
                for feature in target_features:
                    plot_data[feature] = {
                        'class_data': data.groupby('label')[feature].apply(list).to_dict(),
                        'title': f'{feature} by Class',
                        'xlabel': 'Class',
                        'ylabel': feature
                    }
                    
            elif plot_type.lower() in ['scatter', 'scatterplot']:
                if len(target_features) >= 2:
                    plot_data = {
                        'x_values': data[target_features[0]].tolist(),
                        'y_values': data[target_features[1]].tolist(),
                        'labels': data['label'].tolist(),
                        'title': f'{target_features[0]} vs {target_features[1]}',
                        'xlabel': target_features[0],
                        'ylabel': target_features[1]
                    }
                else:
                    return {
                        'status': 'error',
                        'message': 'Scatter plot requires at least 2 features',
                        'data_source': 'Mock data'
                    }
                    
            elif plot_type.lower() in ['correlation', 'correlation_matrix']:
                # Calculate correlation matrix for numeric features
                numeric_data = data[target_features]
                correlation_matrix = numeric_data.corr()
                plot_data = {
                    'correlation_matrix': correlation_matrix.to_dict(),
                    'features': target_features,
                    'title': 'Feature Correlation Matrix',
                    'xlabel': 'Features',
                    'ylabel': 'Features'
                }
                
            elif plot_type.lower() in ['violin', 'violinplot']:
                plot_data = {}
                for feature in target_features:
                    plot_data[feature] = {
                        'class_data': data.groupby('label')[feature].apply(list).to_dict(),
                        'title': f'{feature} Distribution by Class',
                        'xlabel': 'Class',
                        'ylabel': feature
                    }
                    
            else:
                return {
                    'status': 'error',
                    'message': f'Unsupported plot type: {plot_type}',
                    'data_source': 'Mock data'
                }
            
            return {
                'status': 'success',
                'plot_type': plot_type,
                'plot_data': plot_data,
                'features': target_features,
                'filters': filters,
                'sample_count': len(data),
                'data_source': 'Mock data analysis'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error preparing plot data: {str(e)}',
                'data_source': 'Mock data'
            }
    
    def create_plot(self, plot_request: str) -> Dict[str, Any]:
        """Create a plot using the plotting engine."""
        try:
            from ML.plotting_engine import PlottingEngine
            
            # Initialize plotting engine with mock data
            plotting_engine = PlottingEngine(self.feature_matrix_path)
            
            # Generate the plot
            fig = plotting_engine.handle_plot_request(plot_request)
            
            # Save the plot to a temporary file
            import tempfile
            import os
            temp_dir = tempfile.gettempdir()
            plot_filename = f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plot_path = os.path.join(temp_dir, plot_filename)
            
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Close the figure to free memory
            
            return {
                'status': 'success',
                'plot_path': plot_path,
                'plot_type': 'generated',
                'message': f'Plot created successfully: {plot_request}',
                'data_source': 'Plotting engine with mock data'
            }
            
        except ImportError:
            logging.warning("Plotting engine not available")
            return {
                'status': 'error',
                'message': 'Plotting engine not available',
                'data_source': 'Import error'
            }
        except Exception as e:
            logging.error(f"Error creating plot: {e}")
            return {
                'status': 'error',
                'message': f'Error creating plot: {str(e)}',
                'data_source': 'Exception in plotting'
            }
    
    def get_plotting_capabilities(self) -> Dict[str, Any]:
        """Get information about available plotting capabilities."""
        try:
            from ML.plotting_engine import PlottingEngine
            plotting_engine = PlottingEngine(self.feature_matrix_path)
            
            summary = plotting_engine.get_feature_summary()
            
            return {
                'status': 'success',
                'capabilities': {
                    'plot_types': ['boxplot', 'histogram', 'violin', 'correlation', 'timeseries', 'frequency', 'scatter'],
                    'available_sensors': summary['available_sensors'],
                    'feature_categories': summary['feature_categories'],
                    'sample_counts': summary['sample_counts'],
                    'top_features': summary['top_discriminative'][:5]
                },
                'data_source': 'Plotting engine analysis'
            }
            
        except ImportError:
            return {
                'status': 'error',
                'message': 'Plotting engine not available',
                'capabilities': {
                    'plot_types': ['boxplot', 'histogram', 'correlation'],
                    'available_sensors': self.available_sensors,
                    'feature_categories': {},
                    'sample_counts': {},
                    'top_features': []
                },
                'data_source': 'Fallback capabilities'
            }
        except Exception as e:
            logging.error(f"Error getting plotting capabilities: {e}")
            return {
                'status': 'error',
                'message': f'Error: {str(e)}',
                'capabilities': {},
                'data_source': 'Exception in capability check'
            }

    def get_available_features(self) -> List[str]:
        """Get list of available features."""
        if self.mock_data is not None:
            return list(self.mock_data.columns[2:])  # Exclude label and sample
        return self.available_sensors

    def create_plot_from_data(self, plot_type: str, plot_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create an actual matplotlib figure from prepared plot data"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create figure based on plot type
            if plot_type.lower() in ['histogram', 'hist']:
                fig, axes = plt.subplots(1, len(plot_data), figsize=(6*len(plot_data), 5))
                if len(plot_data) == 1:
                    axes = [axes]
                
                for i, (feature, data) in enumerate(plot_data.items()):
                    axes[i].hist(data['values'], bins=data['bins'], alpha=0.7, edgecolor='black')
                    axes[i].set_title(data['title'])
                    axes[i].set_xlabel(data['xlabel'])
                    axes[i].set_ylabel(data['ylabel'])
                    axes[i].grid(True, alpha=0.3)
                
            elif plot_type.lower() in ['boxplot', 'box']:
                fig, axes = plt.subplots(1, len(plot_data), figsize=(6*len(plot_data), 5))
                if len(plot_data) == 1:
                    axes = [axes]
                
                for i, (feature, data) in enumerate(plot_data.items()):
                    class_names = list(data['class_data'].keys())
                    class_values = list(data['class_data'].values())
                    
                    bp = axes[i].boxplot(class_values, labels=class_names, patch_artist=True)
                    axes[i].set_title(data['title'])
                    axes[i].set_xlabel(data['xlabel'])
                    axes[i].set_ylabel(data['ylabel'])
                    axes[i].grid(True, alpha=0.3)
                    
                    # Color the boxes
                    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
                    for patch, color in zip(bp['boxes'], colors[:len(class_names)]):
                        patch.set_facecolor(color)
                
            elif plot_type.lower() in ['scatter', 'scatterplot']:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                
                # Create scatter plot with different colors for each class
                unique_labels = list(set(plot_data['labels']))
                colors = ['blue', 'red', 'green', 'orange']
                
                for i, label in enumerate(unique_labels):
                    mask = [l == label for l in plot_data['labels']]
                    ax.scatter(
                        [plot_data['x_values'][j] for j in range(len(plot_data['x_values'])) if mask[j]],
                        [plot_data['y_values'][j] for j in range(len(plot_data['y_values'])) if mask[j]],
                        label=label, alpha=0.7, c=colors[i % len(colors)]
                    )
                
                ax.set_title(plot_data['title'])
                ax.set_xlabel(plot_data['xlabel'])
                ax.set_ylabel(plot_data['ylabel'])
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            elif plot_type.lower() in ['correlation', 'correlation_matrix']:
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                
                # Convert correlation data to numpy array
                import numpy as np
                features = plot_data['features']
                corr_matrix = np.array([[plot_data['correlation_matrix'][f1][f2] for f2 in features] for f1 in features])
                
                # Create heatmap
                im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                ax.set_xticks(range(len(features)))
                ax.set_yticks(range(len(features)))
                ax.set_xticklabels(features, rotation=45, ha='right')
                ax.set_yticklabels(features)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Correlation Coefficient')
                
                # Add correlation values as text
                for i in range(len(features)):
                    for j in range(len(features)):
                        text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=8)
                
                ax.set_title(plot_data['title'])
                plt.tight_layout()
                
            elif plot_type.lower() in ['violin', 'violinplot']:
                fig, axes = plt.subplots(1, len(plot_data), figsize=(6*len(plot_data), 5))
                if len(plot_data) == 1:
                    axes = [axes]
                
                for i, (feature, data) in enumerate(plot_data.items()):
                    class_names = list(data['class_data'].keys())
                    class_values = list(data['class_data'].values())
                    
                    # Create violin plot
                    parts = axes[i].violinplot(class_values, positions=range(len(class_names)))
                    axes[i].set_title(data['title'])
                    axes[i].set_xlabel(data['xlabel'])
                    axes[i].set_ylabel(data['ylabel'])
                    axes[i].set_xticks(range(len(class_names)))
                    axes[i].set_xticklabels(class_names)
                    axes[i].grid(True, alpha=0.3)
                    
                    # Color the violins
                    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
                    for pc, color in zip(parts['bodies'], colors[:len(class_names)]):
                        pc.set_facecolor(color)
                        pc.set_alpha(0.7)
                
            else:
                return {
                    'status': 'error',
                    'message': f'Unsupported plot type for figure creation: {plot_type}',
                    'data_source': 'Plot creation'
                }
            
            # Save the plot to a temporary file
            import tempfile
            temp_dir = tempfile.gettempdir()
            plot_filename = f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plot_path = os.path.join(temp_dir, plot_filename)
            
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            return {
                'status': 'success',
                'plot_path': plot_path,
                'figure': fig,
                'plot_type': plot_type,
                'message': f'Plot created successfully: {plot_type}',
                'data_source': 'Matplotlib with mock data'
            }
            
        except Exception as e:
            logging.error(f"Error creating plot from data: {e}")
            return {
                'status': 'error',
                'message': f'Error creating plot: {str(e)}',
                'data_source': 'Plot creation error'
            }

    def get_data_quality_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of data quality and reliability"""
        try:
            if self.mock_data is None:
                return {
                    'status': 'error',
                    'message': 'No data loaded for quality assessment',
                    'data_source': 'No data'
                }
            
            # Perform comprehensive data validation
            validation = self._validate_data_quality(self.mock_data, "comprehensive quality assessment")
            
            # Get basic statistics
            total_samples = len(self.mock_data)
            total_features = len(self.mock_data.columns) - 2  # Exclude label and sample columns
            numeric_features = len(self.mock_data.select_dtypes(include=[np.number]).columns)
            
            # Check class distribution
            class_counts = self.mock_data['label'].value_counts().to_dict()
            class_balance = "Balanced" if len(set(class_counts.values())) <= 2 else "Imbalanced"
            
            # Check for missing values
            missing_data = self.mock_data.isnull().sum().sum()
            missing_percentage = (missing_data / (total_samples * total_features)) * 100 if total_features > 0 else 0
            
            # Check data types
            data_types = self.mock_data.dtypes.value_counts().to_dict()
            
            # Assess overall quality
            quality_score = 100
            quality_issues = []
            
            if missing_percentage > 5:
                quality_score -= 20
                quality_issues.append(f"Missing data: {missing_percentage:.1f}%")
            
            if total_samples < 10:
                quality_score -= 30
                quality_issues.append(f"Small sample size: {total_samples} samples")
            
            if numeric_features < 3:
                quality_score -= 15
                quality_issues.append(f"Limited numeric features: {numeric_features}")
            
            if class_balance == "Imbalanced":
                quality_score -= 10
                quality_issues.append("Imbalanced class distribution")
            
            # Determine quality level
            if quality_score >= 80:
                quality_level = "Excellent"
            elif quality_score >= 60:
                quality_level = "Good"
            elif quality_score >= 40:
                quality_level = "Fair"
            else:
                quality_level = "Poor"
            
            return {
                'status': 'success',
                'data_quality_summary': {
                    'overall_quality': quality_level,
                    'quality_score': quality_score,
                    'total_samples': total_samples,
                    'total_features': total_features,
                    'numeric_features': numeric_features,
                    'class_distribution': class_counts,
                    'class_balance': class_balance,
                    'missing_data_percentage': missing_percentage,
                    'data_types': data_types,
                    'quality_issues': quality_issues,
                    'recommendations': self._generate_quality_recommendations(quality_score, quality_issues)
                },
                'validation_details': validation,
                'data_source': 'Mock data quality assessment'
            }
            
        except Exception as e:
            logging.error(f"Error in data quality assessment: {e}")
            return {
                'status': 'error',
                'message': f'Error assessing data quality: {str(e)}',
                'data_source': 'Mock data'
            }
    
    def _generate_quality_recommendations(self, quality_score: int, issues: List[str]) -> List[str]:
        """Generate recommendations for improving data quality"""
        recommendations = []
        
        if quality_score < 80:
            recommendations.append("Consider collecting more samples for better statistical reliability")
        
        if any("Missing data" in issue for issue in issues):
            recommendations.append("Address missing data through imputation or data collection")
        
        if any("Small sample size" in issue for issue in issues):
            recommendations.append("Increase sample size to at least 20-30 samples per class")
        
        if any("Imbalanced" in issue for issue in issues):
            recommendations.append("Consider data augmentation or balanced sampling strategies")
        
        if any("Limited numeric features" in issue for issue in issues):
            recommendations.append("Include more sensor measurements for comprehensive analysis")
        
        if not recommendations:
            recommendations.append("Data quality is excellent - proceed with confidence")
        
        return recommendations

    def _validate_data_quality(self, data: pd.DataFrame, operation: str) -> Dict[str, Any]:
        """Validate data quality before performing operations"""
        try:
            validation_result = {
                'is_valid': True,
                'warnings': [],
                'errors': [],
                'data_info': {}
            }
            
            # Check if data exists
            if data is None or data.empty:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"No data available for {operation}")
                return validation_result
            
            # Check data types
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_columns) == 0:
                validation_result['warnings'].append("No numeric columns found for analysis")
            
            # Check for missing values
            missing_counts = data.isnull().sum()
            if missing_counts.sum() > 0:
                validation_result['warnings'].append(f"Found {missing_counts.sum()} missing values")
            
            # Check sample size
            sample_count = len(data)
            if sample_count < 5:
                validation_result['warnings'].append(f"Small sample size ({sample_count}) may affect statistical reliability")
            
            # Check for infinite values
            inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
            if inf_counts > 0:
                validation_result['errors'].append(f"Found {inf_counts} infinite values")
                validation_result['is_valid'] = False
            
            # Store data information
            validation_result['data_info'] = {
                'sample_count': sample_count,
                'numeric_columns': numeric_columns,
                'missing_values': missing_counts.to_dict(),
                'data_shape': data.shape
            }
            
            return validation_result
            
        except Exception as e:
            return {
                'is_valid': False,
                'errors': [f"Data validation error: {str(e)}"],
                'warnings': [],
                'data_info': {}
            }

# Global instance
ml_interface = MLInterface()
