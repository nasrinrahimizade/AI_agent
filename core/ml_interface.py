"""
ML Interface Wrapper Module

This module provides a clean API interface to the ML layer, abstracting away
the complexity of the underlying statistical and plotting engines.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

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
        """Get statistical measure for a specific column with optional filtering and grouping."""
        if self.initialized and self.ml_agent:
            try:
                # Call real ML agent
                return self.ml_agent.get_statistic(stat, column, filters, group_by)
            except Exception as e:
                logging.error(f"ML agent error: {e}, falling back to stub")
        
        return self._get_stub_statistic(stat, column, filters, group_by)

    def _get_stub_statistic(self, stat: str, column: str, filters: Optional[Dict] = None, 
                           group_by: Optional[str] = None) -> Dict[str, Any]:
        """Stub function that returns realistic mock data based on the CSV."""
        if self.mock_data is None:
            return self._get_fallback_stub_statistic(stat, column, filters, group_by)
        
        try:
            # Apply filters if specified
            data = self.mock_data.copy()
            if filters and 'class' in filters:
                data = data[data['label'].isin(filters['class'])]
            
            if column not in data.columns:
                return {
                    'status': 'error',
                    'message': f'Column {column} not found in dataset',
                    'available_columns': list(data.columns)
                }
            
            # Calculate statistic
            if stat == 'mean':
                result = data[column].mean()
            elif stat == 'median':
                result = data[column].median()
            elif stat == 'std':
                result = data[column].std()
            elif stat == 'variance':
                result = data[column].var()
            elif stat == 'min':
                result = data[column].min()
            elif stat == 'max':
                result = data[column].max()
            elif stat == 'count':
                result = data[column].count()
            else:
                result = data[column].mean()  # fallback
            
            # Group by class if requested
            if group_by == 'class':
                grouped_result = {}
                for class_label in self.available_classes:
                    class_data = data[data['label'] == class_label]
                    if not class_data.empty:
                        if stat == 'mean':
                            grouped_result[class_label] = class_data[column].mean()
                        elif stat == 'median':
                            grouped_result[class_label] = class_data[column].median()
                        elif stat == 'std':
                            grouped_result[class_label] = class_data[column].std()
                        elif stat == 'count':
                            grouped_result[class_label] = class_data[column].count()
                        else:
                            grouped_result[class_label] = class_data[column].mean()
                result = grouped_result
            
            return {
                'status': 'success',
                'statistic': stat,
                'column': column,
                'result': result,
                'filters': filters,
                'group_by': group_by,
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
        """Get data formatted for plotting."""
        if features is None:
            features = ['temperature_mean', 'humidity_mean']
        
        if self.mock_data is not None:
            try:
                data = self.mock_data.copy()
                if filters and 'class' in filters:
                    data = data[data['label'].isin(filters['class'])]
                
                plot_data = {
                    'labels': data['label'].tolist(),
                    'samples': data['sample'].tolist()
                }
                
                for feature in features:
                    if feature in data.columns:
                        plot_data[feature] = data[feature].tolist()
                
                return {
                    'status': 'success',
                    'plot_type': plot_type,
                    'data': plot_data,
                    'data_source': 'Mock data simulation'
                }
            except Exception as e:
                logging.error(f"Error getting plot data: {e}")
        
        return {
            'status': 'success',
            'plot_type': plot_type,
            'data': {
                'labels': ['OK', 'KO'],
                'temperature_mean': [24.6, 22.3],
                'humidity_mean': [45.5, 38.9]
            },
            'data_source': 'Fallback mock data'
        }

    def get_available_features(self) -> List[str]:
        """Get list of available features."""
        if self.mock_data is not None:
            return list(self.mock_data.columns[2:])  # Exclude label and sample
        return self.available_sensors

    def get_top_features(self, n: int = 5) -> Dict[str, Any]:
        """Get top discriminative features."""
        if self.mock_data is not None:
            try:
                # Simple feature importance calculation based on variance
                features = self.available_sensors
                importance_scores = {}
                
                for feature in features:
                    if feature in self.mock_data.columns:
                        # Calculate variance across all classes
                        variance = self.mock_data[feature].var()
                        importance_scores[feature] = variance
                
                # Sort by importance and get top N
                sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
                top_features = sorted_features[:n]
                
                return {
                    'status': 'success',
                    'top_features': [feature for feature, score in top_features],
                    'importance_scores': dict(top_features),
                    'data_source': 'Mock data simulation'
                }
            except Exception as e:
                logging.error(f"Error calculating top features: {e}")
        
        return {
            'status': 'success',
            'top_features': ['temperature_mean', 'humidity_mean', 'pressure_mean', 'acceleration_x_mean', 'gyroscope_z_mean'],
            'importance_scores': {'temperature_mean': 0.85, 'humidity_mean': 0.72, 'pressure_mean': 0.68},
            'data_source': 'Fallback mock data'
        }

# Global instance
ml_interface = MLInterface()
