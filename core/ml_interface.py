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
    def __init__(self, feature_matrix_path: str = "ML/feature_matrix.csv"):
        self.feature_matrix_path = feature_matrix_path
        self.ml_agent = None
        self.initialized = False
        self.mock_data = None
        
        if ML_AVAILABLE:
            try:
                self.ml_agent = StatisticalAIAgent(feature_matrix_path) ##ok
                self.initialized = True
                logging.info("ML Interface initialized with StatisticalAIAgent") ##ok
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
        
        # Extract unique sensor types from actual dataset columns
        self.available_sensors = []
        if self.mock_data is not None:
            # Get all columns except label and sample
            feature_columns = [col for col in self.mock_data.columns if col not in ['label', 'sample']]
            # Extract unique sensor prefixes
            sensor_prefixes = set()
            for col in feature_columns:
                if '_' in col:
                    # Extract sensor name (e.g., 'HTS221_TEMP' from 'HTS221_TEMP_TEMP_mean')
                    parts = col.split('_')
                    if len(parts) >= 2:
                        sensor_prefixes.add(f"{parts[0]}_{parts[1]}")
            self.available_sensors = list(sensor_prefixes)
        else:
            # Fallback list based on your dataset
            self.available_sensors = ['HTS221_TEMP', 'HTS221_HUM', 'LPS22HH_PRESS', 'IIS3DWB_ACC', 
                                    'ISM330DHCX_ACC', 'ISM330DHCX_GYRO', 'IMP23ABSU_MIC', 'IMP34DT05_MIC']
        
        
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
            
            # Validate data quality before analysis and get cleaned data
            validation = self._validate_data_quality(data, f"statistic calculation for {stat}")
            if not validation['is_valid']:
                return {
                    'status': 'error',
                    'message': f'Data quality issues prevent accurate {stat} calculation: {", ".join(validation["errors"])}',
                    'data_source': 'Mock data',
                    'data_quality': 'poor',
                    'validation_details': validation
                }
            
            # Use cleaned data from validation
            cleaned_data = validation['cleaned_data']
            
            # Check if we have enough data for the requested statistic
            if len(cleaned_data) < 2:
                return {
                    'status': 'error',
                    'message': f'Insufficient data for {stat} calculation. Need at least 2 samples, got {len(cleaned_data)}',
                    'data_source': 'Mock data',
                    'data_quality': 'insufficient_samples'
                }
            
            # Get the target column data and remove NaN values for calculation
            column_data = cleaned_data[target_column].dropna()
            
            if len(column_data) == 0:
                return {
                    'status': 'error',
                    'message': f'No valid data remaining in column {target_column} after cleaning',
                    'data_source': 'Mock data',
                    'data_quality': 'no_valid_data'
                }
            
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
                for class_name in cleaned_data['label'].unique():
                    class_data = cleaned_data[cleaned_data['label'] == class_name][target_column].dropna()
                    if len(class_data) > 0:  # Only calculate if we have valid data
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
                    else:
                        grouped_result[class_name] = None  # Mark as no valid data
                
                result = grouped_result
            
            # Add data quality information to response
            response = {
                'status': 'success',
                'result': result,
                'statistic': stat,
                'column': column,  # Original column name requested
                'mapped_column': target_column,  # Actual column used
                'filters': filters,
                'grouping': group_by,
                'sample_count': len(cleaned_data),
                'valid_data_count': len(column_data),
                'data_source': 'Mock data analysis',
                'data_quality': 'good',
                'validation_details': validation,
                'confidence': 'high' if len(column_data) >= 10 else 'medium'
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
            
            # Handle special case where 'KO' means all non-OK classes
            if 'KO' in classes and 'KO' not in self.mock_data['label'].unique():
                # Create a modified dataset where all non-OK classes are labeled as 'KO'
                filtered_data = self.mock_data.copy()
                non_ok_mask = filtered_data['label'] != 'OK'
                filtered_data.loc[non_ok_mask, 'label'] = 'KO'
                
                # Now filter for the requested classes
                filtered_data = filtered_data[filtered_data['label'].isin(classes)]
            else:
                # Standard filtering for exact class matches
                filtered_data = self.mock_data[self.mock_data['label'].isin(classes)]
            
            # Validate data quality before analysis and get cleaned data
            validation = self._validate_data_quality(filtered_data, f"top {n} features analysis")
            
            if not validation['is_valid']:
                return {
                    'status': 'error',
                    'message': f'Data quality issues prevent accurate feature analysis: {", ".join(validation["errors"])}',
                    'data_source': 'Mock data',
                    'data_quality': 'poor',
                    'validation_details': validation
                }
            
            # Use cleaned data from validation
            cleaned_data = validation['cleaned_data']
            
            if len(cleaned_data) == 0:
                return {
                    'status': 'error',
                    'message': f'No data found for classes: {classes}',
                    'data_source': 'Mock data',
                    'data_quality': 'no_data'
                }
            
            # Check if we have enough samples for reliable analysis
            if len(cleaned_data) < 5:
                return {
                    'status': 'error',
                    'message': f'Insufficient data for reliable feature analysis. Need at least 5 samples, got {len(cleaned_data)}',
                    'data_source': 'Mock data',
                    'data_quality': 'insufficient_samples'
                }
            
            # Get feature columns (exclude label and sample)
            feature_columns = [col for col in cleaned_data.columns if col not in ['label', 'sample']]
            
            if len(feature_columns) == 0:
                return {
                    'status': 'error',
                    'message': 'No feature columns found for analysis',
                    'data_source': 'Mock data',
                    'data_quality': 'no_features'
                }
            
            # Calculate discriminative scores for each feature
            feature_scores = {}
            feature_details = {}
            failed_features = []
            
            for feature in feature_columns:
                try:
                    # Get data for this feature and remove NaN values
                    feature_data = cleaned_data[[feature, 'label']].dropna()
                    
                    # Skip if not enough data
                    if len(feature_data) < 3:
                        continue
                    
                    # Get class data
                    class_data = {}
                    class_means = {}
                    valid_classes = []
                    
                    for class_name in classes:
                        class_values = feature_data[feature_data['label'] == class_name][feature]
                        
                        if len(class_values) > 0:
                            class_data[class_name] = class_values
                            class_means[class_name] = class_values.mean()
                            valid_classes.append(class_name)
                    
                    if len(valid_classes) < 2:
                        continue
                    
                    # Simple discriminative score calculation
                    if len(valid_classes) == 2:
                        mean1, mean2 = class_means[valid_classes[0]], class_means[valid_classes[1]]
                        discriminative_score = abs(mean1 - mean2)
                        
                        # Check if score is valid
                        if np.isnan(discriminative_score) or np.isinf(discriminative_score):
                            failed_features.append((feature, f"Invalid score: {discriminative_score}"))
                            continue
                        
                        feature_scores[feature] = discriminative_score
                        feature_details[feature] = {
                            'discriminative_score': discriminative_score,
                            'class_means': class_means,
                            'sample_count': len(feature_data),
                            'valid_classes': valid_classes,
                            'analysis_method': 'Mean difference'
                        }
                    
                except Exception as e:
                    failed_features.append((feature, f"Exception: {str(e)}"))
                    continue
            
            if not feature_scores:
                # Provide detailed debugging information
                debug_info = {
                    'total_features': len(feature_columns),
                    'cleaned_data_shape': cleaned_data.shape,
                    'classes_found': cleaned_data['label'].unique().tolist(),
                    'classes_requested': classes,
                    'failed_features': failed_features,
                    'sample_class_counts': cleaned_data['label'].value_counts().to_dict()
                }
                
                return {
                    'status': 'error',
                    'message': 'Unable to calculate feature scores due to data issues',
                    'data_source': 'Mock data',
                    'data_quality': 'calculation_failed',
                    'debug_info': debug_info,
                    'validation_details': validation
                }
            
            # Sort features by discriminative score (higher = more discriminative)
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get top N features
            top_features = [feature for feature, score in sorted_features[:n]]
            
            # Add data quality information to response
            response = {
                'status': 'success',
                'top_features': top_features,
                'feature_details': {k: v for k, v in feature_details.items() if k in top_features},
                'classes': classes,
                'sample_count': len(cleaned_data),
                'features_analyzed': len(feature_scores),
                'features_skipped': len(feature_columns) - len(feature_scores),
                'data_source': 'Mock data analysis',
                'data_quality': 'good',
                'validation_details': validation,
                'confidence': 'high' if len(cleaned_data) >= 10 else 'medium'
            }
            
            # Add warnings if any
            if validation['warnings']:
                response['warnings'] = validation['warnings']
            
            return response
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error in feature analysis: {str(e)}',
                'data_source': 'Mock data',
                'data_quality': 'analysis_error'
            }
    
    def _map_sensor_to_column(self, sensor_name: str) -> str:
        """Map sensor names to actual column names in the dataset."""
        # If the sensor_name is already a full column name, return it
        if sensor_name in self.mock_data.columns:
            return sensor_name
        
        # Try to find a matching column with _mean suffix (most common case)
        potential_matches = []
        for col in self.mock_data.columns:
            if col.startswith(sensor_name):
                potential_matches.append(col)
        
        # Prefer _mean columns, then _median, then any match
        for suffix in ['_mean', '_median', '_max', '_min', '_std', '_variance']:
            for col in potential_matches:
                if col.endswith(suffix):
                    return col
        
        # If no specific match found, return the first match or original name
        if potential_matches:
            return potential_matches[0]
        
        # Fallback: try common mappings based on your dataset structure
        common_mappings = {
            'temperature': 'HTS221_TEMP_TEMP_mean',
            'humidity': 'HTS221_HUM_HUM_mean',
            'pressure': 'LPS22HH_PRESS_PRESS_mean',
            'acceleration_x': 'IIS3DWB_ACC_A_x_mean',
            'acceleration_y': 'IIS3DWB_ACC_A_y_mean',
            'acceleration_z': 'IIS3DWB_ACC_A_z_mean',
            'gyroscope_x': 'ISM330DHCX_GYRO_G_x_mean',
            'gyroscope_y': 'ISM330DHCX_GYRO_G_y_mean',
            'gyroscope_z': 'ISM330DHCX_GYRO_G_z_mean',
            'microphone': 'IMP23ABSU_MIC_MIC_mean'
        }
        
        return common_mappings.get(sensor_name, sensor_name)
    
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
                # Try to find any column that contains the sensor name
                matching_cols = [col for col in data.columns if column.lower() in col.lower()]
                if matching_cols:
                    actual_column = matching_cols[0]  # Use first match
                else:
                    return {
                        'status': 'error',
                        'message': f'Column {column} (mapped to {actual_column}) not found in dataset',
                        'available_sensors': self.available_sensors
                    }
            
            # Calculate statistic
            column_data = data[actual_column].dropna()
            
            if len(column_data) == 0:
                return {
                    'status': 'error',
                    'message': f'No valid data found in column {actual_column}',
                    'sample_count': 0
                }
            
            if stat.lower() in ['mean', 'average', 'avg']:
                result = column_data.mean()
            elif stat.lower() in ['median', 'med']:
                result = column_data.median()
            elif stat.lower() in ['std', 'standard_deviation']:
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
                result = column_data.mean()  # fallback
            
            # Group by class if requested
            if group_by == 'class':
                grouped_result = {}
                for class_label in data['label'].unique():
                    class_data = data[data['label'] == class_label][actual_column].dropna()
                    if len(class_data) > 0:
                        if stat.lower() in ['mean', 'average', 'avg']:
                            grouped_result[class_label] = class_data.mean()
                        elif stat.lower() in ['median', 'med']:
                            grouped_result[class_label] = class_data.median()
                        elif stat.lower() in ['std', 'standard_deviation']:
                            grouped_result[class_label] = class_data.std()
                        elif stat.lower() in ['variance', 'var']:
                            grouped_result[class_label] = class_data.var()
                        elif stat.lower() in ['count', 'n']:
                            grouped_result[class_label] = len(class_data)
                        elif stat.lower() in ['min', 'minimum']:
                            grouped_result[class_label] = class_data.min()
                        elif stat.lower() in ['max', 'maximum']:
                            grouped_result[class_label] = class_data.max()
                        else:
                            grouped_result[class_label] = class_data.mean()
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
                # More robust feature extraction
                all_columns = list(self.mock_data.columns)
                
                # Identify label and sample columns more flexibly
                exclude_columns = []
                for col in all_columns:
                    if col.lower() in ['label', 'sample', 'class', 'target', 'y']:
                        exclude_columns.append(col)
                
                # Get feature columns (everything except excluded columns)
                feature_columns = [col for col in all_columns if col not in exclude_columns]
                
                # Get class information - try different possible label column names
                label_column = None
                class_info = {}
                
                for possible_label in ['label', 'class', 'target', 'y']:
                    if possible_label in self.mock_data.columns:
                        label_column = possible_label
                        class_info = self.mock_data[possible_label].value_counts().to_dict()
                        break
                
                # If no standard label column found, check first few columns
                if not class_info:
                    for col in all_columns[:3]:  # Check first 3 columns
                        if self.mock_data[col].dtype == 'object' or self.mock_data[col].nunique() < 10:
                            label_column = col
                            class_info = self.mock_data[col].value_counts().to_dict()
                            break
                
                # Get data type information
                data_types = {}
                numeric_features = []
                categorical_features = []
                
                for col in feature_columns:
                    dtype = str(self.mock_data[col].dtype)
                    data_types[col] = dtype
                    
                    if self.mock_data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        numeric_features.append(col)
                    else:
                        categorical_features.append(col)
                
                # Calculate basic statistics
                missing_values = self.mock_data.isnull().sum().sum()
                total_values = len(self.mock_data) * len(self.mock_data.columns)
                missing_percentage = (missing_values / total_values * 100) if total_values > 0 else 0
                
                overview = {
                    'status': 'success',
                    'total_samples': len(self.mock_data),
                    'total_features': len(feature_columns),
                    'numeric_features': len(numeric_features),
                    'categorical_features': len(categorical_features),
                    'classes': class_info,
                    'label_column': label_column,
                    'feature_columns': feature_columns[:20],  # Show first 20 features
                    'total_feature_count': len(feature_columns),
                    'sample_feature_names': feature_columns[:10] if feature_columns else [],
                    'data_shape': list(self.mock_data.shape),
                    'missing_values_total': missing_values,
                    'missing_percentage': round(missing_percentage, 2),
                    'column_info': {
                        'all_columns': all_columns,
                        'excluded_columns': exclude_columns,
                        'data_types_summary': {
                            'numeric': len(numeric_features),
                            'categorical': len(categorical_features)
                        }
                    },
                    'data_source': 'Loaded CSV data'
                }
                
                # Add memory usage info if possible
                try:
                    memory_usage = self.mock_data.memory_usage(deep=True).sum()
                    overview['memory_usage_bytes'] = memory_usage
                    overview['memory_usage_mb'] = round(memory_usage / (1024 * 1024), 2)
                except:
                    pass
                
                return overview
                
            except Exception as e:
                logging.error(f"Error getting dataset overview: {e}")
                import traceback
                error_details = traceback.format_exc()
                
                return {
                    'status': 'error',
                    'message': f'Error analyzing dataset: {str(e)}',
                    'error_details': error_details,
                    'data_source': 'Error in analysis'
                }
        
        # Fallback if no data loaded
        return {
            'status': 'warning',
            'message': 'No dataset loaded, using fallback information',
            'total_samples': 'Unknown',
            'classes': 'Unknown',
            'features': self.available_sensors if hasattr(self, 'available_sensors') else [],
            'data_source': 'Fallback - no data loaded'
        }

    def get_feature_analysis(self, feature: str) -> Dict[str, Any]:
        """Analyze a specific feature across all classes using REAL data."""
        try:
            if self.mock_data is None:
                return {
                    'status': 'error', 
                    'message': 'No data loaded for analysis',
                    'data_source': 'No data'
                }
            
            # Map feature name to actual column
            actual_column = self._map_sensor_to_column(feature)
            
            if actual_column not in self.mock_data.columns:
                return {
                    'status': 'error', 
                    'message': f'Feature {feature} (mapped to {actual_column}) not found in dataset',
                    'available_features': self.get_available_features()[:10]  # Show first 10
                }
            
            # Validate data quality
            validation = self._validate_data_quality(self.mock_data, f"feature analysis for {feature}")
            if not validation['is_valid']:
                return {
                    'status': 'error',
                    'message': f'Data quality issues prevent analysis: {", ".join(validation["errors"])}',
                    'validation_details': validation
                }
            
            cleaned_data = validation['cleaned_data']
            
            # Convert to binary classification (OK vs KO)
            binary_data = cleaned_data.copy()
            binary_data.loc[binary_data['label'] != 'OK', 'label'] = 'KO'
            
            # Calculate real statistics for each class
            analysis = {}
            
            for class_label in ['OK', 'KO']:
                class_data = binary_data[binary_data['label'] == class_label][actual_column].dropna()
                
                if len(class_data) > 0:
                    analysis[class_label] = {
                        'mean': float(class_data.mean()),
                        'std': float(class_data.std()) if len(class_data) > 1 else 0.0,
                        'min': float(class_data.min()),
                        'max': float(class_data.max()),
                        'median': float(class_data.median()),
                        'count': int(len(class_data)),
                        'variance': float(class_data.var()) if len(class_data) > 1 else 0.0
                    }
                else:
                    analysis[class_label] = {
                        'mean': None, 'std': None, 'min': None, 'max': None, 
                        'median': None, 'count': 0, 'variance': None
                    }
            
            # Calculate discriminative power
            if analysis['OK']['mean'] is not None and analysis['KO']['mean'] is not None:
                discriminative_score = abs(analysis['OK']['mean'] - analysis['KO']['mean'])
                relative_difference = discriminative_score / max(abs(analysis['OK']['mean']), abs(analysis['KO']['mean']), 1e-10)
            else:
                discriminative_score = None
                relative_difference = None
            
            return {
                'status': 'success',
                'feature': feature,
                'mapped_column': actual_column,
                'analysis': analysis,
                'discriminative_score': discriminative_score,
                'relative_difference': relative_difference,
                'total_samples': len(cleaned_data),
                'data_source': 'Real data calculation from feature_matrix.csv',
                'validation_details': validation
            }
            
        except Exception as e:
            logging.error(f"Error in feature analysis: {e}")
            return {
                'status': 'error',
                'message': f'Error analyzing feature {feature}: {str(e)}',
                'data_source': 'Calculation error'
            }

    def get_class_comparison(self, feature: str, classes: List[str] = None) -> Dict[str, Any]:
        """Compare feature values between different classes using REAL data."""
        try:
            if self.mock_data is None:
                return {
                    'status': 'error',
                    'message': 'No data loaded for comparison',
                    'data_source': 'No data'
                }
            
            # Default to binary classification
            if classes is None:
                classes = ['OK', 'KO']
            
            # Map feature name to actual column
            actual_column = self._map_sensor_to_column(feature)
            
            if actual_column not in self.mock_data.columns:
                return {
                    'status': 'error',
                    'message': f'Feature {feature} (mapped to {actual_column}) not found in dataset',
                    'available_features': self.get_available_features()[:10]
                }
            
            # Validate data quality
            validation = self._validate_data_quality(self.mock_data, f"class comparison for {feature}")
            if not validation['is_valid']:
                return {
                    'status': 'error',
                    'message': f'Data quality issues prevent comparison: {", ".join(validation["errors"])}',
                    'validation_details': validation
                }
            
            cleaned_data = validation['cleaned_data']
            
            # Handle binary classification if 'KO' is requested but doesn't exist as exact label
            if 'KO' in classes and 'KO' not in cleaned_data['label'].unique():
                # Create binary version
                binary_data = cleaned_data.copy()
                binary_data.loc[binary_data['label'] != 'OK', 'label'] = 'KO'
                comparison_data = binary_data
            else:
                comparison_data = cleaned_data
            
            # Calculate real comparisons
            comparison = {}
            
            for class_label in classes:
                if class_label in comparison_data['label'].values:
                    class_data = comparison_data[comparison_data['label'] == class_label][actual_column].dropna()
                    
                    if len(class_data) > 0:
                        comparison[class_label] = {
                            'mean': float(class_data.mean()),
                            'std': float(class_data.std()) if len(class_data) > 1 else 0.0,
                            'median': float(class_data.median()),
                            'min': float(class_data.min()),
                            'max': float(class_data.max()),
                            'samples': int(len(class_data)),
                            'variance': float(class_data.var()) if len(class_data) > 1 else 0.0,
                            'q25': float(class_data.quantile(0.25)),
                            'q75': float(class_data.quantile(0.75))
                        }
                    else:
                        comparison[class_label] = {
                            'mean': None, 'std': None, 'median': None,
                            'min': None, 'max': None, 'samples': 0,
                            'variance': None, 'q25': None, 'q75': None
                        }
                else:
                    comparison[class_label] = {
                        'mean': None, 'std': None, 'median': None,
                        'min': None, 'max': None, 'samples': 0,
                        'variance': None, 'q25': None, 'q75': None,
                        'note': f'Class {class_label} not found in data'
                    }
            
            # Calculate statistical significance if we have OK and KO
            statistical_tests = {}
            if 'OK' in comparison and 'KO' in comparison:
                if comparison['OK']['samples'] > 0 and comparison['KO']['samples'] > 0:
                    ok_data = comparison_data[comparison_data['label'] == 'OK'][actual_column].dropna()
                    ko_data = comparison_data[comparison_data['label'] == 'KO'][actual_column].dropna()
                    
                    # Simple t-test like comparison
                    try:
                        from scipy import stats
                        t_stat, p_value = stats.ttest_ind(ok_data, ko_data)
                        statistical_tests['t_test'] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        }
                    except ImportError:
                        # Manual calculation if scipy not available
                        pooled_std = np.sqrt(((len(ok_data) - 1) * ok_data.var() + 
                                            (len(ko_data) - 1) * ko_data.var()) / 
                                        (len(ok_data) + len(ko_data) - 2))
                        t_stat = (ok_data.mean() - ko_data.mean()) / (pooled_std * np.sqrt(1/len(ok_data) + 1/len(ko_data)))
                        statistical_tests['t_test'] = {
                            't_statistic': float(t_stat),
                            'note': 'Manual calculation (scipy not available)',
                            'pooled_std': float(pooled_std)
                        }
            
            return {
                'status': 'success',
                'feature': feature,
                'mapped_column': actual_column,
                'comparison': comparison,
                'classes_compared': classes,
                'statistical_tests': statistical_tests,
                'total_samples': len(cleaned_data),
                'data_source': 'Real data calculation from feature_matrix.csv',
                'validation_details': validation
            }
            
        except Exception as e:
            logging.error(f"Error in class comparison: {e}")
            return {
                'status': 'error',
                'message': f'Error comparing classes for feature {feature}: {str(e)}',
                'data_source': 'Calculation error'
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
            # Return actual column names excluding label and sample
            return [col for col in self.mock_data.columns if col not in ['label', 'sample']]
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
                'data_info': {},
                'cleaned_data': data.copy()  # Add cleaned data to result
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
            
            # Handle infinite values - Clean them instead of failing
            cleaned_data = data.copy()
            inf_counts = {}
            total_inf_replaced = 0
            
            for col in numeric_columns:
                # Count infinite values
                inf_mask = np.isinf(cleaned_data[col])
                inf_count = inf_mask.sum()
                
                if inf_count > 0:
                    inf_counts[col] = inf_count
                    total_inf_replaced += inf_count
                    
                    # Replace infinite values with NaN
                    cleaned_data.loc[inf_mask, col] = np.nan
                    
                    validation_result['warnings'].append(
                        f"Replaced {inf_count} infinite values with NaN in column '{col}'"
                    )
            
            if total_inf_replaced > 0:
                validation_result['warnings'].append(
                    f"Total infinite values cleaned: {total_inf_replaced}"
                )
                # Update the cleaned data in the result
                validation_result['cleaned_data'] = cleaned_data
            
            # Check for missing values (including newly created NaNs)
            missing_counts = cleaned_data.isnull().sum()
            if missing_counts.sum() > 0:
                validation_result['warnings'].append(
                    f"Found {missing_counts.sum()} missing values (including cleaned infinite values)"
                )
            
            # Check sample size
            sample_count = len(cleaned_data)
            if sample_count < 5:
                validation_result['warnings'].append(
                    f"Small sample size ({sample_count}) may affect statistical reliability"
                )
            
            # For numeric operations, check if we have enough valid data after cleaning
            valid_data_counts = {}
            for col in numeric_columns:
                valid_count = cleaned_data[col].notna().sum()
                valid_data_counts[col] = valid_count
                
                if valid_count == 0:
                    validation_result['errors'].append(
                        f"No valid data remaining in column '{col}' after cleaning"
                    )
                    validation_result['is_valid'] = False
                elif valid_count < 3:
                    validation_result['warnings'].append(
                        f"Very few valid values ({valid_count}) in column '{col}'"
                    )
            
            # Store data information
            validation_result['data_info'] = {
                'sample_count': sample_count,
                'numeric_columns': numeric_columns,
                'missing_values': missing_counts.to_dict(),
                'infinite_values_cleaned': inf_counts,
                'valid_data_counts': valid_data_counts,
                'data_shape': cleaned_data.shape
            }
            
            return validation_result
            
        except Exception as e:
            return {
                'is_valid': False,
                'errors': [f"Data validation error: {str(e)}"],
                'warnings': [],
                'data_info': {},
                'cleaned_data': data.copy() if data is not None else None
            }

# Global instance
ml_interface = MLInterface()


# Add this test to your main function or create a separate test
def test_dataset_overview():
    """Simple test for the dataset overview function"""
    print("\n" + "=" * 60)
    print("TESTING DATASET OVERVIEW FUNCTION")
    print("=" * 60)
    
    try:
        # Get the overview
        overview = ml_interface.get_dataset_overview()
        
        # Check if the function executed successfully
        if overview['status'] == 'success':
            print("✓ Dataset overview function executed successfully")
            
            # Display key information
            print(f"\nDATASET SUMMARY:")
            print(f"- Status: {overview['status']}")
            print(f"- Total samples: {overview['total_samples']}")
            print(f"- Data shape: {overview.get('data_shape', 'N/A')}")
            print(f"- Total features: {overview['total_features']}")
            print(f"- Numeric features: {overview['numeric_features']}")
            print(f"- Categorical features: {overview['categorical_features']}")
            print(f"- Label column: {overview.get('label_column', 'N/A')}")
            print(f"- Missing values: {overview.get('missing_values_total', 'N/A')} ({overview.get('missing_percentage', 'N/A')}%)")
            print(f"- Memory usage: {overview.get('memory_usage_mb', 'N/A')} MB")
            print(f"- Data source: {overview['data_source']}")
            
            # Display class distribution
            print(f"\nCLASS DISTRIBUTION:")
            if overview['classes']:
                for class_name, count in overview['classes'].items():
                    percentage = (count / overview['total_samples'] * 100) if overview['total_samples'] > 0 else 0
                    print(f"- {class_name}: {count} samples ({percentage:.1f}%)")
            else:
                print("- No class information available")
            
            # Display sample features
            print(f"\nSAMPLE FEATURES (first 10):")
            sample_features = overview.get('sample_feature_names', [])
            if sample_features:
                for i, feature in enumerate(sample_features, 1):
                    print(f"  {i:2d}. {feature}")
            else:
                print("- No feature information available")
            
            if overview['total_feature_count'] > 10:
                print(f"  ... and {overview['total_feature_count'] - 10} more features")
            
            # Display column information
            print(f"\nCOLUMN INFORMATION:")
            col_info = overview.get('column_info', {})
            if col_info:
                print(f"- Total columns: {len(col_info.get('all_columns', []))}")
                print(f"- Excluded columns: {col_info.get('excluded_columns', [])}")
                data_types_summary = col_info.get('data_types_summary', {})
                print(f"- Data types: {data_types_summary.get('numeric', 0)} numeric, {data_types_summary.get('categorical', 0)} categorical")
            
            # Validation checks
            print(f"\nVALIDATION CHECKS:")
            checks_passed = 0
            total_checks = 5
            
            # Check 1: Has data
            if overview['total_samples'] > 0:
                print("✓ Dataset contains samples")
                checks_passed += 1
            else:
                print("✗ Dataset is empty")
            
            # Check 2: Has features
            if overview['total_features'] > 0:
                print("✓ Dataset contains features")
                checks_passed += 1
            else:
                print("✗ No features found")
            
            # Check 3: Has classes
            if overview['classes'] and len(overview['classes']) > 0:
                print("✓ Class information available")
                checks_passed += 1
            else:
                print("✗ No class information found")
            
            # Check 4: Reasonable data quality
            missing_pct = overview.get('missing_percentage', 0)
            if missing_pct < 20:
                print(f"✓ Acceptable missing data level ({missing_pct}%)")
                checks_passed += 1
            else:
                print(f"⚠ High missing data level ({missing_pct}%)")
            
            # Check 5: Multiple classes for classification
            num_classes = len(overview['classes']) if overview['classes'] else 0
            if num_classes >= 2:
                print(f"✓ Multiple classes for classification ({num_classes} classes)")
                checks_passed += 1
            else:
                print(f"⚠ Only {num_classes} class(es) found")
            
            print(f"\nOVERALL ASSESSMENT: {checks_passed}/{total_checks} checks passed")
            
            if checks_passed >= 4:
                print("🎉 Dataset looks good for analysis!")
            elif checks_passed >= 2:
                print("⚠️  Dataset has some issues but may be usable")
            else:
                print("❌ Dataset has significant issues")
            
        elif overview['status'] == 'warning':
            print("⚠️ Dataset overview returned warning")
            print(f"Message: {overview.get('message', 'Unknown warning')}")
            
        else:  # status == 'error'
            print("❌ Dataset overview failed")
            print(f"Error: {overview.get('message', 'Unknown error')}")
            if 'error_details' in overview:
                print("Error details:")
                print(overview['error_details'])
        
        return overview
        
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        return None


# If you want to add this to your existing main function, add this line:


# # Or run it standalone:
# if __name__ == "__main__":
#     # Your existing tests...
    
#     # Add this line at the end:
#     test_dataset_overview()

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING ML INTERFACE WITH UPDATED DATASET COMPATIBILITY")
    print("=" * 80)
    
    # Test 1: Check if data loaded successfully
    print("\n1. DATA LOADING TEST")
    print("-" * 40)
    if ml_interface.mock_data is not None:
        print(f"✓ Data loaded successfully")
        print(f"  - Shape: {ml_interface.mock_data.shape}")
        print(f"  - Classes: {ml_interface.mock_data['label'].value_counts().to_dict()}")
        print(f"  - Sample columns: {list(ml_interface.mock_data.columns[:5])}...")
    else:
        print("✗ Failed to load data")
        exit(1)
    
    # Test 2: Check available sensors
    print("\n2. AVAILABLE SENSORS TEST")
    print("-" * 40)
    available_sensors = ml_interface.available_sensors
    print(f"Available sensors ({len(available_sensors)}):")
    for i, sensor in enumerate(available_sensors[:10], 1):
        print(f"  {i}. {sensor}")
    if len(available_sensors) > 10:
        print(f"  ... and {len(available_sensors) - 10} more")
    
    # Test 3: Check available features
    print("\n3. AVAILABLE FEATURES TEST")
    print("-" * 40)
    available_features = ml_interface.get_available_features()
    print(f"Total features: {len(available_features)}")
    print("Sample features:")
    for i, feature in enumerate(available_features[:10], 1):
        print(f"  {i}. {feature}")
    if len(available_features) > 10:
        print(f"  ... and {len(available_features) - 10} more")
    
    # Test 4: Test sensor name mapping
    print("\n4. SENSOR MAPPING TEST")
    print("-" * 40)
    test_sensors = [
        'temperature',
        'HTS221_TEMP',
        'HTS221_TEMP_TEMP_mean',
        'humidity',
        'HTS221_HUM',
        'pressure',
        'LPS22HH_PRESS',
        'acceleration_x',
        'IIS3DWB_ACC',
        'gyroscope_x',
        'ISM330DHCX_GYRO'
    ]
    
    for sensor in test_sensors:
        mapped = ml_interface._map_sensor_to_column(sensor)
        exists = mapped in ml_interface.mock_data.columns
        status = "✓" if exists else "✗"
        print(f"  {status} '{sensor}' -> '{mapped}' (exists: {exists})")
    
    # Test 5: Basic statistics tests
    print("\n5. BASIC STATISTICS TEST")
    print("-" * 40)
    
    # Test with actual column names from dataset
    test_cases = [
        ('mean', 'HTS221_TEMP_TEMP_mean'),
        ('std', 'HTS221_TEMP_TEMP_mean'),
        ('mean', 'HTS221_HUM_HUM_mean'),
        ('mean', 'LPS22HH_PRESS_PRESS_mean'),
        ('mean', 'IIS3DWB_ACC_A_x_mean'),
        ('mean', 'ISM330DHCX_GYRO_G_x_mean'),
    ]
    
    # Also test with mapped names
    mapped_test_cases = [
        ('mean', 'temperature'),
        ('std', 'temperature'),
        ('mean', 'humidity'),
        ('mean', 'pressure'),
    ]
    
    all_test_cases = test_cases + mapped_test_cases
    
    for stat_type, column in all_test_cases:
        print(f"\nTesting {stat_type} for {column}:")
        result = ml_interface.get_statistic(stat=stat_type, column=column)
        
        if result['status'] == 'success':
            print(f"  ✓ Success: {result['result']:.4f}")
            print(f"    - Mapped to: {result.get('mapped_column', 'N/A')}")
            print(f"    - Sample count: {result.get('sample_count', 'N/A')}")
        else:
            print(f"  ✗ Error: {result['message']}")
    
    # Test 6: Grouped statistics by class
    print("\n6. GROUPED STATISTICS TEST")
    print("-" * 40)
    
    print("Testing mean temperature by class:")
    result = ml_interface.get_statistic(stat='mean', column='HTS221_TEMP_TEMP_mean', group_by='class')
    
    if result['status'] == 'success':
        print("  ✓ Success:")
        for class_name, value in result['result'].items():
            print(f"    - {class_name}: {value:.4f}")
    else:
        print(f"  ✗ Error: {result['message']}")
    
    # Test 7: Filtered statistics
    print("\n7. FILTERED STATISTICS TEST")
    print("-" * 40)
    
    print("Testing mean temperature for OK and KO classes only:")
    result = ml_interface.get_statistic(
        stat='mean', 
        column='HTS221_TEMP_TEMP_mean', 
        filters={'class': ['OK', 'KO']},
        group_by='class'
    )
    
    if result['status'] == 'success':
        print("  ✓ Success:")
        for class_name, value in result['result'].items():
            print(f"    - {class_name}: {value:.4f}")
    else:
        print(f"  ✗ Error: {result['message']}")
    
    # Test 8: Top features analysis
    print("\n8. TOP FEATURES ANALYSIS TEST")
    print("-" * 40)
    
    print("Getting top 5 discriminative features:")
    result = ml_interface.get_top_features(n=5, classes=['OK', 'KO'])
    
    if result['status'] == 'success':
        print("  ✓ Success:")
        for i, feature in enumerate(result['top_features'], 1):
            details = result['feature_details'][feature]
            print(f"    {i}. {feature}")
            print(f"       Discriminative score: {details['discriminative_score']:.4f}")
            print(f"       Analysis method: {details['analysis_method']}")
            print(f"       Class means: {details['class_means']}")
            print(f"       Sample count: {details['sample_count']}")
    else:
        print(f"  ✗ Error: {result['message']}")
    
    # Test 9: Dataset overview
    print("\n9. DATASET OVERVIEW TEST")
    print("-" * 40)
    
    overview = ml_interface.get_dataset_overview()
    if overview['status'] == 'success':
        print("  ✓ Success:")
        print(f"    - Total samples: {overview['total_samples']}")
        print(f"    - Classes: {overview['classes']}")
        # print(f"    - Number of features: {len(overview['features'])}")
    else:
        print(f"  ✗ Error: {overview.get('message', 'Unknown error')}")
    
    # Test 10: Data quality assessment
    print("\n10. DATA QUALITY ASSESSMENT TEST")
    print("-" * 40)
    
    quality_summary = ml_interface.get_data_quality_summary()
    if quality_summary['status'] == 'success':
        summary = quality_summary['data_quality_summary']
        print("  ✓ Success:")
        print(f"    - Overall quality: {summary['overall_quality']}")
        print(f"    - Quality score: {summary['quality_score']}/100")
        print(f"    - Total samples: {summary['total_samples']}")
        print(f"    - Total features: {summary['total_features']}")
        print(f"    - Missing data: {summary['missing_data_percentage']:.2f}%")
        if summary['quality_issues']:
            print("    - Issues:")
            for issue in summary['quality_issues']:
                print(f"      • {issue}")
    else:
        print(f"  ✗ Error: {quality_summary['message']}")
    
    # Test 11: Error handling tests
    print("\n11. ERROR HANDLING TEST")
    print("-" * 40)
    
    # Test with non-existent column
    print("Testing with non-existent column:")
    result = ml_interface.get_statistic(stat='mean', column='NONEXISTENT_SENSOR')
    if result['status'] == 'error':
        print(f"  ✓ Properly handled error: {result['message']}")
    else:
        print(f"  ✗ Should have failed but got: {result}")
    
    # Test with invalid statistic
    print("\nTesting with invalid statistic:")
    result = ml_interface.get_statistic(stat='invalid_stat', column='HTS221_TEMP_TEMP_mean')
    if result['status'] == 'error':
        print(f"  ✓ Properly handled error: {result['message']}")
    else:
        print(f"  ✗ Should have failed but got: {result}")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETED")
    print("=" * 80)
    
    # Summary of key findings
    print(f"\nKEY FINDINGS:")
    print(f"- Dataset shape: {ml_interface.mock_data.shape if ml_interface.mock_data is not None else 'N/A'}")
    print(f"- Available sensors: {len(ml_interface.available_sensors)}")
    print(f"- Available features: {len(ml_interface.get_available_features())}")
    print(f"- Classes in dataset: {list(ml_interface.mock_data['label'].unique()) if ml_interface.mock_data is not None else 'N/A'}")



    # Test feature - using temperature as it's commonly available
    test_feature = 'HTS221_TEMP_TEMP_mean'
    
    print(f"\n1. TESTING FEATURE: {test_feature}")
    print("-" * 50)
    
    # Run the function
    result = ml_interface.get_feature_analysis(test_feature)
    
    # Display results
    if result['status'] == 'success':
        print("✓ Function executed successfully")
        print(f"Feature: {result['feature']}")
        print(f"Mapped column: {result['mapped_column']}")
        print(f"Total samples: {result['total_samples']}")
        
        print("\nClass Analysis Results:")
        for class_name, stats in result['analysis'].items():
            print(f"\n{class_name} Class:")
            if stats['count'] > 0:
                print(f"  - Count: {stats['count']}")
                print(f"  - Mean: {stats['mean']:.6f}")
                print(f"  - Std: {stats['std']:.6f}")
                print(f"  - Min: {stats['min']:.6f}")
                print(f"  - Max: {stats['max']:.6f}")
                print(f"  - Median: {stats['median']:.6f}")
                print(f"  - Variance: {stats['variance']:.6f}")
            else:
                print(f"  - No data for this class")
        
        if result.get('discriminative_score'):
            print(f"\nDiscriminative Score: {result['discriminative_score']:.6f}")
            print(f"Relative Difference: {result['relative_difference']:.6f}")
        
    else:
        print("✗ Function failed:")
        print(f"Error: {result['message']}")
    

    test_dataset_overview()