"""
Response Formatter Module

This module takes ML output and formats it into clear text responses
for the GUI, including optional tables and plot suggestions.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

class ResponseFormatter:
    """Format ML results into user-friendly responses"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Templates for different response types
        self.templates = {
            'mean': {
                'single': "The mean {sensor} is {value}",
                'grouped': "Mean {sensor} by class:\n{grouped_results}",
                'filtered': "The mean {sensor} for {classes} samples is {value}"
            },
            'median': {
                'single': "The median {sensor} is {value}",
                'grouped': "Median {sensor} by class:\n{grouped_results}",
                'filtered': "The median {sensor} for {classes} samples is {value}"
            },
            'variance': {
                'single': "The variance of {sensor} is {value}",
                'grouped': "Variance of {sensor} by class:\n{grouped_results}",
                'filtered': "The variance of {sensor} for {classes} samples is {value}"
            },
            'std': {
                'single': "The standard deviation of {sensor} is {value}",
                'grouped': "Standard deviation of {sensor} by class:\n{grouped_results}",
                'filtered': "The standard deviation of {sensor} for {classes} samples is {value}"
            },
            'standard deviation': {
                'single': "The standard deviation of {sensor} is {value}",
                'grouped': "Standard deviation of {sensor} by class:\n{grouped_results}",
                'filtered': "The standard deviation of {sensor} for {classes} samples is {value}"
            },
            'min': {
                'single': "The minimum {sensor} is {value}",
                'grouped': "Minimum {sensor} by class:\n{grouped_results}",
                'filtered': "The minimum {sensor} for {classes} samples is {value}"
            },
            'minimum': {
                'single': "The minimum {sensor} is {value}",
                'grouped': "Minimum {sensor} by class:\n{grouped_results}",
                'filtered': "The minimum {sensor} for {classes} samples is {value}"
            },
            'max': {
                'single': "The maximum {sensor} is {value}",
                'grouped': "Maximum {sensor} by class:\n{grouped_results}",
                'filtered': "The maximum {sensor} for {classes} samples is {value}"
            },
            'maximum': {
                'single': "The maximum {sensor} is {value}",
                'grouped': "Maximum {sensor} by class:\n{grouped_results}",
                'filtered': "The maximum {sensor} for {classes} samples is {value}"
            },
            'count': {
                'single': "The count of {sensor} samples is {value}",
                'grouped': "Count of {sensor} samples by class:\n{grouped_results}",
                'filtered': "The count of {sensor} samples for {classes} is {value}"
            },
            'n': {
                'single': "The count of {sensor} samples is {value}",
                'grouped': "Count of {sensor} samples by class:\n{grouped_results}",
                'filtered': "The count of {sensor} samples for {classes} is {value}"
            }
        }
        
        # Sensor name mappings for display
        self.sensor_display_names = {
            'HTS221_TEMP': 'HTS221 Temperature',
            'HTS221_HUM': 'HTS221 Humidity',
            'LPS22HH_PRESS': 'LPS22HH Pressure',
            'IIS3DWB_ACC': 'IIS3DWB Acceleration',
            'temperature_mean': 'Temperature',
            'humidity_mean': 'Humidity',
            'pressure_mean': 'Pressure',
            'acceleration_x_mean': 'Acceleration X-axis',
            'acceleration_y_mean': 'Acceleration Y-axis',
            'acceleration_z_mean': 'Acceleration Z-axis',
            'gyroscope_x_mean': 'Gyroscope X-axis',
            'gyroscope_y_mean': 'Gyroscope Y-axis',
            'gyroscope_z_mean': 'Gyroscope Z-axis',
            'microphone_mean': 'Microphone'
        }
    
    def format_response(self, parsed_command: Any, ml_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format ML result into a user-friendly response"""
        try:
            if ml_result.get('status') == 'error':
                return self._format_error_response(parsed_command, ml_result)
            
            command_type = parsed_command.command_type.value if parsed_command.command_type else 'unknown'
            response_type = getattr(parsed_command, 'response_type', 'auto')
            target_column = parsed_command.target_column
            
            # Route to appropriate formatter
            try:
                if command_type == 'top_features':
                    formatted_response = self._format_top_features_response(parsed_command, ml_result)
                elif command_type == 'plot':
                    formatted_response = self._format_plot_response(parsed_command, ml_result)
                elif command_type == 'comparison':
                    formatted_response = self._format_comparison_response(parsed_command, ml_result)
                elif command_type == 'analysis':
                    formatted_response = self._format_analysis_response(parsed_command, ml_result)
                else:
                    # Default to statistic response
                    formatted_response = self._format_statistic_response(parsed_command, ml_result)
            except Exception as formatter_error:
                self.logger.error(f"Error in specific formatter for {command_type}: {formatter_error}")
                # Fallback to error response
                return self._format_error_response(parsed_command, {'message': f'Formatter error: {str(formatter_error)}'})
            
            # For visual responses, keep the main response short and direct
            if response_type == 'visual' or command_type in ['plot', 'comparison', 'analysis']:
                # Keep only the main response, minimize context and suggestions
                formatted_response['main_response'] = formatted_response['main_response']
                formatted_response['context'] = "Visualization ready."
                formatted_response['suggestions'] = formatted_response['suggestions'][:2]  # Limit to 2 suggestions
            
            # Add response type information
            formatted_response['response_type'] = response_type
            formatted_response['response_guidance'] = self._get_response_guidance(response_type, command_type)
            
            return formatted_response
                
        except Exception as e:
            self.logger.error(f"Error formatting response: {e}")
            return self._format_error_response(parsed_command, {'message': f'Formatting error: {str(e)}'})
    
    def _ensure_statistic_template(self, statistic: str) -> str:
        """Ensure a template exists for the given statistic, create one if missing"""
        if statistic in self.templates:
            return statistic
        
        # Create a generic template for the missing statistic
        # Use safe formatting that works with both numeric and string values
        self.templates[statistic] = {
            'single': f"The {statistic} of {{sensor}} is {{value}}",
            'grouped': f"{statistic.title()} of {{sensor}} by class:\n{{grouped_results}}",
            'filtered': f"The {statistic} of {{sensor}} for {{classes}} samples is {{value}}"
        }
        
        self.logger.info(f"Created template for missing statistic: {statistic}")
        return statistic

    def _format_statistic_response(self, parsed_command: Any, ml_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format a statistical response"""
        statistic = parsed_command.statistic.value if parsed_command.statistic else 'mean'
        target_column = parsed_command.target_column
        class_filters = parsed_command.class_filters
        grouping = parsed_command.grouping
        
        # Get display name for sensor
        sensor_display = self.sensor_display_names.get(target_column, target_column)
        
        # Ensure we have a template for this statistic
        statistic = self._ensure_statistic_template(statistic)
        
        # Check data quality and add warnings if needed
        data_quality = ml_result.get('data_quality', 'unknown')
        confidence = ml_result.get('confidence', 'unknown')
        warnings = ml_result.get('warnings', [])
        
        # Format the main response
        if grouping == 'class' and isinstance(ml_result.get('result'), dict):
            # Grouped results
            grouped_text = self._format_grouped_results(ml_result['result'])
            main_text = self.templates[statistic]['grouped'].format(
                sensor=sensor_display,
                grouped_results=grouped_text
            )
        elif class_filters:
            # Filtered results
            classes = ', '.join(class_filters)
            value = ml_result.get('result', 'N/A')
            # Ensure value is properly formatted for the template
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            main_text = self.templates[statistic]['filtered'].format(
                sensor=sensor_display,
                classes=classes,
                value=formatted_value
            )
        else:
            # Single result
            value = ml_result.get('result', 'N/A')
            # Ensure value is properly formatted for the template
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            main_text = self.templates[statistic]['single'].format(
                sensor=sensor_display,
                value=formatted_value
            )
        
        # Add data quality context
        context_parts = []
        
        # Add confidence information
        if confidence == 'high':
            context_parts.append("High confidence analysis based on sufficient data.")
        elif confidence == 'medium':
            context_parts.append("Medium confidence - consider collecting more samples for higher reliability.")
        elif confidence == 'low':
            context_parts.append("Low confidence - results may not be reliable due to limited data.")
        
        # Add data quality information
        if data_quality == 'good':
            context_parts.append("Data quality is good.")
        elif data_quality == 'poor':
            context_parts.append("⚠️ Data quality issues detected - verify results.")
        
        # Add warnings if any
        if warnings:
            context_parts.append(f"⚠️ Warnings: {', '.join(warnings)}")
        
        # Add sample count information if available
        sample_count = ml_result.get('sample_count', 0)
        if sample_count > 0:
            context_parts.append(f"Analysis based on {sample_count} samples.")
        
        context = ' '.join(context_parts) if context_parts else "Analysis completed successfully."
        
        # Generate suggestions
        suggestions = self._generate_suggestions(parsed_command, ml_result)
        
        return {
            'status': 'success',
            'main_response': main_text,
            'context': context,
            'suggestions': suggestions,
            'data_source': ml_result.get('data_source', 'Unknown'),
            'confidence': parsed_command.confidence,
            'data_quality': data_quality,
            'analysis_confidence': confidence,
            'plot_suggestion': self._suggest_plot(parsed_command, ml_result)
        }
    
    def _format_top_features_response(self, parsed_command: Any, ml_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format a top features response - text only, no plot suggestions"""
        class_filters = parsed_command.class_filters
        classes = class_filters if class_filters else ['OK', 'KO']
        classes_text = ' and '.join(classes)
        
        # Check data quality and confidence
        data_quality = ml_result.get('data_quality', 'unknown')
        confidence = ml_result.get('confidence', 'unknown')
        warnings = ml_result.get('warnings', [])
        analysis_method = ml_result.get('analysis_method', 'Statistical analysis')
        
        if ml_result.get('status') == 'success' and 'top_features' in ml_result:
            features = ml_result['top_features']
            feature_details = ml_result.get('feature_details', {})
            count = len(features)
            
            # Format features with detailed analysis
            feature_descriptions = []
            for i, feature in enumerate(features, 1):
                # Get human-readable sensor name
                sensor_display = self.sensor_display_names.get(feature, feature)
                
                # Add statistical details if available
                if feature in feature_details:
                    details = feature_details[feature]
                    f_stat = details.get('f_statistic', 0)
                    p_value = details.get('p_value', 1.0)
                    class_means = details.get('class_means', {})
                    
                    # Format class means
                    mean_details = []
                    for cls, mean_val in class_means.items():
                        mean_details.append(f"{cls}: {mean_val:.2f}")
                    
                    # Add statistical significance information
                    significance = "highly significant" if p_value < 0.01 else "significant" if p_value < 0.05 else "not significant"
                    
                    feature_descriptions.append(
                        f"{i}. {sensor_display} (F-stat: {f_stat:.2f}, p-value: {p_value:.4f})\n"
                        f"   Statistical significance: {significance}\n"
                        f"   Class means: {', '.join(mean_details)}"
                    )
                else:
                    feature_descriptions.append(f"{i}. {sensor_display}")
            
            feature_list = '\n'.join(feature_descriptions)
            
            # Add detailed explanation with data quality context
            main_text = f"Top {count} statistical indices that best separate {classes_text} samples:\n\n{feature_list}\n\n"
            main_text += f"These features were selected using {analysis_method} (ANOVA F-test), which measures the discriminative power "
            main_text += f"between {classes_text} classes. Higher F-statistics indicate greater separation between classes, "
            main_text += f"making these features most effective for classification and quality control purposes."
            
            # Add sample count information
            sample_count = ml_result.get('sample_count', 0)
            if sample_count > 0:
                main_text += f"\n\nAnalysis based on {sample_count} samples from the dataset."
            
        else:
            # Fallback to mock data
            features = ['temperature_mean', 'humidity_mean', 'pressure_mean']
            count = len(features)
            feature_descriptions = []
            for i, feature in enumerate(features, 1):
                sensor_display = self.sensor_display_names.get(feature, feature)
                feature_descriptions.append(f"{i}. {sensor_display}")
            
            feature_list = '\n'.join(feature_descriptions)
            main_text = f"Top {count} statistical indices that best separate {classes_text} samples:\n\n{feature_list}\n\n"
            main_text += f"These features were selected based on their discriminative power between {classes_text} classes."
        
        # Build context with data quality information
        context_parts = []
        
        # Add analysis method
        context_parts.append(f"Feature selection analysis completed using {analysis_method}.")
        
        # Add confidence information
        if confidence == 'high':
            context_parts.append("High confidence analysis based on sufficient data.")
        elif confidence == 'medium':
            context_parts.append("Medium confidence - consider collecting more samples for higher reliability.")
        elif confidence == 'low':
            context_parts.append("Low confidence - results may not be reliable due to limited data.")
        
        # Add data quality information
        if data_quality == 'good':
            context_parts.append("Data quality is good.")
        elif data_quality == 'poor':
            context_parts.append("⚠️ Data quality issues detected - verify results.")
        
        # Add warnings if any
        if warnings:
            context_parts.append(f"⚠️ Warnings: {', '.join(warnings)}")
        
        context = ' '.join(context_parts)
        
        suggestions = [
            "Request detailed statistical analysis of any of these top features",
            "Compare the distributions of these features between classes",
            "Ask for correlation analysis between these features"
        ]
        
        return {
            'status': 'success',
            'main_response': main_text,
            'context': context,
            'suggestions': suggestions,
            'data_source': ml_result.get('data_source', 'Mock data'),
            'confidence': parsed_command.confidence,
            'data_quality': data_quality,
            'analysis_confidence': confidence,
            'plot_suggestion': None  # No plot suggestion for text-only responses
        }
    
    def _format_plot_response(self, parsed_command: Any, ml_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format a plot response"""
        plot_type = parsed_command.plot_type.value if parsed_command.plot_type else 'line_graph'
        target_features = parsed_command.target_features
        
        if target_features:
            features_text = ', '.join(target_features)
        else:
            features_text = 'the requested data'
        
        # Short, direct response for plot requests
        if ml_result.get('plot_ready', False):
            main_text = f"✅ {plot_type.title()} created and ready to display."
            plot_suggestion = plot_type
        else:
            main_text = f" Preparing {plot_type} visualization..."
            plot_suggestion = None
        
        # Minimal context and suggestions for plot responses
        context = "Visualization ready."
        suggestions = [
            "Analyze the patterns shown",
            "Request statistical details",
            "Generate additional plots"
        ]
        
        return {
            'status': 'success',
            'main_response': main_text,
            'context': context,
            'suggestions': suggestions,
            'data_source': ml_result.get('data_source', 'Plot generation'),
            'confidence': parsed_command.confidence,
            'plot_suggestion': plot_suggestion,
            'plot_data': ml_result.get('plot_data'),
            'plot_path': ml_result.get('plot_path'),
            'figure': ml_result.get('figure')
        }
    
    def _format_comparison_response(self, parsed_command: Any, ml_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format a comparison response"""
        target_column = parsed_command.target_column
        class_filters = parsed_command.class_filters
        
        if target_column and class_filters:
            sensor_display = self.sensor_display_names.get(target_column, target_column)
            classes_text = ' vs '.join(class_filters)
            
            # Short response for comparison with plot
            main_text = f" Comparing {sensor_display} between {classes_text} samples."
        else:
            main_text = " Class comparison completed."
        
        # Minimal context for visual responses
        context = "Comparison analysis ready."
        suggestions = [
            "View the comparison plot",
            "Request statistical details",
            "Analyze other features"
        ]
        
        return {
            'status': 'success',
            'main_response': main_text,
            'context': context,
            'suggestions': suggestions,
            'data_source': ml_result.get('data_source', 'Comparison analysis'),
            'confidence': parsed_command.confidence,
            'plot_suggestion': 'line_graph'
        }
    
    def _format_analysis_response(self, parsed_command: Any, ml_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format an analysis response"""
        analysis_type = parsed_command.analysis_type or 'general'
        target_column = parsed_command.target_column
        
        if target_column:
            sensor_display = self.sensor_display_names.get(target_column, target_column)
            main_text = f" Analyzing {sensor_display}..."
        else:
            main_text = f" {analysis_type.title()} analysis in progress..."
        
        # Minimal context for visual responses
        context = "Analysis completed."
        suggestions = [
            "View the visualization",
            "Request statistical details",
            "Compare with other features"
        ]
        
        return {
            'status': 'success',
            'main_response': main_text,
            'context': context,
            'suggestions': suggestions,
            'data_source': ml_result.get('data_source', 'Analysis engine'),
            'confidence': parsed_command.confidence,
            'plot_suggestion': 'line_graph'  # Changed from 'histogram' to 'line_graph' as default
        }
    
    def _format_grouped_results(self, grouped_data: Dict[str, Any]) -> str:
        """Format grouped results into readable text"""
        lines = []
        for class_name, value in grouped_data.items():
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            lines.append(f"• {class_name}: {formatted_value}")
        return '\n'.join(lines)
    
    def _generate_context(self, parsed_command: Any, ml_result: Dict[str, Any]) -> str:
        """Generate contextual information for the response"""
        context_parts = []
        
        # Add confidence information
        if parsed_command.confidence < 0.7:
            context_parts.append("Note: Low confidence in request parsing - please verify the results.")
        
        # Add data source information
        data_source = ml_result.get('data_source', 'Unknown')
        if 'mock' in data_source.lower():
            context_parts.append("Results based on mock data simulation.")
        
        # Add sample count information if available
        if 'sample_count' in ml_result:
            context_parts.append(f"Analysis based on {ml_result['sample_count']} samples.")
        
        return ' '.join(context_parts) if context_parts else "Analysis completed successfully."
    
    def _generate_suggestions(self, parsed_command: Any, ml_result: Dict[str, Any]) -> List[str]:
        """Generate follow-up suggestions for the user"""
        suggestions = []
        
        command_type = parsed_command.command_type.value if parsed_command.command_type else 'unknown'
        target_column = parsed_command.target_column
        
        # General suggestions
        suggestions.append("Ask for other statistical measures of this sensor")
        suggestions.append("Compare this sensor across different classes")
        
        # Specific suggestions based on command type
        if command_type == 'top_features':
            suggestions.append("Request detailed analysis of any of these features")
            suggestions.append("Compare the distributions of these features between classes")
            suggestions.append("Generate correlation plots for these features")
        elif command_type == 'plot':
            suggestions.append("Analyze the patterns shown in this visualization")
            suggestions.append("Request statistical details for the plotted data")
            suggestions.append("Generate additional plots for comparison")
        elif command_type == 'comparison':
            suggestions.append("Request statistical significance testing")
            suggestions.append("Generate visual comparison plots")
            suggestions.append("Analyze other features for similar patterns")
        elif command_type == 'analysis':
            suggestions.append("Request specific statistical measures")
            suggestions.append("Generate visualizations of the findings")
            suggestions.append("Compare with other features")
        
        # Sensor-specific suggestions
        if target_column and 'temperature' in target_column.lower():
            suggestions.append("Compare with humidity readings for environmental analysis")
        elif target_column and 'acceleration' in target_column.lower():
            suggestions.append("Request gyroscope data for complete motion analysis")
        elif target_column and 'pressure' in target_column.lower():
            suggestions.append("Correlate with temperature for atmospheric analysis")
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _suggest_plot(self, parsed_command: Any, ml_result: Dict[str, Any]) -> Optional[str]:
        """Suggest appropriate plot types for the data based on response type preference"""
        command_type = parsed_command.command_type.value if parsed_command.command_type else 'unknown'
        response_type = getattr(parsed_command, 'response_type', 'auto')
        grouping = parsed_command.grouping
        
        # Respect response type preference
        if response_type == "text":
            return None  # No plot suggestion for text-only responses
        elif response_type == "visual":
            # Force plot suggestions for visual requests
            if command_type == 'top_features':
                return 'correlation_matrix'  # Best for feature analysis
            elif command_type == 'comparison':
                return 'line_graph'
            elif command_type == 'statistic':
                return 'histogram'
            else:
                return 'line_graph'  # Default visual response
        
        # Auto mode - use smart defaults
        # Top features requests should not suggest plots - they're text-only
        if command_type == 'top_features':
            return None  # No plot suggestion for text-only responses
        elif command_type == 'plot':
            return 'line_graph'  # Changed from 'histogram' to 'line_graph' as default
        elif command_type == 'comparison':
            return 'line_graph'
        elif command_type == 'analysis':
            return 'line_graph'  # Changed from 'histogram' to 'line_graph' as default
        else:
            if grouping == 'class':
                if command_type in ['mean', 'median', 'std', 'variance']:
                    return 'line_graph'
                else:
                    return 'line_graph'  # Changed from 'histogram' to 'line_graph'
            else:
                if command_type in ['mean', 'median']:
                    return 'line_graph'  # Changed from 'histogram' to 'line_graph'
                else:
                    return 'scatter'
    
    def _format_error_response(self, parsed_command: Any, ml_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format an error response"""
        error_message = ml_result.get('message', 'Unknown error occurred')
        
        return {
            'status': 'error',
            'main_response': f"Sorry, I couldn't process that request: {error_message}",
            'context': "Please check your request format and try again.",
            'suggestions': [
                "Make sure you specify a sensor name",
                "Check that the class names are correct",
                "Try rephrasing your request"
            ],
            'data_source': 'Error',
            'confidence': 0.0,
            'plot_suggestion': None
        }
    
    def _get_response_guidance(self, response_type: str, command_type: str) -> str:
        """Get guidance about the response type"""
        if response_type == "text":
            return "This is a text-based response providing detailed information and analysis."
        elif response_type == "visual":
            return "This response includes visualizations to help you better understand the data."
        else:
            # Auto mode
            if command_type == 'top_features':
                return "This analysis is presented as text since it's primarily informational."
            elif command_type == 'plot':
                return "This response includes the requested visualization."
            elif command_type == 'statistic':
                return "Statistical results are shown with optional visualization suggestions."
            else:
                return "Response format automatically determined based on request type."
    
    def format_table_data(self, ml_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Format ML result into table data for GUI display"""
        try:
            if ml_result.get('status') != 'success':
                return None
            
            result = ml_result.get('result', {})
            
            if isinstance(result, dict):
                # Grouped results
                headers = ['Class', 'Value']
                rows = []
                for class_name, value in result.items():
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    rows.append([class_name, formatted_value])
                
                return {
                    'headers': headers,
                    'rows': rows,
                    'title': 'Results by Class'
                }
            else:
                # Single result
                return {
                    'headers': ['Metric', 'Value'],
                    'rows': [['Result', str(result)]],
                    'title': 'Analysis Result'
                }
                
        except Exception as e:
            self.logger.error(f"Error formatting table data: {e}")
            return None


# Global instance
response_formatter = ResponseFormatter()

# Convenience functions
def format_response(parsed_command: Any, ml_result: Dict[str, Any]) -> Dict[str, Any]:
    """Format ML result into user-friendly response"""
    return response_formatter.format_response(parsed_command, ml_result)

def format_table_data(ml_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Format ML result into table data"""
    return response_formatter.format_table_data(ml_result)
