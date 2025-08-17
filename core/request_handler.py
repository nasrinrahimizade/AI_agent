"""
Request Handler Module

This module integrates the unified parser, ML interface, and response formatter
to handle the complete flow: User input → parse → route to ML → format → send back to GUI.
"""

import logging
from typing import Dict, List, Any, Optional
from .unified_parser import parse_command, UnifiedParsedCommand
from .ml_interface import ml_interface
from .response_formatter import format_response, format_table_data

class RequestHandler:
    """Main handler for processing user requests through the ML pipeline"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def handle_request(self, user_request: str) -> Dict[str, Any]:
        """
        Main method to handle a user request through the complete pipeline
        """
        try:
            # 1. Parse the request using unified parser
            parsed_command = parse_command(user_request)
            
            # 2. Route to appropriate ML function
            ml_result = self._route_request(parsed_command)
            
            # 3. Format the response
            formatted_response = format_response(parsed_command, ml_result)
            
            # 4. Add metadata
            formatted_response['parsed_command'] = {
                'command_type': parsed_command.command_type.value if parsed_command.command_type else 'unknown',
                'response_type': getattr(parsed_command, 'response_type', 'auto'),
                'target_column': parsed_command.target_column,
                'class_filters': parsed_command.class_filters,
                'grouping': parsed_command.grouping,
                'confidence': parsed_command.confidence
            }
            
            # 5. Add table data if available
            table_data = format_table_data(ml_result)
            if table_data:
                formatted_response['table_data'] = table_data
            
            return formatted_response
            
        except Exception as e:
            self.logger.error(f"Error handling request: {e}")
            return {
                'status': 'error',
                'main_response': f"Sorry, I encountered an error processing your request: {str(e)}",
                'context': "Please try rephrasing your request or contact support if the problem persists.",
                'suggestions': [
                    "Check that your request includes a sensor name",
                    "Make sure class names are spelled correctly",
                    "Try using simpler language"
                ],
                'data_source': 'Error',
                'confidence': 0.0,
                'plot_suggestion': None,
                'response_type': 'text',
                'response_guidance': 'Error response provided as text.'
            }
    
    def _route_request(self, parsed_command: UnifiedParsedCommand) -> Dict[str, Any]:
        """
        Route the parsed command to the appropriate ML function
        
        Args:
            parsed_command: Parsed command object
            
        Returns:
            Result from ML interface
        """
        try:
            command_type = parsed_command.command_type
            target_column = parsed_command.target_column
            class_filters = parsed_command.class_filters
            grouping = parsed_command.grouping
            
            # Handle different command types
            if command_type.value == 'top_features':
                # Handle top features request
                classes = class_filters if class_filters else ['OK', 'KO']
                count = parsed_command.filters.get('count', 3)  # Default to 3
                
                return ml_interface.get_top_features(n=count, classes=classes)
            
            elif command_type.value == 'statistic':
                # Handle statistical requests
                stat_type = parsed_command.statistic.value if parsed_command.statistic else 'mean'
                filters = {'class': class_filters} if class_filters else {}
                
                return ml_interface.get_statistic(
                    stat=stat_type,
                    column=target_column,
                    filters=filters,
                    group_by=grouping
                )
            
            elif command_type.value == 'plot':
                # Handle plot requests
                plot_type = parsed_command.plot_type.value if parsed_command.plot_type else 'line_graph'
                features = parsed_command.target_features if parsed_command.target_features else [target_column]
                
                # First get the plot data
                plot_data_result = ml_interface.get_plot_data(
                    plot_type=plot_type,
                    features=features,
                    filters={'class': class_filters} if class_filters else {}
                )
                
                if plot_data_result['status'] == 'success':
                    # Now create the actual plot
                    plot_creation_result = ml_interface.create_plot_from_data(
                        plot_type=plot_type,
                        plot_data=plot_data_result['plot_data']
                    )
                    
                    if plot_creation_result['status'] == 'success':
                        # Combine both results
                        return {
                            'status': 'success',
                            'plot_type': plot_type,
                            'plot_data': plot_data_result['plot_data'],
                            'plot_path': plot_creation_result['plot_path'],
                            'figure': plot_creation_result.get('figure'),
                            'features': features,
                            'filters': {'class': class_filters} if class_filters else {},
                            'sample_count': plot_data_result['sample_count'],
                            'data_source': 'Plot creation with mock data',
                            'plot_ready': True
                        }
                    else:
                        return plot_creation_result
                else:
                    return plot_data_result
            
            elif command_type.value == 'comparison':
                # Handle comparison requests
                if target_column:
                    return ml_interface.get_class_comparison(
                        feature=target_column,
                        classes=class_filters
                    )
                else:
                    return {
                        'status': 'error',
                        'message': 'No target feature specified for comparison'
                    }
            
            elif command_type.value == 'analysis':
                # Handle analysis requests
                if target_column:
                    return ml_interface.get_feature_analysis(feature=target_column)
                else:
                    return ml_interface.get_dataset_overview()
            
            else:
                # Default to statistic if command type is unknown
                filters = {'class': class_filters} if class_filters else {}
                return ml_interface.get_statistic(
                    stat='mean',
                    column=target_column or 'temperature_mean',
                    filters=filters,
                    group_by=grouping
                )
                
        except Exception as e:
            self.logger.error(f"Error routing request: {e}")
            return {
                'status': 'error',
                'message': f'Error routing request: {str(e)}'
            }
    
    def get_available_capabilities(self) -> Dict[str, Any]:
        """Get information about available capabilities for the GUI"""
        try:
            from .unified_parser import get_available_sensors, get_available_classes, get_available_stats, get_available_plots, get_available_commands
            
            # Get available sensors and classes
            available_sensors = get_available_sensors()
            available_classes = get_available_classes()
            available_stats = get_available_stats()
            available_plots = get_available_plots()
            available_commands = get_available_commands()
            
            # Get plotting capabilities
            plotting_capabilities = ml_interface.get_plotting_capabilities()
            
            return {
                'status': 'success',
                'capabilities': {
                    'command_types': available_commands,
                    'statistical_measures': available_stats,
                    'plot_types': available_plots,
                    'available_sensors': available_sensors,
                    'available_classes': available_classes,
                    'plotting_capabilities': plotting_capabilities.get('capabilities', {}),
                    'example_requests': [
                        "What is the mean temperature for OK samples from HTS221_TEMP?",
                        "Calculate the median humidity for KO_HIGH_2mm samples.",
                        "Show the variance of acceleration in IIS3DWB_ACC for OK and KO.",
                        "Give me the standard deviation of pressure from LPS22HH_PRESS for each class.",
                        "List the top 3 statistical indices that best separate OK and KO samples.",
                        "Create a line graph comparing temperature across all classes",
                        "Show me a histogram of humidity data",
                        "Generate a correlation matrix between sensors"
                    ]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting capabilities: {e}")
            return {
                'status': 'error',
                'message': f'Error getting capabilities: {str(e)}',
                'capabilities': {}
            }
    
    def validate_request(self, user_request: str) -> Dict[str, Any]:
        """
        Validate a user request without processing it
        
        Args:
            user_request: Natural language request from user
            
        Returns:
            Validation result with suggestions
        """
        try:
            # Parse the request using unified parser
            parsed_command = parse_command(user_request)
            
            validation_result = {
                'is_valid': parsed_command.confidence > 0.5,
                'confidence': parsed_command.confidence,
                'parsed_components': {
                    'command_type': parsed_command.command_type.value if parsed_command.command_type else 'unknown',
                    'statistic': parsed_command.statistic.value if parsed_command.statistic else None,
                    'target_column': parsed_command.target_column,
                    'plot_type': parsed_command.plot_type.value if parsed_command.plot_type else None,
                    'target_features': parsed_command.target_features,
                    'class_filters': parsed_command.class_filters,
                    'grouping': parsed_command.grouping
                },
                'suggestions': []
            }
            
            # Generate suggestions based on parsing confidence
            if parsed_command.confidence < 0.7:
                validation_result['suggestions'].append("Consider being more specific about the sensor name")
                validation_result['suggestions'].append("Make sure to specify the statistical measure you want")
                validation_result['suggestions'].append("Check that class names are spelled correctly")
            
            if not parsed_command.target_features and not parsed_command.target_column:
                validation_result['suggestions'].append("Please specify which sensor you want to analyze")
            
            if parsed_command.command_type.value == 'statistic' and not parsed_command.statistic:
                validation_result['suggestions'].append("Please specify what statistic you want (mean, median, variance, etc.)")
            
            if parsed_command.command_type.value == 'plot' and not parsed_command.plot_type:
                validation_result['suggestions'].append("Please specify what type of plot you want (line graph, histogram, etc.)")
            
            # Add specific suggestions based on parsed components
            if parsed_command.target_column and 'HTS221' in parsed_command.target_column:
                validation_result['suggestions'].append("HTS221 sensor detected - you can ask for temperature or humidity data")
            
            if parsed_command.target_column and 'LPS22HH' in parsed_command.target_column:
                validation_result['suggestions'].append("LPS22HH pressure sensor detected - you can ask for pressure statistics")
            
            if parsed_command.target_column and 'IIS3DWB' in parsed_command.target_column:
                validation_result['suggestions'].append("IIS3DWB accelerometer detected - you can specify X, Y, or Z axis")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating request: {e}")
            return {
                'is_valid': False,
                'confidence': 0.0,
                'error': str(e),
                'suggestions': ["Please try rephrasing your request"]
            }


# Global instance
request_handler = RequestHandler()

# Convenience functions
def handle_request(user_request: str) -> Dict[str, Any]:
    """Handle a user request through the complete pipeline"""
    return request_handler.handle_request(user_request)

def get_capabilities() -> Dict[str, Any]:
    """Get available capabilities"""
    return request_handler.get_available_capabilities()

def validate_request(user_request: str) -> Dict[str, Any]:
    """Validate a user request"""
    return request_handler.validate_request(user_request)
