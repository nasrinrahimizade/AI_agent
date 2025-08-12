"""
Request Parser Module

This module parses natural language requests and extracts:
- Statistical type (mean, median, variance, std, top features)
- Target column (sensor name)
- Filters (class names)
- Additional parameters
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ParsedRequest:
    """Structured representation of a parsed request"""
    stat_type: str  # mean, median, variance, std, top_features
    target_column: str  # sensor name or feature
    filters: Dict[str, Any]  # class filters, etc.
    group_by: Optional[str] = None  # grouping parameter
    confidence: float = 0.0  # confidence in parsing
    original_request: str = ""
    error_message: Optional[str] = None

class RequestParser:
    """Parse natural language requests for ML operations"""
    
    def __init__(self):
        # Statistical measures and their synonyms
        self.stat_patterns = {
            'mean': [
                r'\b(?:what is the |get |show |calculate |give me |find the )?mean\b',
                r'\baverage\b',
                r'\bavg\b',
                r'\barithmetic mean\b'
            ],
            'median': [
                r'\b(?:what is the |get |show |calculate |give me |find the )?median\b',
                r'\bmiddle value\b',
                r'\b50th percentile\b'
            ],
            'variance': [
                r'\b(?:what is the |get |show |calculate |give me |find the )?variance\b',
                r'\bvar\b',
                r'\bspread\b'
            ],
            'std': [
                r'\b(?:what is the |get |show |calculate |give me |find the )?(?:standard deviation|std)\b',
                r'\bdeviation\b'
            ],
            'top_features': [
                r'\b(?:what are |get |show |list |give me |find )?(?:the )?top\b',
                r'\bbest\b',
                r'\bmost (?:important|discriminative|separating)\b',
                r'\bstatistical indices\b'
            ]
        }
        
        # Sensor mapping with common names and abbreviations
        self.sensor_mapping = {
            # Temperature sensors
            'HTS221_TEMP': ['hts221_temp', 'hts221', 'temperature', 'temp', 'thermal'],
            'temperature_mean': ['temperature', 'temp', 'thermal', 'heat'],
            
            # Humidity sensors
            'HTS221_HUM': ['hts221_hum', 'hts221_humidity', 'humidity', 'hum', 'moisture'],
            'humidity_mean': ['humidity', 'hum', 'moisture', 'wetness'],
            
            # Pressure sensors
            'LPS22HH_PRESS': ['lps22hh_press', 'lps22hh', 'pressure', 'press', 'barometric'],
            'pressure_mean': ['pressure', 'press', 'barometric', 'force'],
            
            # Acceleration sensors
            'IIS3DWB_ACC': ['iis3dwb_acc', 'iis3dwb', 'acceleration', 'acc', 'motion'],
            'acceleration_x_mean': ['acceleration_x', 'acc_x', 'accel_x', 'motion_x'],
            'acceleration_y_mean': ['acceleration_y', 'acc_y', 'accel_y', 'motion_y'],
            'acceleration_z_mean': ['acceleration_z', 'acc_z', 'accel_z', 'motion_z'],
            
            # Gyroscope sensors
            'gyroscope_x_mean': ['gyroscope_x', 'gyro_x', 'rotation_x', 'angular_x'],
            'gyroscope_y_mean': ['gyroscope_y', 'gyro_y', 'rotation_y', 'angular_y'],
            'gyroscope_z_mean': ['gyroscope_z', 'gyro_z', 'rotation_z', 'angular_z'],
            
            # Microphone
            'microphone_mean': ['microphone', 'mic', 'audio', 'sound', 'acoustic']
        }
        
        # Class patterns
        self.class_patterns = {
            'OK': [r'\bOK\b', r'\bok\b', r'\bgood\b', r'\bnormal\b'],
            'KO': [r'\bKO\b', r'\bko\b', r'\bbad\b', r'\bfaulty\b'],
            'KO_HIGH_2mm': [r'\bKO_HIGH_2mm\b', r'\bko_high_2mm\b', r'\bhigh\b', r'\bhigh_2mm\b'],
            'KO_LOW_2mm': [r'\bKO_LOW_2mm\b', r'\bko_low_2mm\b', r'\blow\b', r'\blow_2mm\b']
        }
        
        # Compile patterns for efficiency
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        for stat_type, patterns in self.stat_patterns.items():
            self.stat_patterns[stat_type] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        
        for class_name, patterns in self.class_patterns.items():
            self.class_patterns[class_name] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def parse_request(self, request: str) -> ParsedRequest:
        """Parse a natural language request and extract components"""
        request_lower = request.lower().strip()
        
        # Initialize result
        parsed = ParsedRequest(
            stat_type="",
            target_column="",
            filters={},
            original_request=request
        )
        
        try:
            # 1. Detect statistical type
            parsed.stat_type = self._detect_stat_type(request_lower)
            
            # 2. Extract target column
            parsed.target_column = self._extract_target_column(request_lower)
            
            # 3. Extract filters (class names)
            parsed.filters = self._extract_filters(request_lower)
            
            # 4. Determine grouping
            parsed.group_by = self._determine_grouping(request_lower, parsed.filters)
            
            # 5. Calculate confidence
            parsed.confidence = self._calculate_confidence(parsed)
            
            # 6. Validate and set defaults
            self._validate_and_set_defaults(parsed)
            
        except Exception as e:
            logging.error(f"Error parsing request: {e}")
            parsed.error_message = f"Error parsing request: {str(e)}"
            parsed.confidence = 0.0
        
        return parsed
    
    def _detect_stat_type(self, request_lower: str) -> str:
        """Detect the statistical type from the request"""
        for stat_type, patterns in self.stat_patterns.items():
            for pattern in patterns:
                if pattern.search(request_lower):
                    return stat_type
        
        # Default to mean if no specific stat type detected
        return "mean"
    
    def _extract_target_column(self, request_lower: str) -> str:
        """Extract the target column/sensor from the request"""
        # First, try to find exact sensor names
        for sensor_name, aliases in self.sensor_mapping.items():
            for alias in aliases:
                if alias.lower() in request_lower:
                    return sensor_name
        
        # If no exact match, try to infer from context
        if any(word in request_lower for word in ['temperature', 'temp', 'thermal']):
            return 'temperature_mean'
        elif any(word in request_lower for word in ['humidity', 'hum', 'moisture']):
            return 'humidity_mean'
        elif any(word in request_lower for word in ['pressure', 'press', 'barometric']):
            return 'pressure_mean'
        elif any(word in request_lower for word in ['acceleration', 'acc', 'motion']):
            return 'acceleration_x_mean'  # Default to X-axis
        elif any(word in request_lower for word in ['gyroscope', 'gyro', 'rotation']):
            return 'gyroscope_x_mean'  # Default to X-axis
        elif any(word in request_lower for word in ['microphone', 'mic', 'audio']):
            return 'microphone_mean'
        
        # Default fallback
        return 'temperature_mean'
    
    def _extract_filters(self, request_lower: str) -> Dict[str, Any]:
        """Extract filters from the request"""
        filters = {}
        detected_classes = []
        
        # Detect class names
        for class_name, patterns in self.class_patterns.items():
            for pattern in patterns:
                if pattern.search(request_lower):
                    detected_classes.append(class_name)
                    break
        
        # Handle special cases
        if 'each class' in request_lower or 'for each class' in request_lower:
            detected_classes = ['OK', 'KO', 'KO_HIGH_2mm', 'KO_LOW_2mm']
        elif 'ok and ko' in request_lower or 'ok vs ko' in request_lower:
            detected_classes = ['OK', 'KO']
        
        if detected_classes:
            filters['class'] = detected_classes
        
        return filters
    
    def _determine_grouping(self, request_lower: str, filters: Dict[str, Any]) -> Optional[str]:
        """Determine if grouping is requested"""
        if 'each class' in request_lower or 'for each class' in request_lower:
            return 'class'
        elif 'grouped by' in request_lower:
            return 'class'
        elif filters.get('class') and len(filters['class']) > 1:
            return 'class'
        
        return None
    
    def _calculate_confidence(self, parsed: ParsedRequest) -> float:
        """Calculate confidence score for the parsing"""
        confidence = 0.0
        
        # Base confidence for having a stat type
        if parsed.stat_type:
            confidence += 0.3
        
        # Confidence for having a target column
        if parsed.target_column:
            confidence += 0.3
        
        # Confidence for having filters
        if parsed.filters:
            confidence += 0.2
        
        # Confidence for proper grouping
        if parsed.group_by:
            confidence += 0.2
        
        # Bonus for specific sensor names
        if any(sensor in parsed.target_column for sensor in ['HTS221', 'LPS22HH', 'IIS3DWB']):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _validate_and_set_defaults(self, parsed: ParsedRequest):
        """Validate parsed request and set sensible defaults"""
        # Ensure we have a stat type
        if not parsed.stat_type:
            parsed.stat_type = 'mean'
        
        # Ensure we have a target column
        if not parsed.target_column:
            parsed.target_column = 'temperature_mean'
        
        # Set default grouping for class comparisons
        if parsed.filters.get('class') and len(parsed.filters['class']) > 1 and not parsed.group_by:
            parsed.group_by = 'class'
        
        # Handle top features requests
        if parsed.stat_type == 'top_features':
            if not parsed.filters.get('class'):
                parsed.filters['class'] = ['OK', 'KO']  # Default comparison
            parsed.group_by = 'class'
    
    def get_available_sensors(self) -> List[str]:
        """Get list of available sensor names"""
        return list(self.sensor_mapping.keys())
    
    def get_available_classes(self) -> List[str]:
        """Get list of available class names"""
        return list(self.class_patterns.keys())
    
    def get_available_stats(self) -> List[str]:
        """Get list of available statistical measures"""
        return list(self.stat_patterns.keys())


# Global instance
request_parser = RequestParser()

# Convenience function
def parse_request(request: str) -> ParsedRequest:
    """Parse a natural language request"""
    return request_parser.parse_request(request)

