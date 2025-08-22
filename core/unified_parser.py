"""
Unified Parser Module

This module combines the functionality of both Request Parser and Command Parser
to handle natural language requests for ML operations and general AI commands.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

class CommandType(Enum):
    """Types of commands the system can handle"""
    STATISTIC = "statistic"
    PLOT = "plot"
    COMPARISON = "comparison"
    ANALYSIS = "analysis"
    TOP_FEATURES = "top_features"
    FEATURE_ANALYSIS = "feature_analysis"
    CLASSIFICATION = "classification"
    CORRELATION = "correlation"
    UNKNOWN = "unknown"

class StatisticType(Enum):
    """Statistical measures available"""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    STD = "std"
    VARIANCE = "variance"
    MIN = "min"
    MAX = "max"
    RANGE = "range"
    IQR = "iqr"
    COUNT = "count"
    SUM = "sum"

class PlotType(Enum):
    """Types of plots available"""
    LINE_GRAPH = "line_graph"
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    VIOLIN = "violin"
    CORRELATION = "correlation"
    HEATMAP = "heatmap"
    TIMESERIES = "timeseries"
    FREQUENCY = "frequency"
    BAR = "bar"
    LINE = "line"

@dataclass
class UnifiedParsedCommand:
    """Unified representation of a parsed command"""
    # Command identification
    command_type: CommandType
    original_request: str
    confidence: float = 0.0
    
    # Response type preference
    response_type: str = "auto"  # "text", "visual", or "auto"
    
    # Statistical operations
    statistic: Optional[StatisticType] = None
    target_column: Optional[str] = None
    
    # Plot operations
    plot_type: Optional[PlotType] = None
    target_features: List[str] = None
    
    # Analysis operations
    analysis_type: Optional[str] = None
    
    # Common parameters
    filters: Dict[str, Any] = None
    class_filters: List[str] = None
    grouping: Optional[str] = None
    comparison_classes: List[str] = None
    
    # Additional context
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values"""
        if self.target_features is None:
            self.target_features = []
        if self.filters is None:
            self.filters = {}
        if self.class_filters is None:
            self.class_filters = []
        if self.comparison_classes is None:
            self.comparison_classes = []

class UnifiedParser:
    """Unified parser for both ML operations and general AI commands"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Statistical patterns and synonyms
        self.stat_patterns = {
            StatisticType.MEAN: [
                r'\b(?:what is the |get |show |calculate |give me |find the )?mean\b',
                r'\baverage\b', r'\bavg\b', r'\barithmetic mean\b'
            ],
            StatisticType.MEDIAN: [
                r'\b(?:what is the |get |show |calculate |give me |find the )?median\b',
                r'\bmiddle value\b', r'\b50th percentile\b'
            ],
            StatisticType.VARIANCE: [
                r'\b(?:what is the |get |show |calculate |give me |find the )?variance\b',
                r'\bvar\b', r'\bspread\b'
            ],
            StatisticType.STD: [
                r'\b(?:what is the |get |show |calculate |give me |find the )?(?:standard deviation|std)\b',
                r'\bdeviation\b'
            ],
            StatisticType.MIN: [
                r'\b(?:what is the |get |show |calculate |give me |find the )?min(?:imum)?\b',
                r'\bsmallest\b', r'\blowest\b'
            ],
            StatisticType.MAX: [
                r'\b(?:what is the |get |show |calculate |give me |find the )?max(?:imum)?\b',
                r'\bbiggest\b', r'\bhighest\b'
            ],
            StatisticType.COUNT: [
                r'\b(?:what is the |get |show |calculate |give me |find the )?count\b',
                r'\bnumber of\b', r'\btotal\b'
            ]
        }
        
        # Plot patterns
        self.plot_patterns = {
                    PlotType.LINE_GRAPH: [
            r'\b(?:show |create |generate |make |display )?(?:a )?line graph\b',
            r'\b(?:line|linegraph|line-graph)\b'
        ],
            PlotType.HISTOGRAM: [
                r'\b(?:show |create |generate |make |display )?(?:a )?histogram\b',
                r'\b(?:hist|distribution plot)\b'
            ],
            PlotType.SCATTER: [
                r'\b(?:show |create |generate |make |display )?(?:a )?scatter(?: plot)?\b',
                r'\b(?:scatter plot|scatterplot)\b'
            ],
            PlotType.CORRELATION: [
                r'\b(?:show |create |generate |make |display )?(?:a )?correlation(?: matrix)?\b',
                r'\b(?:correlation plot|corr matrix)\b'
            ],
            PlotType.VIOLIN: [
                r'\b(?:show |create |generate |make |display )?(?:a )?violin(?: plot)?\b'
            ],
            PlotType.HEATMAP: [
                r'\b(?:show |create |generate |make |display )?(?:a )?heatmap\b',
                r'\b(?:heat map|heat-map)\b'
            ]
        }
        
        # Command type patterns
        self.command_patterns = {
            CommandType.STATISTIC: [
                r'\b(?:what is|get|show|calculate|give me|find|compute)\b',
                r'\b(?:mean|median|variance|std|min|max|count)\b'
            ],
            CommandType.PLOT: [
                r'\b(?:show|create|generate|make|display|plot)\b',
                r'\b(?:line graph|histogram|scatter|correlation|heatmap)\b'
            ],
            CommandType.COMPARISON: [
                r'\b(?:compare|comparison|vs|versus|between)\b',
                r'\b(?:difference|similarity|contrast)\b'
            ],
            CommandType.ANALYSIS: [
                r'\b(?:analyze|analysis|examine|investigate|study)\b',
                r'\b(?:pattern|trend|relationship|insight)\b'
            ],
            CommandType.TOP_FEATURES: [
                r'\b(?:top|best|most|discriminative|separating|separate)\b',
                r'\b(?:features|indices|characteristics|sensors|measures|statistics)\b',
                r'\b(?:important)\s+(?:feature|features)\b',
                r'\b(?:list|show|get|find|identify)\b.*\b(?:top|best|most)\b',
                r'\b(?:top|best|most)\b.*\b(?:features|indices|characteristics|sensors)\b'
            ],
            CommandType.FEATURE_ANALYSIS: [
                r'\b(?:feature|sensor|characteristic)\b',
                r'\b(?:analysis|examination|investigation)\b'
            ]
        }
        
        # Sensor mapping with common names and abbreviations
        self.sensor_mapping = {
            # Temperature sensors
            'HTS221_TEMP': ['hts221_temp', 'hts221', 'temperature', 'temp', 'thermal'],
            'temperature_mean': ['temperature', 'temp', 'thermal', 'heat', 'hts221_temp', 'hts221', 'hts221 temperature'],
            
            # Humidity sensors
            'HTS221_HUM': ['hts221_hum', 'hts221_humidity', 'hts221 humidity', 'humidity', 'hum', 'moisture', 'wetness'],
            'humidity_mean': ['humidity', 'hum', 'moisture', 'wetness', 'hts221_hum', 'hts221_humidity', 'hts221 humidity'],
            
            # Pressure sensors
            'LPS22HH_PRESS': ['lps22hh_press', 'lps22hh', 'lps22hh pressure', 'pressure', 'press', 'barometric', 'force'],
            'pressure_mean': ['pressure', 'press', 'barometric', 'force', 'lps22hh_press', 'lps22hh', 'lps22hh pressure'],
            
            # Acceleration sensors
            'IIS3DWB_ACC': ['iis3dwb_acc', 'iis3dwb', 'iis3dwb acceleration', 'acceleration_x', 'acc_x', 'accel_x', 'motion_x', 'acceleration_y', 'acc_y', 'accel_y', 'motion_y', 'acceleration_z', 'acc_z', 'accel_z', 'motion_z'],
            'acceleration_x_mean': ['acceleration_x', 'acc_x', 'accel_x', 'motion_x', 'iis3dwb_acc', 'iis3dwb', 'iis3dwb acceleration'],
            'acceleration_y_mean': ['acceleration_y', 'acc_y', 'accel_y', 'motion_y', 'iis3dwb_acc', 'iis3dwb', 'iis3dwb acceleration'],
            'acceleration_z_mean': ['acceleration_z', 'acc_z', 'accel_z', 'motion_z', 'iis3dwb_acc', 'iis3dwb', 'iis3dwb acceleration'],
            
            # Gyroscope sensors
            'gyroscope_x_mean': ['gyroscope_x', 'gyro_x', 'rotation_x', 'angular_x', 'iis3dwb_gyro', 'iis3dwb', 'iis3dwb gyroscope'],
            'gyroscope_y_mean': ['gyroscope_y', 'gyro_y', 'rotation_y', 'angular_y', 'iis3dwb_gyro', 'iis3dwb', 'iis3dwb gyroscope'],
            'gyroscope_z_mean': ['gyroscope_z', 'gyro_z', 'rotation_z', 'angular_z', 'iis3dwb_gyro', 'iis3dwb', 'iis3dwb gyroscope'],
            
            # Microphone
            'microphone_mean': ['microphone', 'mic', 'audio', 'sound', 'acoustic', 'iis3dwb_mic', 'iis3dwb', 'iis3dwb microphone']
        }
        
        # Class patterns
        self.class_patterns = {
            'OK': [r'\bOK\b', r'\bok\b', r'\bgood\b', r'\bnormal\b'],
            'KO': [r'\bKO\b', r'\bko\b', r'\bbad\b', r'\bfaulty\b'],
            'KO_HIGH_2mm': [r'\bKO_HIGH_2mm\b', r'\bko_high_2mm\b', r'\bhigh\b', r'\bhigh_2mm\b'],
            'KO_LOW_2mm': [r'\bKO_LOW_2mm\b', r'\bko_low_2mm\b', r'\blow\b', r'\blow_2mm\b'],
            'KO_LOW_4mm': [r'\bKO_LOW_4mm\b', r'\bko_low_4mm\b', r'\blow 4mm\b', r'\blow_4mm\b']
        }
        
        # Compile patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        for stat_type, patterns in self.stat_patterns.items():
            self.stat_patterns[stat_type] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        
        for plot_type, patterns in self.plot_patterns.items():
            self.plot_patterns[plot_type] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        
        for command_type, patterns in self.command_patterns.items():
            self.command_patterns[command_type] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        
        for class_name, patterns in self.class_patterns.items():
            self.class_patterns[class_name] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def parse_command(self, text: str) -> UnifiedParsedCommand:
        """
        Parse a natural language command and extract all components
        
        Args:
            text: Natural language command from user
            
        Returns:
            UnifiedParsedCommand with all parsed components
        """
        text_lower = text.lower().strip()
        
        # Initialize result
        parsed = UnifiedParsedCommand(
            command_type=CommandType.UNKNOWN,
            original_request=text
        )
        
        try:
            # 1. Detect command type and response type preference
            parsed.command_type, parsed.response_type = self._detect_command_type(text_lower)
            
            # 2. Parse based on command type
            if parsed.command_type == CommandType.STATISTIC:
                self._parse_statistic_command(text_lower, parsed)
            elif parsed.command_type == CommandType.PLOT:
                self._parse_plot_command(text_lower, parsed)
            elif parsed.command_type == CommandType.COMPARISON:
                self._parse_comparison_command(text_lower, parsed)
            elif parsed.command_type == CommandType.ANALYSIS:
                self._parse_analysis_command(text_lower, parsed)
            elif parsed.command_type == CommandType.TOP_FEATURES:
                self._parse_top_features_command(text_lower, parsed)
            elif parsed.command_type == CommandType.FEATURE_ANALYSIS:
                self._parse_feature_analysis_command(text_lower, parsed)
            
            # 3. Extract common components
            self._parse_target_features(text_lower, parsed)
            self._parse_class_filters(text_lower, parsed)
            self._parse_grouping(text_lower, parsed)
            
            # 4. Calculate confidence
            parsed.confidence = self._calculate_confidence(parsed)
            
            # 5. Validate and set defaults
            self._validate_and_set_defaults(parsed)
            
        except Exception as e:
            self.logger.error(f"Error parsing command: {e}")
            parsed.error_message = f"Error parsing command: {str(e)}"
            parsed.confidence = 0.0
        
        return parsed
    
    def _detect_command_type(self, text_lower: str) -> Tuple[CommandType, str]:
        """Detect the primary command type and response type preference from the text"""
        # First, check for TOP_FEATURES with high priority (clear indicators)
        if any(word in text_lower for word in ['top', 'best', 'most']) and any(word in text_lower for word in ['features', 'indices', 'characteristics', 'sensors']):
            # Additional check for separation/discrimination context
            if any(word in text_lower for word in ['separate', 'separating', 'discriminative', 'discriminate', 'distinguish']):
                return CommandType.TOP_FEATURES, "text"
        
        # Check for text-only requests (lists, analysis, information)
        text_indicators = [
            'list', 'show me', 'tell me', 'what is', 'what are', 'describe', 'explain',
            'analyze', 'analysis', 'examine', 'investigate', 'study', 'find', 'identify',
            'compare', 'comparison', 'versus', 'vs', 'between', 'among'
        ]
        
        # Check for visual requests (plots, charts, graphs)
        visual_indicators = [
            'plot', 'chart', 'graph', 'visualize', 'visualization', 'display', 'show',
            'create', 'generate', 'make', 'draw', 'render', 'illustrate'
        ]
        
        # Determine response type preference
        response_type = "auto"
        if any(indicator in text_lower for indicator in text_indicators):
            response_type = "text"
        elif any(indicator in text_lower for indicator in visual_indicators):
            response_type = "visual"
        
        # Check for specific command types with stricter conditions
        # Statistics: require BOTH an action verb and a statistic term
        stat_actions = re.search(r"\b(?:what is|get|show|calculate|give me|find|compute)\b", text_lower)
        stat_terms = re.search(r"\b(?:mean|median|variance|std|min|max|count)\b", text_lower)
        if stat_actions and stat_terms:
            return CommandType.STATISTIC, "text"

        # Plot: presence of plot intent words and known plot types
        if any(p.search(text_lower) for p in self.plot_patterns.get(PlotType.LINE_GRAPH, [])) \
           or any(p.search(text_lower) for p in self.plot_patterns.get(PlotType.HISTOGRAM, [])) \
           or any(p.search(text_lower) for p in self.plot_patterns.get(PlotType.SCATTER, [])) \
           or any(p.search(text_lower) for p in self.plot_patterns.get(PlotType.CORRELATION, [])) \
           or any(p.search(text_lower) for p in self.plot_patterns.get(PlotType.VIOLIN, [])) \
           or any(p.search(text_lower) for p in self.plot_patterns.get(PlotType.HEATMAP, [])):
            return CommandType.PLOT, "visual"

        # Generic plot requests (no specific type mentioned)
        if any(p.search(text_lower) for p in self.command_patterns.get(CommandType.PLOT, [])):
            return CommandType.PLOT, "visual"

        # Comparison: keywords
        if any(p.search(text_lower) for p in self.command_patterns.get(CommandType.COMPARISON, [])):
            return CommandType.COMPARISON, response_type

        # Analysis: keywords
        if any(p.search(text_lower) for p in self.command_patterns.get(CommandType.ANALYSIS, [])):
            return CommandType.ANALYSIS, response_type

        # Top features: prior check above may have returned; otherwise check patterns
        if any(p.search(text_lower) for p in self.command_patterns.get(CommandType.TOP_FEATURES, [])):
            return CommandType.TOP_FEATURES, "text"
        
        # Check for statistical patterns
        for stat_type, patterns in self.stat_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    return CommandType.STATISTIC, "text"
        
        # Check for plot patterns
        for plot_type, patterns in self.plot_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    return CommandType.PLOT, "visual"
        
        # Default fallback
        return CommandType.UNKNOWN, "text"
    
    def _parse_statistic_command(self, text_lower: str, parsed: UnifiedParsedCommand):
        """Parse statistical command components"""
        # Detect statistic type
        for stat_type, patterns in self.stat_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    parsed.statistic = stat_type
                    break
            if parsed.statistic:
                break
        
        # Extract target column
        parsed.target_column = self._extract_target_column(text_lower)
    
    def _parse_plot_command(self, text_lower: str, parsed: UnifiedParsedCommand):
        """Parse plot command components"""
        # Detect plot type
        for plot_type, patterns in self.plot_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    parsed.plot_type = plot_type
                    break
            if parsed.plot_type:
                break
    
    def _parse_comparison_command(self, text_lower: str, parsed: UnifiedParsedCommand):
        """Parse comparison command components"""
        parsed.command_type = CommandType.COMPARISON
        
        # Extract comparison classes
        detected_classes = []
        for class_name, patterns in self.class_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    detected_classes.append(class_name)
                    break
        
        if detected_classes:
            parsed.comparison_classes = detected_classes
            parsed.class_filters = detected_classes
    
    def _parse_analysis_command(self, text_lower: str, parsed: UnifiedParsedCommand):
        """Parse analysis command components"""
        parsed.command_type = CommandType.ANALYSIS
        
        # Detect analysis type
        if 'pattern' in text_lower or 'trend' in text_lower:
            parsed.analysis_type = 'pattern'
        elif 'relationship' in text_lower or 'correlation' in text_lower:
            parsed.analysis_type = 'correlation'
        elif 'outlier' in text_lower or 'anomaly' in text_lower:
            parsed.analysis_type = 'outlier'
        else:
            parsed.analysis_type = 'general'
    
    def _parse_top_features_command(self, text_lower: str, parsed: UnifiedParsedCommand):
        """Parse top features command components"""
        parsed.command_type = CommandType.TOP_FEATURES
        
        # Extract count if specified
        count_match = re.search(r'\b(\d+)\b', text_lower)
        if count_match:
            parsed.filters['count'] = int(count_match.group(1))
        else:
            parsed.filters['count'] = 3  # Default
    
    def _parse_feature_analysis_command(self, text_lower: str, parsed: UnifiedParsedCommand):
        """Parse feature analysis command components"""
        parsed.command_type = CommandType.FEATURE_ANALYSIS
        parsed.target_column = self._extract_target_column(text_lower)
    
    def _parse_target_features(self, text_lower: str, parsed: UnifiedParsedCommand):
        """Extract target features from the text"""
        # Detect vendor-specific sensor codes (e.g., HTS221, LPS22HH, STTS751)
        vendor_tokens = re.findall(r"\b[A-Z]{2,}[A-Z0-9]*\d+[A-Z0-9]*\b", text_lower.upper())
        for tok in vendor_tokens:
            if tok not in parsed.target_features:
                parsed.target_features.append(tok)
            if parsed.target_column is None:
                parsed.target_column = tok
        
        # First, try to find exact sensor names
        for sensor_name, aliases in self.sensor_mapping.items():
            for alias in aliases:
                if alias.lower() in text_lower:
                    if parsed.target_column is None:
                        parsed.target_column = sensor_name
                    if sensor_name not in parsed.target_features:
                        parsed.target_features.append(sensor_name)
        
        # If no exact match, try to infer from context
        if not parsed.target_features:
            if any(word in text_lower for word in ['temperature', 'temp', 'thermal']):
                parsed.target_features.append('temperature_mean')
            elif any(word in text_lower for word in ['humidity', 'hum', 'moisture']):
                parsed.target_features.append('humidity_mean')
            elif any(word in text_lower for word in ['pressure', 'press', 'barometric']):
                parsed.target_features.append('pressure_mean')
            elif any(word in text_lower for word in ['acceleration', 'acc', 'motion']):
                parsed.target_features.append('acceleration_x_mean')
            elif any(word in text_lower for word in ['gyroscope', 'gyro', 'rotation']):
                parsed.target_features.append('gyroscope_x_mean')
            elif any(word in text_lower for word in ['microphone', 'mic', 'audio']):
                parsed.target_features.append('microphone_mean')
    
    def _parse_class_filters(self, text_lower: str, parsed: UnifiedParsedCommand):
        """Extract class filters from the text"""
        detected_classes = []
        
        # Detect class names
        for class_name, patterns in self.class_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    detected_classes.append(class_name)
                    break
        
        # Handle special cases
        if 'each class' in text_lower or 'for each class' in text_lower:
            detected_classes = ['OK', 'KO', 'KO_HIGH_2mm', 'KO_LOW_2mm']
        elif 'ok and ko' in text_lower or 'ok vs ko' in text_lower:
            detected_classes = ['OK', 'KO']
        
        if detected_classes:
            parsed.class_filters = detected_classes
            parsed.filters['class'] = detected_classes
    
    def _parse_grouping(self, text_lower: str, parsed: UnifiedParsedCommand):
        """Determine if grouping is requested"""
        if 'each class' in text_lower or 'for each class' in text_lower:
            parsed.grouping = 'class'
        elif 'grouped by' in text_lower:
            parsed.grouping = 'class'
        elif parsed.class_filters and len(parsed.class_filters) > 1:
            parsed.grouping = 'class'
    
    def _extract_target_column(self, text_lower: str) -> str:
        """Extract the target column/sensor from the text"""
        # First, try to find exact sensor names
        for sensor_name, aliases in self.sensor_mapping.items():
            for alias in aliases:
                if alias.lower() in text_lower:
                    return sensor_name
        
        # If no exact match, try to infer from context
        if any(word in text_lower for word in ['temperature', 'temp', 'thermal']):
            return 'temperature_mean'
        elif any(word in text_lower for word in ['humidity', 'hum', 'moisture']):
            return 'humidity_mean'
        elif any(word in text_lower for word in ['pressure', 'press', 'barometric']):
            return 'pressure_mean'
        elif any(word in text_lower for word in ['acceleration', 'acc', 'motion']):
            return 'acceleration_x_mean'
        elif any(word in text_lower for word in ['gyroscope', 'gyro', 'rotation']):
            return 'gyroscope_x_mean'
        elif any(word in text_lower for word in ['microphone', 'mic', 'audio']):
            return 'microphone_mean'
        
        # Default fallback
        return 'temperature_mean'
    
    def _calculate_confidence(self, parsed: UnifiedParsedCommand) -> float:
        """Calculate confidence score for the parsing"""
        confidence = 0.0
        
        # Base confidence for having a command type
        if parsed.command_type != CommandType.UNKNOWN:
            confidence += 0.3
        
        # Confidence for having target features
        if parsed.target_features:
            confidence += 0.2
        
        # Confidence for having class filters
        if parsed.class_filters:
            confidence += 0.2
        
        # Confidence for proper grouping
        if parsed.grouping:
            confidence += 0.1
        
        # Confidence for specific components based on command type
        if parsed.command_type == CommandType.STATISTIC and parsed.statistic:
            confidence += 0.2
        elif parsed.command_type == CommandType.PLOT and parsed.plot_type:
            confidence += 0.2
        elif parsed.command_type == CommandType.TOP_FEATURES:
            confidence += 0.2
        
        # Bonus for specific sensor names
        if parsed.target_column and any(sensor in parsed.target_column for sensor in ['HTS221', 'LPS22HH', 'IIS3DWB']):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _validate_and_set_defaults(self, parsed: UnifiedParsedCommand):
        """Validate parsed command and set sensible defaults"""
        # Set default grouping for class comparisons
        if parsed.class_filters and len(parsed.class_filters) > 1 and not parsed.grouping:
            parsed.grouping = 'class'
        
        # Handle top features requests
        if parsed.command_type == CommandType.TOP_FEATURES:
            if not parsed.class_filters:
                parsed.class_filters = ['OK', 'KO']
                parsed.filters['class'] = ['OK', 'KO']
            parsed.grouping = 'class'
        
        # Ensure we have at least one target feature
        if not parsed.target_features and parsed.target_column:
            parsed.target_features = [parsed.target_column]
    
    def get_available_sensors(self) -> List[str]:
        """Get list of available sensor names"""
        return list(self.sensor_mapping.keys())
    
    def get_available_classes(self) -> List[str]:
        """Get list of available class names"""
        return list(self.class_patterns.keys())
    
    def get_available_stats(self) -> List[str]:
        """Get list of available statistical measures"""
        return [stat.value for stat in StatisticType]
    
    def get_available_plots(self) -> List[str]:
        """Get list of available plot types"""
        return [plot.value for plot in PlotType]
    
    def get_available_commands(self) -> List[str]:
        """Get list of available command types"""
        return [cmd.value for cmd in CommandType]


# Global instance
unified_parser = UnifiedParser()

# Convenience functions
def parse_command(text: str) -> UnifiedParsedCommand:
    """Parse a natural language command"""
    return unified_parser.parse_command(text)

def get_available_sensors() -> List[str]:
    """Get available sensors"""
    return unified_parser.get_available_sensors()

def get_available_classes() -> List[str]:
    """Get available classes"""
    return unified_parser.get_available_classes()

def get_available_stats() -> List[str]:
    """Get available statistics"""
    return unified_parser.get_available_stats()

def get_available_plots() -> List[str]:
    """Get available plot types"""
    return unified_parser.get_available_plots()

