"""
Natural Language Command Parser

This module parses natural language requests and converts them into structured
commands that can be sent to the ML interface.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class StatisticType(Enum):
    """Types of statistical measures"""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    STD = "std"
    VARIANCE = "variance"
    MIN = "min"
    MAX = "max"
    RANGE = "range"
    IQR = "iqr"
    SKEWNESS = "skewness"
    KURTOSIS = "kurtosis"
    COUNT = "count"
    SUM = "sum"
    TOP_FEATURES = "top_features"
    DISCRIMINATIVE = "discriminative_features"

class PlotType(Enum):
    """Types of plots"""
    HISTOGRAM = "histogram"
    BOXPLOT = "boxplot"
    SCATTER = "scatter"
    CORRELATION = "correlation"
    TIMESERIES = "timeseries"
    FREQUENCY = "frequency"
    LINE = "line"
    BAR = "bar"
    PIE = "pie"

@dataclass
class ParsedCommand:
    """Structured representation of a parsed command"""
    command_type: str  # 'statistic', 'plot', 'analysis', 'comparison'
    statistic: Optional[str] = None
    target_features: List[str] = None
    class_filters: List[str] = None
    grouping: Optional[str] = None
    plot_type: Optional[str] = None
    filters: Dict[str, Any] = None
    comparison_classes: List[str] = None
    analysis_type: Optional[str] = None
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.target_features is None:
            self.target_features = []
        if self.class_filters is None:
            self.class_filters = []
        if self.filters is None:
            self.filters = {}
        if self.comparison_classes is None:
            self.comparison_classes = []

class CommandParser:
    """
    Parser for natural language commands related to data analysis
    """
    
    def __init__(self):
        """Initialize the command parser with patterns and keywords"""
        
        # Statistical keywords and their variations
        self.statistic_patterns = {
            StatisticType.MEAN: [r'\b(?:mean|average|avg|arithmetic\s+mean)\b'],
            StatisticType.MEDIAN: [r'\b(?:median|middle\s+value|50th\s+percentile)\b'],
            StatisticType.MODE: [r'\b(?:mode|most\s+common|frequent)\b'],
            StatisticType.STD: [r'\b(?:std|standard\s+deviation|deviation|spread)\b'],
            StatisticType.VARIANCE: [r'\b(?:variance|var|variation)\b'],
            StatisticType.MIN: [r'\b(?:min|minimum|lowest|smallest)\b'],
            StatisticType.MAX: [r'\b(?:max|maximum|highest|largest)\b'],
            StatisticType.RANGE: [r'\b(?:range|span|difference\s+between\s+min\s+and\s+max)\b'],
            StatisticType.IQR: [r'\b(?:iqr|interquartile\s+range|quartile\s+range)\b'],
            StatisticType.SKEWNESS: [r'\b(?:skewness|skew|asymmetry)\b'],
            StatisticType.KURTOSIS: [r'\b(?:kurtosis|peakedness)\b'],
            StatisticType.COUNT: [r'\b(?:count|number|total\s+count)\b'],
            StatisticType.SUM: [r'\b(?:sum|total|addition)\b'],
            StatisticType.TOP_FEATURES: [r'\b(?:top\s+features?|best\s+features?|most\s+important\s+features?|discriminative\s+features?)\b'],
            StatisticType.DISCRIMINATIVE: [r'\b(?:discriminative|discriminating|separating|distinguishing)\b']
        }
        
        # Sensor and feature keywords
        self.sensor_patterns = {
            'accelerometer': [r'\b(?:accelerometer|acc|acceleration|motion|movement|vibration|shake)\b'],
            'gyroscope': [r'\b(?:gyroscope|gyro|rotation|angular|orientation|spin)\b'],
            'magnetometer': [r'\b(?:magnetometer|mag|magnetic|compass|north|field|direction)\b'],
            'temperature': [r'\b(?:temperature|temp|thermal|heat|cold|warm|cool|thermometer)\b'],
            'pressure': [r'\b(?:pressure|press|barometric|force|stress|load|atmospheric)\b'],
            'humidity': [r'\b(?:humidity|hum|moisture|wetness|damp|dry|relative\s+humidity)\b'],
            'microphone': [r'\b(?:microphone|mic|audio|sound|acoustic|noise|voice)\b']
        }
        
        # Class labels and filters
        self.class_patterns = {
            'OK': [r'\b(?:ok|good|normal|pass|acceptable|valid)\b'],
            'KO': [r'\b(?:ko|bad|fail|reject|defective|invalid)\b'],
            'KO_HIGH_2mm': [r'\b(?:ko\s+high\s+2mm|high\s+2mm|high\s+defect|high\s+failure)\b'],
            'KO_LOW_2mm': [r'\b(?:ko\s+low\s+2mm|low\s+2mm|low\s+defect|low\s+failure)\b']
        }
        
        # Plot type patterns
        self.plot_patterns = {
            PlotType.HISTOGRAM: [r'\b(?:histogram|hist|distribution|frequency|bins|count\s+plot)\b'],
            PlotType.BOXPLOT: [r'\b(?:boxplot|box|quartile|whisker|outlier|box\s+plot)\b'],
            PlotType.SCATTER: [r'\b(?:scatter|scatterplot|scatter\s+plot|point\s+plot|dots|correlation\s+plot)\b'],
            PlotType.CORRELATION: [r'\b(?:correlation|corr|relationship|association|connection|correlation\s+matrix)\b'],
            PlotType.TIMESERIES: [r'\b(?:time\s+series|timeseries|trend|temporal|over\s+time|timeline|time\s+plot)\b'],
            PlotType.FREQUENCY: [r'\b(?:frequency|fft|spectrum|fourier|domain|oscillation|frequency\s+analysis)\b'],
            PlotType.LINE: [r'\b(?:line|lineplot|line\s+chart|trend\s+line|continuous|line\s+graph)\b'],
            PlotType.BAR: [r'\b(?:bar|barplot|bar\s+chart|categorical|comparison|bar\s+graph)\b'],
            PlotType.PIE: [r'\b(?:pie|pie\s+chart|proportion|percentage|share|pie\s+graph)\b']
        }
        
        # Grouping patterns
        self.grouping_patterns = {
            'class': [r'\b(?:by\s+class|per\s+class|group\s+by\s+class|class\s+wise|class\s+grouping)\b'],
            'sensor': [r'\b(?:by\s+sensor|per\s+sensor|group\s+by\s+sensor|sensor\s+wise|sensor\s+grouping)\b'],
            'feature': [r'\b(?:by\s+feature|per\s+feature|group\s+by\s+feature|feature\s+wise)\b'],
            'time': [r'\b(?:by\s+time|per\s+time|group\s+by\s+time|temporal\s+grouping)\b']
        }
        
        # Comparison patterns
        self.comparison_patterns = [
            r'\b(?:compare|comparison|versus|vs|between|difference|similar|different)\b',
            r'\b(?:ok\s+vs\s+ko|good\s+vs\s+bad|pass\s+vs\s+fail)\b',
            r'\b(?:class\s+comparison|group\s+comparison|category\s+comparison)\b'
        ]
        
        # Analysis patterns
        self.analysis_patterns = [
            r'\b(?:analyze|analysis|examine|investigate|study|explore)\b',
            r'\b(?:statistical\s+analysis|data\s+analysis|feature\s+analysis)\b',
            r'\b(?:pattern\s+analysis|trend\s+analysis|correlation\s+analysis)\b'
        ]
        
        # Compile all patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all regex patterns for efficient matching"""
        self.compiled_statistic_patterns = {
            stat_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for stat_type, patterns in self.statistic_patterns.items()
        }
        
        self.compiled_sensor_patterns = {
            sensor: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for sensor, patterns in self.sensor_patterns.items()
        }
        
        self.compiled_class_patterns = {
            class_name: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for class_name, patterns in self.class_patterns.items()
        }
        
        self.compiled_plot_patterns = {
            plot_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for plot_type, patterns in self.plot_patterns.items()
        }
        
        self.compiled_grouping_patterns = {
            group_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for group_type, patterns in self.grouping_patterns.items()
        }
        
        self.compiled_comparison_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.comparison_patterns]
        self.compiled_analysis_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.analysis_patterns]
    
    def parse_command(self, text: str) -> ParsedCommand:
        """
        Parse natural language text into a structured command
        
        Args:
            text: Natural language input text
            
        Returns:
            ParsedCommand object with structured information
        """
        text_lower = text.lower()
        
        # Initialize command
        command = ParsedCommand(command_type='unknown')
        
        # Detect command type
        command.command_type = self._detect_command_type(text_lower)
        
        # Parse based on command type
        if command.command_type == 'statistic':
            self._parse_statistic_command(text_lower, command)
        elif command.command_type == 'plot':
            self._parse_plot_command(text_lower, command)
        elif command.command_type == 'comparison':
            self._parse_comparison_command(text_lower, command)
        elif command.command_type == 'analysis':
            self._parse_analysis_command(text_lower, command)
        
        # Parse common elements
        self._parse_target_features(text_lower, command)
        self._parse_class_filters(text_lower, command)
        self._parse_grouping(text_lower, command)
        
        # Calculate confidence
        command.confidence = self._calculate_confidence(command)
        
        return command
    
    def _detect_command_type(self, text: str) -> str:
        """Detect the type of command being requested"""
        # Check for plot requests first
        for plot_type, patterns in self.compiled_plot_patterns.items():
            if any(pattern.search(text) for pattern in patterns):
                return 'plot'
        
        # Check for comparison requests
        if any(pattern.search(text) for pattern in self.compiled_comparison_patterns):
            return 'comparison'
        
        # Check for analysis requests
        if any(pattern.search(text) for pattern in self.compiled_analysis_patterns):
            return 'analysis'
        
        # Check for statistic requests
        for stat_type, patterns in self.compiled_statistic_patterns.items():
            if any(pattern.search(text) for pattern in patterns):
                return 'statistic'
        
        # Default to analysis if unclear
        return 'analysis'
    
    def _parse_statistic_command(self, text: str, command: ParsedCommand):
        """Parse statistic-specific command elements"""
        # Find the requested statistic
        for stat_type, patterns in self.compiled_statistic_patterns.items():
            if any(pattern.search(text) for pattern in patterns):
                command.statistic = stat_type.value
                break
        
        # Check for specific feature mentions
        if 'feature' in text or 'column' in text:
            # Extract feature names (this could be enhanced with more sophisticated parsing)
            words = text.split()
            for i, word in enumerate(words):
                if word in ['feature', 'column', 'variable'] and i + 1 < len(words):
                    command.target_features.append(words[i + 1])
    
    def _parse_plot_command(self, text: str, command: ParsedCommand):
        """Parse plot-specific command elements"""
        # Find the requested plot type
        for plot_type, patterns in self.compiled_plot_patterns.items():
            if any(pattern.search(text) for pattern in patterns):
                command.plot_type = plot_type.value
                break
        
        # Check for plot-specific features
        if 'of' in text:
            # Extract what to plot (e.g., "show histogram of temperature")
            parts = text.split('of')
            if len(parts) > 1:
                target_text = parts[1].strip()
                command.target_features.extend(self._extract_features_from_text(target_text))
    
    def _parse_comparison_command(self, text: str, command: ParsedCommand):
        """Parse comparison-specific command elements"""
        # Extract comparison classes
        for class_name, patterns in self.compiled_class_patterns.items():
            if any(pattern.search(text) for pattern in patterns):
                command.comparison_classes.append(class_name)
        
        # Check for "vs" or "versus" patterns
        vs_patterns = [r'\bvs\b', r'\bversus\b', r'\bbetween\b']
        for pattern in vs_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract classes around the vs/versus
                parts = text.split(match.group())
                if len(parts) == 2:
                    left_part = parts[0].strip()
                    right_part = parts[1].strip()
                    command.comparison_classes.extend(self._extract_classes_from_text(left_part))
                    command.comparison_classes.extend(self._extract_classes_from_text(right_part))
    
    def _parse_analysis_command(self, text: str, command: ParsedCommand):
        """Parse analysis-specific command elements"""
        # Determine analysis type
        if 'pattern' in text:
            command.analysis_type = 'pattern'
        elif 'trend' in text:
            command.analysis_type = 'trend'
        elif 'correlation' in text:
            command.analysis_type = 'correlation'
        elif 'feature' in text:
            command.analysis_type = 'feature'
        else:
            command.analysis_type = 'general'
    
    def _parse_target_features(self, text: str, command: ParsedCommand):
        """Parse target features from text"""
        # Check for sensor mentions
        for sensor, patterns in self.compiled_sensor_patterns.items():
            if any(pattern.search(text) for pattern in patterns):
                command.target_features.append(sensor)
        
        # Check for specific feature names (could be enhanced with actual feature list)
        feature_keywords = ['mean', 'std', 'max', 'min', 'variance', 'skewness', 'kurtosis']
        for keyword in feature_keywords:
            if keyword in text:
                # Look for sensor + feature combinations
                for sensor in self.sensor_patterns.keys():
                    if sensor in text and keyword in text:
                        command.target_features.append(f"{sensor}_{keyword}")
    
    def _parse_class_filters(self, text: str, command: ParsedCommand):
        """Parse class filters from text"""
        for class_name, patterns in self.compiled_class_patterns.items():
            if any(pattern.search(text) for pattern in patterns):
                command.class_filters.append(class_name)
    
    def _parse_grouping(self, text: str, command: ParsedCommand):
        """Parse grouping information from text"""
        for group_type, patterns in self.compiled_grouping_patterns.items():
            if any(pattern.search(text) for pattern in patterns):
                command.grouping = group_type
                break
    
    def _extract_features_from_text(self, text: str) -> List[str]:
        """Extract feature names from text"""
        features = []
        # This is a simple extraction - could be enhanced with more sophisticated NLP
        words = text.split()
        for word in words:
            if any(sensor in word.lower() for sensor in self.sensor_patterns.keys()):
                features.append(word)
        return features
    
    def _extract_classes_from_text(self, text: str) -> List[str]:
        """Extract class names from text"""
        classes = []
        for class_name, patterns in self.compiled_class_patterns.items():
            if any(pattern.search(text) for pattern in patterns):
                classes.append(class_name)
        return classes
    
    def _calculate_confidence(self, command: ParsedCommand) -> float:
        """Calculate confidence score for the parsed command"""
        confidence = 0.0
        
        # Base confidence for command type detection
        if command.command_type != 'unknown':
            confidence += 0.3
        
        # Confidence for specific elements
        if command.statistic:
            confidence += 0.2
        if command.target_features:
            confidence += 0.2
        if command.plot_type:
            confidence += 0.2
        if command.comparison_classes:
            confidence += 0.2
        if command.grouping:
            confidence += 0.1
        
        # Cap at 1.0
        return min(confidence, 1.0)
    
    def get_available_statistics(self) -> List[str]:
        """Get list of available statistical measures"""
        return [stat.value for stat in StatisticType]
    
    def get_available_sensors(self) -> List[str]:
        """Get list of available sensors"""
        return list(self.sensor_patterns.keys())
    
    def get_available_classes(self) -> List[str]:
        """Get list of available classes"""
        return list(self.class_patterns.keys())
    
    def get_available_plot_types(self) -> List[str]:
        """Get list of available plot types"""
        return [plot.value for plot in PlotType]


# Global instance for easy access
command_parser = CommandParser()

# Convenience functions
def parse_command(text: str) -> ParsedCommand:
    """Parse natural language text into a structured command"""
    return command_parser.parse_command(text)

def get_available_statistics() -> List[str]:
    """Get list of available statistical measures"""
    return command_parser.get_available_statistics()

def get_available_sensors() -> List[str]:
    """Get list of available sensors"""
    return command_parser.get_available_sensors()

def get_available_classes() -> List[str]:
    """Get list of available classes"""
    return command_parser.get_available_classes()

def get_available_plot_types() -> List[str]:
    """Get list of available plot types"""
    return command_parser.get_available_plot_types()
