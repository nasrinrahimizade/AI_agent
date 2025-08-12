"""
Response Formatter Module

This module takes ML results and formats them into user-friendly responses,
including text summaries, follow-up suggestions, and plot requests.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

class ResponseFormatter:
    """
    Formats ML results into user-friendly responses with summaries,
    explanations, and follow-up suggestions.
    """
    
    def __init__(self):
        """Initialize the response formatter"""
        self.logger = logging.getLogger(__name__)
        
        # Response templates
        self.templates = {
            'statistic_summary': {
                'mean': "The mean value of {feature} is {value:.2f}",
                'median': "The median value of {feature} is {value:.2f}",
                'std': "The standard deviation of {feature} is {value:.2f}",
                'variance': "The variance of {feature} is {value:.2f}",
                'min': "The minimum value of {feature} is {value:.2f}",
                'max': "The maximum value of {feature} is {value:.2f}",
                'range': "The range of {feature} is {value:.2f}",
                'count': "There are {value} samples for {feature}",
                'top_features': "The top discriminative features are: {features}",
                'discriminative_features': "The most discriminative features between classes are: {features}"
            },
            
            'comparison_summary': {
                'class_comparison': "Comparing {class1} vs {class2} for {feature}:",
                'statistical_significance': "The difference is {significance} (p-value: {p_value:.4f})",
                'effect_size': "Effect size: {effect_size}",
                'interpretation': "This suggests {interpretation}"
            },
            
            'plot_suggestion': {
                'histogram': "A histogram would show the distribution of {feature}",
                'boxplot': "A boxplot would compare {feature} across classes",
                'scatter': "A scatter plot would show the relationship between {feature1} and {feature2}",
                'correlation': "A correlation matrix would show feature relationships",
                'timeseries': "A time series plot would show temporal patterns in {feature}"
            },
            
            'follow_up_suggestions': {
                'statistic': [
                    "Would you like to see the distribution of this feature?",
                    "Should I compare this across different classes?",
                    "Would you like to see how this relates to other features?"
                ],
                'comparison': [
                    "Would you like to see a visual comparison?",
                    "Should I analyze other features for similar patterns?",
                    "Would you like to see the statistical significance details?"
                ],
                'analysis': [
                    "Would you like me to generate a plot for this analysis?",
                    "Should I look for similar patterns in other features?",
                    "Would you like to see the top discriminative features?"
                ]
            }
        }
        
        # Statistical significance interpretations
        self.significance_interpretations = {
            'high': 'very strong evidence of a difference',
            'medium': 'moderate evidence of a difference',
            'low': 'weak evidence of a difference',
            'none': 'no significant difference'
        }
        
        # Effect size interpretations
        self.effect_size_interpretations = {
            'large': 'a substantial difference between groups',
            'medium': 'a moderate difference between groups',
            'small': 'a small but meaningful difference between groups',
            'negligible': 'a negligible difference between groups'
        }
    
    def format_statistic_response(self, result: Dict[str, Any], command: Any) -> Dict[str, Any]:
        """
        Format a statistical result into a user-friendly response
        
        Args:
            result: Result from ML interface
            command: Parsed command object
            
        Returns:
            Formatted response dictionary
        """
        try:
            if result['status'] != 'success':
                return self._format_error_response(result, "statistic")
            
            statistic = command.statistic if hasattr(command, 'statistic') else 'unknown'
            feature = command.target_features[0] if command.target_features else 'the feature'
            
            # Format the main result
            if statistic == 'top_features' or statistic == 'discriminative_features':
                response = self._format_top_features_response(result, feature)
            else:
                response = self._format_basic_statistic_response(result, statistic, feature)
            
            # Add follow-up suggestions
            response['follow_up_suggestions'] = self._get_follow_up_suggestions('statistic', command)
            
            # Add plot suggestions if appropriate
            response['plot_suggestions'] = self._get_plot_suggestions(statistic, command.target_features)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error formatting statistic response: {e}")
            return self._format_error_response(result, "statistic")
    
    def format_comparison_response(self, result: Dict[str, Any], command: Any) -> Dict[str, Any]:
        """
        Format a comparison result into a user-friendly response
        
        Args:
            result: Result from ML interface
            command: Parsed command object
            
        Returns:
            Formatted response dictionary
        """
        try:
            if result['status'] != 'success':
                return self._format_error_response(result, "comparison")
            
            feature = command.target_features[0] if command.target_features else 'the feature'
            classes = command.comparison_classes if hasattr(command, 'comparison_classes') else ['OK', 'KO']
            
            response = {
                'type': 'comparison',
                'title': f"Comparison: {feature} across classes",
                'summary': self._format_comparison_summary(result, feature, classes),
                'details': self._format_comparison_details(result),
                'interpretation': self._format_comparison_interpretation(result),
                'follow_up_suggestions': self._get_follow_up_suggestions('comparison', command),
                'plot_suggestions': self._get_plot_suggestions('comparison', command.target_features)
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error formatting comparison response: {e}")
            return self._format_error_response(result, "comparison")
    
    def format_analysis_response(self, result: Dict[str, Any], command: Any) -> Dict[str, Any]:
        """
        Format an analysis result into a user-friendly response
        
        Args:
            result: Result from ML interface
            command: Parsed command object
            
        Returns:
            Formatted response dictionary
        """
        try:
            if result['status'] != 'success':
                return self._format_error_response(result, "analysis")
            
            analysis_type = getattr(command, 'analysis_type', 'general')
            features = command.target_features if hasattr(command, 'target_features') else []
            
            response = {
                'type': 'analysis',
                'title': f"{analysis_type.title()} Analysis",
                'summary': self._format_analysis_summary(result, analysis_type),
                'key_findings': self._extract_key_findings(result),
                'insights': self._extract_insights(result),
                'follow_up_suggestions': self._get_follow_up_suggestions('analysis', command),
                'plot_suggestions': self._get_plot_suggestions(analysis_type, features)
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error formatting analysis response: {e}")
            return self._format_error_response(result, "analysis")
    
    def format_plot_response(self, result: Dict[str, Any], command: Any) -> Dict[str, Any]:
        """
        Format a plot result into a user-friendly response
        
        Args:
            result: Result from ML interface
            command: Parsed command object
            
        Returns:
            Formatted response dictionary
        """
        try:
            if result['status'] != 'success':
                return self._format_error_response(result, "plot")
            
            plot_type = getattr(command, 'plot_type', 'unknown')
            features = command.target_features if hasattr(command, 'target_features') else []
            
            response = {
                'type': 'plot',
                'title': f"{plot_type.title()} Generated",
                'summary': f"I've created a {plot_type} showing {', '.join(features) if features else 'the requested data'}",
                'plot_data': result.get('plot_data', {}),
                'follow_up_suggestions': [
                    "Would you like to analyze this visualization further?",
                    "Should I generate additional plots for comparison?",
                    "Would you like to see statistical details for this data?"
                ]
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error formatting plot response: {e}")
            return self._format_error_response(result, "plot")
    
    def _format_basic_statistic_response(self, result: Dict[str, Any], statistic: str, feature: str) -> Dict[str, Any]:
        """Format a basic statistical result"""
        result_data = result.get('result', {})
        
        if isinstance(result_data, dict):
            # Grouped results (e.g., by class)
            if 'OK' in result_data and 'KO' in result_data:
                summary = f"Mean {statistic} for {feature}: OK = {result_data['OK']:.2f}, KO = {result_data['KO']:.2f}"
                details = result_data
            else:
                summary = f"{statistic.title()} for {feature}: {result_data}"
                details = result_data
        else:
            summary = f"{statistic.title()} for {feature}: {result_data}"
            details = result_data
        
        return {
            'type': 'statistic',
            'title': f"{statistic.title()} Analysis: {feature}",
            'summary': summary,
            'details': details,
            'statistic_type': statistic,
            'feature': feature
        }
    
    def _format_top_features_response(self, result: Dict[str, Any], feature: str) -> Dict[str, Any]:
        """Format a top features result"""
        features_data = result.get('result', [])
        
        if isinstance(features_data, list) and features_data:
            top_features = []
            for i, feat in enumerate(features_data[:5], 1):  # Show top 5
                if isinstance(feat, dict):
                    name = feat.get('feature_name', f'Feature_{i}')
                    score = feat.get('separation_score', 0)
                    significance = feat.get('statistical_significance', 'Unknown')
                    top_features.append(f"{i}. {name} (Score: {score:.3f}, {significance})")
                else:
                    top_features.append(f"{i}. {feat}")
            
            summary = f"Top discriminative features:\n" + "\n".join(top_features)
        else:
            summary = "No discriminative features found"
            top_features = []
        
        return {
            'type': 'statistic',
            'title': "Top Discriminative Features",
            'summary': summary,
            'details': features_data,
            'top_features': top_features
        }
    
    def _format_comparison_summary(self, result: Dict[str, Any], feature: str, classes: List[str]) -> str:
        """Format a comparison summary"""
        result_data = result.get('result', {})
        
        if 'class1_stats' in result_data and 'class2_stats' in result_data:
            class1_stats = result_data['class1_stats']
            class2_stats = result_data['class2_stats']
            
            summary = f"Comparing {classes[0] if classes else 'Class 1'} vs {classes[1] if len(classes) > 1 else 'Class 2'} for {feature}:\n"
            summary += f"• {classes[0] if classes else 'Class 1'}: Mean = {class1_stats.get('mean', 'N/A'):.2f}, Std = {class1_stats.get('std', 'N/A'):.2f}\n"
            summary += f"• {classes[1] if len(classes) > 1 else 'Class 2'}: Mean = {class2_stats.get('mean', 'N/A'):.2f}, Std = {class2_stats.get('std', 'N/A'):.2f}"
            
            if 'difference' in result_data:
                summary += f"\n• Difference: {result_data['difference']:.2f}"
            
            return summary
        
        return f"Comparison results for {feature} across classes"
    
    def _format_comparison_details(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format comparison details"""
        result_data = result.get('result', {})
        details = {}
        
        if 'p_value' in result_data:
            details['p_value'] = result_data['p_value']
            details['statistical_significance'] = result_data.get('statistical_significance', 'Unknown')
        
        if 'effect_size' in result_data:
            details['effect_size'] = result_data['effect_size']
        
        return details
    
    def _format_comparison_interpretation(self, result: Dict[str, Any]) -> str:
        """Format comparison interpretation"""
        result_data = result.get('result', {})
        
        interpretation = "Based on the analysis: "
        
        if 'statistical_significance' in result_data:
            significance = result_data['statistical_significance']
            if significance == 'Significant':
                interpretation += "There is a statistically significant difference between the classes. "
            else:
                interpretation += "There is no statistically significant difference between the classes. "
        
        if 'effect_size' in result_data:
            effect_size = result_data['effect_size']
            interpretation += f"The effect size is {effect_size}, indicating "
            interpretation += self.effect_size_interpretations.get(effect_size.lower(), "a meaningful difference")
        
        return interpretation
    
    def _format_analysis_summary(self, result: Dict[str, Any], analysis_type: str) -> str:
        """Format an analysis summary"""
        if analysis_type == 'feature':
            return "Feature analysis completed. I've examined the dataset for patterns and relationships."
        elif analysis_type == 'pattern':
            return "Pattern analysis completed. I've identified recurring patterns in the data."
        elif analysis_type == 'trend':
            return "Trend analysis completed. I've examined temporal patterns and trends."
        elif analysis_type == 'correlation':
            return "Correlation analysis completed. I've examined relationships between features."
        else:
            return "Analysis completed. I've examined the dataset for insights and patterns."
    
    def _extract_key_findings(self, result: Dict[str, Any]) -> List[str]:
        """Extract key findings from analysis result"""
        findings = []
        result_data = result.get('result', {})
        
        # Look for common analysis result patterns
        if 'best_discriminative_features' in result_data:
            features = result_data['best_discriminative_features']
            if features:
                top_feature = features[0]
                if isinstance(top_feature, dict):
                    name = top_feature.get('feature_name', 'Unknown')
                    score = top_feature.get('separation_score', 0)
                    findings.append(f"Top discriminative feature: {name} (separation score: {score:.3f})")
        
        if 'class_distribution' in result_data:
            dist = result_data['class_distribution']
            findings.append(f"Class distribution: {', '.join([f'{k}: {v}' for k, v in dist.items()])}")
        
        if 'total_samples' in result_data:
            findings.append(f"Total samples analyzed: {result_data['total_samples']}")
        
        return findings
    
    def _extract_insights(self, result: Dict[str, Any]) -> List[str]:
        """Extract insights from analysis result"""
        insights = []
        result_data = result.get('result', {})
        
        # Look for statistical insights
        if 'statistical_insights' in result_data:
            insights.extend(result_data['statistical_insights'])
        
        # Generate insights based on data
        if 'best_discriminative_features' in result_data:
            features = result_data['best_discriminative_features']
            if len(features) > 0:
                insights.append(f"Found {len(features)} discriminative features that can help distinguish between classes")
        
        if 'model_performance' in result_data:
            performance = result_data['model_performance']
            if performance:
                best_model = max(performance.items(), key=lambda x: x[1].get('accuracy', 0))
                insights.append(f"Best performing model: {best_model[0]} with {best_model[1].get('accuracy', 0):.2f} accuracy")
        
        return insights
    
    def _get_follow_up_suggestions(self, response_type: str, command: Any) -> List[str]:
        """Get follow-up suggestions based on response type"""
        suggestions = self.templates['follow_up_suggestions'].get(response_type, [])
        
        # Customize suggestions based on command content
        if hasattr(command, 'target_features') and command.target_features:
            feature = command.target_features[0]
            suggestions.append(f"Would you like to analyze {feature} in more detail?")
        
        if hasattr(command, 'comparison_classes') and command.comparison_classes:
            classes = ' vs '.join(command.comparison_classes)
            suggestions.append(f"Would you like to see a visual comparison of {classes}?")
        
        return suggestions
    
    def _get_plot_suggestions(self, plot_type: str, features: List[str]) -> List[str]:
        """Get plot suggestions based on plot type and features"""
        suggestions = []
        
        if plot_type == 'histogram':
            for feature in features:
                suggestions.append(f"Generate histogram of {feature}")
        elif plot_type == 'boxplot':
            for feature in features:
                suggestions.append(f"Generate boxplot comparing {feature} across classes")
        elif plot_type == 'scatter':
            if len(features) >= 2:
                suggestions.append(f"Generate scatter plot of {features[0]} vs {features[1]}")
        elif plot_type == 'correlation':
            suggestions.append("Generate correlation matrix of all features")
        elif plot_type == 'comparison':
            for feature in features:
                suggestions.append(f"Generate comparison plot of {feature} across classes")
        
        return suggestions
    
    def _format_error_response(self, result: Dict[str, Any], response_type: str) -> Dict[str, Any]:
        """Format an error response"""
        return {
            'type': 'error',
            'title': f"Error in {response_type}",
            'summary': result.get('message', 'An error occurred'),
            'error_details': result.get('error_details', ''),
            'suggestions': [
                "Please check your request and try again",
                "Make sure the data is properly loaded",
                "Try a different analysis approach"
            ]
        }
    
    def create_plot_request(self, command: Any, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a plot request based on the command and result
        
        Args:
            command: Parsed command object
            result: ML result
            
        Returns:
            Plot request dictionary or None
        """
        try:
            if not hasattr(command, 'target_features') or not command.target_features:
                return None
            
            feature = command.target_features[0]
            
            # Determine appropriate plot type based on command
            if hasattr(command, 'plot_type') and command.plot_type:
                plot_type = command.plot_type
            elif hasattr(command, 'command_type'):
                if command.command_type == 'comparison':
                    plot_type = 'boxplot'
                elif command.command_type == 'statistic':
                    plot_type = 'histogram'
                else:
                    plot_type = 'scatter'
            else:
                plot_type = 'histogram'
            
            # Create plot request
            plot_request = {
                'type': 'plot_request',
                'plot_type': plot_type,
                'features': command.target_features,
                'filters': getattr(command, 'filters', {}),
                'grouping': getattr(command, 'grouping', None),
                'description': f"Generate {plot_type} for {feature}"
            }
            
            return plot_request
            
        except Exception as e:
            self.logger.error(f"Error creating plot request: {e}")
            return None


# Global instance for easy access
response_formatter = ResponseFormatter()

# Convenience functions
def format_statistic_response(result: Dict[str, Any], command: Any) -> Dict[str, Any]:
    """Format a statistical result"""
    return response_formatter.format_statistic_response(result, command)

def format_comparison_response(result: Dict[str, Any], command: Any) -> Dict[str, Any]:
    """Format a comparison result"""
    return response_formatter.format_comparison_response(result, command)

def format_analysis_response(result: Dict[str, Any], command: Any) -> Dict[str, Any]:
    """Format an analysis result"""
    return response_formatter.format_analysis_response(result, command)

def format_plot_response(result: Dict[str, Any], command: Any) -> Dict[str, Any]:
    """Format a plot result"""
    return response_formatter.format_plot_response(result, command)

def create_plot_request(command: Any, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create a plot request"""
    return response_formatter.create_plot_request(command, result)
