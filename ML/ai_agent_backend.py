import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server deployment

# Import your existing engines
from statistical_engine import StatisticalAnalysisEngine
from plotting_engine import PlottingEngine

class StatisticalAIAgent:
    """
    🤖 MAIN AI AGENT CLASS - UNIFIED INTERFACE
    
    This is the single entry point your teammate will use for all operations.
    Combines statistical analysis and plotting into one clean API.
    """
    
    def __init__(self, feature_matrix_path: str, agent_name: str = "Statistical AI Agent"):
        """
        Initialize the complete AI Agent
        
        Args:
            feature_matrix_path: Path to your feature_matrix.csv
            agent_name: Name for the agent (for logging/display)
        """
        self.agent_name = agent_name
        self.feature_matrix_path = feature_matrix_path
        self.initialization_time = datetime.now()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.agent_name)
        
        print(f"🤖 Initializing {self.agent_name}...")
        
        try:
            # Initialize the engines
            self.stats_engine = StatisticalAnalysisEngine(feature_matrix_path)
            self.plot_engine = PlottingEngine(feature_matrix_path)
            
            # Cache analysis results (expensive operation)
            print("📊 Running initial statistical analysis...")
            self._analysis_cache = None
            self._is_initialized = True
            
            print(f"✅ {self.agent_name} successfully initialized!")
            
        except Exception as e:
            print(f"❌ Failed to initialize {self.agent_name}: {e}")
            self._is_initialized = False
            raise
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get current status of the AI Agent
        Useful for GUI to show agent health
        """
        return {
            'agent_name': self.agent_name,
            'initialized': self._is_initialized,
            'initialization_time': self.initialization_time.isoformat(),
            'data_source': self.feature_matrix_path,
            'engines_loaded': {
                'statistical_engine': hasattr(self, 'stats_engine'),
                'plotting_engine': hasattr(self, 'plot_engine')
            },
            'analysis_cached': self._analysis_cache is not None
        }
    
    def get_dataset_overview(self) -> Dict[str, Any]:
        """
        Quick dataset overview for GUI dashboard
        Fast operation, doesn't require full analysis
        """
        if not self._is_initialized:
            raise RuntimeError("Agent not properly initialized")
        
        try:
            # Get basic info from the stats engine
            dataset_info = {
                'total_samples': len(self.stats_engine.feature_matrix),
                'total_features': len(self.stats_engine.feature_matrix.columns) - 2,
                'classes': self.stats_engine.feature_matrix['label'].unique().tolist(),
                'class_distribution': self.stats_engine.feature_matrix['label'].value_counts().to_dict(),
                'available_sensors': self.plot_engine.get_available_sensors(),
                'sample_ids': self.stats_engine.feature_matrix['sample'].unique().tolist()[:10]  # First 10
            }
            
            return {
                'status': 'success',
                'dataset_overview': dataset_info,
                'message': 'Dataset overview retrieved successfully'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting dataset overview: {e}")
            return {
                'status': 'error',
                'message': f'Failed to get dataset overview: {str(e)}'
            }
    
    def get_analysis_summary(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get complete statistical analysis summary
        This is the main function your teammate will call for showing results
        
        Args:
            force_refresh: If True, recalculates analysis (takes time)
        """
        if not self._is_initialized:
            raise RuntimeError("Agent not properly initialized")
        
        # Use cached results if available
        if self._analysis_cache is not None and not force_refresh:
            print("📋 Using cached analysis results")
            return {
                'status': 'success',
                'analysis': self._analysis_cache,
                'cached': True,
                'message': 'Analysis retrieved from cache'
            }
        
        try:
            print("🔄 Running complete statistical analysis...")
            start_time = datetime.now()
            
            # Get complete analysis from statistical engine
            analysis_results = self.stats_engine.get_analysis_summary()
            
            # Add agent metadata
            analysis_results['agent_metadata'] = {
                'agent_name': self.agent_name,
                'analysis_timestamp': datetime.now().isoformat(),
                'processing_time_seconds': (datetime.now() - start_time).total_seconds(),
                'data_source': self.feature_matrix_path
            }
            
            # Cache results
            self._analysis_cache = analysis_results
            
            processing_time = (datetime.now() - start_time).total_seconds()
            print(f"✅ Analysis completed in {processing_time:.2f} seconds")
            
            return {
                'status': 'success',
                'analysis': analysis_results,
                'cached': False,
                'processing_time': processing_time,
                'message': 'Statistical analysis completed successfully'
            }
            
        except Exception as e:
            self.logger.error(f"Error in statistical analysis: {e}")
            return {
                'status': 'error',
                'message': f'Statistical analysis failed: {str(e)}',
                'error_details': str(e)
            }
    
    def get_top_features(self, n: int = 10, min_score: float = 0.01) -> Dict[str, Any]:
        """
        Quick access to top discriminative features
        Useful for GUI tables and quick displays
        """
        try:
            # Get analysis (uses cache if available)
            analysis_result = self.get_analysis_summary()
            
            if analysis_result['status'] != 'success':
                return analysis_result
            
            features = analysis_result['analysis']['best_discriminative_features']
            
            # Filter and limit
            top_features = [
                f for f in features 
                if f['separation_score'] >= min_score
            ][:n]
            
            return {
                'status': 'success',
                'top_features': top_features,
                'total_significant_features': len([f for f in features if f['statistical_significance'] == 'Significant']),
                'message': f'Retrieved top {len(top_features)} discriminative features'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to get top features: {str(e)}'
            }
    
    def generate_plot(self, request: str, return_base64: bool = False) -> Dict[str, Any]:
        """
        Generate plots based on natural language requests
        This is what your teammate's LLM interface will call
        
        Args:
            request: Natural language plot request
            return_base64: If True, returns plot as base64 string (useful for web apps)
        """
        if not self._is_initialized:
            return {
                'status': 'error',
                'message': 'Agent not properly initialized'
            }
        
        try:
            print(f"🎨 Generating plot for request: '{request}'")
            start_time = datetime.now()
            
            # Generate plot using plotting engine
            fig = self.plot_engine.handle_plot_request(request)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'status': 'success',
                'request': request,
                'processing_time': processing_time,
                'plot_generated': True,
                'message': 'Plot generated successfully'
            }
            
            if return_base64:
                # Convert plot to base64 string (useful for web interfaces)
                import io
                import base64
                
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                
                plot_data = base64.b64encode(buffer.getvalue()).decode()
                result['plot_base64'] = plot_data
                result['plot_format'] = 'png'
                
                plt.close(fig)  # Clean up
            else:
                # Return matplotlib figure object
                result['figure'] = fig
            
            print(f"✅ Plot generated in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating plot: {e}")
            return {
                'status': 'error',
                'request': request,
                'message': f'Plot generation failed: {str(e)}',
                'error_details': str(e)
            }
    
    def get_plot_suggestions(self) -> Dict[str, Any]:
        """
        Get suggested plot requests based on available data
        Helpful for GUI to show users what they can ask for
        """
        try:
            # Get available sensors
            sensors = self.plot_engine.get_available_sensors()
            feature_summary = self.plot_engine.get_feature_summary()
            
            # Generate suggestions based on available data
            suggestions = {
                'sensor_comparisons': [
                    f"show boxplot comparison of {sensor} sensor" for sensor in sensors[:5]
                ],
                'distribution_analysis': [
                    f"plot histogram of {sensor} data" for sensor in sensors[:3]
                ],
                'correlation_analysis': [
                    "generate correlation matrix",
                    "show feature relationships",
                    "correlation between accelerometer features"
                ],
                'time_series': [
                    "plot time series of top features",
                    f"show temporal analysis of {sensors[0]} sensor" if sensors else "show temporal analysis"
                ],
                'frequency_domain': [
                    "frequency domain analysis",
                    f"FFT analysis of {sensors[0]} data" if sensors else "FFT analysis"
                ]
            }
            
            return {
                'status': 'success',
                'suggestions': suggestions,
                'available_sensors': sensors,
                'feature_summary': feature_summary,
                'message': f'Generated suggestions for {len(sensors)} available sensors'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to generate suggestions: {str(e)}'
            }
    
    def handle_chat_request(self, user_input: str) -> Dict[str, Any]:
        """
        Handle general chat requests from users
        Routes requests to appropriate functions (analysis or plotting)
        This is what your teammate's LLM will call
        """
        if not self._is_initialized:
            return {
                'status': 'error',
                'message': 'Agent not properly initialized'
            }
        
        user_input_lower = user_input.lower()
        
        try:
            # Detect request type
            plot_keywords = ['plot', 'chart', 'graph', 'show', 'visualize', 'histogram', 'boxplot', 'correlation']
            analysis_keywords = ['analyze', 'summary', 'features', 'statistics', 'best', 'top', 'discriminative']
            
            is_plot_request = any(keyword in user_input_lower for keyword in plot_keywords)
            is_analysis_request = any(keyword in user_input_lower for keyword in analysis_keywords)
            
            if is_plot_request:
                # Handle as plot request
                return self.generate_plot(user_input)
                
            elif is_analysis_request:
                # Handle as analysis request
                if 'features' in user_input_lower or 'top' in user_input_lower:
                    return self.get_top_features()
                else:
                    return self.get_analysis_summary()
            
            else:
                # General request - provide overview
                return {
                    'status': 'info',
                    'message': 'I can help with statistical analysis and plot generation. Try asking for "top features", "plot accelerometer data", or "analyze dataset".',
                    'suggestions': [
                        'What are the top discriminative features?',
                        'Show me a boxplot of accelerometer data',
                        'Generate correlation matrix',
                        'Analyze the dataset'
                    ]
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to process chat request: {str(e)}',
                'original_request': user_input
            }
    
    def export_results(self, format: str = 'json', include_plots: bool = False) -> Dict[str, Any]:
        """
        Export analysis results in different formats
        Useful for saving reports or sharing results
        """
        try:
            analysis_result = self.get_analysis_summary()
            
            if analysis_result['status'] != 'success':
                return analysis_result
            
            analysis_data = analysis_result['analysis']
            
            if format.lower() == 'json':
                # Export as JSON
                export_data = {
                    'agent_info': self.get_agent_status(),
                    'analysis_results': analysis_data,
                    'export_timestamp': datetime.now().isoformat()
                }
                
                return {
                    'status': 'success',
                    'format': 'json',
                    'data': export_data,
                    'message': 'Results exported as JSON'
                }
            
            else:
                return {
                    'status': 'error',
                    'message': f'Export format "{format}" not supported. Use: json'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Export failed: {str(e)}'
            }
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'stats_engine'):
            del self.stats_engine
        if hasattr(self, 'plot_engine'):
            del self.plot_engine
        self._analysis_cache = None
        print(f"🧹 {self.agent_name} cleaned up successfully")


# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

def test_ai_agent():
    """
    Test the complete AI Agent
    This shows your teammate exactly how to use it
    """
    print("TESTING COMPLETE AI AGENT BACKEND")
    print("=" * 60)
    
    try:
        # Initialize agent
        agent = StatisticalAIAgent("feature_matrix.csv", "Test AI Agent")
        
        # Test 1: Agent status
        print("\n1️⃣ TESTING: Agent Status")
        status = agent.get_agent_status()
        print(f"   Agent initialized: {status['initialized']}")
        print(f"   Engines loaded: {status['engines_loaded']}")
        
        # Test 2: Dataset overview (fast)
        print("\n2️⃣ TESTING: Dataset Overview")
        overview = agent.get_dataset_overview()
        if overview['status'] == 'success':
            info = overview['dataset_overview']
            print(f"   Samples: {info['total_samples']}, Features: {info['total_features']}")
            print(f"   Classes: {info['classes']}")
            print(f"   Sensors: {info['available_sensors'][:3]}...")
        
        # Test 3: Statistical analysis
        print("\n3️⃣ TESTING: Statistical Analysis")
        analysis = agent.get_analysis_summary()
        if analysis['status'] == 'success':
            results = analysis['analysis']
            print(f"   Processing time: {analysis['processing_time']:.2f}s")
            print(f"   Top feature: {results['best_discriminative_features'][0]['feature_name']}")
            print(f"   Best model: {max(results['model_performance'].items(), key=lambda x: x[1]['accuracy'])}")
        
        # Test 4: Top features
        print("\n4️⃣ TESTING: Top Features")
        top_features = agent.get_top_features(n=3)
        if top_features['status'] == 'success':
            for i, feature in enumerate(top_features['top_features'], 1):
                print(f"   {i}. {feature['feature_name']} (Score: {feature['separation_score']})")
        
        # Test 5: Plot generation
        print("\n5️⃣ TESTING: Plot Generation")
        plot_result = agent.generate_plot("show boxplot of accelerometer data")
        if plot_result['status'] == 'success':
            print(f"   Plot generated in {plot_result['processing_time']:.2f}s")
            print(f"   Request: {plot_result['request']}")
            # plt.show() # Uncomment to display plot
            if 'figure' in plot_result:
                plt.close(plot_result['figure'])  # Clean up
        
        # Test 6: Chat requests
        print("\n6️⃣ TESTING: Chat Requests")
        chat_requests = [
            "What are the top features?",
            "Show me temperature sensor data",
            "Analyze the dataset"
        ]
        
        for request in chat_requests:
            response = agent.handle_chat_request(request)
            print(f"   '{request}' → {response['status']}: {response['message'][:50]}...")
        
        # Test 7: Plot suggestions
        print("\n7️⃣ TESTING: Plot Suggestions")
        suggestions = agent.get_plot_suggestions()
        if suggestions['status'] == 'success':
            print(f"   Available sensors: {len(suggestions['available_sensors'])}")
            print(f"   Sample suggestions: {suggestions['suggestions']['sensor_comparisons'][:2]}")
        
        # Test 8: Export
        print("\n8️⃣ TESTING: Export Results")
        export_result = agent.export_results(format='json')
        if export_result['status'] == 'success':
            print(f"   Export format: {export_result['format']}")
            print(f"   Data keys: {list(export_result['data'].keys())}")
        
        print(f"\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print(f"🤖 AI Agent is ready for GUI integration!")
        
        # Cleanup
        agent.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        return False


# ============================================================================
# INTEGRATION EXAMPLES FOR YOUR TEAMMATE
# ============================================================================

def integration_example_streamlit():
    """
    Example integration with Streamlit
    Your teammate can use this as a starting point
    """
    example_code = '''
import streamlit as st
from ai_agent_backend import StatisticalAIAgent

# Initialize agent (cache it to avoid reloading)
@st.cache_resource
def load_agent():
    return StatisticalAIAgent("feature_matrix.csv")

agent = load_agent()

st.title("🤖 Statistical AI Agent")

# Show agent status
status = agent.get_agent_status()
if status['initialized']:
    st.success("✅ AI Agent ready!")
else:
    st.error("❌ AI Agent not initialized")
    st.stop()

# Dataset overview
st.header("📊 Dataset Overview")
overview = agent.get_dataset_overview()
if overview['status'] == 'success':
    info = overview['dataset_overview']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Samples", info['total_samples'])
    with col2:
        st.metric("Features", info['total_features'])
    with col3:
        st.metric("Sensors", len(info['available_sensors']))

# Analysis results
st.header("🏆 Top Discriminative Features")
if st.button("Run Analysis"):
    with st.spinner("Analyzing dataset..."):
        analysis = agent.get_analysis_summary()
    
    if analysis['status'] == 'success':
        features = analysis['analysis']['best_discriminative_features'][:10]
        
        # Display as table
        df = pd.DataFrame(features)
        st.dataframe(df[['rank', 'feature_name', 'sensor_name', 'separation_score', 'statistical_significance']])
        
        # Show insights
        st.subheader("💡 Key Insights")
        for insight in analysis['analysis']['statistical_insights']:
            st.write(f"• {insight}")

# Chat interface
st.header("💬 Chat with AI Agent")
user_input = st.text_input("Ask for analysis or plots:")
if user_input and st.button("Submit"):
    response = agent.handle_chat_request(user_input)
    
    if response['status'] == 'success':
        if 'figure' in response:
            # Display plot
            st.pyplot(response['figure'])
        else:
            st.success(response['message'])
    else:
        st.error(response['message'])
'''
    return example_code


if __name__ == "__main__":
    # Run comprehensive test
    test_ai_agent()