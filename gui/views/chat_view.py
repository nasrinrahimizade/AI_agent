import asyncio
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel, 
                               QHBoxLayout, QScrollArea, QFrame, QSizePolicy)
from PySide6.QtGui import QTextCursor, QPixmap, QFont
from PySide6.QtCore import Qt, QEvent, QSize
from datetime import datetime
import re
from core.transformers_backend import Chatbot
from core.ml_plotter import get_plotting_engine, prepare_example_plot, prepare_boxplot_accelerometer, \
                           prepare_temperature_histogram, prepare_correlation_matrix, \
                           prepare_time_series_analysis, prepare_frequency_domain_plot, \
                           prepare_scatter_plot_features
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

class ChatView(QWidget):
    def __init__(self, parent=None, plot_view=None, open_dataset_callback=None):
        super().__init__(parent)
        self.plot_view = plot_view  # Store reference to plot view
        self.open_dataset_callback = open_dataset_callback

        # Initialize plotting engine
        self.plotting_engine = get_plotting_engine()
        
        # Setup UI
        self.setup_ui()
        
        # Initialize chatbot with model directory
        model_dir = "Llama-3.2-1B"
        self.chatbot = Chatbot(model_dir=model_dir)
        
        # Plot request mapping for backward compatibility
        self.plot_mapping = {
            'boxplot': prepare_boxplot_accelerometer,
            'histogram': prepare_temperature_histogram,
            'correlation': prepare_correlation_matrix,
            'time series': prepare_time_series_analysis,
            'frequency': prepare_frequency_domain_plot,
            'scatter': prepare_scatter_plot_features,
            'default': prepare_example_plot
        }

        # Strict allowlist of plot-triggering phrases (lowercase, punctuation-insensitive)
        self.allowed_plot_requests_specific = {
            # Basic plot requests routed to specific plot functions
            'show me a boxplot': 'boxplot',
            'create a histogram': 'histogram',
            'display correlation matrix': 'correlation',
            'generate time series analysis': 'time series',
            'show frequency domain plot': 'frequency',
            'plot scatter relationships': 'scatter',
        }

        # Requests that should be handled by the natural language plotting engine
        self.allowed_plot_requests_engine = set([
            # Sensor-specific
            'show me accelerometer data',
            'analyze temperature sensors',
            'compare pressure readings',
            'display humidity distribution',
            # Advanced
            'show me the most discriminative features',
            'create a comparison between ok and ko conditions',
            'generate a feature relationship matrix',
        ])

    def set_dataframe(self, df):
        # Optional: if ChatView needs the dataframe, store it for context or future use
        self._dataframe = df

    def setup_ui(self):
        """Setup the chat interface UI"""
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        
        # Title
        title = QLabel("AI Chat Assistant")
        title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title)
        
        # Create scroll area for chat content
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Chat content widget
        self.chat_content = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_content)
        self.chat_layout.addStretch()  # Push content to top
        
        self.scroll_area.setWidget(self.chat_content)
        self.layout.addWidget(self.scroll_area)
        
        # Input area
        input_layout = QHBoxLayout()

        self.input_field = QTextEdit()
        self.input_field.setMaximumHeight(100)

        # Set input font to match chat message font size
        input_font = QFont()
        input_font.setPointSize(12)
        self.input_field.setFont(input_font)

        self.input_field.setPlaceholderText("Type your message here...")
        input_layout.addWidget(self.input_field)

        # Right side button layout (vertical)
        button_layout = QVBoxLayout()
        
        # Send button
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self._send_message)
        button_layout.addWidget(self.send_button)

        # Add dataset button under send button
        if self.open_dataset_callback:
            self.dataset_button = QPushButton("+")
            self.dataset_button.setToolTip("Open Dataset")
            self.dataset_button.setMaximumWidth(40)
            self.dataset_button.clicked.connect(self.open_dataset_callback)
            button_layout.addWidget(self.dataset_button)
        
        input_layout.addLayout(button_layout)

        self.layout.addLayout(input_layout)
        
        # Install event filter for Enter key handling
        self.input_field.installEventFilter(self)
        
        # Welcome message
        self._append_message("AI Assistant", "Hello! I'm your AI assistant. I can help you analyze sensor data and create visualizations.", Qt.AlignLeft)

    def eventFilter(self, obj, event):
        """Handle Enter key press in input field"""
        if obj is self.input_field and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Return and not (event.modifiers() & Qt.ShiftModifier):
                self._send_message()
                return True
        return super().eventFilter(obj, event)

    def _append_message(self, sender: str, message: str, alignment=Qt.AlignLeft, plot_fig=None):
        """Append a message to the chat display"""
        # Create message container
        message_container = QWidget()
        message_layout = QVBoxLayout(message_container)
        
        # Format the message with proper timestamp
        timestamp = datetime.now().strftime("%H:%M")
        message_text = f"[{timestamp}] {sender}: {message}"
        
        # Create message label
        message_label = QLabel(message_text)
        message_label.setWordWrap(True)
        message_label.setAlignment(alignment)
        
        # Set chat font
        chat_font = QFont()
        chat_font.setPointSize(12)
        message_label.setFont(chat_font)
        
        message_layout.addWidget(message_label)
        
        # Add plot if provided
        if plot_fig:
            # Create a canvas for the plot
            canvas = FigureCanvas(plot_fig)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            canvas.setMinimumSize(400, 300)
            message_layout.addWidget(canvas)
            # Draw the plot
            canvas.draw()
        
        # Add message to chat layout (before the stretch)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, message_container)
        
        # Scroll to bottom
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )

    def _send_message(self):
        """Send the current message"""
        message = self.input_field.toPlainText().strip()
        if not message:
            return
        
        # Display user message
        self._append_message("You", message, Qt.AlignRight)
        
        # Clear input field
        self.input_field.clear()
        
        # Get AI response
        self._ai_reply(message)

    def _ai_reply(self, user_message: str):
        """Get AI response and handle plot triggers"""
        try:
            # Get AI response using the correct method
            ai_response = self.chatbot.generate(user_message)
            
            # Check for new trigger markers first (e.g., [TRIGGER_PLOT:histogram])
            plot_fig = self._check_trigger_markers(ai_response)
            
            # If no trigger markers found, check old plot triggers for backward compatibility
            if not plot_fig:
                plot_fig = self._check_plot_triggers(ai_response, user_message)
            
            # Clean the response by removing trigger markers before displaying
            clean_response = self._clean_response_from_triggers(ai_response)
            
            # Display AI response with plot if available
            self._append_message("AI Assistant", clean_response, Qt.AlignLeft, plot_fig)
            
        except Exception as e:
            error_msg = f"Error getting AI response: {str(e)}"
            self._append_message("System", error_msg, Qt.AlignLeft)

    def _check_plot_triggers(self, ai_response: str, user_message: str):
        """Check AI response for keywords that should trigger plot display"""
        response_lower = ai_response.lower()
        user_lower = user_message.lower()

        # Normalize user text for strict allowlist matching
        normalized_user = self._normalize_text(user_lower)
        
        # Strict allowlist: specific plot functions
        if normalized_user in self.allowed_plot_requests_specific:
            plot_type = self.allowed_plot_requests_specific[normalized_user]
            return self.trigger_specific_plot(plot_type)

        # Strict allowlist: engine-handled requests
        if normalized_user in self.allowed_plot_requests_engine:
            return self.trigger_natural_plot_request(user_message)
        
        return None

    def _detect_natural_plot_request(self, user_message: str, ai_response: str) -> str:
        """Detect natural language plot requests"""
        text = user_message.lower()

        # Action verbs indicating intent to produce something
        action_keywords = ['show', 'display', 'plot', 'visualize', 'create', 'generate', 'draw', 'render']
        # Plot-related nouns/types
        noun_keywords = [
            'plot', 'chart', 'graph', 'visualization',
            'boxplot', 'histogram', 'correlation', 'scatter', 'time series',
            'frequency', 'fft', 'spectrum', 'distribution', 'comparison', 'matrix'
        ]

        # Word-boundary match for single-word actions to avoid matching 'created' for 'create'
        def has_action_intent(t: str) -> bool:
            for word in action_keywords:
                if ' ' in word:  # simple substring for multi-word phrases
                    if word in t:
                        return True
                else:
                    if re.search(rf"\\b{re.escape(word)}\\b", t):
                        return True
            return False

        def has_plot_noun(t: str) -> bool:
            for word in noun_keywords:
                if word in t:
                    return True
            return False

        if has_action_intent(text) and has_plot_noun(text):
            return user_message

        return None

    def _detect_plot_type(self, response_lower: str) -> str:
        """Detect specific plot type from user message (requires plot intent)"""
        # Require explicit plot intent to avoid accidental triggers on casual mentions
        intent_words = ['show', 'display', 'plot', 'visualize', 'create', 'generate', 'draw', 'render', 'chart', 'graph', 'visualization']
        has_intent = any(
            (re.search(rf"\\b{re.escape(w)}\\b", response_lower) if ' ' not in w else w in response_lower)
            for w in intent_words
        )
        if not has_intent:
            return None

        if any(word in response_lower for word in ['boxplot', 'accelerometer', 'acceleration']):
            return 'boxplot'
        elif any(word in response_lower for word in ['histogram', 'distribution']):
            return 'histogram'
        elif any(word in response_lower for word in ['correlation', 'matrix', 'relationships']):
            return 'correlation'
        elif any(word in response_lower for word in ['time series', 'temporal', 'over time']):
            return 'time series'
        elif any(word in response_lower for word in ['frequency', 'fft', 'spectrum']):
            return 'frequency'
        elif any(word in response_lower for word in ['scatter', 'features', 'relationships']):
            return 'scatter'
        return None

    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing punctuation and collapsing whitespace for robust matching."""
        # Remove non-word, non-space characters (Python re doesn't support Unicode \p classes)
        text = re.sub(r"[^\w\s]", " ", text)
        # Collapse multiple spaces and trim
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def trigger_plot_display(self):
        """
        Generate default plot and display it in the chat
        """
        try:
            fig = prepare_example_plot()
            return fig
        except Exception as e:
            self._append_message("System", f"Error displaying plot: {str(e)}", Qt.AlignLeft)
            return None

    def trigger_natural_plot_request(self, request: str):
        """Handle natural language plot requests using the plotting engine"""
        try:
            # Use the plotting engine to handle the request
            fig = self.plotting_engine.handle_plot_request(request)
            return fig
        except Exception as e:
            self._append_message("System", f"Error generating plot: {str(e)}", Qt.AlignLeft)
            return None

    def trigger_specific_plot(self, plot_type: str):
        """
        Generate and display a specific type of plot
        """
        try:
            plot_function = self.plot_mapping.get(plot_type, prepare_example_plot)
            fig = plot_function()
            return fig
        except Exception as e:
            self._append_message("System", f"Error displaying {plot_type} plot: {str(e)}", Qt.AlignLeft)
            return None

    def show_sales_plot(self):
        """Alias for trigger_plot_display for backward compatibility"""
        return self.trigger_plot_display()

    def _check_trigger_markers(self, ai_response: str):
        """Check AI response for new trigger markers like [TRIGGER_PLOT:histogram]"""
        # Look for trigger markers in the response
        trigger_pattern = r'\[TRIGGER_PLOT:(\w+)\]'
        analysis_pattern = r'\[TRIGGER_ANALYSIS:(\w+)\]'
        
        # Check for plot triggers
        plot_match = re.search(trigger_pattern, ai_response)
        if plot_match:
            plot_type = plot_match.group(1)
            return self._handle_plot_trigger(plot_type)
        
        # Check for analysis triggers
        analysis_match = re.search(analysis_pattern, ai_response)
        if analysis_match:
            analysis_type = analysis_match.group(1)
            return self._handle_analysis_trigger(analysis_type)
        
        return None

    def _handle_plot_trigger(self, plot_type: str):
        """Handle plot triggers based on plot type"""
        try:
            if plot_type in ['histogram', 'boxplot', 'scatter', 'correlation', 'timeseries', 'line', 'bar', 'pie']:
                # Basic plot types - use existing specific plot functions
                if plot_type == 'histogram':
                    return self.trigger_specific_plot('histogram')
                elif plot_type == 'boxplot':
                    return self.trigger_specific_plot('boxplot')
                elif plot_type == 'correlation':
                    return self.trigger_specific_plot('correlation')
                elif plot_type == 'timeseries':
                    return self.trigger_specific_plot('time series')
                elif plot_type == 'scatter':
                    return self.trigger_specific_plot('scatter')
                else:
                    # For other plot types, use the plotting engine
                    return self.trigger_natural_plot_request(f"create {plot_type}")
            
            elif plot_type in ['temperature_analysis', 'pressure_analysis', 'humidity_analysis', 'motion_analysis', 'magnetic_analysis']:
                # Sensor-specific analysis - use plotting engine
                sensor_map = {
                    'temperature_analysis': 'analyze temperature sensors',
                    'pressure_analysis': 'analyze pressure readings',
                    'humidity_analysis': 'analyze humidity distribution',
                    'motion_analysis': 'analyze accelerometer data',
                    'magnetic_analysis': 'analyze magnetometer data'
                }
                return self.trigger_natural_plot_request(sensor_map.get(plot_type, f"analyze {plot_type}"))
            
            elif plot_type == 'general_visualization':
                # General visualization request
                return self.trigger_natural_plot_request("create a general visualization")
            
            else:
                # Unknown plot type - use plotting engine
                return self.trigger_natural_plot_request(f"create {plot_type}")
                
        except Exception as e:
            self._append_message("System", f"Error handling plot trigger '{plot_type}': {str(e)}", Qt.AlignLeft)
            return None

    def _handle_analysis_trigger(self, analysis_type: str):
        """Handle analysis triggers"""
        try:
            if analysis_type == 'statistical_analysis':
                return self.trigger_natural_plot_request("perform statistical analysis")
            elif analysis_type == 'comparison_analysis':
                return self.trigger_natural_plot_request("create comparison analysis")
            elif analysis_type == 'trend_analysis':
                return self.trigger_natural_plot_request("analyze trends and patterns")
            elif analysis_type == 'correlation_analysis':
                return self.trigger_natural_plot_request("analyze correlations")
            else:
                return self.trigger_natural_plot_request(f"perform {analysis_type}")
        except Exception as e:
            self._append_message("System", f"Error handling analysis trigger '{analysis_type}': {str(e)}", Qt.AlignLeft)
            return None

    def _clean_response_from_triggers(self, ai_response: str) -> str:
        """Remove trigger markers from AI response before displaying to user"""
        # Remove all trigger markers
        clean_response = re.sub(r'\[TRIGGER_PLOT:\w+\]', '', ai_response)
        clean_response = re.sub(r'\[TRIGGER_ANALYSIS:\w+\]', '', clean_response)
        
        # Clean up any extra whitespace
        clean_response = re.sub(r'\s+', ' ', clean_response).strip()
        
        return clean_response
