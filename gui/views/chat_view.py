import asyncio
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel, 
                               QHBoxLayout, QScrollArea, QFrame, QSizePolicy)
from PySide6.QtGui import QTextCursor, QPixmap, QFont
from PySide6.QtCore import Qt, QEvent, QSize
from datetime import datetime
import re
from core.transformers_backend import Chatbot
from core.ml_plotter import get_plotting_engine, prepare_example_plot, prepare_line_graph_accelerometer, \
                           prepare_temperature_histogram, prepare_correlation_matrix, \
                           prepare_time_series_analysis, prepare_frequency_domain_plot, \
                           prepare_scatter_plot_features
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from core.ml_interface import ml_interface
# from PySide6.QtCore import Qt, QEvent, QSize, QThread, pyqtSignal
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QEvent, QSize, QThread, Signal

class AIWorkerThread(QThread):
    # Signals to communicate with main thread
    response_ready = Signal(str)    # AI response text
    plot_ready = Signal(object)     # Plot figure object
    error_occurred = Signal(str)    # Error message
    
    def __init__(self, chatbot, plotting_engine, parent=None):
        super().__init__(parent)
        self.chatbot = chatbot
        self.plotting_engine = plotting_engine
        self.user_message = ""
        self.chat_view = None
        
    def set_request(self, user_message, chat_view):
        self.user_message = user_message
        self.chat_view = chat_view
        
    def run(self):
        """This runs in the background thread"""
        try:
            # Generate AI response
            ai_response = self.chatbot.generate(self.user_message)
            self.response_ready.emit(ai_response)
            
            # Check for plot triggers and generate plot if needed
            plot_fig = None
            
            # Check for trigger markers
            plot_fig = self.chat_view._check_trigger_markers(ai_response)
            
            # If no trigger markers, check old plot triggers
            if not plot_fig:
                plot_fig = self.chat_view._check_plot_triggers(ai_response, self.user_message)
            
            # Emit plot if generated
            if plot_fig:
                self.plot_ready.emit(plot_fig)
            else:
                self.plot_ready.emit(None)
                
        except Exception as e:
            self.error_occurred.emit(str(e))
            
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
        
        # Create worker thread
        self.ai_worker = AIWorkerThread(self.chatbot, self.plotting_engine, self)
        
        # Connect signals
        self.ai_worker.response_ready.connect(self._on_response_ready)
        self.ai_worker.plot_ready.connect(self._on_plot_ready)
        self.ai_worker.error_occurred.connect(self._on_error)
        
        # Track loading state
        self.is_loading = False
        self.pending_response = ""
        self.pending_plot = None

        # Plot request mapping for backward compatibility
        self.plot_mapping = {
            'line graph': prepare_line_graph_accelerometer,
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
            'show me a line graph': 'line graph',
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
            'generate a plot of accelerometer data',
            'show me humidity data',
            'create a line graph of temperature data',
            'analyze accelerometer data',
            'show temperature data',
            'display pressure data',
            'create humidity plot',
            'generate accelerometer plot',
            'show me temperature sensors',
            'analyze humidity sensors',
            'compare accelerometer readings',
            # Advanced
            'show me the most discriminative features',
            'create a comparison between ok and ko conditions',
            'generate a feature relationship matrix',
        ])

    def _show_loading_indicator(self):
        """Show loading indicator"""
        from PySide6.QtCore import QTimer
        
        # Create loading message
        timestamp = datetime.now().strftime("%H:%M")
        loading_text = f"[{timestamp}] AI Assistant: Thinking..."
        
        self.loading_label = QLabel(loading_text)
        self.loading_label.setWordWrap(True)
        self.loading_label.setAlignment(Qt.AlignLeft)
        
        # Style as loading
        loading_font = QFont()
        loading_font.setPointSize(12)
        loading_font.setItalic(True)
        self.loading_label.setFont(loading_font)
        self.loading_label.setStyleSheet("color: #666666;")
        
        # Create container
        self.loading_container = QWidget()
        loading_layout = QVBoxLayout(self.loading_container)
        loading_layout.addWidget(self.loading_label)
        
        # Add to chat
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, self.loading_container)
        
        # Animate dots
        self.dot_count = 0
        self.dots_timer = QTimer()
        self.dots_timer.timeout.connect(self._update_loading_dots)
        self.dots_timer.start(500)
        
        # Scroll to bottom
        self._scroll_to_bottom()

    def _update_loading_dots(self):
        """Update loading animation"""
        dots = "." * (self.dot_count % 4)
        timestamp = datetime.now().strftime("%H:%M")
        self.loading_label.setText(f"[{timestamp}] AI Assistant: Thinking{dots}")
        self.dot_count += 1

    def _hide_loading_indicator(self):
        """Remove loading indicator"""
        if hasattr(self, 'dots_timer'):
            self.dots_timer.stop()
            self.dots_timer.deleteLater()
        if hasattr(self, 'loading_container'):
            self.loading_container.setParent(None)
            self.loading_container.deleteLater()

    def _scroll_to_bottom(self):
        """Scroll chat to bottom"""
        QApplication.processEvents()  # Process pending events
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )

    def set_dataframe(self, df):
        # Store the dataframe
        self._dataframe = df
        
        # Get dataset overview from ML interface
        overview = ml_interface.get_dataset_overview()
        
        # Display the overview in chat
        self._display_dataset_overview(overview)

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
        
        # Set chat font (default, no font changes for tables) and preserve wrapping
        chat_font = QFont()
        chat_font.setPointSize(12)
        message_label.setFont(chat_font)
        message_label.setTextFormat(Qt.PlainText)
        message_label.setWordWrap(True)
        # Enable text selection and copy (Ctrl+C)
        message_label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        message_label.setFocusPolicy(Qt.StrongFocus)
        
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
        """Send message using async worker thread"""
        message = self.input_field.toPlainText().strip()
        if not message:
            return
        
        # Prevent multiple simultaneous requests
        if self.is_loading:
            return
        
        # Display user message
        self._append_message("You", message, Qt.AlignRight)
        
        # Clear input field
        self.input_field.clear()
        
        # Show loading and start async processing
        self.is_loading = True
        self._show_loading_indicator()
        
        # Start worker thread
        self.ai_worker.set_request(message, self)
        self.ai_worker.start()

    # def _ai_reply(self, user_message: str):
    #     """Get AI response and handle plot triggers"""
    #     try:
    #         # Get AI response using the correct method
    #         ai_response = self.chatbot.generate(user_message)
            
    #         # Check for new trigger markers first (e.g., [TRIGGER_PLOT:histogram])
    #         plot_fig = self._check_trigger_markers(ai_response)
            
    #         # If no trigger markers found, check old plot triggers for backward compatibility
    #         if not plot_fig:
    #             plot_fig = self._check_plot_triggers(ai_response, user_message)
            
    #         # Clean the response by removing trigger markers before displaying
    #         clean_response = self._clean_response_from_triggers(ai_response)
            
    #         # Display AI response with plot if available
    #         self._append_message("AI Assistant", clean_response, Qt.AlignLeft, plot_fig)
            
    #     except Exception as e:
    #         error_msg = f"Error getting AI response: {str(e)}"
    #         self._append_message("System", error_msg, Qt.AlignLeft)

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
        
        # NEW: Detect natural language plot requests that weren't caught by allowlists
        if self._is_natural_plot_request(user_lower):
            return self.trigger_natural_plot_request(user_message)
        
        return None

    def _is_natural_plot_request(self, user_message: str) -> bool:
        """Detect if a user message is a natural language plot request"""
        user_lower = user_message.lower()
        
        # Action verbs indicating intent to produce something
        action_keywords = ['show', 'display', 'plot', 'visualize', 'create', 'generate', 'draw', 'render', 'make']
        
        # Sensor-related keywords
        sensor_keywords = ['temperature', 'temp', 'humidity', 'hum', 'pressure', 'press', 'accelerometer', 'acc', 
                          'gyroscope', 'gyro', 'magnetometer', 'mag', 'microphone', 'mic', 'sensor', 'data']
        
        # Plot-related keywords
        plot_keywords = ['plot', 'chart', 'graph', 'visualization', 'line', 'histogram', 'correlation', 'scatter', 
                        'time series', 'frequency', 'fft', 'spectrum', 'distribution', 'comparison', 'matrix']
        
        # Check if message contains action intent
        has_action = any(keyword in user_lower for keyword in action_keywords)
        
        # Check if message contains sensor or plot keywords
        has_sensor_or_plot = any(keyword in user_lower for keyword in sensor_keywords + plot_keywords)
        
        # Additional check for specific patterns
        specific_patterns = [
            'generate a plot',
            'show me',
            'create a',
            'display',
            'analyze',
            'compare'
        ]
        has_specific_pattern = any(pattern in user_lower for pattern in specific_patterns)
        
        return (has_action and has_sensor_or_plot) or has_specific_pattern

    def _detect_natural_plot_request(self, user_message: str, ai_response: str) -> str:
        """Detect natural language plot requests"""
        text = user_message.lower()

        # Action verbs indicating intent to produce something
        action_keywords = ['show', 'display', 'plot', 'visualize', 'create', 'generate', 'draw', 'render']
        # Plot-related nouns/types
        noun_keywords = [
            'plot', 'chart', 'graph', 'visualization', 'histogram', 'correlation', 'scatter', 'time series',
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

        if any(word in response_lower for word in ['line graph', 'accelerometer', 'acceleration']):
            return 'line graph'
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
            if plot_type in ['histogram', 'line graph', 'scatter', 'correlation', 'timeseries', 'line', 'bar', 'pie']:
                # Basic plot types - use existing specific plot functions
                if plot_type == 'histogram':
                    return self.trigger_specific_plot('histogram')
                elif plot_type == 'line graph' or plot_type == 'line':
                    return self.trigger_specific_plot('line graph')
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
        
        # Strip any code blocks or inline code the model may have generated
        clean_response = re.sub(r"```[\s\S]*?```", "", clean_response)  # fenced code
        clean_response = re.sub(r"`[^`]*`", "", clean_response)          # inline code
        clean_response = re.sub(r"<pre[\s\S]*?>[\s\S]*?</pre>", "", clean_response, flags=re.IGNORECASE)
        
        # Clean up any extra whitespace
        clean_response = re.sub(r'\s+', ' ', clean_response).strip()
        
        return clean_response
    
    def _display_dataset_overview(self, overview):
        """Display dataset overview in the chat"""
        if overview['status'] == 'success':
            # Format the overview message
            message = self._format_overview_message(overview)
            self._append_message("System", message, Qt.AlignLeft)
        else:
            error_msg = f"Could not analyze dataset: {overview.get('message', 'Unknown error')}"
            self._append_message("System", error_msg, Qt.AlignLeft)

    def _format_overview_message(self, overview):
        """Format the overview data into a readable message"""
        msg = "📊 **Dataset Overview**\n\n"
        
        # Basic info
        msg += f"• **Samples**: {overview['total_samples']}\n"
        msg += f"• **Features**: {overview['total_features']}\n"
        msg += f"• **Data Quality**: {overview.get('data_quality_summary', {}).get('overall_quality', 'Unknown')}\n\n"
        
        # Class distribution
        if overview.get('classes'):
            msg += "**Class Distribution:**\n"
            for class_name, count in overview['classes'].items():
                percentage = (count / overview['total_samples'] * 100) if overview['total_samples'] > 0 else 0
                msg += f"• {class_name}: {count} samples ({percentage:.1f}%)\n"
            msg += "\n"
        
        # Sample features
        if overview.get('sample_feature_names'):
            msg += "**Sample Features:**\n"
            for feature in overview['sample_feature_names'][:5]:
                msg += f"• {feature}\n"
            if overview['total_feature_count'] > 5:
                msg += f"• ... and {overview['total_feature_count'] - 5} more\n"
        
        msg += "\n💡 You can now ask me to analyze specific features or create visualizations!"
        
        return msg
    def _on_response_ready(self, ai_response):
        """Handle AI response ready signal"""
        # Store response and clean it
        self.pending_response = self._clean_response_from_triggers(ai_response)

    def _on_plot_ready(self, plot_fig):
        """Handle plot ready signal"""
        self.pending_plot = plot_fig
        
        # Once we have both response and plot status, display everything
        self._finalize_response()

    def _on_error(self, error_message):
        """Handle error signal"""
        self._hide_loading_indicator()
        self.is_loading = False
        
        error_msg = f"Error getting AI response: {error_message}"
        self._append_message("System", error_msg, Qt.AlignLeft)

    def _finalize_response(self):
        """Display the final response with plot if available"""
        # Hide loading
        self._hide_loading_indicator()
        
        # Display response with plot
        self._append_message("AI Assistant", self.pending_response, Qt.AlignLeft, self.pending_plot)
        
        # Reset state
        self.is_loading = False
        self.pending_response = ""
        self.pending_plot = None
