import os
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict, deque
import time

# Import our new unified system
from .unified_parser import parse_command
from .request_handler import handle_request

class Chatbot:
    def __init__(self, model_dir: str, prompt_choice: str = None, history_limit: int = 7):
        # — Load system prompts & memory‐extraction patterns from JSON (or fallback) —
        base_dir = os.path.dirname(__file__)
        prompts_path = os.path.join(base_dir, "prompt.json")
        with open(prompts_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Select persona prompt
        choice = prompt_choice or config.get("default_choice", "default")
        self.system_prompt = config.get("system_prompts", {}).get(choice, "")

        # Compile dynamic memory patterns
        self.memory_patterns = [re.compile(p, re.IGNORECASE) for p in config.get("memory_patterns", [])]
        self.memory = {}  # dynamic key→value store for extracted facts

        # Enhanced conversation context tracking with learning capabilities
        self.conversation_context = {
            'current_topic': None,
            'user_expertise_level': 'intermediate',  # beginner, intermediate, expert
            'preferred_analysis_type': None,  # statistical, visual, comparative
            'recent_questions': deque(maxlen=10),
            'data_insights_shared': set(),
            'plot_preferences': set(),
            'conversation_flow': [],  # Track conversation progression
            'user_satisfaction_signals': [],  # Track positive/negative feedback
            'topic_transitions': defaultdict(int),  # How often user switches topics
            'response_preferences': defaultdict(int),  # What types of responses user prefers
            'session_start_time': time.time(),
            'conversation_depth': 0,  # How deep into analysis user goes
            'repeated_questions': defaultdict(int),  # Track repeated questions
            'successful_interactions': set()  # Track what worked well
        }

        # Load model in 4bit for faster inference
        # bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        
        # Load tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        # quantization_config=bnb_config
        ).to("cuda")

        self.device = self.model.device

        # Enhanced sampling & repetition settings with adaptive parameters
        self.do_sample = True  # Enable sampling for more diverse responses
        self.base_temperature = 0.7  # Base temperature for response generation
        self.temperature = 0.7  # Current temperature (can be adjusted)
        self.top_k = 40
        self.top_p = 0.85
        self.repetition_penalty = 1.3  # Increased to prevent repetition

        # Conversation state with enhanced tracking
        self.history = []           # list of (AI, User)
        self.history_limit = history_limit
        
        # Add missing attributes for ML response tracking
        self.user_messages = []     # Track user messages for ML responses
        self.ai_messages = []       # Track AI messages for ML responses
        
        # Enhanced response quality tracking and learning
        self.response_metrics = {
            'total_responses': 0,
            'plot_triggers': 0,
            'analysis_requests': 0,
            'user_satisfaction_signals': [],
            'response_times': [],
            'conversation_lengths': [],
            'topic_engagement': defaultdict(int),
            'successful_patterns': defaultdict(int),
            'failed_patterns': defaultdict(int)
        }

        # Learning parameters
        self.learning_rate = 0.1
        self.min_confidence_threshold = 0.3
        self.max_context_length = 2000  # Maximum context length for model input
        
        # Pattern learning for plot requests - DEPRECATED: Using unified parser now
        self.learned_plot_patterns = {}
        
        # Comprehensive command trigger list - DEPRECATED: Using unified parser now
        self.command_triggers = {}

    def _extract_memory(self, text: str):
        """Enhanced memory extraction with pattern learning"""
        for pat in self.memory_patterns:
            m = pat.search(text)
            if not m:
                continue
            gd = m.groupdict()
            if "field" in gd and gd["field"]:
                field = gd["field"].lower()
            else:
                # if the pattern was "i am ...", treat as 'name'
                if pat.pattern.lower().startswith("i am"):
                    field = "name"
                else:
                    field = "info"
            value = gd.get("value", m.group(0)).strip()
            key = re.sub(r"\s+", "_", field)
            self.memory[key] = value
            
        # Learn new patterns from user input
        self._learn_new_patterns(text)

    def _learn_new_patterns(self, text: str):
        """Enhanced pattern learning with plot request detection and memory extraction - DEPRECATED: Using unified parser now"""
        # This method is deprecated - we now use the unified parser
        pass

    def _analyze_user_input(self, user_input: str) -> Dict:
        """Enhanced user input analysis with learning capabilities"""
        analysis = {
            'intent': 'general',
            'confidence': 0.0,
            'analysis_type': None,
            'sensor_mentioned': None,
            'plot_request': False,
            'specific_plot_type': False,
            'expertise_level': None,
            'urgency': 'normal',
            'emotional_tone': 'neutral',
            'complexity_level': 'medium',
            'follow_up_question': False,
            'clarification_needed': False,
            'is_general_conversation': False,
            'commands_detected': [],
            'prompt_adjustments': []
        }
        
        text_lower = user_input.lower()
        
        # First, check if this is general conversation (greetings, casual talk)
        general_conversation_keywords = [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'how\'s it going', 'what\'s up', 'nice to meet you',
            'thanks', 'thank you', 'bye', 'goodbye', 'see you', 'good night',
            'how was your day', 'nice weather', 'have a good day', 'take care',
            'morning', 'afternoon', 'evening', 'night', 'sup', 'yo', 'greetings'
        ]
        
        # Check for exact matches first (more reliable)
        if any(keyword == text_lower.strip() for keyword in ['hi', 'hello', 'hey', 'sup', 'yo']):
            analysis['is_general_conversation'] = True
            analysis['intent'] = 'general_conversation'
            analysis['confidence'] = 0.95
            analysis['expertise_level'] = 'beginner'
            return analysis
        
        # Check for partial matches
        if any(keyword in text_lower for keyword in general_conversation_keywords):
            analysis['is_general_conversation'] = True
            analysis['intent'] = 'general_conversation'
            analysis['confidence'] = 0.9
            analysis['expertise_level'] = 'beginner'
            return analysis
        
        # Command detection and prompt adjustment - DEPRECATED: Using unified parser now
        analysis['commands_detected'] = []
        analysis['prompt_adjustments'] = []
        
        # Contextual intent recognition based on conversation history - DEPRECATED: Using unified parser now
        analysis['contextual_plot_request'] = False
        
        # Enhanced intent detection with confidence scoring for data-related questions
        intent_keywords = {
            'visualization': ['plot', 'show', 'display', 'visualize', 'chart', 'graph', 'draw', 'create', 'histogram', 'line graph', 'scatter', 'correlation', 'time series'],
            'analysis': ['analyze', 'compare', 'difference', 'statistics', 'mean', 'std', 'correlation', 'pattern'],
            'help': ['help', 'explain', 'what is', 'how to', 'why', 'when'],
            'exploration': ['explore', 'investigate', 'find', 'discover', 'look for'],
            'comparison': ['compare', 'versus', 'vs', 'difference', 'similar', 'different'],
            'prediction': ['predict', 'forecast', 'trend', 'future', 'next']
        }
        
        for intent, keywords in intent_keywords.items():
            matches = sum(1 for word in keywords if word in text_lower)
            if matches > 0:
                confidence = min(0.8, matches * 0.2)
                if confidence > analysis['confidence']:
                    analysis['intent'] = intent
                    analysis['confidence'] = confidence
                    
                if intent == 'visualization':
                    analysis['plot_request'] = True
                    # Check for specific plot types
                    specific_plots = ['histogram', 'line graph', 'scatter', 'correlation', 'time series', 'line', 'bar']
                    if any(plot in text_lower for plot in specific_plots):
                        analysis['specific_plot_type'] = True
                elif intent == 'analysis':
                    analysis['analysis_type'] = 'statistical'
        
        # Enhanced plot request detection using learned patterns and contextual clues - DEPRECATED: Using unified parser now
        analysis['plot_request'] = False
        analysis['specific_plot_type'] = False
        
        # Enhanced sensor detection with context
        sensors = {
            'accelerometer': ['acc', 'acceleration', 'motion', 'movement', 'vibration', 'shake'],
            'temperature': ['temp', 'thermal', 'heat', 'cold', 'warm', 'cool'],
            'pressure': ['press', 'barometric', 'force', 'stress', 'load'],
            'humidity': ['hum', 'moisture', 'wetness', 'damp', 'dry'],
            'magnetometer': ['mag', 'magnetic', 'compass', 'north', 'field'],
            'gyroscope': ['gyro', 'rotation', 'angular', 'spin', 'turn'],
            'microphone': ['mic', 'audio', 'sound', 'noise', 'acoustic']
        }
        
        for sensor, keywords in sensors.items():
            if any(keyword in text_lower for keyword in keywords):
                analysis['sensor_mentioned'] = sensor
                analysis['confidence'] += 0.2
                break
        
        # Enhanced expertise level detection with learning
        expertise_indicators = {
            'expert': ['p-value', 'statistical significance', 'correlation coefficient', 'fft', 'frequency domain', 'anova', 'regression', 'hypothesis testing'],
            'intermediate': ['mean', 'average', 'distribution', 'compare', 'variance', 'standard deviation', 'correlation'],
            'beginner': ['what is', 'how does', 'explain', 'simple', 'basic', 'easy']
        }
        
        for level, indicators in expertise_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                analysis['expertise_level'] = level
                break
        else:
            analysis['expertise_level'] = 'beginner'
            
        # Detect emotional tone and urgency
        urgency_words = ['urgent', 'quick', 'fast', 'asap', 'now', 'immediately', 'hurry']
        if any(word in text_lower for word in urgency_words):
            analysis['urgency'] = 'high'
            
        # Detect complexity and follow-up questions
        if len(user_input.split()) > 25:
            analysis['complexity_level'] = 'high'
        elif len(user_input.split()) < 8:
            analysis['complexity_level'] = 'low'
            
        # Check if this is a follow-up question
        follow_up_indicators = ['what about', 'how about', 'also', 'additionally', 'furthermore', 'moreover']
        if any(indicator in text_lower for indicator in follow_up_indicators):
            analysis['follow_up_question'] = True
            
        # Check if clarification might be needed
        if analysis['confidence'] < 0.4 and not analysis['follow_up_question']:
            analysis['clarification_needed'] = True
            
        return analysis

    def _update_conversation_context(self, user_input: str, analysis: Dict):
        """Simplified conversation context update - focused on essential tracking"""
        # Update current topic if sensor is mentioned
        if analysis['sensor_mentioned']:
            self.conversation_context['current_topic'] = analysis['sensor_mentioned']
            
        # Update expertise level with confidence
        if analysis['expertise_level'] and analysis['confidence'] > 0.6:
            self.conversation_context['user_expertise_level'] = analysis['expertise_level']
            
        # Track recent questions (keep only last 3)
        question_context = {
            'text': user_input[:60],  # Shorter text
            'intent': analysis['intent'],
            'sensor': analysis['sensor_mentioned']
        }
        self.conversation_context['recent_questions'].append(question_context)
        
        # Keep only last 3 questions
        if len(self.conversation_context['recent_questions']) > 3:
            self.conversation_context['recent_questions'].popleft()
            
        # Simple conversation depth tracking
        if analysis['intent'] in ['analysis', 'visualization']:
            self.conversation_context['conversation_depth'] += 1

    def _generate_contextual_prompt(self, user_input: str, analysis: Dict) -> str:
        """Focused contextual prompt generation - minimal context for better focus"""
        # For general conversation, use a completely different, simple prompt
        if analysis.get('is_general_conversation', False):
            return "You are Peak, a friendly AI assistant. Respond naturally and briefly to casual conversation. Keep responses to ONE LINE maximum with no explanations."
        
        # For data questions, use the full sensor data prompt
        base_prompt = self.system_prompt
        
        # Add minimal but focused context
        focused_context = self._generate_focused_context(analysis)
        
        return base_prompt + focused_context

    def _generate_focused_context(self, analysis: Dict) -> str:
        """Generate minimal, focused context for the current question"""
        context_parts = []
        
        # Only mention current topic if it's directly relevant
        if analysis.get('sensor_mentioned'):
            sensor = analysis['sensor_mentioned']
            context_parts.append(f"\n\nFOCUS: The user is asking about {sensor} sensor data.")
        
        # Only add expertise level if it significantly affects the response
        if analysis.get('expertise_level') == 'beginner':
            context_parts.append("\n\nRESPONSE STYLE: Keep explanation simple and clear.")
        elif analysis.get('expertise_level') == 'expert':
            context_parts.append("\n\nRESPONSE STYLE: Use technical terminology and advanced concepts.")
        
        # Only add plot guidance if visualization is requested
        if analysis.get('plot_request', False):
            # Check if specific plot types are mentioned
            if analysis.get('specific_plot_type'):
                context_parts.append("\n\nPLOT REQUEST: User requested specific plot type. Create it immediately with 'show plot'. DO NOT ask questions or explain what the plot type is.")
            else:
                context_parts.append("\n\nPLOT REQUEST: Include 'show plot' in response. DO NOT ask questions about the plot type - just create it.")
        
        # Add command-specific prompt adjustments
        if analysis.get('prompt_adjustments'):
            for adjustment in analysis['prompt_adjustments']:
                context_parts.append(f"\n\n{adjustment}")
        
        return ''.join(context_parts)

    def _summarize(self, entries):
        """Create a brief summary of earlier user turns."""
        texts = [txt for spk, txt in entries if spk == "User"]
        return "Previous topics: " + ", ".join(texts[-3:])

    def _prune_history(self):
        """Keep only the most recent turns, plus one summary entry if needed."""
        max_entries = self.history_limit * 2
        if len(self.history) > max_entries:
            old = self.history[:-max_entries]
            summary = self._summarize(old)
            # tag the summary as its own speaker to avoid confusion
            self.history = [("Summary", summary)] + self.history[-max_entries:]

    def _build_prompt(self, user_input: str) -> str:
        # 1) Analyze user input for context
        analysis = self._analyze_user_input(user_input)
        
        # 2) Update conversation context
        self._update_conversation_context(user_input, analysis)
        
        # 3) Extract any new memory from this turn (skip for general conversation)
        if not analysis.get('is_general_conversation', False):
            self._extract_memory(user_input)

        # 4) Append the user's message
        self.history.append(("User", user_input))
        self._prune_history()

        # 5) Generate contextual prompt
        contextual_prompt = self._generate_contextual_prompt(user_input, analysis)
        
        # 6) Assemble the system prompt + injected memory + history
        lines = [f"System: {contextual_prompt}"]
        
        # Only add sensor-related memory for data questions
        if not analysis.get('is_general_conversation', False):
            for k, v in self.memory.items():
                lines.append(f"(Note: user {k} is {v}.)")
        
        for speaker, txt in self.history:
            lines.append(f"{speaker}: {txt}")
        lines.append("AI:")
        return "\n".join(lines)

    def generate(self, user_input: str, max_new_tokens: int =250) -> str:
        start_time = time.time()
        
        # IMPORTANT: This method has TWO response paths:
        # 1. ML SYSTEM PATH: For data analysis requests (returns early, no decode)
        # 2. LLM FALLBACK PATH: For general conversation (uses decode function)
        
        # 1) Use our new unified parser to detect command type and response preference
        try:
            parsed_command = parse_command(user_input)
            
            # 2) Check if this is a data analysis request that should use our ML system
            if parsed_command.command_type.value != 'unknown':
                # Use our request handler for structured data analysis
                response_data = handle_request(user_input)
                
                if response_data.get('status') == 'success':
                    # Format the response based on the response type
                    response_type = response_data.get('response_type', 'auto')
                    
                    if response_type == 'text':
                        # Text-only response - no plots, no context, no suggestions
                        response = response_data['main_response']
                    elif response_type == 'visual':
                        # Visual response - include plot trigger
                        response = response_data['main_response']
                        # Prefer the actual plot_type returned from the pipeline over suggestion
                        effective_plot = (response_data.get('plot_type') or response_data.get('plot_suggestion') or '').lower()
                        # Normalize aliases
                        if effective_plot in ['line', 'line graph', 'line_graph']:
                            response = "📈 Creating line graph..."
                            trigger = 'line_graph'
                        elif effective_plot in ['hist', 'histogram']:
                            response = "📊 Creating histogram..."
                            trigger = 'histogram'
                        elif effective_plot in ['scatter', 'scatterplot']:
                            response = "💫 Creating scatter plot..."
                            trigger = 'scatter'
                        elif effective_plot in ['correlation', 'correlation_matrix']:
                            response = "🔗 Creating correlation matrix..."
                            trigger = 'correlation'
                        elif effective_plot in ['timeseries', 'time series']:
                            response = "🕒 Creating time series plot..."
                            trigger = 'timeseries'
                        elif effective_plot in ['frequency', 'fft']:
                            response = "📡 Creating frequency domain plot..."
                            trigger = 'frequency'
                        else:
                            # Fallback to suggestion if available
                            plot_suggestion = response_data.get('plot_suggestion')
                            if plot_suggestion == 'line_graph':
                                response = "📈 Creating line graph..."; trigger = 'line_graph'
                            elif plot_suggestion == 'histogram':
                                response = "📊 Creating histogram..."; trigger = 'histogram'
                            elif plot_suggestion == 'scatter':
                                response = "💫 Creating scatter plot..."; trigger = 'scatter'
                            elif plot_suggestion == 'correlation':
                                response = "🔗 Creating correlation matrix..."; trigger = 'correlation'
                            elif plot_suggestion == 'timeseries':
                                response = "🕒 Creating time series plot..."; trigger = 'timeseries'
                            elif plot_suggestion == 'frequency':
                                response = "📡 Creating frequency domain plot..."; trigger = 'frequency'
                            else:
                                trigger = (plot_suggestion or 'line_graph')
                                response = f"📊 Creating {trigger}..."

                        response += f" [TRIGGER_PLOT:{trigger}]"
                    else:
                        # Auto mode - check if plot is suggested
                        response = response_data['main_response']
                        effective_plot = (response_data.get('plot_type') or response_data.get('plot_suggestion') or '').lower()
                        if effective_plot in ['line', 'line graph', 'line_graph']:
                            response = "📈 Creating line graph..."; trigger = 'line_graph'
                        elif effective_plot in ['hist', 'histogram']:
                            response = "📊 Creating histogram..."; trigger = 'histogram'
                        elif effective_plot in ['scatter', 'scatterplot']:
                            response = "💫 Creating scatter plot..."; trigger = 'scatter'
                        elif effective_plot in ['correlation', 'correlation_matrix']:
                            response = "🔗 Creating correlation matrix..."; trigger = 'correlation'
                        elif effective_plot in ['timeseries', 'time series']:
                            response = "🕒 Creating time series plot..."; trigger = 'timeseries'
                        elif effective_plot in ['frequency', 'fft']:
                            response = "📡 Creating frequency domain plot..."; trigger = 'frequency'
                        else:
                            trigger = (response_data.get('plot_suggestion') or 'line_graph')
                            response = f"📊 Creating {trigger}..."

                        response += f" [TRIGGER_PLOT:{trigger}]"
                    
                    # Update conversation context and return
                    self.user_messages.append(user_input)
                    self.ai_messages.append(response)
                    
                    # Also update the main history for consistency
                    self.history.append(("User", user_input))
                    self.history.append(("AI", response))
                    self._prune_history()
                    
                    # Update response metrics for ML responses
                    response_time = time.time() - start_time
                    ml_analysis = {
                        'intent': 'ml_analysis',
                        'confidence': 0.9,
                        'expertise_level': 'intermediate',
                        'sensor_mentioned': None,
                        'plot_request': response_type == 'visual',
                        'specific_plot_type': False,
                        'urgency': 'normal',
                        'emotional_tone': 'neutral',
                        'complexity_level': 'medium',
                        'follow_up_question': False,
                        'clarification_needed': False,
                        'is_general_conversation': False,
                        'commands_detected': [],
                        'prompt_adjustments': []
                    }
                    self._update_response_metrics(response, ml_analysis, response_time)
                    
                    return response
                    
        except Exception as e:
            # Fallback to old system if there's an error
            pass
        
        # 3) Fallback to old LLM-based system for non-data requests
        # Note: The decode function below is ONLY for LLM-generated responses, 
        # NOT for ML responses which are handled above and returned directly
        
        # 1) Analyze user input for context (only if no command detected)
        analysis = self._analyze_user_input(user_input)
        
        # 2) Update conversation context
        self._update_conversation_context(user_input, analysis)
        
        # 3) Extract any new memory from this turn
        self._extract_memory(user_input)

        # 4) Append the user's message
        self.history.append(("User", user_input))
        self._prune_history()

        # 5) Generate contextual prompt with adaptive learning
        contextual_prompt = self._generate_contextual_prompt(user_input, analysis)
        
        # 6) Assemble the system prompt + injected memory + history
        lines = [f"System: {contextual_prompt}"]
        for k, v in self.memory.items():
            lines.append(f"(Note: user {k} is {v}.)")
        for speaker, txt in self.history:
            lines.append(f"{speaker}: {txt}")
        lines.append("AI:")
        
        prompt = "\n".join(lines)
        
        # 7) Adaptive token adjustment based on context and user behavior
        max_new_tokens = self._calculate_adaptive_tokens(user_input, analysis, max_new_tokens)
        
        # 8) Adaptive temperature based on user behavior and conversation state
        self._adjust_temperature(analysis)
        
        # 9) Generate response with enhanced parameters
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,  # Prevent repetition of 3-grams
        )

        # DECODE FUNCTION: This is ONLY for LLM-generated responses, NOT for ML responses
        # ML responses are handled above and returned directly without going through decode
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # strip out the prompt echo - use more robust method to avoid losing first character
        if decoded.startswith(prompt):
            reply = decoded[len(prompt):].strip()
        else:
            # Fallback: look for the last "AI:" marker and take everything after it
            ai_marker = "AI:"
            if ai_marker in decoded:
                last_ai_index = decoded.rfind(ai_marker)
                reply = decoded[last_ai_index + len(ai_marker):].strip()
                # Sometimes the model adds a newline or space after AI:, so we need to handle that
                if reply.startswith('\n'):
                    reply = reply[1:].strip()
                elif reply.startswith(' '):
                    reply = reply.lstrip()
            else:
                # If all else fails, try to find where the prompt ends by looking for common patterns
                prompt_lines = prompt.split('\n')
                if len(prompt_lines) > 1:
                    # Look for the last line of the prompt (usually "AI:")
                    last_prompt_line = prompt_lines[-1].strip()
                    if last_prompt_line in decoded:
                        last_index = decoded.rfind(last_prompt_line)
                        reply = decoded[last_index + len(last_prompt_line):].strip()
                        # Handle potential newline or space after the marker
                        if reply.startswith('\n'):
                            reply = reply[1:].strip()
                        elif reply.startswith(' '):
                            reply = reply.lstrip()
                    else:
                        # Final fallback: use the original method but be more careful
                        reply = decoded[len(prompt):].strip()
                else:
                    reply = decoded[len(prompt):].strip()
        
        # Ensure the first character is preserved
        if reply and len(reply) > 0:
            if reply[0].isspace():
                reply = reply.lstrip()
        
        # Additional safety check: ensure we have a valid response
        if not reply or len(reply.strip()) == 0:
            # Last resort: try to extract anything that looks like a response
            # Look for content after the last "AI:" or similar marker
            markers = ["AI:", "Assistant:", "Bot:"]
            for marker in markers:
                if marker in decoded:
                    marker_index = decoded.rfind(marker)
                    potential_reply = decoded[marker_index + len(marker):].strip()
                    if potential_reply and len(potential_reply) > 0:
                        reply = potential_reply
                        break
            
            # If still no reply, try to find any content that looks like a response
            if not reply or len(reply.strip()) == 0:
                # Look for the last colon and take everything after it
                if ':' in decoded:
                    last_colon_index = decoded.rfind(':')
                    potential_reply = decoded[last_colon_index + 1:].strip()
                    if potential_reply and len(potential_reply) > 0:
                        reply = potential_reply
                
                # If still no reply, try to find content after the last newline
                if not reply or len(reply.strip()) == 0:
                    if '\n' in decoded:
                        lines = decoded.split('\n')
                        # Look for the last non-empty line that doesn't look like a prompt
                        for line in reversed(lines):
                            line = line.strip()
                            if line and not line.startswith(('System:', 'User:', 'AI:', 'Note:')):
                                reply = line
                                break
        
        # 10) Enhanced response cleaning and improvement with learning
        reply = self._clean_response(reply)
        reply = self._enhance_response_quality(reply, user_input, analysis)
        
        # Stop at any new speaker label (including USER : with space)
        reply = re.split(r"(?i)\b(?:User|System|Developer|AI|USER)\s*:", reply)[0].strip()

        # 11) Update response metrics and learn from interaction
        response_time = time.time() - start_time
        self._update_response_metrics(reply, analysis, response_time)
        self._learn_from_interaction(user_input, reply, analysis)
        
        # 12) Add conversation flow entry for AI response
        ai_flow_entry = {
            'type': 'ai_response',
            'intent': analysis['intent'],
            'topic': analysis['sensor_mentioned'],
            'response_length': len(reply),
            'response_time': response_time,
            'timestamp': time.time()
        }
        self.conversation_context['conversation_flow'].append(ai_flow_entry)

        self.history.append(("AI", reply))
        return reply

    def _calculate_adaptive_tokens(self, user_input: str, analysis: Dict, base_tokens: int) -> int:
        """Calculate adaptive token count based on user behavior and context - optimized for single-line responses"""
        # Handle general conversation with very short responses
        if analysis.get('is_general_conversation', False):
            return 30  # Very short for greetings and casual conversation
        
        # For data questions, enforce single-line responses
        return 80  # Maximum tokens for single-line technical responses

    def _adjust_temperature(self, analysis: Dict):
        """Adjust temperature based on user behavior and conversation state"""
        base_temp = self.base_temperature
        
        # Adjust based on expertise level
        if analysis['expertise_level'] == 'beginner':
            self.temperature = base_temp * 0.9  # More focused for beginners
        elif analysis['expertise_level'] == 'expert':
            self.temperature = base_temp * 1.1  # More creative for experts
        else:
            self.temperature = base_temp
            
        # Adjust based on urgency
        if analysis['urgency'] == 'high':
            self.temperature = base_temp * 0.8  # More focused for urgent requests
            
        # Adjust based on conversation depth
        depth = self.conversation_context['conversation_depth']
        if depth > 3:
            self.temperature = base_temp * 1.1  # More creative for deep conversations
            
        # Ensure temperature stays within reasonable bounds
        self.temperature = max(0.3, min(1.2, self.temperature))

    def _update_response_metrics(self, reply: str, analysis: Dict, response_time: float):
        """Update response metrics with enhanced tracking"""
        self.response_metrics['total_responses'] += 1
        self.response_metrics['response_times'].append(response_time)
        
        # Track plot triggers
        if 'plot' in reply.lower() or 'visualization' in reply.lower():
            self.response_metrics['plot_triggers'] += 1
            
        # Track analysis requests
        if analysis['intent'] == 'analysis':
            self.response_metrics['analysis_requests'] += 1
            
        # Track conversation lengths
        self.response_metrics['conversation_lengths'].append(len(reply))
        
        # Track successful patterns
        if response_time < 2.0 and len(reply) > 50:  # Quick, substantial response
            pattern_key = f"{analysis['intent']}_{analysis['expertise_level']}"
            self.response_metrics['successful_patterns'][pattern_key] += 1
            
        # Keep metrics lists manageable
        if len(self.response_metrics['response_times']) > 100:
            self.response_metrics['response_times'] = self.response_metrics['response_times'][-50:]
        if len(self.response_metrics['conversation_lengths']) > 100:
            self.response_metrics['conversation_lengths'] = self.response_metrics['conversation_lengths'][-50:]

    def _learn_from_interaction(self, user_input: str, ai_response: str, analysis: Dict):
        """Learn from user interactions to improve future responses"""
        # Track successful interactions
        if len(ai_response) > 30 and analysis['confidence'] > 0.5:
            success_key = f"{analysis['intent']}_{analysis['sensor_mentioned'] or 'general'}"
            self.conversation_context['successful_interactions'].add(success_key)
            
        # Learn user preferences
        if analysis['intent'] == 'visualization':
            self.conversation_context['plot_preferences'].add(analysis['sensor_mentioned'] or 'general')
            
        # Track response preferences
        response_type = 'detailed' if len(ai_response) > 100 else 'concise'
        self.conversation_context['response_preferences'][response_type] += 1
        
        # Learn from repeated questions
        question_hash = hash(user_input.lower().strip())
        repeat_count = self.conversation_context['repeated_questions'].get(question_hash, 0)
        if repeat_count > 1:
            # User is asking the same question - learn what they need
            self._learn_from_repeated_questions(user_input, ai_response)

    def _learn_from_repeated_questions(self, user_input: str, ai_response: str):
        """Learn from repeated questions to provide better responses"""
        # Analyze what might be missing from previous responses
        text_lower = user_input.lower()
        
        if 'explain' in text_lower or 'what is' in text_lower:
            # User needs more explanation
            self.conversation_context['response_preferences']['detailed'] += 2
            
        if 'example' in text_lower or 'show me' in text_lower:
            # User needs concrete examples
            self.conversation_context['response_preferences']['examples'] += 2
            
        if 'how to' in text_lower or 'steps' in text_lower:
            # User needs step-by-step guidance
            self.conversation_context['response_preferences']['step_by_step'] += 2

    def _enhance_response_quality(self, reply: str, user_input: str, analysis: Dict) -> str:
        """Enhanced response quality improvement - optimized for single-line responses"""
        if not reply:
            return reply
        
        # For general conversation, don't add technical enhancements
        if analysis.get('is_general_conversation', False):
            return reply
            
        # Enhanced plot trigger detection and GUI integration
        if analysis.get('plot_request', False):
            # Add plot trigger if missing
            if 'plot' not in reply.lower() and 'show plot' not in reply.lower():
                reply += " show plot"
            
            # Add GUI trigger for specific plot types
            if analysis.get('specific_plot_type', False):
                plot_type = self._determine_plot_type(analysis)
                if plot_type:
                    reply += f" TRIGGER_PLOT:{plot_type}"
            else:
                reply += " TRIGGER_PLOT:general_visualization"
        
        # Ensure response is single line
        reply = reply.replace('\n', ' ').replace('\r', ' ')
        
        return reply

    def _clean_response(self, reply: str) -> str:
        """Clean and optimize response for concise, direct communication"""
        if not reply:
            return reply
            
        # Remove any code blocks or inline code to avoid showing code in chat
        # Triple backtick fenced blocks (any language)
        reply = re.sub(r"```[\s\S]*?```", "", reply)
        # Inline code wrapped in single backticks
        reply = re.sub(r"`[^`]*`", "", reply)
        # HTML <pre> blocks if present
        reply = re.sub(r"<pre[\s\S]*?>[\s\S]*?</pre>", "", reply, flags=re.IGNORECASE)

        # Preserve single newlines for table formatting, collapse excessive spaces
        # Replace Windows/Mac newlines with \n
        reply = reply.replace('\r\n', '\n').replace('\r', '\n')
        # Collapse multiple blank lines to a single blank line
        reply = re.sub(r'\n\n\n+', '\n\n', reply)
        # For each line, collapse internal runs of spaces > 2 to 1 (keep at most 2 to help alignment)
        reply = '\n'.join([re.sub(r' {3,}', '  ', line).rstrip() for line in reply.split('\n')])
        reply = reply.strip()
        
        # Remove repetitive sentences
        sentences = reply.split('.')
        unique_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not any(self._similar_sentences(sentence, existing) for existing in unique_sentences):
                unique_sentences.append(sentence)
        
        # Reconstruct with only unique, meaningful sentences
        cleaned_reply = '. '.join(unique_sentences)
        
        # Remove verbose phrases and unnecessary words
        verbose_patterns = [
            r'\b(it is important to note that|it should be mentioned that|it is worth noting that)\b',
            r'\b(as you can see|as shown|as demonstrated|as illustrated)\b',
            r'\b(this analysis reveals|this data shows|this indicates that)\b',
            r'\b(in conclusion|to summarize|in summary)\b',
            r'\b(furthermore|moreover|additionally|also)\b',
            r'\b(please note that|keep in mind that|remember that)\b'
        ]
        
        for pattern in verbose_patterns:
            cleaned_reply = re.sub(pattern, '', cleaned_reply, flags=re.IGNORECASE)
        
        # Clean up any double spaces or punctuation artifacts
        cleaned_reply = re.sub(r'\s+', ' ', cleaned_reply)
        cleaned_reply = re.sub(r'\.\s*\.', '.', cleaned_reply)
        cleaned_reply = re.sub(r'\s*,\s*', ', ', cleaned_reply)
        
        # Ensure the response ends with proper punctuation
        if cleaned_reply and not cleaned_reply.endswith(('.', '!', '?')):
            cleaned_reply += '.'
            
        return cleaned_reply.strip()
    
    def _similar_sentences(self, sent1: str, sent2: str, threshold: float = 0.8) -> bool:
        """Check if two sentences are very similar (to detect repetition)."""
        # Simple similarity check based on word overlap
        words1 = set(re.findall(r'\b\w+\b', sent1.lower()))
        words2 = set(re.findall(r'\b\w+\b', sent2.lower()))
        
        if not words1 or not words2:
            return False
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        return similarity > threshold

    def get_conversation_insights(self) -> Dict:
        """Get comprehensive insights about the current conversation for debugging/improvement"""
        # Calculate average response time
        avg_response_time = 0
        if self.response_metrics['response_times']:
            avg_response_time = sum(self.response_metrics['response_times']) / len(self.response_metrics['response_times'])
            
        # Calculate average conversation length
        avg_conversation_length = 0
        if self.response_metrics['conversation_lengths']:
            avg_conversation_length = sum(self.response_metrics['conversation_lengths']) / len(self.response_metrics['conversation_lengths'])
            
        # Get session duration
        session_duration = time.time() - self.conversation_context['session_start_time']
        
        # Get most engaged topics
        top_topics = sorted(
            self.response_metrics['topic_engagement'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # Get most successful patterns
        top_patterns = sorted(
            self.response_metrics['successful_patterns'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            'conversation_context': self.conversation_context,
            'response_metrics': {
                **self.response_metrics,
                'avg_response_time': avg_response_time,
                'avg_conversation_length': avg_conversation_length,
                'session_duration': session_duration
            },
            'memory': self.memory,
            'current_topic': self.conversation_context['current_topic'],
            'user_expertise_level': self.conversation_context['user_expertise_level'],
            'top_engaged_topics': top_topics,
            'top_successful_patterns': top_patterns,
            'conversation_flow_summary': self._summarize_conversation_flow(),
            'learning_insights': self._get_learning_insights()
        }

    def _summarize_conversation_flow(self) -> Dict:
        """Summarize the conversation flow for insights"""
        if not self.conversation_context['conversation_flow']:
            return {}
            
        flow = self.conversation_context['conversation_flow']
        
        # Count different types of interactions
        user_inputs = [f for f in flow if f['type'] == 'user_input']
        ai_responses = [f for f in flow if f['type'] == 'ai_response']
        
        # Analyze topic transitions
        topic_transitions = {}
        for i in range(len(user_inputs) - 1):
            current_topic = user_inputs[i].get('topic', 'general')
            next_topic = user_inputs[i + 1].get('topic', 'general')
            if current_topic != next_topic:
                transition = f"{current_topic} -> {next_topic}"
                topic_transitions[transition] = topic_transitions.get(transition, 0) + 1
                
        # Analyze intent patterns
        intent_counts = {}
        for input_entry in user_inputs:
            intent = input_entry.get('intent', 'general')
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
        return {
            'total_interactions': len(flow),
            'user_inputs': len(user_inputs),
            'ai_responses': len(ai_responses),
            'topic_transitions': topic_transitions,
            'intent_distribution': intent_counts,
            'conversation_depth': self.conversation_context['conversation_depth']
        }

    def _get_learning_insights(self) -> Dict:
        """Get insights about what the AI has learned from the conversation"""
        return {
            'successful_interactions': list(self.conversation_context['successful_interactions']),
            'plot_preferences': list(self.conversation_context['plot_preferences']),
            'response_preferences': dict(self.conversation_context['response_preferences']),
            'topic_transitions': dict(self.conversation_context['topic_transitions']),
            'repeated_questions_count': len([k for k, v in self.conversation_context['repeated_questions'].items() if v > 1]),
            'expertise_level_changes': self.conversation_context['user_expertise_level'],
            'learning_rate': self.learning_rate
        }

    def adapt_to_user_feedback(self, feedback_type: str, feedback_data: Dict = None):
        """Adapt AI behavior based on user feedback"""
        if feedback_type == 'positive':
            # Reinforce successful patterns
            if feedback_data and 'intent' in feedback_data and 'sensor' in feedback_data:
                pattern_key = f"{feedback_data['intent']}_{feedback_data['sensor']}"
                self.response_metrics['successful_patterns'][pattern_key] += 2
                
        elif feedback_type == 'negative':
            # Learn from failed patterns
            if feedback_data and 'intent' in feedback_data and 'sensor' in feedback_data:
                pattern_key = f"{feedback_data['intent']}_{feedback_data['sensor']}"
                self.response_metrics['failed_patterns'][pattern_key] += 1
                
        elif feedback_type == 'clarification_needed':
            # Increase explanation level
            self.conversation_context['response_preferences']['detailed'] += 1
            
        elif feedback_type == 'too_complex':
            # Simplify future responses
            self.conversation_context['response_preferences']['concise'] += 1
            
        # Adjust learning rate based on feedback frequency
        feedback_count = len(self.conversation_context['user_satisfaction_signals'])
        if feedback_count > 10:
            self.learning_rate = min(0.2, self.learning_rate * 1.1)  # Increase learning rate

    def get_conversation_summary(self) -> str:
        """Generate a human-readable summary of the conversation"""
        insights = self.get_conversation_insights()
        
        summary_parts = []
        
        # Session overview
        duration_minutes = insights['response_metrics']['session_duration'] / 60
        summary_parts.append(f"**Session Overview:** {duration_minutes:.1f} minutes, {insights['response_metrics']['total_responses']} interactions")
        
        # Current topic
        if insights['current_topic']:
            summary_parts.append(f"**Current Focus:** {insights['current_topic']} sensor analysis")
            
        # Expertise level
        summary_parts.append(f"**User Expertise:** {insights['user_expertise_level']} level")
        
        # Top topics
        if insights['top_engaged_topics']:
            topics = [f"{topic} ({count})" for topic, count in insights['top_engaged_topics'][:3]]
            summary_parts.append(f"**Most Discussed:** {', '.join(topics)}")
            
        # Conversation depth
        depth = insights['conversation_flow_summary']['conversation_depth']
        if depth > 3:
            summary_parts.append(f"**Analysis Depth:** Deep exploration ({depth} analysis interactions)")
        elif depth > 1:
            summary_parts.append(f"**Analysis Depth:** Moderate exploration ({depth} analysis interactions)")
        else:
            summary_parts.append(f"**Analysis Depth:** Initial exploration ({depth} analysis interactions)")
            
        # Learning insights
        if insights['learning_insights']['repeated_questions_count'] > 0:
            summary_parts.append(f"**Learning:** {insights['learning_insights']['repeated_questions_count']} repeated questions indicate areas for improvement")
            
        return "\n".join(summary_parts)

    def suggest_improvements(self) -> List[str]:
        """Suggest improvements based on conversation analysis"""
        insights = self.get_conversation_insights()
        suggestions = []
        
        # Check for repeated questions
        repeated_count = insights['learning_insights']['repeated_questions_count']
        if repeated_count > 2:
            suggestions.append("Consider providing more detailed explanations for complex topics")
            
        # Check response time
        avg_response_time = insights['response_metrics']['avg_response_time']
        if avg_response_time > 3.0:
            suggestions.append("Response times are high - consider optimizing model parameters")
            
        # Check conversation depth
        depth = insights['conversation_flow_summary']['conversation_depth']
        if depth < 2:
            suggestions.append("Encourage deeper analysis by suggesting follow-up questions")
            
        # Check topic diversity
        topic_count = len(insights['top_engaged_topics'])
        if topic_count < 2:
            suggestions.append("Suggest exploring different sensor types for comprehensive analysis")
            
        # Check expertise level
        expertise = insights['user_expertise_level']
        if expertise == 'beginner' and depth > 3:
            suggestions.append("User shows interest in deep analysis - consider introducing advanced concepts gradually")
            
        return suggestions

    def reset_memory(self):
        """Clear the conversation history and all stored facts with enhanced reset"""
        self.history.clear()
        self.memory.clear()
        
        # Enhanced context reset
        self.conversation_context = {
            'current_topic': None,
            'user_expertise_level': 'intermediate',
            'preferred_analysis_type': None,
            'recent_questions': deque(maxlen=10),
            'data_insights_shared': set(),
            'plot_preferences': set(),
            'conversation_flow': [],
            'user_satisfaction_signals': [],
            'topic_transitions': defaultdict(int),
            'response_preferences': defaultdict(int),
            'session_start_time': time.time(),
            'conversation_depth': 0,
            'repeated_questions': defaultdict(int),
            'successful_interactions': set()
        }
        
        # Enhanced metrics reset
        self.response_metrics = {
            'total_responses': 0,
            'plot_triggers': 0,
            'analysis_requests': 0,
            'user_satisfaction_signals': [],
            'response_times': [],
            'conversation_lengths': [],
            'topic_engagement': defaultdict(int),
            'successful_patterns': defaultdict(int),
            'failed_patterns': defaultdict(int)
        }
        
        # Reset adaptive parameters
        self.temperature = self.base_temperature

    def export_conversation_data(self) -> Dict:
        """Export conversation data for analysis or backup"""
        return {
            'timestamp': time.time(),
            'session_duration': time.time() - self.conversation_context['session_start_time'],
            'conversation_insights': self.get_conversation_insights(),
            'memory': self.memory,
            'history': self.history[-20:],  # Last 20 interactions
            'learning_insights': self._get_learning_insights(),
            'suggestions': self.suggest_improvements()
        }

    def _detect_commands(self, text_lower: str) -> List[str]:
        """Detect commands in user input - DEPRECATED: Use unified parser instead"""
        # This method is deprecated - we now use the unified parser
        return []
    
    def detect_command(self, user_input: str) -> Optional[str]:
        """Detect if user input matches any command trigger - DEPRECATED: Use unified parser instead"""
        # This method is deprecated - we now use the unified parser
        return None

    def _generate_prompt_adjustments(self, commands: List[str]) -> List[str]:
        """Generate prompt adjustments based on detected commands - DEPRECATED: Use unified parser instead"""
        # This method is deprecated - we now use the unified parser
        return []

    def _determine_plot_type(self, analysis: Dict) -> str:
        """Determine the specific plot type based on analysis and commands - DEPRECATED: Using unified parser now"""
        # This method is deprecated - we now use the unified parser
        return 'general_visualization'

    def _detect_contextual_plot_request(self, user_input: str, analysis: Dict) -> bool:
        """Detect plot requests based on conversation context and indirect phrases - DEPRECATED: Using unified parser now"""
        # This method is deprecated - we now use the unified parser
        return False

    def _get_recent_conversation_context(self) -> Dict:
        """Get context from recent conversation turns - DEPRECATED: Using unified parser now"""
        # This method is deprecated - we now use the unified parser
        return {
            'plot_related': False,
            'repeated_topic': False,
            'current_topic': None,
            'topic_frequency': {}
        }
