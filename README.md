# 🤖 Statistical AI Agent - Advanced Data Analysis Platform

A comprehensive, AI-powered desktop application for statistical analysis, machine learning, and data visualization with natural language interface. Built with PySide6 and powered by Llama-3.2-1B for intelligent data science workflows.

### **Enhanced AI Capabilities**
- **Professional Data Scientist Persona**: Expert-level AI assistant specialized in sensor data analysis
- **Advanced Natural Language Processing**: Intelligent plot request detection and generation
- **Memory & Learning**: AI remembers user preferences and learns from interactions
- **Statistical Expertise**: Focus on statistical significance and data-driven insights
- **Smart Response Cleaning**: Multi-layered defense against AI artifacts and instruction leakage
- **Conversation History Tracking**: Enhanced context awareness for better future responses
- **Validation-First Approach**: Pre-validates requests before generating responses

### **Improved Architecture**
- **Unified AI Agent Interface**: Single entry point for all ML and statistical operations
- **Enhanced Prompt System**: Comprehensive, professional prompts for data science tasks
- **Better Error Handling**: Fallback data generation and graceful error recovery
- **Performance Optimization**: Cached analysis results and efficient data processing
- **Unified Parser System**: Advanced natural language command parsing and routing
- **Response Formatting Engine**: Professional formatting with error handling
- **Multi-Layer Defense**: Comprehensive response cleaning and sanitization

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GUI Layer     │    │   Core Layer    │    │   ML Layer      │
│   (PySide6)     │◄──►│   (AI Logic)    │◄──►│   (Analysis)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Three-Tier Architecture**
- **GUI Layer**: Modern PySide6 desktop interface with integrated chat and plotting views
- **Core Layer**: AI chatbot, unified parser, request handler, response formatter, and ML interface
- **ML Layer**: Statistical analysis, feature selection, plotting engine, and machine learning

## 🎯 **Key Features**

### **🤖 AI-Powered Analysis**
- **Natural Language Interface**: Ask for plots and analysis in plain English
- **Intelligent Plot Generation**: Automatic visualization based on conversation context
- **Statistical Insights**: Professional data science analysis and recommendations
- **Learning Capability**: AI remembers your preferences and improves over time
- **Smart Intent Detection**: Distinguishes between general conversation and data analysis requests
- **Capability Recognition**: Automatically detects skills/capabilities questions
- **Context-Aware Responses**: Adapts responses based on conversation history and user expertise

### **📊 Advanced Visualization**
- **Multiple Plot Types**: Histograms, line graphs, scatter plots, correlation matrices, violin plots
- **Time Series Analysis**: Temporal pattern detection and trend analysis
- **Frequency Domain**: FFT analysis for signal processing applications
- **Sensor-Specific**: Specialized analysis for accelerometer, gyroscope, magnetometer, temperature, pressure, humidity, microphone
- **Real-Time Plot Generation**: Dynamic plot creation with GUI integration
- **Plot Validation**: Ensures requested plots can actually be created before confirmation

### **🔬 Statistical Engine**
- **Feature Discrimination**: Identify most important features between classes
- **Statistical Significance**: Hypothesis testing and confidence intervals
- **Class Comparison**: OK vs KO analysis with detailed metrics (supports 4-class structure)
- **Performance Metrics**: Accuracy, precision, recall, and F1 scores
- **Advanced Statistical Tests**: T-tests, ANOVA, correlation analysis, feature importance ranking
- **Multi-Class Support**: Handles OK, KO_HIGH_2mm, KO_LOW_2mm, KO_LOW_4mm classifications

### **💻 Modern Interface**
- **Chat-Based UI**: Natural conversation with AI for data analysis
- **Integrated Plotting**: Plots generated directly within the chat interface
- **Real-Time Updates**: Dynamic plot generation and analysis results
- **Professional Design**: Clean, intuitive interface for data scientists
- **Multi-View Support**: Chat, Help, and integrated plotting interface
- **Background Processing**: Non-blocking AI responses with worker threads

### **🧠 Advanced AI Features**
- **Response Sanitization**: Multi-layered cleaning to remove AI artifacts
- **Speaker Label Removal**: Eliminates "AI:", "User:" artifacts from responses
- **Instruction Leakage Prevention**: Removes prompt instructions and formatting artifacts
- **Self-Conversation Detection**: Prevents AI from talking to itself
- **Smart Emoji Integration**: Contextually relevant emojis for better user experience
- **Conversation Flow Tracking**: Monitors topic transitions and user preferences

## 🛠️ **Technical Stack**

### **Core Technologies**
- **GUI Framework**: PySide6 (Qt for Python) with modern UI components
- **AI Model**: Llama-3.2-1B with transformers library and CUDA support
- **Data Processing**: pandas, numpy, scipy for statistical analysis
- **Visualization**: matplotlib, seaborn for professional charts
- **Machine Learning**: scikit-learn for feature selection and classification
- **Natural Language Processing**: Advanced regex patterns and intent detection

### **AI & ML Capabilities**
- **Natural Language Understanding**: Advanced prompt engineering and response generation
- **Memory Management**: Conversation context and user preference learning
- **Pattern Recognition**: Intelligent detection of analysis and plot requests
- **Statistical Expertise**: Professional data science knowledge and insights
- **Response Quality Control**: Multi-stage validation and cleaning pipeline

## 📁 **Project Structure**

```
Statistical-AI-Agent/
├── 📱 gui/                          # User Interface Layer
│   ├── main_window.py               # Main application window
│   ├── views/                       # Different analysis views
│   │   ├── chat_view.py            # AI chat interface with integrated plots
│   │   └── help_view.py            # Help and documentation
│   └── resources/                   # UI resources and icons
├── 🧠 core/                         # Core AI and Business Logic
│   ├── transformers_backend.py      # AI chatbot with Llama-3.2-1B
│   ├── unified_parser.py           # Advanced natural language command parser
│   ├── request_handler.py          # Request routing and handling
│   ├── response_formatter.py       # AI response formatting and error handling
│   ├── ml_interface.py             # ML layer API wrapper with validation
│   ├── data_loader.py              # Data ingestion and preprocessing
│   ├── ml_plotter.py               # Advanced plotting engine integration
│   └── prompt.json                 # Enhanced AI prompt system
├── 🤖 ML/                          # Machine Learning Layer
│   ├── ai_agent_backend.py         # Unified AI agent interface
│   ├── statistical_engine.py       # Statistical analysis engine
│   ├── plotting_engine.py          # ML-powered visualization
│   ├── feature_matrix.csv          # Dataset for analysis
│   └── requirements.txt            # ML dependencies
├── 🎯 Llama-3.2-1B/               # AI model directory
├── 📋 requirements.txt             # Main project dependencies
├── 🚀 main.py                      # Application entry point
└── 📖 README.md                    # This documentation
```

## 🚀 **Quick Start**

### **1. Installation**
```bash
# Clone the repository
git clone <your-repo-url>
cd AI-agent

# Install dependencies
pip install -r requirements.txt

# Download AI model (if not included)
Place Llama-3.2-1B model in the Llama-3.2-1B/ directory
```

### **2. Launch Application**
```bash
python main.py
```

### **3. Start Analyzing**
1. **Load Data**: Use File → Open Dataset to load your CSV data
2. **Chat with AI**: Ask for analysis in natural language
3. **Generate Plots**: Request visualizations through conversation
4. **Get Insights**: Receive professional statistical analysis

## 💬 **AI Commands & Examples**

### **📈 Plot Generation**
```
"Show me a line graph of accelerometer data"
"Create a histogram of temperature readings"
"Display correlation matrix between sensors"
"Generate time series analysis of pressure data"
"Show frequency spectrum of vibration data"
"Create scatter plot of feature relationships"
"Generate violin plot for sensor comparison"
```

### **🔍 Statistical Analysis**
```
"What are the most discriminative features?"
"Compare OK vs KO samples statistically"
"Show me feature importance ranking"
"Analyze the dataset for patterns"
"Give me a statistical summary"
"Calculate correlation between temperature and humidity"
```

### **📊 Sensor-Specific Analysis**
```
"Analyze accelerometer sensor patterns"
"Show temperature sensor distribution"
"Compare pressure readings between classes"
"Display humidity sensor correlations"
"Analyze microphone frequency data"
"Compare gyroscope data across classes"
"Analyze magnetometer patterns"
```

### **🎯 Advanced Requests**
```
"Find the best features for classification"
"Show me statistical significance tests"
"Generate comprehensive analysis report"
"Identify outliers in the dataset"
"Recommend next analysis steps"
"Compare all sensor types for discrimination"
"Analyze feature relationships and dependencies"
```

### **💬 General Conversation**
```
"What are your skills?"
"What can you do?"
"How are you today?"
"Thanks for the help"
```

## 🔧 **Configuration & Customization**

### **AI Prompt System**
The enhanced `core/prompt.json` provides:
- **Professional Data Scientist Persona**: Expert-level AI responses
- **Memory Patterns**: Learning user preferences and analysis patterns
- **Response Templates**: Consistent, professional communication
- **Keyword Mapping**: Intelligent detection of analysis requests
- **Project Profile**: Comprehensive understanding of application capabilities
- **Conversation Flow**: Structured conversation management

### **Core Modules**
- **Unified Parser**: Advanced natural language understanding and command routing
- **Request Handler**: Intelligent request processing and ML operation coordination
- **Response Formatter**: Professional formatting with comprehensive error handling
- **ML Interface**: Clean API wrapper with validation and dataset catalog support
- **AI Backend**: Enhanced chatbot with response cleaning and conversation tracking

### **Statistical Engine**
- **Feature Selection**: Identifies most discriminative features using multiple algorithms
- **Class Comparison**: Detailed multi-class analysis with statistical significance
- **Performance Metrics**: Comprehensive ML model evaluation and validation
- **Insight Generation**: Actionable business recommendations and next steps
- **Data Quality Assessment**: Automatic validation and quality scoring

### **Plotting Engine**
- **Natural Language Processing**: Understands plot requests in plain English
- **Sensor-Specific Visualization**: Specialized plots for different sensor types
- **Statistical Plotting**: Advanced statistical visualizations and comparisons
- **Real-Time Generation**: Dynamic plot creation with GUI integration
- **Plot Validation**: Ensures requested visualizations can be created

## 📊 **Data Requirements**

### **Supported Formats**
- **CSV Files**: Primary data format with pandas compatibility
- **Sensor Data**: Accelerometer, gyroscope, magnetometer, temperature, pressure, humidity, microphone
- **Classification Labels**: OK/KO or custom class labels (supports 4-class structure)
- **Feature Matrix**: Numerical features with statistical measures

### **Data Structure**
```csv
sample,label,feature1,feature2,feature3,...
Sample_001,OK,0.123,0.456,0.789,...
Sample_002,KO_HIGH_2mm,0.234,0.567,0.890,...
Sample_003,KO_LOW_2mm,0.345,0.678,0.901,...
Sample_004,KO_LOW_4mm,0.456,0.789,0.012,...
```

### **Supported Sensors**
- **Environmental**: HTS221 (Temperature, Humidity), LPS22HH (Pressure, Temperature), STTS751 (Temperature)
- **Motion**: IIS2DH, IIS3DWB, ISM330DHCX (Accelerometer, Gyroscope)
- **Magnetic**: IIS2MDC (Magnetometer)
- **Audio**: IMP23ABSU, IMP34DT05 (Microphone)

## 🎨 **Visualization Types**

### **Statistical Plots**
- **Line Graphs**: Class comparison and trend analysis (DEFAULT)
- **Histograms**: Distribution analysis and pattern recognition
- **Scatter Plots**: Feature relationship exploration
- **Correlation Matrices**: Feature association analysis
- **Violin Plots**: Distribution comparison across classes
- **Bar Charts**: Categorical data visualization

### **Advanced Visualizations**
- **Time Series**: Temporal pattern analysis
- **Frequency Domain**: FFT-based signal analysis
- **Feature Importance**: Discriminative feature ranking
- **Class Comparison**: Statistical significance visualization
- **Multi-Sensor Analysis**: Cross-sensor correlation and comparison
- **Statistical Significance**: P-value visualization and confidence intervals

## 🔍 **Troubleshooting**

### **Common Issues**
1. **Model Loading**: Ensure Llama-3.2-1B model is properly downloaded
2. **CUDA Support**: Install appropriate torch version for your GPU
3. **Dependencies**: Use exact versions from requirements.txt
4. **Data Loading**: Check CSV format and file permissions
5. **Plot Generation**: Verify requested sensors and features exist in dataset

### **Performance Tips**
- **GPU Acceleration**: Enable CUDA for faster AI inference
- **Data Caching**: Large datasets are automatically cached
- **Memory Management**: Close unused plots to free memory
- **Batch Processing**: Process multiple requests efficiently
- **Background Processing**: AI responses run in separate threads

## 🚀 **Future Enhancements**

### **Planned Features**
- **Real-Time Data Streaming**: Live sensor data analysis
- **Advanced ML Models**: Deep learning integration
- **Cloud Deployment**: Web-based analysis platform
- **API Integration**: RESTful endpoints for external access
- **Multi-Language Support**: Internationalization features
- **Advanced Statistical Tests**: More comprehensive hypothesis testing
- **Custom Plot Types**: User-defined visualization templates

### **Extensibility**
- **Plugin System**: Custom analysis modules
- **Custom Plots**: User-defined visualization types
- **Data Connectors**: Database and API integrations
- **Export Formats**: Multiple output format support
- **Batch Analysis**: Automated analysis workflows

### **Code Standards**
- **Documentation**: Comprehensive docstrings and comments
- **Performance**: Optimize for large datasets and real-time use
- **Error Handling**: Comprehensive validation and graceful degradation

## 🙏 **Acknowledgments**

- **Llama-3.2-1B**: Meta's open-source language model
- **PySide6**: Qt for Python framework
- **Transformers**: Hugging Face's AI library
- **Scientific Python**: pandas, numpy, scipy, matplotlib ecosystem
- **Scikit-learn**: Machine learning algorithms and tools
---