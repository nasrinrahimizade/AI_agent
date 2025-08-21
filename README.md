# 🤖 Statistical AI Agent - Advanced Data Analysis Platform

A comprehensive, AI-powered desktop application for statistical analysis, machine learning, and data visualization with natural language interface. Built with PySide6 and powered by Llama-3.2-1B for intelligent data science workflows.

### **Enhanced AI Capabilities**
- **Professional Data Scientist Persona**: Expert-level AI assistant specialized in sensor data analysis
- **Advanced Natural Language Processing**: Intelligent plot request detection and generation
- **Memory & Learning**: AI remembers user preferences and learns from interactions
- **Statistical Expertise**: Focus on statistical significance and data-driven insights

### **Improved Architecture**
- **Unified AI Agent Interface**: Single entry point for all ML and statistical operations
- **Enhanced Prompt System**: Comprehensive, professional prompts for data science tasks
- **Better Error Handling**: Fallback data generation and graceful error recovery
- **Performance Optimization**: Cached analysis results and efficient data processing

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GUI Layer     │    │   Core Layer    │    │   ML Layer      │
│   (PySide6)     │◄──►│   (AI Logic)    │◄──►│   (Analysis)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Three-Tier Architecture**
- **GUI Layer**: Modern PySide6 desktop interface with multiple analysis views
- **Core Layer**: AI chatbot, plotting engine, and data processing
- **ML Layer**: Statistical analysis, feature selection, and machine learning

## 🎯 **Key Features**

### **🤖 AI-Powered Analysis**
- **Natural Language Interface**: Ask for plots and analysis in plain English
- **Intelligent Plot Generation**: Automatic visualization based on conversation context
- **Statistical Insights**: Professional data science analysis and recommendations
- **Learning Capability**: AI remembers your preferences and improves over time

### **📊 Advanced Visualization**
- **Multiple Plot Types**: Histograms, line graphs, scatter plots, correlation matrices
- **Time Series Analysis**: Temporal pattern detection and trend analysis
- **Frequency Domain**: FFT analysis for signal processing applications
- **Sensor-Specific**: Specialized analysis for accelerometer, temperature, pressure, humidity

### **🔬 Statistical Engine**
- **Feature Discrimination**: Identify most important features between classes
- **Statistical Significance**: Hypothesis testing and confidence intervals
- **Class Comparison**: OK vs KO analysis with detailed metrics
- **Performance Metrics**: Accuracy, precision, recall, and F1 scores

### **💻 Modern Interface**
- **Chat-Based UI**: Natural conversation with AI for data analysis
- **Multiple Views**: Chat, Help, and integrated plotting interface
- **Real-Time Updates**: Dynamic plot generation and analysis results
- **Professional Design**: Clean, intuitive interface for data scientists

## 🛠️ **Technical Stack**

### **Core Technologies**
- **GUI Framework**: PySide6 (Qt for Python) with modern UI components
- **AI Model**: Llama-3.2-1B with transformers library and CUDA support
- **Data Processing**: pandas, numpy, scipy for statistical analysis
- **Visualization**: matplotlib, seaborn for professional charts
- **Machine Learning**: scikit-learn for feature selection and classification

### **AI & ML Capabilities**
- **Natural Language Understanding**: Advanced prompt engineering and response generation
- **Memory Management**: Conversation context and user preference learning
- **Pattern Recognition**: Intelligent detection of analysis and plot requests
- **Statistical Expertise**: Professional data science knowledge and insights

## 📁 **Project Structure**

```
Statistical-AI-Agent/
├── 📱 gui/                          # User Interface Layer
│   ├── main_window.py               # Main application window
│   ├── views/                       # Different analysis views
│   │   ├── chat_view.py            # AI chat interface with plots
│   │   └── help_view.py            # Help and documentation
│   └── resources/                   # UI resources and icons
├── 🧠 core/                         # Core AI and Business Logic
│   ├── transformers_backend.py      # AI chatbot with Llama-3.2-1B
│   ├── ml_plotter.py               # Advanced plotting engine
│   ├── data_loader.py              # Data ingestion and preprocessing
│   ├── ml_interface.py             # ML layer API wrapper
│   ├── command_parser.py           # Natural language command parser
│   ├── response_formatter.py       # AI response formatting
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
pip install -r ML/requirements.txt

# Download AI model (if not included)
# Place Llama-3.2-1B model in the Llama-3.2-1B/ directory
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
```

### **🔍 Statistical Analysis**
```
"What are the most discriminative features?"
"Compare OK vs KO samples statistically"
"Show me feature importance ranking"
"Analyze the dataset for patterns"
"Give me a statistical summary"
```

### **📊 Sensor-Specific Analysis**
```
"Analyze accelerometer sensor patterns"
"Show temperature sensor distribution"
"Compare pressure readings between classes"
"Display humidity sensor correlations"
"Analyze microphone frequency data"
```

### **🎯 Advanced Requests**
```
"Find the best features for classification"
"Show me statistical significance tests"
"Generate comprehensive analysis report"
"Identify outliers in the dataset"
"Recommend next analysis steps"
```

## 🔧 **Configuration & Customization**

### **AI Prompt System**
The enhanced `core/prompt.json` provides:
- **Professional Data Scientist Persona**: Expert-level AI responses
- **Memory Patterns**: Learning user preferences and analysis patterns
- **Response Templates**: Consistent, professional communication
- **Keyword Mapping**: Intelligent detection of analysis requests

### **Core Modules**
- **ML Interface**: Clean API wrapper for ML layer operations
- **Command Parser**: Natural language understanding for user queries
- **Response Formatter**: Professional formatting of analysis results
- **Plotting Engine**: Advanced visualization capabilities

### **Statistical Engine**
- **Feature Selection**: Identifies most discriminative features
- **Class Comparison**: Detailed OK vs KO analysis
- **Performance Metrics**: Comprehensive ML model evaluation
- **Insight Generation**: Actionable business recommendations

## 📊 **Data Requirements**

### **Supported Formats**
- **CSV Files**: Primary data format with pandas compatibility
- **Sensor Data**: Accelerometer, gyroscope, magnetometer, temperature, pressure, humidity
- **Classification Labels**: OK/KO or custom class labels
- **Feature Matrix**: Numerical features with statistical measures

### **Data Structure**
```csv
sample,label,feature1,feature2,feature3,...
Sample_001,OK,0.123,0.456,0.789,...
Sample_002,KO,0.234,0.567,0.890,...
```

## 🎨 **Visualization Types**

### **Statistical Plots**
- **Line Graphs**: Class comparison and trend analysis
- **Histograms**: Distribution analysis and pattern recognition
- **Scatter Plots**: Feature relationship exploration
- **Correlation Matrices**: Feature association analysis

### **Advanced Visualizations**
- **Time Series**: Temporal pattern analysis
- **Frequency Domain**: FFT-based signal analysis
- **Feature Importance**: Discriminative feature ranking
- **Class Comparison**: Statistical significance visualization

## 🔍 **Troubleshooting**

### **Common Issues**
1. **Model Loading**: Ensure Llama-3.2-1B model is properly downloaded
2. **CUDA Support**: Install appropriate torch version for your GPU
3. **Dependencies**: Use exact versions from requirements.txt
4. **Data Loading**: Check CSV format and file permissions

### **Performance Tips**
- **GPU Acceleration**: Enable CUDA for faster AI inference
- **Data Caching**: Large datasets are automatically cached
- **Memory Management**: Close unused plots to free memory
- **Batch Processing**: Process multiple requests efficiently

## 🚀 **Future Enhancements**

### **Planned Features**
- **Real-Time Data Streaming**: Live sensor data analysis
- **Advanced ML Models**: Deep learning integration
- **Cloud Deployment**: Web-based analysis platform
- **API Integration**: RESTful endpoints for external access
- **Multi-Language Support**: Internationalization features

### **Extensibility**
- **Plugin System**: Custom analysis modules
- **Custom Plots**: User-defined visualization types
- **Data Connectors**: Database and API integrations

### **Development Setup**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

### **Code Standards**
- **Documentation**: Comprehensive docstrings and comments
- **Performance**: Optimize for large datasets and real-time use

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**

- **Llama-3.2-1B**: Meta's open-source language model
- **PySide6**: Qt for Python framework
- **Transformers**: Hugging Face's AI library
- **Scientific Python**: pandas, numpy, scipy, matplotlib ecosystem
---

**🎯 Ready to transform your data analysis workflow? Launch the Statistical AI Agent and experience the future of intelligent data science!** 
