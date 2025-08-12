# Statistical AI Agent

A comprehensive data analysis and visualization application with AI-powered chat interface for sensor data analysis.

## Overview

The Statistical AI Agent is a desktop application built with PySide6 that provides an interactive interface for analyzing sensor data through various views and an AI-powered chat assistant. The application is structured into two main components: **GUI** (user interface) and **Core** (business logic and AI functionality).

## Architecture

### GUI Component (`gui/`)

The GUI component handles all user interface elements and provides multiple views for data analysis:

#### Main Window (`gui/main_window.py`)
- **Purpose**: Central application window that manages all views and navigation
- **Features**:
  - Menu bar with File operations (Open Dataset, Exit)
  - Toolbar with quick navigation buttons for all views
  - Status bar for application feedback
  - Stacked widget system for switching between different views
  - Default view: Chat interface (index 4)

#### Views (`gui/views/`)

**1. Chat View (`chat_view.py`)**
- **Purpose**: AI-powered chat interface for natural language data analysis
- **Features**:
  - Real-time chat with AI assistant using Llama-3.2-1B model
  - Natural language plot request detection
  - Automatic plot generation based on conversation context
  - Integration with plotting engine for visualizations
  - Support for various plot types: boxplots, histograms, correlation matrices, time series, frequency analysis, scatter plots
  - Enter key handling for message sending
  - Timestamp-based message formatting

**2. Data View (`data_view.py`)**
- **Purpose**: Display raw dataset in tabular format
- **Features**:
  - QTableView for displaying pandas DataFrame
  - Placeholder for future DataFrame model integration

**3. Statistics View (`stats_view.py`)**
- **Purpose**: Display statistical analysis of the dataset
- **Features**:
  - Placeholder for comprehensive statistical computations
  - Basic row count display
  - Extensible for mean, std, and other statistical measures

**4. Model View (`model_view.py`)**
- **Purpose**: Machine learning model interface
- **Features**:
  - Placeholder for ML model operations and training interface

**5. Plot View (`plot_view.py`)**
- **Purpose**: Display matplotlib visualizations
- **Features**:
  - FigureCanvas integration for matplotlib plots
  - Dynamic plot switching and display
  - Integration with plotting engine for advanced visualizations

#### Resources (`gui/resources/`)
- **UI Files**: Qt Designer files for interface layouts
- **Icons**: Application icon resources

### Core Component (`core/`)

The core component handles all business logic, AI functionality, and data processing:

#### AI Backend (`core/transformers_backend.py`)
- **Purpose**: AI chatbot implementation using transformers library
- **Features**:
  - Llama-3.2-1B model integration with CUDA support
  - 4-bit quantization for optimized inference
  - Dynamic memory extraction from user conversations
  - Conversation history management with configurable limits
  - Advanced text generation with temperature, top-k, top-p sampling
  - Repetition penalty to prevent response loops
  - Natural language understanding for data analysis requests

#### Plotting Engine (`core/ml_plotter.py`)
- **Purpose**: Advanced statistical visualization system
- **Features**:
  - Natural language plot request parsing
  - Multiple plot types: boxplots, histograms, correlation matrices, time series, frequency analysis, scatter plots
  - Sensor-specific analysis (accelerometer, temperature, pressure, humidity)
  - Statistical significance testing between OK/KO classes
  - Feature discriminative analysis
  - Fallback data generation for testing
  - Integration with matplotlib and seaborn

#### Data Loader (`core/data_loader.py`)
- **Purpose**: Data ingestion and preprocessing
- **Features**:
  - CSV file loading with pandas
  - Extensible data cleaning pipeline
  - Support for various data formats

#### Prompt Configuration (`core/prompt.json`)
- **Purpose**: AI system prompt and configuration management
- **Features**:
  - Specialized prompts for data analysis assistant
  - Plot request detection patterns
  - Natural language understanding for visualization requests
  - Context-aware responses for sensor data analysis

## Key Features

### AI-Powered Analysis
- Natural language interface for data analysis
- Automatic plot generation based on conversation context
- Specialized knowledge in sensor data analysis
- Real-time statistical insights

### Multi-View Interface
- **Data View**: Raw data exploration
- **Statistics View**: Statistical analysis and summaries
- **Model View**: Machine learning operations
- **Plot View**: Advanced visualizations
- **Chat View**: AI-powered conversational analysis

### Advanced Plotting
- Boxplots for feature comparison
- Histograms for distribution analysis
- Correlation matrices for feature relationships
- Time series analysis for temporal patterns
- Frequency domain analysis using FFT
- Scatter plots for feature relationships
- Sensor-specific analysis (accelerometer, temperature, pressure, humidity)

### Data Management
- CSV file loading and processing
- Automatic data preparation and cleaning
- Feature matrix handling for ML operations
- Binary classification support (OK vs KO)

## Technical Stack

- **GUI Framework**: PySide6 (Qt for Python)
- **AI Model**: Llama-3.2-1B (transformers library)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Statistical Analysis**: scipy
- **Machine Learning**: scikit-learn integration ready

## Usage

1. **Launch Application**: Run `main.py` to start the application
2. **Load Data**: Use File → Open Dataset to load CSV sensor data
3. **Navigate Views**: Use toolbar buttons to switch between different analysis views
4. **Chat Analysis**: Use the Chat view for natural language data analysis
5. **Generate Plots**: Request visualizations through the chat interface or plot view

## File Structure

```
Statistical-AI-Agent/
├── gui/
│   ├── main_window.py          # Main application window
│   ├── views/
│   │   ├── chat_view.py        # AI chat interface
│   │   ├── data_view.py        # Data table display
│   │   ├── stats_view.py       # Statistical analysis
│   │   ├── model_view.py       # ML model interface
│   │   └── plot_view.py        # Visualization display
│   └── resources/              # UI resources and icons
├── core/
│   ├── transformers_backend.py # AI chatbot implementation
│   ├── ml_plotter.py          # Advanced plotting engine
│   ├── data_loader.py         # Data ingestion
│   └── prompt.json            # AI configuration
├── Llama-3.2-1B/             # AI model directory
└── main.py                    # Application entry point
```

## Development Notes

- The application uses a modular architecture with clear separation between GUI and core logic
- AI responses are optimized for sensor data analysis with specialized prompts
- Plot generation is triggered through natural language detection in the chat interface
- The system supports both OK and KO classification for sensor data analysis
- All views are extensible for additional functionality

## Component Integration: How GUI, Core, and ML Work Together

The application follows a three-tier architecture where each component has specific responsibilities and communicates through well-defined interfaces:

### **Integration Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GUI Layer     │    │   Core Layer    │    │   ML Layer      │
│   (PySide6)     │◄──►│   (AI Logic)    │◄──►│   (Analysis)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Data Flow and Communication**

#### **1. Application Initialization**
```python
# main.py - Entry point
from ML.ai_agent_backend import StatisticalAIAgent
agent = StatisticalAIAgent("ML/feature_matrix.csv")  # ML integration
window = MainWindow(agent)  # GUI with ML agent
```

#### **2. GUI ↔ Core Communication**
- **GUI** (`gui/main_window.py`) creates the main application window
- **Core** (`core/transformers_backend.py`) provides AI chatbot functionality
- **Core** (`core/ml_plotter.py`) handles plotting requests from GUI
- **Core** (`core/data_loader.py`) manages data loading for all views

#### **3. Core ↔ ML Communication**
- **Core** plotting engine references `ML/feature_matrix.csv` for data
- **Core** AI prompts reference ML data sources
- **ML** (`ML/ai_agent_backend.py`) provides unified interface for all operations

### **Component Responsibilities**

#### **GUI Layer (`gui/`)**
- **User Interface**: PySide6-based desktop application
- **View Management**: 5 different views (Chat, Data, Stats, Model, Plot)
- **Event Handling**: User interactions, file loading, view switching
- **Plot Display**: Integration with matplotlib for visualization rendering

#### **Core Layer (`core/`)**
- **AI Chatbot**: Llama-3.2-1B model integration with transformers
- **Plot Generation**: Natural language parsing and matplotlib figure creation
- **Data Processing**: CSV loading and preprocessing
- **Prompt Management**: Specialized AI prompts for data analysis

#### **ML Layer (`ML/`)**
- **Statistical Analysis**: Advanced feature discrimination and class analysis
- **Unified Agent Interface**: Single entry point for all ML operations
- **Plotting Engine**: Comprehensive visualization system
- **Data Management**: Feature matrix handling and sensor data processing

### **Integration Points**

#### **1. Data Flow**
```
User Input (GUI) → Chat View → AI Backend (Core) → ML Agent (ML) → Statistical Engine (ML) → Plotting Engine (ML) → Plot View (GUI)
```

#### **2. File Loading**
```
File Dialog (GUI) → Data Loader (Core) → Feature Matrix (ML) → All Views (GUI)
```

#### **3. Plot Generation**
```
Natural Language (GUI Chat) → Plot Request Detection (Core) → Plotting Engine (ML) → Matplotlib Figure → Plot View (GUI)
```

#### **4. Statistical Analysis**
```
Analysis Request (GUI) → Statistical Engine (ML) → Results Cache → Statistics View (GUI)
```

### **Key Integration Features**

#### **Unified AI Agent Interface**
- **ML** provides `StatisticalAIAgent` class as single entry point
- **Core** AI chatbot integrates with ML agent for comprehensive analysis
- **GUI** uses ML agent for all statistical and plotting operations

#### **Natural Language Processing**
- **Core** AI understands sensor data analysis requests
- **ML** plotting engine parses natural language plot requests
- **GUI** chat interface triggers automatic plot generation

#### **Data Consistency**
- All components reference `ML/feature_matrix.csv` as primary data source
- **ML** statistical engine provides consistent analysis across all views
- **Core** plotting engine ensures visualization consistency

#### **Performance Optimization**
- **ML** agent caches expensive statistical analysis results
- **Core** AI model uses 4-bit quantization for faster inference
- **GUI** uses efficient Qt widgets for responsive interface

### **Cross-Component Communication**

#### **GUI → Core**
- Chat messages sent to AI backend
- File loading requests to data loader
- Plot display requests to plotting engine

#### **Core → ML**
- Statistical analysis requests to ML agent
- Plot generation requests to ML plotting engine
- Data preprocessing requests to ML statistical engine

#### **ML → GUI**
- Analysis results displayed in statistics view
- Generated plots displayed in plot view
- Dataset overview shown in data view

### **Error Handling and Fallbacks**
- **ML** provides fallback data generation if feature matrix unavailable
- **Core** AI handles conversation gracefully with error recovery
- **GUI** shows appropriate error messages for failed operations

### **Extensibility Points**
- **ML** statistical engine supports new analysis methods
- **Core** plotting engine supports new visualization types
- **GUI** views can be extended with new functionality
- **AI** prompts can be customized for different analysis domains

## Dependencies

### **Required Dependencies**
```bash
# Core Dependencies
pip install PySide6 qasync
pip install transformers torch accelerate bitsandbytes
pip install pandas numpy scipy scikit-learn
pip install matplotlib seaborn
```

### **System Requirements**
- **Python**: 3.8+
- **RAM**: 8GB+ (16GB+ recommended)
- **GPU**: NVIDIA with CUDA 11.8+ (optional)
- **Storage**: 5GB+ free space

## AI Plot Commands

The AI assistant understands natural language requests for generating plots. Here are the key commands:

### **Plot Types**
```
"Show me a boxplot of accelerometer data"
"Create a histogram of temperature readings"
"Display correlation matrix"
"Generate time series analysis"
"Show frequency spectrum"
"Create scatter plot"
```

### **Sensor Analysis**
```
"Show me accelerometer data"
"Analyze temperature patterns"
"Display pressure sensor data"
"Plot humidity distribution"
```

### **Statistical Analysis**
```
"What are the best features?"
"Compare OK vs KO samples"
"Show feature importance"
"Analyze the dataset"
```

### **Quick Commands**
```
"Give me a quick overview"
"Show me the most important plots"
"Create comprehensive analysis"
"Generate dashboard"
```

**Tips**: Be specific with sensor names and analysis types. The AI will automatically generate plots and display them in the Plot View. ALL dependencies exist in requirement.txt 
