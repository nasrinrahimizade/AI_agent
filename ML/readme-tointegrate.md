# 🤖 Statistical AI Agent - Quick Integration Guide

**For GUI/LLM Developer - Everything you need in 5 minutes**

---

##  Files You Need

```
project_folder/
├── feature_matrix.csv          # Sensor data (PROVIDED)
├── statistical_engine.py      # Statistical analysis (PROVIDED) 
├── plotting_engine.py         # Plot generation (PROVIDED)
├── ai_agent_backend.py        # Main agent interface (PROVIDED)
└── your_gui.py                # Your GUI code (YOUR WORK)
```

---

##  Basic Setup

### 1. Install Dependencies
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

### 2. Initialize the Agent
```python
from ai_agent_backend import StatisticalAIAgent

# Initialize once - this takes 2-3 seconds
agent = StatisticalAIAgent("feature_matrix.csv")
```

---

##  Main Functions (Everything You Need)

### **Dashboard Info** (Fast - Use for GUI startup)
```python
# Get basic dataset info for dashboard
overview = agent.get_dataset_overview()
print(f"Samples: {overview['dataset_overview']['total_samples']}")
print(f"Sensors: {overview['dataset_overview']['available_sensors']}")
```

### **Statistical Analysis** (Use for main results display)
```python
# Get complete analysis results
analysis = agent.get_analysis_summary()

if analysis['status'] == 'success':
    results = analysis['analysis']
    
    # Top discriminative features (for tables)
    top_features = results['best_discriminative_features'][:10]
    
    # Model performance (for metrics display)
    models = results['model_performance']
    
    # Human-readable insights (for text display)
    insights = results['statistical_insights']
```

### **Quick Top Features** (Use for summary tables)
```python
# Just get top N features without full analysis
top_features = agent.get_top_features(n=5)

for feature in top_features['top_features']:
    print(f"{feature['feature_name']}: {feature['separation_score']}")
```

### **Plot Generation** (Use for chat interface)
```python
# Generate plots from natural language
fig = agent.generate_plot("show line graph of accelerometer data")

if fig['status'] == 'success':
    # Display the matplotlib figure in your GUI
    plt.show(fig['figure'])  # or embed in tkinter/streamlit
```

### **Chat Handler** (Use for LLM integration)
```python
# Handle any user request
response = agent.handle_chat_request("What are the best features?")

if response['status'] == 'success':
    if 'figure' in response:
        # It's a plot request
        display_plot(response['figure'])
    else:
        # It's text/data response
        show_message(response['message'])
```

---

##  GUI Integration Examples

### **Streamlit** (Recommended - Easiest)
```python
import streamlit as st

@st.cache_resource  # Cache the agent
def load_agent():
    return StatisticalAIAgent("feature_matrix.csv")

agent = load_agent()

# Dashboard
st.title("Statistical AI Agent")
overview = agent.get_dataset_overview()
st.metric("Samples", overview['dataset_overview']['total_samples'])

# Analysis results
if st.button("Analyze"):
    analysis = agent.get_analysis_summary()
    st.dataframe(analysis['analysis']['best_discriminative_features'][:5])

# Chat interface
user_input = st.text_input("Ask for plots or analysis:")
if user_input:
    response = agent.handle_chat_request(user_input)
    if 'figure' in response:
        st.pyplot(response['figure'])
```

### **Tkinter**
```python
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AIAgentGUI:
    def __init__(self):
        self.agent = StatisticalAIAgent("feature_matrix.csv")
        self.root = tk.Tk()
        self.setup_gui()
    
    def generate_plot(self):
        request = self.entry.get()
        response = self.agent.generate_plot(request)
        
        if response['status'] == 'success':
            # Clear previous plot
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
            
            # Add new plot
            canvas = FigureCanvasTkAgg(response['figure'], self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()
```

---

##  Complete Example App

```python
import streamlit as st
from ai_agent_backend import StatisticalAIAgent

# Initialize
agent = StatisticalAIAgent("feature_matrix.csv")

st.title(" Statistical AI Agent")

# Sidebar - Dataset Info
with st.sidebar:
    st.header(" Dataset")
    overview = agent.get_dataset_overview()
    if overview['status'] == 'success':
        info = overview['dataset_overview']
        st.write(f"**Samples:** {info['total_samples']}")
        st.write(f"**Features:** {info['total_features']}")
        st.write(f"**Classes:** {', '.join(info['classes'])}")

# Main area - Analysis
tab1, tab2 = st.tabs([" Analysis", " Chat"])

with tab1:
    if st.button("🔍 Run Analysis"):
        with st.spinner("Analyzing..."):
            analysis = agent.get_analysis_summary()
        
        if analysis['status'] == 'success':
            results = analysis['analysis']
            
            # Show insights
            st.subheader("💡 Key Insights")
            for insight in results['statistical_insights']:
                st.write(f"• {insight}")
            
            # Show top features
            st.subheader("🏆 Top Features")
            features_df = pd.DataFrame(results['best_discriminative_features'][:10])
            st.dataframe(features_df[['feature_name', 'sensor_name', 'separation_score']])

with tab2:
    st.subheader(" Ask the AI Agent")
    
    # Suggestions
    suggestions = agent.get_plot_suggestions()
    if suggestions['status'] == 'success':
        st.write("**Try these:**")
        for suggestion in suggestions['suggestions']['sensor_comparisons'][:3]:
            if st.button(suggestion):
                response = agent.generate_plot(suggestion)
                if response['status'] == 'success':
                    st.pyplot(response['figure'])
    
    # Manual input
    user_input = st.text_input("Or type your request:")
    if user_input:
        response = agent.handle_chat_request(user_input)
        
        if response['status'] == 'success':
            if 'figure' in response:
                st.pyplot(response['figure'])
            else:
                st.success(response['message'])
        else:
            st.error(response['message'])
```

---

##  Supported Plot Requests

**The agent understands these types of requests:**

| Request | What it generates |
|---------|------------------|
| `"show line graph of accelerometer"` | OK vs KO comparison |
| `"plot histogram of temperature"` | Distribution comparison |
| `"correlation matrix"` | Feature relationships |
| `"time series analysis"` | Temporal plots |
| `"frequency domain plot"` | FFT analysis |
| `"scatter plot features"` | Feature pair relationships |

---

##  Response Format

**All functions return consistent format:**
```python
{
    'status': 'success' or 'error',
    'message': 'Human readable message',
    'data': {...},  # Actual results
    # Additional fields depending on function
}
```

**Always check status:**
```python
result = agent.some_function()
if result['status'] == 'success':
    # Use result['data'] or specific fields
    process_results(result)
else:
    # Handle error
    show_error(result['message'])
```

---

##  Important Notes

1. **Initialization**: Takes 2-3 seconds, do it once and cache the agent
2. **Analysis**: First call to `get_analysis_summary()` takes time, subsequent calls are cached
3. **Memory**: Close matplotlib figures after displaying: `plt.close(fig)`
4. **Errors**: Always check `response['status']` before using data

---

##  Test Everything Works

```python
# Quick test
agent = StatisticalAIAgent("feature_matrix.csv")

# Test 1: Basic info
overview = agent.get_dataset_overview()
print("✅ Overview:", overview['status'])

# Test 2: Analysis
analysis = agent.get_analysis_summary()
print("✅ Analysis:", analysis['status'])

# Test 3: Plot
plot = agent.generate_plot("line graph accelerometer")
print("✅ Plot:", plot['status'])

# If all show 'success', you're ready! 🎉
```

---

##  You're Ready!

The agent handles all the complex statistical analysis and plotting. You just need to:

1. **Call the functions** above in your GUI
2. **Display the results** in tables, charts, text
3. **Handle the chat interface** for user requests

The backend is complete - focus on making a great user interface! 