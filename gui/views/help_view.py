from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextBrowser, QScrollArea
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt


class HelpView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        title = QLabel("AI Agent Help & Capabilities")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Create scrollable area for better navigation
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setReadOnly(True)
        body_font = QFont()
        body_font.setPointSize(11)
        browser.setFont(body_font)

        # Comprehensive help content with all new features
        browser.setHtml(
            """
            <div>
                <h2>🚀 Getting Started</h2>
                <ol>
                    <li><b>Load Data:</b> Use <b>File → Open Dataset</b> to load your CSV data</li>
                    <li><b>Chat with AI:</b> Use the Chat interface to ask for analysis in natural language</li>
                    <li><b>View Results:</b> Analysis results and plots appear in the integrated interface</li>
                </ol>

                <h2>⚡ Quick Start - Line Graphs & Sensor Analysis</h2>
                <p><b>Line graphs are now the default! The app automatically creates accurate sensor-specific visualizations:</b></p>
                <ul>
                    <li><b>🕒 Line Graphs (Default):</b> "Generate a plot of accelerometer data" - Automatically creates line graphs</li>
                    <li><b>🎯 Sensor-Specific:</b> "Show me humidity data" - Shows only humidity sensor data, no more or less</li>
                    <li><b>📊 Precise Data:</b> "Create a line graph of temperature data" - Shows only temperature sensors</li>
                </ul>

                <h2>🎯 NEW: Smart Sensor Detection & Line Graphs</h2>
                <h3>Automatic Sensor Recognition</h3>
                <p>The app now intelligently detects sensor types and creates precise line graphs:</p>
                <ul>
                    <li><b>Temperature Requests:</b> "Show temperature data" → Only temperature sensors (HTS221, LPS22HH, STTS751)</li>
                    <li><b>Humidity Requests:</b> "Show humidity data" → Only humidity sensors (HTS221)</li>
                    <li><b>Accelerometer Requests:</b> "Show accelerometer data" → Only accelerometer sensors (IIS2DH, IIS3DWB, ISM330DHCX)</li>
                    <li><b>Pressure Requests:</b> "Show pressure data" → Only pressure sensors (LPS22HH)</li>
                    <li><b>Gyroscope Requests:</b> "Show gyroscope data" → Only gyroscope sensors (ISM330DHCX)</li>
                    <li><b>Magnetometer Requests:</b> "Show magnetometer data" → Only magnetometer sensors (IIS2MDC)</li>
                    <li><b>Microphone Requests:</b> "Show microphone data" → Only microphone sensors (IMP23ABSU, IMP34DT05)</li>
                </ul>

                <h3>Line Graph Features</h3>
                <ul>
                    <li><b>Multi-Class Comparison:</b> Shows OK vs KO_HIGH_2mm vs KO_LOW_2mm vs KO_LOW_4mm</li>
                    <li><b>Sensor-Specific Data:</b> Only displays data from the requested sensor type</li>
                    <li><b>Statistical Accuracy:</b> Uses actual CSV data with proper statistical measures</li>
                    <li><b>Interactive Display:</b> Plots appear directly in the chat interface</li>
                </ul>

                <h2>📊 Statistical Analysis Requests</h2>
                <h3>Basic Statistics</h3>
                <ul>
                    <li><b>"What is the mean temperature for KO_HIGH_2mm samples from HTS221_TEMP?"</b> - Get average temperature for specific class and sensor</li>
                    <li><b>"Calculate the median humidity for KO_LOW_2mm samples"</b> - Get middle value for specific class</li>
                    <li><b>"Give me the standard deviation of pressure from LPS22HH_PRESS for each class"</b> - Get spread across all classes</li>
                    <li><b>"List the top 3 statistical indices that best separate KO_HIGH_2mm and KO_LOW_2mm samples"</b> - Find most discriminative features</li>
                </ul>

                <h3>Sensor-Specific Requests</h3>
                <ul>
                    <li><b>HTS221 Temperature Sensor:</b> "What is the mean HTS221_TEMP for KO_HIGH_2mm samples?"</li>
                    <li><b>HTS221 Humidity Sensor:</b> "Show me humidity stats from HTS221_HUM for each class"</li>
                    <li><b>LPS22HH Pressure Sensor:</b> "Calculate pressure variance from LPS22HH_PRESS for KO_LOW_2mm samples"</li>
                    <li><b>IIS3DWB Accelerometer:</b> "Compare acceleration data from IIS3DWB_ACC between KO_HIGH_2mm and KO_LOW_2mm"</li>
                </ul>

                <h3>Grouped Analysis</h3>
                <ul>
                    <li><b>"Show me temperature statistics grouped by class"</b> - Compare across KO_HIGH_2mm/KO_LOW_2mm/KO_LOW_4mm groups</li>
                    <li><b>"What is the mean humidity for each class?"</b> - Class-specific averages</li>
                    <li><b>"Compare pressure readings between classes"</b> - Multi-class comparison</li>
                </ul>

                <h2>🔍 Feature Analysis</h2>
                <h3>Feature Importance</h3>
                <ul>
                    <li><b>"What are the most discriminative features?"</b> - Find best classification features</li>
                    <li><b>"Which features separate OK from KO samples best?"</b> - Feature discrimination</li>
                </ul>

                <h3>Detailed Feature Analysis</h3>
                <ul>
                    <li><b>"Analyze the temperature sensor patterns"</b> - Comprehensive temperature analysis</li>
                    <li><b>"Show me humidity sensor distribution"</b> - Statistical distribution analysis</li>
                    <li><b>"Compare acceleration sensors between classes"</b> - Multi-axis comparison</li>
                </ul>

                <h2>📈 Visualization Requests</h2>
                <h3>Line Graphs (Default)</h3>
                <ul>
                    <li><b>"Generate a plot of accelerometer data"</b> - Creates line graph of all accelerometer sensors</li>
                    <li><b>"Show me humidity data"</b> - Creates line graph of humidity sensor data</li>
                    <li><b>"Create a line graph of temperature data"</b> - Creates line graph of all temperature sensors</li>
                    <li><b>"Display pressure data"</b> - Creates line graph of pressure sensor data</li>
                    <li><b>"Show gyroscope data"</b> - Creates line graph of gyroscope sensor data</li>
                </ul>

                <h3>Other Plot Types</h3>
                <ul>
                    <li><b>"Create a histogram of humidity data"</b> - Distribution analysis across all classes</li>
                    <li><b>"Generate a scatter plot of temperature vs humidity"</b> - Feature relationships</li>
                    <li><b>"Display correlation matrix between sensors"</b> - Feature associations</li>
                    <li><b>"Show frequency domain analysis"</b> - FFT analysis for signal processing</li>
                </ul>

                <h2>🔬 Sensor-Specific Analysis</h2>
                <h3>Temperature & Environmental</h3>
                <ul>
                    <li><b>"Analyze temperature sensor patterns"</b> - Temperature analysis with line graphs</li>
                    <li><b>"Show humidity sensor distribution"</b> - Humidity analysis with line graphs</li>
                    <li><b>"Compare pressure readings between classes"</b> - Pressure analysis with line graphs</li>
                </ul>

                <h3>Motion & Vibration</h3>
                <ul>
                    <li><b>"Show me accelerometer sensor patterns"</b> - Acceleration analysis with line graphs</li>
                    <li><b>"Analyze gyroscope data"</b> - Rotation analysis with line graphs</li>
                    <li><b>"Display magnetometer readings"</b> - Magnetic field analysis with line graphs</li>
                    <li><b>"Show frequency spectrum of vibration data"</b> - Vibration analysis with FFT</li>
                </ul>

                <h3>Audio & Signal</h3>
                <ul>
                    <li><b>"Analyze microphone frequency data"</b> - Audio signal analysis with FFT</li>
                    <li><b>"Show me microphone signal patterns"</b> - Audio pattern analysis with line graphs</li>
                </ul>

                <h2>📋 Class Comparison & Classification</h2>
                <h3>Multi-Class Analysis</h3>
                <ul>
                    <li><b>"Compare KO_HIGH_2mm vs KO_LOW_2mm samples statistically"</b> - Binary classification analysis</li>
                    <li><b>"Show me differences between all classes"</b> - Multi-class comparison</li>
                    <li><b>"Analyze KO_HIGH_2mm vs KO_LOW_4mm"</b> - Subclass comparison</li>
                    <li><b>"Which features best separate the classes?"</b> - Classification feature selection</li>
                </ul>

                <h3>Statistical Significance</h3>
                <ul>
                    <li><b>"Are the differences between classes statistically significant?"</b> - Hypothesis testing</li>
                    <li><b>"Show me confidence intervals for the comparisons"</b> - Statistical confidence</li>
                    <li><b>"What is the effect size between KO_HIGH_2mm and KO_LOW_2mm?"</b> - Practical significance</li>
                </ul>

                <h2>🎯 Advanced AI Requests</h2>
                <h3>Machine Learning Insights</h3>
                <ul>
                    <li><b>"Find the best features for classification"</b> - Feature selection</li>
                    <li><b>"Show me statistical significance tests"</b> - Statistical testing</li>
                    <li><b>"Generate comprehensive analysis report"</b> - Full dataset analysis</li>
                    <li><b>"Identify outliers in the dataset"</b> - Anomaly detection</li>
                </ul>

                <h3>Predictive Analysis</h3>
                <ul>
                    <li><b>"Recommend next analysis steps"</b> - AI-guided workflow</li>
                    <li><b>"What patterns should I investigate next?"</b> - Pattern discovery</li>
                    <li><b>"Show me the most interesting findings"</b> - Insight discovery</li>
                </ul>

                <h2>💡 Natural Language Tips</h2>
                <div>
                    <h4>How to Talk to Your AI Agent:</h4>
                    <ul>
                        <li><b>Be Specific:</b> "Show me temperature data for OK samples" vs "Show me data"</li>
                        <li><b>Use Natural Language:</b> "What's the average humidity?" works just as well as "Get mean humidity"</li>
                        <li><b>Ask for Comparisons:</b> "Compare OK vs KO" or "Show differences between classes"</li>
                        <li><b>Request Visualizations:</b> "Create a line graph" or "Show me a histogram"</li>
                        <li><b>Ask for Insights:</b> "What patterns do you see?" or "What's interesting about this data?"</li>
                    </ul>
                </div>

                <h2>🔧 Available Data Features</h2>
                <div>
                    <h4>Your Dataset Includes:</h4>
                    <ul>
                        <li><b>Environmental Sensors:</b> Temperature, Humidity, Pressure</li>
                        <li><b>Motion Sensors:</b> Accelerometer (X, Y, Z), Gyroscope (X, Y, Z)</li>
                        <li><b>Audio Sensors:</b> Microphone</li>
                        <li><b>Classification Labels:</b> OK, KO, KO_HIGH_2mm, KO_LOW_2mm</li>
                        <li><b>Statistical Measures:</b> Mean, Median, Standard Deviation, Variance, Min, Max, Count</li>
                    </ul>
                </div>

                <h2>🚀 Pro Tips</h2>
                <div>
                    <ul>
                        <li><b>Start Simple:</b> Begin with basic statistics like "What is the mean temperature?"</li>
                        <li><b>Build Complexity:</b> Add filters like "Show me temperature for OK samples only"</li>
                        <li><b>Request Visualizations:</b> Ask for plots to better understand your data</li>
                        <li><b>Compare Classes:</b> Use "Compare OK vs KO" to find differences</li>
                        <li><b>Ask for Insights:</b> Let the AI suggest what to investigate next</li>
                        <li><b>Use Natural Language:</b> Talk to the AI like you would to a colleague</li>
                    </ul>
                </div>

                <h2>🔧 Line Graph Troubleshooting</h2>
                <div>
                    <h4>Default Behavior:</h4>
                    <ul>
                        <li><b>🕒 Line Graphs are Default:</b> Simple requests like "Create a plot" or "Show me data" now generate line graphs</li>
                        <li><b>📊 Other Plot Types:</b> Explicitly request "histogram", "scatter plot", or "frequency analysis" for different visualizations</li>
                        <li><b>🎯 Smart Detection:</b> The system automatically detects sensor types and creates precise visualizations</li>
                    </ul>
                    
                    <h4>Common Issues & Solutions:</h4>
                    <ul>
                        <li><b>Line Graph Not Showing:</b> Try "Generate a plot of [sensor] data" or "Show me [sensor] data"</li>
                        <li><b>Want Different Plot Type:</b> Explicitly say "Create a histogram" or "Show scatter plot"</li>
                        <li><b>Wrong Sensor Data:</b> Be specific: "Show me temperature data" not just "Show me data"</li>
                        <li><b>Class Comparison:</b> Add "between classes" to compare patterns across OK/KO groups</li>
                        <li><b>Frequency Analysis:</b> Use "Show frequency domain analysis" for FFT plots</li>
                    </ul>
                    
                    <h4>Best Practices:</h4>
                    <ul>
                        <li><b>For Motion Analysis:</b> Use "Show accelerometer data" for line graphs, "frequency analysis" for FFT</li>
                        <li><b>For Environmental Monitoring:</b> Use "Show temperature data" for line graphs</li>
                        <li><b>For Audio Analysis:</b> Use "Show microphone data" for line graphs, "frequency analysis" for FFT</li>
                        <li><b>For Pattern Detection:</b> Combine both line graphs and frequency analysis</li>
                    </ul>
                </div>

                <h2>🎯 Smart Response System</h2>
                <h3>🕒 Line Graphs as Default</h3>
                <p><b>NEW:</b> Line graphs are now the default visualization! The system automatically generates line graphs for most requests unless you specify otherwise:</p>
                <ul>
                    <li><b>Default Behavior:</b> "Create a plot" → Line graph</li>
                    <li><b>Smart Detection:</b> "Show me the data" → Line graph</li>
                    <li><b>Sensor Analysis:</b> "Analyze temperature" → Line graph</li>
                    <li><b>Override Default:</b> "Create a histogram" → Histogram (explicit request)</li>
                </ul>

                <h3>Automatic Response Type Detection</h3>
                <p>The system automatically detects whether you want a text response or visual response based on your request:</p>
                
                <h4>📝 Text Responses (No Plots)</h4>
                <ul>
                    <li><b>Lists & Analysis:</b> "List the top 3 features", "Show me the analysis", "Tell me about..."</li>
                    <li><b>Information Requests:</b> "What is the mean", "Describe the differences", "Explain the patterns"</li>
                    <li><b>Comparative Analysis:</b> "Compare OK vs KO", "Analyze the relationships", "Find the best features"</li>
                </ul>
                
                <h4>📊 Visual Responses (With Plots)</h4>
                <ul>
                    <li><b>Plot Requests:</b> "Create a line graph", "Show me a histogram", "Generate a scatter plot"</li>
                    <li><b>Visual Commands:</b> "Plot the data", "Display the chart", "Visualize the results"</li>
                    <li><b>Chart Generation:</b> "Make a graph", "Draw a plot", "Show the visualization"</li>
                </ul>
                
                <h4>🔄 Auto Mode (Smart Defaults)</h4>
                <ul>
                    <li><b>Statistics:</b> Text with optional plot suggestions</li>
                    <li><b>Top Features:</b> Always text (informational)</li>
                    <li><b>Comparisons:</b> Text with comparison plot suggestions</li>
                    <li><b>Analysis:</b> Text with relevant visualization options</li>
                </ul>

                <h2>🔍 NEW: Advanced Sensor Detection</h2>
                <h3>Precise Sensor Mapping</h3>
                <p>The app now uses exact sensor-to-feature mapping for accurate data selection:</p>
                <ul>
                    <li><b>Temperature Sensors:</b> HTS221_TEMP_TEMP_mean, LPS22HH_TEMP_TEMP_mean, STTS751_TEMP_TEMP_mean</li>
                    <li><b>Humidity Sensors:</b> HTS221_HUM_HUM_mean</li>
                    <li><b>Pressure Sensors:</b> LPS22HH_PRESS_PRESS_mean</li>
                    <li><b>Accelerometer Sensors:</b> IIS2DH_ACC_A_x/y/z_mean, IIS3DWB_ACC_A_x/y/z_mean, ISM330DHCX_ACC_A_x/y/z_mean</li>
                    <li><b>Gyroscope Sensors:</b> ISM330DHCX_GYRO_G_x/y/z_mean</li>
                    <li><b>Magnetometer Sensors:</b> IIS2MDC_MAG_M_x/y/z_mean</li>
                    <li><b>Microphone Sensors:</b> IMP23ABSU_MIC_MIC_mean, IMP34DT05_MIC_MIC_mean</li>
                </ul>

                <h3>Smart Feature Filtering</h3>
                <ul>
                    <li><b>Exact Matching:</b> Only shows features for the requested sensor type</li>
                    <li><b>No Cross-Contamination:</b> Temperature requests won't show accelerometer data</li>
                    <li><b>Multi-Axis Support:</b> Accelerometer requests show X, Y, Z axes together</li>
                    <li><b>Statistical Measures:</b> All features use proper statistical calculations (mean, std, etc.)</li>
                </ul>
            </div>
            """
        )

        content_layout.addWidget(browser)
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)


