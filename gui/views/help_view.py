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

                <h2>📊 Statistical Analysis Requests</h2>
                <h3>Basic Statistics</h3>
                <ul>
                    <li><b>"What is the mean temperature?"</b> - Get average temperature across all samples</li>
                    <li><b>"Show me the standard deviation of humidity"</b> - Get variability measure</li>
                    <li><b>"What is the median pressure?"</b> - Get middle value of pressure data</li>
                    <li><b>"Count the total samples"</b> - Get total number of data points</li>
                    <li><b>"What are the min and max values for acceleration?"</b> - Get range of values</li>
                </ul>

                <h3>Grouped Analysis</h3>
                <ul>
                    <li><b>"Show me temperature statistics grouped by class"</b> - Compare across OK/KO groups</li>
                    <li><b>"What is the mean humidity for each class?"</b> - Class-specific averages</li>
                    <li><b>"Compare pressure readings between classes"</b> - Multi-class comparison</li>
                </ul>

                <h2>🔍 Feature Analysis</h2>
                <h3>Feature Importance</h3>
                <ul>
                    <li><b>"What are the most discriminative features?"</b> - Find best classification features</li>
                    <li><b>"Show me the top 5 important features"</b> - Ranked feature importance</li>
                    <li><b>"Which features separate OK from KO samples best?"</b> - Feature discrimination</li>
                </ul>

                <h3>Detailed Feature Analysis</h3>
                <ul>
                    <li><b>"Analyze the temperature sensor patterns"</b> - Comprehensive temperature analysis</li>
                    <li><b>"Show me humidity sensor distribution"</b> - Statistical distribution analysis</li>
                    <li><b>"Compare acceleration sensors between classes"</b> - Multi-axis comparison</li>
                </ul>

                <h2>📈 Visualization Requests</h2>
                <h3>Statistical Plots</h3>
                <ul>
                    <li><b>"Create a boxplot of temperature by class"</b> - Class comparison visualization</li>
                    <li><b>"Show me a histogram of humidity data"</b> - Distribution analysis</li>
                    <li><b>"Generate a scatter plot of temperature vs humidity"</b> - Feature relationships</li>
                    <li><b>"Display correlation matrix between sensors"</b> - Feature associations</li>
                </ul>

                <h3>Advanced Visualizations</h3>
                <ul>
                    <li><b>"Show me time series analysis of pressure data"</b> - Temporal patterns</li>
                    <li><b>"Generate frequency spectrum of vibration data"</b> - Signal analysis</li>
                    <li><b>"Create a heatmap of feature correlations"</b> - Correlation visualization</li>
                    <li><b>"Plot feature importance ranking"</b> - Feature selection visualization</li>
                </ul>

                <h2>🔬 Sensor-Specific Analysis</h2>
                <h3>Temperature & Environmental</h3>
                <ul>
                    <li><b>"Analyze temperature sensor patterns"</b> - Temperature analysis</li>
                    <li><b>"Show humidity sensor distribution"</b> - Humidity analysis</li>
                    <li><b>"Compare pressure readings between classes"</b> - Pressure analysis</li>
                </ul>

                <h3>Motion & Vibration</h3>
                <ul>
                    <li><b>"Show me accelerometer sensor patterns"</b> - Acceleration analysis</li>
                    <li><b>"Analyze gyroscope data"</b> - Rotation analysis</li>
                    <li><b>"Display magnetometer readings"</b> - Magnetic field analysis</li>
                    <li><b>"Show frequency spectrum of vibration data"</b> - Vibration analysis</li>
                </ul>

                <h3>Audio & Signal</h3>
                <ul>
                    <li><b>"Analyze microphone frequency data"</b> - Audio signal analysis</b></li>
                    <li><b>"Show me microphone signal patterns"</b> - Audio pattern analysis</li>
                </ul>

                <h2>📋 Class Comparison & Classification</h2>
                <h3>Multi-Class Analysis</h3>
                <ul>
                    <li><b>"Compare OK vs KO samples statistically"</b> - Binary classification analysis</li>
                    <li><b>"Show me differences between all classes"</b> - Multi-class comparison</li>
                    <li><b>"Analyze KO_HIGH_2mm vs KO_LOW_2mm"</b> - Subclass comparison</li>
                    <li><b>"Which features best separate the classes?"</b> - Classification feature selection</li>
                </ul>

                <h3>Statistical Significance</h3>
                <ul>
                    <li><b>"Are the differences between classes statistically significant?"</b> - Hypothesis testing</li>
                    <li><b>"Show me confidence intervals for the comparisons"</b> - Statistical confidence</li>
                    <li><b>"What is the effect size between OK and KO?"</b> - Practical significance</li>
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
                        <li><b>Request Visualizations:</b> "Create a boxplot" or "Show me a histogram"</li>
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
            </div>
            """
        )

        content_layout.addWidget(browser)
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)


