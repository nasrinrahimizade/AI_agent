from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextBrowser
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt


class HelpView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        title = QLabel("Help & Instructions")
        title_font = QFont()
        title_font.setPointSize(17)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setReadOnly(True)
        # Slightly larger body font for readability
        body_font = QFont()
        body_font.setPointSize(12)
        browser.setFont(body_font)

        # Provide structured guidance and the exact supported commands
        browser.setHtml(
            """
            <h3>Getting Started</h3>
            <ol>
              <li>Open a dataset via <b>File → Open Dataset</b> (CSV).</li>
              <li>Use <b>Chat</b> to request analyses or plots; results appear in <b>Plots</b>.</li>
            </ol>

            <h3>Basic Plot Requests</h3>
            <ul>
              <li>"Show me a boxplot"</li>
              <li>"Create a histogram"</li>
              <li>"Display correlation matrix"</li>
              <li>"Generate time series analysis"</li>
              <li>"Show frequency domain plot"</li>
              <li>"Plot scatter relationships"</li>
            </ul>

            <h3>Sensor-Specific Requests</h3>
            <ul>
              <li>"Show me accelerometer data"</li>
              <li>"Analyze temperature sensors"</li>
              <li>"Compare pressure readings"</li>
              <li>"Display humidity distribution"</li>
            </ul>

            <h3>Advanced Requests</h3>
            <ul>
              <li>"Show me the most discriminative features"</li>
              <li>"Create a comparison between OK and KO conditions"</li>
              <li>"Generate a feature relationship matrix"</li>
            </ul>

            <h3>Tips</h3>
            <ul>
              <li>Use the exact phrases above to trigger plot generation.</li>
              <li>Case and punctuation are flexible; the wording must match.</li>
              <li>Switch between <b>Chat</b>, <b>Plots</b>, and <b>Help</b> using the top toolbar.</li>
            </ul>
            """
        )

        layout.addWidget(browser)


