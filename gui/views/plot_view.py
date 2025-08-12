from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import QWidget, QVBoxLayout

class PlotView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.canvas = None

    def show_plot(self, fig):
        if self.canvas:
            self.layout.removeWidget(self.canvas)
            self.canvas.deleteLater()

        self.canvas = FigureCanvas(fig)
        self.layout.addWidget(self.canvas)
        self.canvas.draw()

    def set_dataframe(self, df):
        """
        df is a pandas.DataFrame.
        Here you can plot things like time series or histograms.
        """
        # This method can be used for other plotting needs
        pass
