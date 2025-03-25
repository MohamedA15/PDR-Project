import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QLabel, QComboBox, QTextEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import random
import threading

class IndoorNavGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Indoor Navigation System")
        self.setGeometry(100, 100, 800, 600)
        
        self.initUI()
    
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        
        # Route selection dropdown
        self.route_selector = QComboBox()
        self.route_selector.addItems(["Route 1", "Route 2", "Route 3"])
        layout.addWidget(QLabel("Select Route:"))
        layout.addWidget(self.route_selector)
        
        # Start button
        self.start_button = QPushButton("Run Navigation")
        self.start_button.clicked.connect(self.run_navigation)
        layout.addWidget(self.start_button)
        
        # Matplotlib canvas for real-time graph updates
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Console output
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        layout.addWidget(QLabel("Console Output:"))
        layout.addWidget(self.console)
        
        central_widget.setLayout(layout)
    
    def run_navigation(self):
        self.console.append("Starting navigation for: " + self.route_selector.currentText())
        threading.Thread(target=self.simulate_data, daemon=True).start()
    
    def simulate_data(self):
        x_data, y_data = [], []
        for i in range(20):  # Simulating 20 steps
            x_data.append(i)
            y_data.append(random.uniform(0, 10))  # Random values for now
            
            self.ax.clear()
            self.ax.plot(x_data, y_data, marker='o', linestyle='-')
            self.canvas.draw()
            
            self.console.append(f"Step {i+1}: Position ({x_data[-1]}, {y_data[-1]:.2f})")
            QApplication.processEvents()
            
        self.console.append("Navigation Complete!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IndoorNavGUI()
    window.show()
    sys.exit(app.exec_())