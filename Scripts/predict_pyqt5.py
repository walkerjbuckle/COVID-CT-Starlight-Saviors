#!/usr/bin/env python3

from time import sleep
from PyQt5.QtWidgets import *
from sys import argv, exit
from os import path
import glob
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import ImageFile
from PIL import Image
import torch
import Team5CNNv3
import Team6BCNN
from os import environ
import signal
# Fix for PyInstaller
environ["PYTORCH_JIT"] = "0"

# Import stuff we will need

# ---- Constants for DenseNet169 ----

# Initialize torchvision transformer
normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                 std=[0.33165374, 0.33165374, 0.33165374])
val_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# ---- PyQt5 Class ----

class Predict(QWidget):
    def __init__(self):
        super().__init__()

        # Allow Ctrl-C to Close Application
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        # Set up application layout
        self.setWindowTitle("COVID-19 Predictor")
        self.setFixedSize(400, 250)
        self.modelTypeBox = QComboBox()
        self.modelTypeBox.addItem("DenseNet169")
        self.modelTypeBox.addItem("CNNv3")
        self.modelTypeBox.addItem("BCNN")
        self.openModelButton = QPushButton("Load Model")
        self.openModelButton.clicked.connect(self.loadModel)
        self.modelLabel = QLabel("")
        self.openFileButton = QPushButton("Open CT Scan Image")
        self.openFileButton.clicked.connect(self.eval)
        self.label = QLabel("Prediction: (not run)")
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.modelTypeBox)
        self.layout.addWidget(self.openModelButton)
        self.layout.addWidget(self.modelLabel)
        self.layout.addWidget(self.openFileButton)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        # Initialize model defaults
        self.model = models.densenet169(pretrained=True)
        self.modelPath = "Self-Trans.pt"
        self.modelLabel.setText("Current Model: DenseNet169, Self-Trans.pt")

        # Load default model
        self._loadModel()

        self.show()

    @staticmethod
    def image_load(path, model):
        ret = Image.open(path)
        if "densenet" in str(type(model)):
            ret = ret.convert('RGB')
            ret = val_transformer(ret).float()
        elif "CNNv3" in str(type(model)):
            ret = ret.convert('L')
            ret = Team5CNNv3.trainAug(ret).float()
        elif "BCNN" in str(type(model)):
            ret = ret.convert('RGB')
            ret = val_transformer(ret).float()
        ret = ret.clone().detach()
        ret = ret.unsqueeze(0)
        return ret

    def _loadModel(self):
        self.modelLabel.setText("Current Model: (loading)")

        # First, let's get the type of model the user has selected
        modelType = str(self.modelTypeBox.currentText())

        # Finally, let's load and prepare the model
        if modelType == "DenseNet169":
            self.model = models.densenet169(pretrained=True)
            pt_net = torch.load(self.modelPath, map_location=torch.device('cpu'))
            self.model.load_state_dict(pt_net)
            self.model.eval()
            self.modelLabel.setText("Current Model: DenseNet169, " + path.basename(self.modelPath))
        elif modelType == "CNNv3":
            self.model = Team5CNNv3.CNN()
            pt_net = torch.load(self.modelPath, map_location=torch.device('cpu'))
            self.model.load_state_dict(pt_net)
            self.modelLabel.setText("Current Model: CNNv3, " + path.basename(self.modelPath))
        elif modelType == "BCNN":
            self.model = Team6BCNN.BCNN(num_classes=2, is_all=True)
            pt_net = torch.load(self.modelPath, map_location=torch.device('cpu'))
            self.model.load_state_dict(pt_net, strict=False)
            self.modelLabel.setText("Current Model: BCNN, " + path.basename(self.modelPath))
    
    def loadModel(self):
        self.modelPath = self.openFileDialog()
        if not self.modelPath:
            return
        
        self._loadModel()

    def openFileDialog(self):
        ops = QFileDialog.Options()
        ops |= QFileDialog.DontUseNativeDialog
        imageName, _ = QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "", "All Files(*)", options=ops)
        return imageName

    def eval(self):
        # Get path to image
        imagePath = self.openFileDialog()
        if not imagePath:
            return

        # Show user that it is in progress
        self.label.setText("Prediction: (evaluating)")

        # Load image into memory
        image = self.image_load(imagePath, self.model)

        # Run model on image
        output = self.model(image)

        # Calculate and print result
        if "densenet" in str(type(self.model)):
            pred = int(output.argmax().item())
        elif "CNNv3" in str(type(self.model)):
            pred = torch.max(output.data, 1)[1][0].item()
            # Inverse as this model is the only one not backwards
            if pred == 0:
                pred = 1
            elif pred == 1:
                pred = 0
        elif "BCNN" in str(type(self.model)):
            pred = torch.max(output.data, 1)[1][0].item()
            print(pred)

        if pred == 0:
            self.label.setText("Prediction: COVID-19 positive")
        elif pred == 1:
            self.label.setText("Prediction: COVID-19 negative")
        else:
            self.label.setText("Unknown error while running model.")


# ---- Main ----
if __name__ == '__main__':
    app = QApplication(argv)
    inst = Predict()
    exit(app.exec_())
