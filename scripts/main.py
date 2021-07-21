import sys
import os
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QIntValidator, QImage
from PyQt5.QtCore import *
from scripts.dataset import create_data_loader
from scripts.models import AutoEncoderConv
import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
from torchvision import datasets
from torchvision import transforms
import PIL.Image as Image
from skimage.measure import compare_ssim as ssim
import time
import matplotlib.pyplot as plt


class Trainer(QThread):
    change_value = pyqtSignal(int)

    def __init__(self):
        super(Trainer, self).__init__()
        self.train_path = ''
        self.num_epoch = 0
        self.batch_size = 0
        self.learning_rate = 0.0

    def set_parameter(self, train_path, num_epoch, batch_size, learning_rate):
        self.train_path = train_path
        self.num_epoch = int(num_epoch)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)

    def run(self):
        print('train run')
        transform = transforms.Compose([
            transforms.Resize(size=(512, 512)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(self.train_path, transform=transform)
        dataloader = create_data_loader(dataset, _batch_size=self.batch_size, _shuffle=True)
        myWindow.write_log('<-- set data')

        model = AutoEncoderConv()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if (device == 'cuda') and (torch.cuda.device_count() > 1):
            model = nn.DataParallel(model)

        model.to(device)
        print('set model (device:{})'.format(device))
        myWindow.write_log('<--- set model (device:{})'.format(device))

        loss_func = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        model.train()
        myWindow.write_log('<---- start train')

        for epoch in range(self.num_epoch):
            loss_arr = []
            percent = (epoch + 1) / self.num_epoch * 100.0
            self.change_value.emit(percent)
            for batch, data in enumerate(dataloader):
                image, _ = data
                image = image.to(device)

                output = model(image)
                loss = loss_func(output, image)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_arr.append(loss.item())
                print('epoch : {}/{} | batch : {}/{} | loss : {:.4f}'.format(
                    epoch + 1, self.num_epoch, batch + 1, len(dataloader), np.mean(loss_arr)))
                myWindow.write_log('epoch : {}/{} | batch : {}/{} | loss : {:.4f}'.format(
                    epoch + 1, self.num_epoch, batch + 1, len(dataloader), np.mean(loss_arr)))

        torch.save(model, '../train_model.pth')
        myWindow.write_log('end train')
        return


# connect ui file
form_class = uic.loadUiType("../my_ui.ui")[0]


def get_file_paths(_path):
    lst_file_paths = []
    for (root, dirs, files) in os.walk(_path):
        for file_name in files:
            lst_file_paths.append(root + '/' + file_name)

    return lst_file_paths


class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super(WindowClass, self).__init__()
        self.setupUi(self)

        self.btn_load.clicked.connect(self.load_data)
        self.btn_train.clicked.connect(self.train_start)
        self.btn_load_image.clicked.connect(self.load_test_image)
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_predict.clicked.connect(self.predict)

        self.table_load.itemSelectionChanged.connect(self.change_select)

        self.train_path = ''
        self.load_image_path = ''

        self.edit_epoch.setValidator(QIntValidator())
        self.edit_batch_size.setValidator(QIntValidator())
        self.edit_learning_rate.setValidator(QIntValidator())
        self.edit_epoch.setText('30')
        self.edit_batch_size.setText('8')
        self.edit_learning_rate.setText('0.001')
        self.edit_threshold.setText('100')

        self.progressBar.setValue(0)

        self.trainer = Trainer()
        self.trainer.change_value.connect(self.progressBar.setValue)

        self.model = AutoEncoderConv()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        file_path = QFileDialog.getOpenFileName(self, caption='select model', directory='../')
        file_path = file_path[0]
        if file_path == '':
            return

        self.model = torch.load(file_path)
        self.model.to(self.device)
        self.model.eval()

        self.write_log('model load complete')
        self.edit_load_model.setText(file_path)

    def load_test_image(self):
        file_path = QFileDialog.getOpenFileName(self, caption='test image', directory='../dataset/')
        self.load_image_path = file_path[0]
        if file_path[0] != '':
            self.show_image_from_path(self.label_test_view, file_path[0])
        self.edit_load_image.setText(file_path[0])

    def load_data(self):
        folder_path = QFileDialog.getExistingDirectory(self, caption='load_data', directory='../')
        self.train_path = folder_path
        self.edit_load.setText(folder_path)
        file_paths = get_file_paths(folder_path)

        # set table widget
        self.table_load.setRowCount(file_paths.__len__())
        for index, file_path in enumerate(file_paths):
            self.table_load.setItem(index, 0, QTableWidgetItem(file_path))

        self.table_load.resizeColumnsToContents()

    def train_start(self):
        train_path = self.train_path
        epoch_num = self.edit_epoch.text()
        batch_size = self.edit_batch_size.text()
        learning_rate = self.edit_learning_rate.text()

        self.trainer.set_parameter(train_path, epoch_num, batch_size, learning_rate)
        self.trainer.start()

    def change_select(self):
        image_path = self.table_load.currentItem().text()
        self.show_image_from_path(viewer=self.label_view, image_path=image_path)

    def show_image_from_path(self, viewer, image_path):
        pixmap = QPixmap(image_path)
        self.show_pixmap(viewer, pixmap)

    def show_pixmap(self, viewer, pixmap):
        pixmap = pixmap.scaled(viewer.width(), viewer.height(), Qt.KeepAspectRatio)
        viewer.setPixmap(pixmap)

    def write_log(self, str_log):
        self.list_log.insertItem(0, str_log)

    def predict(self):
        start_time = time.time()

        if self.load_image_path == '':
            return

        active_ssim = self.checkBox_ssim.isChecked()
        threshold_val = int(self.edit_threshold.text())

        with torch.no_grad():
            pil_image = Image.open(self.load_image_path)

            org_width = pil_image.width
            org_height = pil_image.height

            transform = transforms.Compose([
                transforms.Resize(size=(512, 512)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])

            data = transform(pil_image)
            data = torch.unsqueeze(data, 0)
            data.to(self.device)

            output = self.model(data)

            image = data.cpu()
            output = output.cpu()

            image = torch.squeeze(image[0])
            output = torch.squeeze(output[0])

            eval_time = time.time() - start_time

            # cal diff
            if active_ssim:
                (score, diff) = ssim(image.numpy(), output.numpy(), win_size=13, full=True)   # data 에 따라서 단순 diff or ssim
            else:
                diff = cv.absdiff(image.numpy(), output.numpy())

            diff = (diff * 255).astype('uint8')
            if active_ssim:
                _, mask = cv.threshold(diff, threshold_val, 255, cv.THRESH_BINARY_INV)
            else:
                _, mask = cv.threshold(diff, threshold_val, 255, cv.THRESH_BINARY)
            kernel = np.ones((3, 3))
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

            _image = image.numpy()
            _image = cv.cvtColor(_image, cv.COLOR_GRAY2RGB)
            index = np.where(mask > 0)
            _mask = np.zeros(_image.shape, dtype=np.float32)
            _mask[index] = [1.0, 0, 0]
            _image = cv.addWeighted(_image, 1.0, _mask, 0.5, 0)

            _image = cv.resize(_image, (org_width, org_height))
            diff_count = np.count_nonzero(_mask)
            if diff_count < 4500:
                cv.putText(_image, "OK", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            else:
                cv.putText(_image, "NG", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            # to tensor
            mask = torch.from_numpy(mask)
            diff = torch.from_numpy(diff)
            result_image = torch.from_numpy(_image)
            result_image = np.transpose(result_image, (2, 0, 1))

            # to PIL Image
            image = transforms.ToPILImage()(image)
            out_image = transforms.ToPILImage()(output)
            mask = transforms.ToPILImage()(mask)
            diff = transforms.ToPILImage()(diff)
            result_image = transforms.ToPILImage()(result_image)

            # plt.subplot(321)
            # plt.imshow(image, cmap='gray')
            # plt.subplot(322)
            # plt.imshow(out_image, cmap='gray')
            # plt.subplot(323)
            # plt.imshow(result_image)
            # plt.subplot(324)
            # plt.imshow(mask, cmap='gray')
            # plt.subplot(325)
            # plt.imshow(diff, cmap='jet')
            # # plt.subplot(326)
            # # plt.imshow(test_mask, cmap='gray')
            # plt.show()

            qim = result_image.tobytes("raw", "RGB")
            qim = QImage(qim, result_image.size[0], result_image.size[1], QImage.Format_RGB888)
            pix = QPixmap.fromImage(qim)
            self.show_pixmap(self.label_test_view, pix)
            self.write_log('eval time : {0:0.4f}    diff_count : {1}'.format(eval_time, diff_count))


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # create window instance
    myWindow = WindowClass()

    # show window
    myWindow.show()

    # event loop
    app.exec_()
