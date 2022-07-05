import sys
from PyQt5.QtWidgets import QWidget,QApplication,QMessageBox
from PyQt5.QtCore import QTimer
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
import ui_test
import cv2


class InitForm(QWidget):
    def __init__(self):
        super().__init__()
        self.timer_camera = QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.cap = cv2.VideoCapture()  # 视频流
        self.CAM_NUM = 0  # 为0时表示视频流来自笔记本内置摄像头

        self.ui = ui_test.Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle("测试")
        self.init_slots()

    def init_slots(self):
        self.ui.button_to_show_cam.clicked.connect(self.on_btn_show_frame)
        self.timer_camera.timeout.connect(self.show_camera)  # 若定时器结束，则调用show_camera()

        self.ui.button_to_recognition.clicked.connect(self.on_btn_to_recognition)

    def on_btn_show_frame(self):
        print('open camera')
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM,cv2.CAP_DSHOW)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if flag == False:  # flag表示open()成不成功
                msg = QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QMessageBox.Ok)
            else:
                self.timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.ui.label_to_show.clear()  # 清空视频显示区域

    def show_camera(self):
        flag, self.image = self.cap.read()  # 从视频流中读取

        show = cv2.resize(self.image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.ui.label_to_show.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage


    def on_btn_to_recognition(self):
        self.ui.text_result.setText("停止")
        pixmap = QPixmap('img.png')
        self.ui.label_to_show_2.setPixmap(pixmap)
        self.ui.label_to_show_2.setScaledContents(True)
        print('hello world')



    def closeEvent(self, event):
        print("窗体关闭")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    w1 = InitForm()
    w1.show()
    sys.exit(app.exec_())
