# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_test.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1080, 680)
        self.label_to_show = QtWidgets.QLabel(Form)
        self.label_to_show.setGeometry(QtCore.QRect(510, 40, 431, 311))
        self.label_to_show.setText("")
        self.label_to_show.setObjectName("label_to_show")
        self.layoutWidget = QtWidgets.QWidget(Form)
        self.layoutWidget.setGeometry(QtCore.QRect(100, 170, 95, 241))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.button_to_show_cam = QtWidgets.QPushButton(self.layoutWidget)
        self.button_to_show_cam.setObjectName("button_to_show_cam")
        self.verticalLayout.addWidget(self.button_to_show_cam)
        self.button_to_recognition = QtWidgets.QPushButton(self.layoutWidget)
        self.button_to_recognition.setObjectName("button_to_recognition")
        self.verticalLayout.addWidget(self.button_to_recognition)
        self.layoutWidget1 = QtWidgets.QWidget(Form)
        self.layoutWidget1.setGeometry(QtCore.QRect(40, 540, 211, 41))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.text_result = QtWidgets.QTextEdit(self.layoutWidget1)
        self.text_result.setObjectName("text_result")
        self.horizontalLayout.addWidget(self.text_result)
        self.label_to_show_2 = QtWidgets.QLabel(Form)
        self.label_to_show_2.setGeometry(QtCore.QRect(510, 360, 431, 231))
        self.label_to_show_2.setText("")
        self.label_to_show_2.setObjectName("label_to_show_2")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.button_to_show_cam.setText(_translate("Form", "??????"))
        self.button_to_recognition.setText(_translate("Form", "??????"))
        self.label_2.setText(_translate("Form", "??????????????????"))
