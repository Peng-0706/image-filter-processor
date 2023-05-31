"""
Created on 2023年5月27日
@author:liubochen
@description:未添加功能的界面
"""
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import sys
from PyQt5 import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class Ui_MainWindow(QWidget):
    def setupUi(self, MainWindow):

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setMinimumSize(QtCore.QSize(1200, 800))
        self.centralwidget.setMaximumSize(QtCore.QSize(1200, 800))
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)

        self.setWindowFlags(Qt.CustomizeWindowHint)  # 去掉标题栏的代码

        self.frame.setStyleSheet("#frame{\n"
"    background-color: rgb(56, 57, 60);\n"
"    border-radius:20px;\n"
"}")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setContentsMargins(0, 0, 0, -1)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setStyleSheet("#frame_2{\n"
"    background-color: rgb(86, 88, 93);\n"
"    border-top-left-radius:20px;\n"
"    border-top-right-radius:20px;\n"
"}")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.frame_5 = QtWidgets.QFrame(self.frame_2)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.open_Button = QtWidgets.QPushButton(self.frame_5)
        self.open_Button.setGeometry(QtCore.QRect(50, 20, 61, 23))
        self.open_Button.setStyleSheet("#open_Button{\n"
"background-color: rgb(103, 103, 103);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.open_Button.setObjectName("open_Button")
        self.save_Button = QtWidgets.QPushButton(self.frame_5)
        self.save_Button.setGeometry(QtCore.QRect(150, 20, 75, 23))
        self.save_Button.setStyleSheet("#save_Button{\n"
"background-color: rgb(103, 103, 103);\n"
"border-radius:10px;\n"
"}")
        self.save_Button.setObjectName("save_Button")
        self.horizontalLayout_2.addWidget(self.frame_5)
        self.frame_4 = QtWidgets.QFrame(self.frame_2)
        self.frame_4.setMinimumSize(QtCore.QSize(50, 10))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_4)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.frame_6 = QtWidgets.QFrame(self.frame_4)
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.horizontalLayout_3.addWidget(self.frame_6)
        self.frame_7 = QtWidgets.QFrame(self.frame_4)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.x_Button = QtWidgets.QPushButton(self.frame_7)
        self.x_Button.setGeometry(QtCore.QRect(180, 10, 21, 21))
        self.x_Button.setStyleSheet("QPushButton{\n"
"background-color: rgb(103, 103, 103);\n"
"}\n"
"\n"
"")
        self.x_Button.setObjectName("x_Button")
        self.horizontalLayout_3.addWidget(self.frame_7)
        self.horizontalLayout_2.addWidget(self.frame_4)
        self.verticalLayout.addWidget(self.frame_2)
        self.frame_3 = QtWidgets.QFrame(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(10)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setStyleSheet("#frame{\n"
"background-color: rgb(56, 57, 60);\n"
"border-radius:20px\n"
"\n"
"}")
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.frame_8 = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(10)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_8.sizePolicy().hasHeightForWidth())
        self.frame_8.setSizePolicy(sizePolicy)
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.frame_8)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.show_label = QtWidgets.QLabel(self.frame_8)
        self.show_label.setStyleSheet("border-color: rgb(93, 94, 82);")
        self.show_label.setText("")
        self.show_label.setObjectName("show_label")
        self.horizontalLayout_7.addWidget(self.show_label)
        self.horizontalLayout_4.addWidget(self.frame_8)
        self.frame_9 = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(6)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_9.sizePolicy().hasHeightForWidth())
        self.frame_9.setSizePolicy(sizePolicy)
        self.frame_9.setStyleSheet("background-color: rgb(59, 59, 59);")
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_9)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame_10 = QtWidgets.QFrame(self.frame_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(6)
        sizePolicy.setHeightForWidth(self.frame_10.sizePolicy().hasHeightForWidth())
        self.frame_10.setSizePolicy(sizePolicy)
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_10)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame_12 = QtWidgets.QFrame(self.frame_10)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.frame_12.sizePolicy().hasHeightForWidth())
        self.frame_12.setSizePolicy(sizePolicy)
        self.frame_12.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame_12)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.frame_17 = QtWidgets.QFrame(self.frame_12)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_17.sizePolicy().hasHeightForWidth())
        self.frame_17.setSizePolicy(sizePolicy)
        self.frame_17.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_17.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_17.setObjectName("frame_17")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_17)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.ruidu_Button = QtWidgets.QPushButton(self.frame_17)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ruidu_Button.sizePolicy().hasHeightForWidth())
        self.ruidu_Button.setSizePolicy(sizePolicy)
        self.ruidu_Button.setMaximumSize(QtCore.QSize(80, 20))
        self.ruidu_Button.setStyleSheet("#ruidu_Button{\n"
"background-color: rgb(103, 103, 103);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.ruidu_Button.setObjectName("ruidu_Button")
        self.verticalLayout_5.addWidget(self.ruidu_Button)
        self.liangdu_Button = QtWidgets.QPushButton(self.frame_17)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.liangdu_Button.sizePolicy().hasHeightForWidth())
        self.liangdu_Button.setSizePolicy(sizePolicy)
        self.liangdu_Button.setMaximumSize(QtCore.QSize(80, 20))
        self.liangdu_Button.setStyleSheet("#liangdu_Button{\n"
"background-color: rgb(103, 103, 103);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.liangdu_Button.setObjectName("liangdu_Button")
        self.verticalLayout_5.addWidget(self.liangdu_Button)
        self.duibidu_Button = QtWidgets.QPushButton(self.frame_17)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.duibidu_Button.sizePolicy().hasHeightForWidth())
        self.duibidu_Button.setSizePolicy(sizePolicy)
        self.duibidu_Button.setMaximumSize(QtCore.QSize(80, 20))
        self.duibidu_Button.setStyleSheet("#duibidu_Button{\n"
"background-color: rgb(103, 103, 103);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.duibidu_Button.setObjectName("duibidu_Button")
        self.verticalLayout_5.addWidget(self.duibidu_Button)
        self.horizontalLayout_5.addWidget(self.frame_17)
        self.frame_18 = QtWidgets.QFrame(self.frame_12)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(10)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_18.sizePolicy().hasHeightForWidth())
        self.frame_18.setSizePolicy(sizePolicy)
        self.frame_18.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_18.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_18.setObjectName("frame_18")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.frame_18)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.ruidutiao = QtWidgets.QSlider(self.frame_18)
        self.ruidutiao.setStyleSheet("QSlider::handle:horizontal{ \n"
"        width:  50px; \n"
"        height: 50px; \n"
"        margin-top: -20px; \n"
"        margin-left: 0px; \n"
"        margin-bottom: -20px; \n"
"        margin-right: 0px; \n"
"         \n"
"} \n"
"")
        self.ruidutiao.setMaximum(10)
        self.ruidutiao.setPageStep(1)
        self.ruidutiao.setOrientation(QtCore.Qt.Horizontal)
        self.ruidutiao.setObjectName("ruidutiao")
        self.verticalLayout_7.addWidget(self.ruidutiao)
        self.liangdutiao = QtWidgets.QSlider(self.frame_18)
        self.liangdutiao.setMinimum(-10)
        self.liangdutiao.setMaximum(10)
        self.liangdutiao.setPageStep(1)
        self.liangdutiao.setOrientation(QtCore.Qt.Horizontal)
        self.liangdutiao.setObjectName("liangdutiao")
        self.verticalLayout_7.addWidget(self.liangdutiao)
        self.dubidutiao = QtWidgets.QSlider(self.frame_18)
        self.dubidutiao.setMaximum(300)
        self.dubidutiao.setSingleStep(10)
        self.dubidutiao.setProperty("value", 100)
        self.dubidutiao.setOrientation(QtCore.Qt.Horizontal)
        self.dubidutiao.setObjectName("dubidutiao")
        self.verticalLayout_7.addWidget(self.dubidutiao)
        self.horizontalLayout_5.addWidget(self.frame_18)
        self.verticalLayout_3.addWidget(self.frame_12)
        self.frame_13 = QtWidgets.QFrame(self.frame_10)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(6)
        sizePolicy.setHeightForWidth(self.frame_13.sizePolicy().hasHeightForWidth())
        self.frame_13.setSizePolicy(sizePolicy)
        self.frame_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_13.setObjectName("frame_13")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.frame_13)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.frame_19 = QtWidgets.QFrame(self.frame_13)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(8)
        sizePolicy.setHeightForWidth(self.frame_19.sizePolicy().hasHeightForWidth())
        self.frame_19.setSizePolicy(sizePolicy)
        self.frame_19.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_19.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_19.setObjectName("frame_19")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_19)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.lunkuo_Button = QtWidgets.QPushButton(self.frame_19)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lunkuo_Button.sizePolicy().hasHeightForWidth())
        self.lunkuo_Button.setSizePolicy(sizePolicy)
        self.lunkuo_Button.setMaximumSize(QtCore.QSize(80, 20))
        self.lunkuo_Button.setStyleSheet("#lunkuo_Button{\n"
"background-color: rgb(103, 103, 103);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.lunkuo_Button.setObjectName("lunkuo_Button")
        self.gridLayout_2.addWidget(self.lunkuo_Button, 0, 0, 1, 1)
        self.sumiao_Button = QtWidgets.QPushButton(self.frame_19)
        self.sumiao_Button.setMaximumSize(QtCore.QSize(80, 20))
        self.sumiao_Button.setStyleSheet("#sumiao_Button{\n"
"background-color: rgb(103, 103, 103);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.sumiao_Button.setObjectName("sumiao_Button")
        self.gridLayout_2.addWidget(self.sumiao_Button, 0, 1, 1, 1)
        self.miaobianhua_Button = QtWidgets.QPushButton(self.frame_19)
        self.miaobianhua_Button.setMaximumSize(QtCore.QSize(80, 20))
        self.miaobianhua_Button.setStyleSheet("#miaobianhua_Button{\n"
"background-color: rgb(103, 103, 103);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.miaobianhua_Button.setObjectName("miaobianhua_Button")
        self.gridLayout_2.addWidget(self.miaobianhua_Button, 0, 2, 1, 1)
        self.masaike_Button = QtWidgets.QPushButton(self.frame_19)
        self.masaike_Button.setMaximumSize(QtCore.QSize(80, 20))
        self.masaike_Button.setStyleSheet("#masaike_Button{\n"
"background-color: rgb(103, 103, 103);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.masaike_Button.setObjectName("masaike_Button")
        self.gridLayout_2.addWidget(self.masaike_Button, 1, 0, 1, 1)
        self.keli_Button = QtWidgets.QPushButton(self.frame_19)
        self.keli_Button.setMaximumSize(QtCore.QSize(80, 20))
        self.keli_Button.setStyleSheet("#keli_Button{\n"
"background-color: rgb(103, 103, 103);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.keli_Button.setObjectName("keli_Button")
        self.gridLayout_2.addWidget(self.keli_Button, 1, 1, 1, 1)
        self.verticalLayout_6.addWidget(self.frame_19)
        self.frame_20 = QtWidgets.QFrame(self.frame_13)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(3)
        sizePolicy.setHeightForWidth(self.frame_20.sizePolicy().hasHeightForWidth())
        self.frame_20.setSizePolicy(sizePolicy)
        self.frame_20.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_20.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_20.setObjectName("frame_20")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.frame_20)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setSpacing(0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.frame_21 = QtWidgets.QFrame(self.frame_20)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_21.sizePolicy().hasHeightForWidth())
        self.frame_21.setSizePolicy(sizePolicy)
        self.frame_21.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_21.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_21.setObjectName("frame_21")
        self.qiangdu_Button = QtWidgets.QPushButton(self.frame_21)
        self.qiangdu_Button.setGeometry(QtCore.QRect(10, 10, 80, 20))
        self.qiangdu_Button.setMaximumSize(QtCore.QSize(80, 20))
        self.qiangdu_Button.setStyleSheet("#qiangdu_Button{\n"
"background-color: rgb(103, 103, 103);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.qiangdu_Button.setObjectName("qiangdu_Button")
        self.horizontalLayout_8.addWidget(self.frame_21)
        self.frame_22 = QtWidgets.QFrame(self.frame_20)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(7)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_22.sizePolicy().hasHeightForWidth())
        self.frame_22.setSizePolicy(sizePolicy)
        self.frame_22.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_22.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_22.setObjectName("frame_22")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.frame_22)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalSlider_3 = QtWidgets.QSlider(self.frame_22)
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setObjectName("horizontalSlider_3")
        self.verticalLayout_8.addWidget(self.horizontalSlider_3)
        self.horizontalLayout_8.addWidget(self.frame_22)
        self.verticalLayout_6.addWidget(self.frame_20)
        self.verticalLayout_3.addWidget(self.frame_13)
        self.verticalLayout_2.addWidget(self.frame_10)
        self.frame_11 = QtWidgets.QFrame(self.frame_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.frame_11.sizePolicy().hasHeightForWidth())
        self.frame_11.setSizePolicy(sizePolicy)
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_11)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.frame_15 = QtWidgets.QFrame(self.frame_11)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(8)
        sizePolicy.setHeightForWidth(self.frame_15.sizePolicy().hasHeightForWidth())
        self.frame_15.setSizePolicy(sizePolicy)
        self.frame_15.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_15.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_15.setObjectName("frame_15")
        self.gridLayout = QtWidgets.QGridLayout(self.frame_15)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.shuicai_Button = QtWidgets.QPushButton(self.frame_15)
        self.shuicai_Button.setMaximumSize(QtCore.QSize(80, 20))
        self.shuicai_Button.setStyleSheet("#shuicai_Button{\n"
"background-color: rgb(103, 103, 103);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.shuicai_Button.setObjectName("shuicai_Button")
        self.gridLayout.addWidget(self.shuicai_Button, 0, 0, 1, 1)
        self.fudiao_Button = QtWidgets.QPushButton(self.frame_15)
        self.fudiao_Button.setMinimumSize(QtCore.QSize(40, 20))
        self.fudiao_Button.setMaximumSize(QtCore.QSize(80, 20))
        self.fudiao_Button.setStyleSheet("#fudiao_Button{\n"
"background-color: rgb(103, 103, 103);\n"
"border-radius:10px;\n"
"}\n"
"\n"
"\n"
"")
        self.fudiao_Button.setAutoRepeatInterval(100)
        self.fudiao_Button.setObjectName("fudiao_Button")
        self.gridLayout.addWidget(self.fudiao_Button, 1, 0, 1, 1)
        self.maoboli_Button_8 = QtWidgets.QPushButton(self.frame_15)
        self.maoboli_Button_8.setMaximumSize(QtCore.QSize(80, 20))
        self.maoboli_Button_8.setStyleSheet("#maoboli_Button_8{\n"
"background-color: rgb(103, 103, 103);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.maoboli_Button_8.setObjectName("maoboli_Button_8")
        self.gridLayout.addWidget(self.maoboli_Button_8, 2, 0, 1, 1)
        self.fugu_Button_7 = QtWidgets.QPushButton(self.frame_15)
        self.fugu_Button_7.setMaximumSize(QtCore.QSize(80, 20))
        self.fugu_Button_7.setStyleSheet("#fugu_Button_7{\n"
"background-color: rgb(103, 103, 103);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.fugu_Button_7.setObjectName("fugu_Button_7")
        self.gridLayout.addWidget(self.fugu_Button_7, 3, 0, 1, 1)
        self.dipian_pushButton_9 = QtWidgets.QPushButton(self.frame_15)
        self.dipian_pushButton_9.setMaximumSize(QtCore.QSize(80, 20))
        self.dipian_pushButton_9.setStyleSheet("#dipian_pushButton_9{\n"
"background-color: rgb(103, 103, 103);\n"
"border-radius:10px;\n"
"}\n"
"")
        self.dipian_pushButton_9.setObjectName("dipian_pushButton_9")
        self.gridLayout.addWidget(self.dipian_pushButton_9, 4, 0, 1, 1)
        self.verticalLayout_4.addWidget(self.frame_15)
        self.frame_14 = QtWidgets.QFrame(self.frame_11)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(self.frame_14.sizePolicy().hasHeightForWidth())
        self.frame_14.setSizePolicy(sizePolicy)
        self.frame_14.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_14.setObjectName("frame_14")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.frame_14)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.frame_16 = QtWidgets.QFrame(self.frame_14)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(4)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_16.sizePolicy().hasHeightForWidth())
        self.frame_16.setSizePolicy(sizePolicy)
        self.frame_16.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_16.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_16.setObjectName("frame_16")
        self.yes_pushButton_4 = QtWidgets.QPushButton(self.frame_16)
        self.yes_pushButton_4.setGeometry(QtCore.QRect(110, 0, 75, 21))
        self.yes_pushButton_4.setMaximumSize(QtCore.QSize(100, 40))
        self.yes_pushButton_4.setStyleSheet("#yes_pushButton_4{\n"
"background-color: rgb(103, 103, 103);\n"
"border-radius:4px;\n"
"}\n"
"")
        self.yes_pushButton_4.setObjectName("yes_pushButton_4")
        self.horizontalLayout_6.addWidget(self.frame_16)
        self.verticalLayout_4.addWidget(self.frame_14)
        self.verticalLayout_2.addWidget(self.frame_11)
        self.horizontalLayout_4.addWidget(self.frame_9)
        self.verticalLayout.addWidget(self.frame_3)
        self.horizontalLayout.addWidget(self.frame)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.x_Button.clicked.connect(MainWindow.close) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.open_Button.setText(_translate("MainWindow", "打开文件"))
        self.save_Button.setText(_translate("MainWindow", "保存文件"))
        self.x_Button.setText(_translate("MainWindow", "×"))
        self.ruidu_Button.setText(_translate("MainWindow", "锐度"))
        self.liangdu_Button.setText(_translate("MainWindow", "亮度"))
        self.duibidu_Button.setText(_translate("MainWindow", "对比度"))
        self.lunkuo_Button.setText(_translate("MainWindow", "轮廓图像"))
        self.sumiao_Button.setText(_translate("MainWindow", "素描图像"))
        self.miaobianhua_Button.setText(_translate("MainWindow", "描边化图像"))
        self.masaike_Button.setText(_translate("MainWindow", "马赛克图像"))
        self.keli_Button.setText(_translate("MainWindow", "颗粒图像"))
        self.qiangdu_Button.setText(_translate("MainWindow", "强度"))
        self.shuicai_Button.setText(_translate("MainWindow", "水彩滤镜"))
        self.fudiao_Button.setText(_translate("MainWindow", "浮雕滤镜"))
        self.maoboli_Button_8.setText(_translate("MainWindow", "毛玻璃滤镜"))
        self.fugu_Button_7.setText(_translate("MainWindow", "复古滤镜"))
        self.dipian_pushButton_9.setText(_translate("MainWindow", "底片滤镜"))
        self.yes_pushButton_4.setText(_translate("MainWindow", "确定"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
