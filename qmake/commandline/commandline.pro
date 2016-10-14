#-------------------------------------------------
#
# Project created by QtCreator 2016-10-13T15:33:36
#
#-------------------------------------------------

QT       -= core gui

TARGET = commandline
TEMPLATE = lib
CONFIG -= qt
CONFIG += c++11

DEFINES += COMMANDLINE_LIBRARY

SOURCES +=

HEADERS += \
    ../../include/commandline/argumenthandlers.h \
    ../../include/commandline/commandlineoption.h \
    ../../include/commandline/commandlineparser.h

INCLUDEPATH += \
    ../../include/commandline

unix {
    target.path = /usr/lib
    INSTALLS += target
}
