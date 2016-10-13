#-------------------------------------------------
#
# Project created by QtCreator 2016-10-01T13:09:35
#
#-------------------------------------------------

QT       -= core gui

TARGET = argon2-opencl
TEMPLATE = lib
CONFIG -= qt
CONFIG += c++11

DEFINES += ARGON2OPENCL_LIBRARY

LIBS += -lOpenCL

SOURCES += \
    ../../lib/argon2-opencl/globalcontext.cpp \
    ../../lib/argon2-opencl/programcontext.cpp \
    ../../lib/argon2-opencl/processingunit.cpp \
    ../../lib/argon2-opencl/device.cpp \
    ../../lib/argon2-opencl/kernelloader.cpp \
    ../../lib/argon2-opencl/argon2params.cpp \
    ../../lib/argon2-opencl/blake2b.cpp

HEADERS += \
    ../../include/argon2-opencl/cl.hpp \
    ../../include/argon2-opencl/opencl.h \
    ../../include/argon2-opencl/device.h \
    ../../include/argon2-opencl/programcontext.h \
    ../../include/argon2-opencl/globalcontext.h \
    ../../include/argon2-opencl/processingunit.h \
    ../../include/argon2-opencl/argon2-common.h\
    ../../lib/argon2-opencl/kernelloader.h \
    ../../include/argon2-opencl/argon2params.h \
    ../../lib/argon2-opencl/blake2b.h

OTHER_FILES += \
    ../../data/kernels/argon2_kernel.cl

INCLUDEPATH += \
    ../../include/argon2-opencl \
    ../../lib/argon2-opencl

unix {
    target.path = /usr/lib
    INSTALLS += target
}
