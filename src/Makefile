#
# Makefile for blobs library for OpenCV Version 4 and its examples
#

CFLAGS= `pkg-config --cflags opencv gtk+-2.0` -I. 
LDFLAGS= `pkg-config --libs opencv gtk+-2.0` -L. -lcvblob 
ARCHFLAGS= -arch i386
CXX=g++

CPPFILES= \
	cvaux.cpp\
	cvblob.cpp\
	cvcolor.cpp\
	cvcontour.cpp\
	cvlabel.cpp\
	cvtrack.cpp

.SUFFIXES: .cpp.o
.cpp.o:	; echo 'Compiling $*.cpp' ; $(CXX) $(CFLAGS) $(ARCHFLAGS) -c $*.cpp

.SILENT:

libcvblob.a: $(CPPFILES:.cpp=.o)
	ar ru libcvblob.a $(CPPFILES:.cpp=.o) 2> /dev/null
	ranlib libcvblob.a
#
	@echo Copy include files...
#
	@echo Cleaning objects...
	rm -f $(CPPFILES:.cpp=.o)

all: clean

clean:
	@echo Cleaning...
	rm -f $(CFILES:.cpp=.o)
	rm -f libcvblob.a

