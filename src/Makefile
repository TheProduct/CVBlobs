#
# Makefile for blobs library for OpenCV Version 4 and its examples
#

CFLAGS= `pkg-config --cflags opencv gtk+-2.0` -I. 
LDFLAGS= `pkg-config --libs opencv gtk+-2.0` -L. -lblob 
ARCHFLAGS= -arch i386
CXX=g++

CPPFILES= \
	Blob.cpp\
	BlobResult.cpp \
	BlobExtraction.cpp

.SUFFIXES: .cpp.o
.cpp.o:	; echo 'Compiling $*.cpp' ; $(CXX) $(CFLAGS) $(ARCHFLAGS) -c $*.cpp

.SILENT:

libblob.a: $(CPPFILES:.cpp=.o)
	ar ru libblob.a $(CPPFILES:.cpp=.o) 2> /dev/null
	ranlib libblob.a
#
#	@echo Build tests, examples and tools...
#	$(CXX) -g blobdemo.cpp $(LDFLAGS) $(CFLAGS) -o blobdemo
#	$(CXX) -g blobdemo2.cpp $(LDFLAGS) $(CFLAGS) -o blobdemo2
#
	@echo Copy include files...
#
	@echo Cleaning objects...
	rm -f $(CPPFILES:.cpp=.o)

all: clean

clean:
	@echo Cleaning...
	rm -f $(CFILES:.cpp=.o)
	rm -f libblob.a

