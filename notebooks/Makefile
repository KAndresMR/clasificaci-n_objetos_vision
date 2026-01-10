CXX = g++
CXXFLAGS = -std=c++17 -O2 \
	-I/usr/local/include/opencv4

LDFLAGS = -L/usr/local/lib \
	-lopencv_core \
	-lopencv_imgproc \
	-lopencv_imgcodecs \
	-lopencv_highgui \
	-lopencv_video \
	-lopencv_videoio \
	-lopencv_cudaarithm \
	-lopencv_cudaimgproc \
	-lopencv_cudafilters \
	-lopencv_cudev

TARGET = parte1C
SRC = main.cpp

all:
	$(CXX) $(SRC) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET)

run:
	./$(TARGET)

clean:
	rm -f $(TARGET)
