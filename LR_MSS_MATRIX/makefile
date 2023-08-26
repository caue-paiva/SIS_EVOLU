CC = clang++
CFLAGS = -I c:/c_cpp_libs/eigen-3.4.0

all: AG_LR_INTE

AG_LR_INTE.o: AG_LR_INTE.cpp MA_LR.cpp
	$(CC) $(CFLAGS) -c AG_LR_INTE.cpp  MA_LR.cpp

AG_LR_INTE: AG_LR_INTE.o  MA_LR.o
	$(CC) -o AG_LR_INTE.exe AG_LR_INTE.o MA_LR.o

MA_LR.o: MA_LR.cpp
	$(CC) $(CFLAGS) -c MA_LR.cpp

clean:
	del -f *.o AG_LR_INTE.exe
