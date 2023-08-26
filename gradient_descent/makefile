all:  evolu2.o ma.o
	clang++ -o evolu2.exe evolu2.o ma.o

evolu2.o: evolu2.cpp ma.o
	clang++ -c evolu2.cpp

ma.o: ma.cpp
	clang++ -c ma.cpp

clean: 
	del ma.o evolu2.o evolu2.exe
