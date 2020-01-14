g++ -std=c++11 -g -O2 \
-I/home/jesse/MIT/MEng/iRRAM/installed/include \
-Xlinker -rpath -Xlinker /home/jesse/MIT/MEng/iRRAM/installed/lib \
-I/home/jesse/MIT/MEng/irram.sh/packages/irram-random \
 plot.cc \
-L/home/jesse/MIT/MEng/iRRAM/installed/lib \
 -liRRAM -lmpfr -lgmp -lm -lpthread -o test
