# Hight-Frequency-Trading-C++-Code
#Code made by : SimonZxZ
Compile:
    g++ -O3 -march=native -mtune=native -std=c++20 \
        -fno-exceptions -fno-rtti \
        -falign-functions=64 -falign-loops=64 \
        -funroll-loops -fprefetch-loop-arrays \
        -lrt -lpthread -o hft_engine hft_engine.cpp
