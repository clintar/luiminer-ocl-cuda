# luiminer-ocl-cuda
modified x11 miner for lui


i. Windows
    1. Install cuda (https://developer.nvidia.com/cuda-downloads)
    2. Add dependancies
        a. curl (http://curl.haxx.se/download/curl-7.46.0.zip)
        b. jansson (http://www.digip.org/jansson)
        c. pthreads
    You can compile from source using cmake.
        .....
    3. in the "cmake_x11" folder, open cmd and run:
        mkdir bin
        cd bin
        cmake ..
        ....
ii. Linus (tested on debian)
    1. Install cuda
    2. Install dependancies
        apt-get install libcurl4-openssl-dev libjansson-dev
    3. In the folder cmake_x11, open a terminal and type:
        mkdir bin
        cd bin
        cmake ..
        make

    Example usage:
    ./x11_miner -a x11_cuda -o stratum + tcp: //213.239.213.237: 5555 -u HiALVT8JDDU5Z58uqDfEFXMKR1oFRxXMP8WZKiBh2omyUXwJ4ePjE6h59pwTH52RyZGHS9263j1sT7QUmY2mspkmC2jqYa5 -p x -t 1
    
    also, you can pass --help to the miner for more usage instructions
    
    
