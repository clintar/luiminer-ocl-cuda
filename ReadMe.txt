x11 miner (x32)

���������:

i. Windows
	1. ���������� cuda (https://developer.nvidia.com/cuda-downloads)
	2. ��������� �����������:
		�) curl (http://curl.haxx.se/download/curl-7.46.0.zip)
		�) jansson (http://www.digip.org/jansson)
		�) pthreads
	����� ������� ������ �� ���������� ��� ������ cmake, � ����� ������������ ��� ��������� ���������� �� "/cmake_x11/for_win/".
	��� �����, ��������, ��� VS2012, ���������� "include" ���������� ����������� � "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\include" ,
	� ���������� "lib" - � "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\lib"
	3. � ����� "cmake_x11" ��������� cmd, � ���:
		mkdir bin
		cd bin
		cmake ..
	� ����� "bin" ��������� ������, ������� F7, ����

ii. Linux (tested at debian)
	1. ���������� cuda (� ���� ����� ���������� 100500 ��������, ��� ������ �� ����, ���, ����������� ���������� ����� � �������� ������)
	2. ��������� �����������:
		apt-get install libcurl4-openssl-dev libjansson-dev
	3. � ����� "cmake_x11" ��������� ��������, � ���:
		mkdir bin
		cd ./bin
		cmake ..
	� ����� "bin" � ��� ����� ��������, � ���� ����� ������� ������� make, ����

�������������:

� ������� ������ "x11 --help", �� ����������. ������ ������� ������� �������� � ����:
./x11 -a x11_cuda -o stratum+tcp://213.239.213.237:5555 -u HiALVT8JDDU5Z58uqDfEFXMKR1oFRxXMP8WZKiBh2omyUXwJ4ePjE6h59pwTH52RyZGHS9263j1sT7QUmY2mspkmC2jqYa5 -p x -t 1

��� ������������� opencl, ��� *.cl-����� ������ ������ � ����� ����� � ����������� �������� �������