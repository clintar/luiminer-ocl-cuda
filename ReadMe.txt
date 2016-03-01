x11 miner (x32)

Установка:

i. Windows
	1. Установить cuda (https://developer.nvidia.com/cuda-downloads)
	2. Разрешить зависимости:
		а) curl (http://curl.haxx.se/download/curl-7.46.0.zip)
		б) jansson (http://www.digip.org/jansson)
		в) pthreads
	Можно собрать самому из исходников при помощи cmake, а можно использовать уже собранные библиотеки из "/cmake_x11/for_win/".
	Для этого, например, для VS2012, содержимое "include" необходимо скопировать в "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\include" ,
	а содержимое "lib" - в "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\lib"
	3. в папке "cmake_x11" открываем cmd, в ней:
		mkdir bin
		cd bin
		cmake ..
	В папке "bin" открываем солюшн, жмакаем F7, ждем

ii. Linux (tested at debian)
	1. Установить cuda (в этих ваших ентернетах 100500 мануалов, как именно на твой, бро, дистрибутив установить дрова и кудошный тулкит)
	2. Разрешить зависимости:
		apt-get install libcurl4-openssl-dev libjansson-dev
	3. в папке "cmake_x11" открываем терминал, в ней:
		mkdir bin
		cd ./bin
		cmake ..
	В папке "bin" у нас лежит мейкфайл, в этой папке вбиваем команду make, ждем

Использование:

В консоли вбивай "x11 --help", всё стандартно. Пример команды запуска майнинга в пуле:
./x11 -a x11_cuda -o stratum+tcp://213.239.213.237:5555 -u HiALVT8JDDU5Z58uqDfEFXMKR1oFRxXMP8WZKiBh2omyUXwJ4ePjE6h59pwTH52RyZGHS9263j1sT7QUmY2mspkmC2jqYa5 -p x -t 1

Для использования opencl, все *.cl-файлы должны лежать в одной папке с исполняемым файликом майнера