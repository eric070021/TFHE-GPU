all:
	@echo "OpenFHE has converted to CMake"
	@echo "Try this:"
	@echo mkdir build
	@echo cd build
	@echo cmake ..
	@echo make
	@echo make install \(to install in 'installed'\)

create_debug:
	mkdir -p ./build_debug && \
	cd ./build_debug && \
	cmake .. -DCMAKE_BUILD_TYPE=Debug

clean_debug:
	rm -rf ./build_debug/*

build_debug:
	cd ./build_debug && \
	make -j16

cache_clean_debug:
	cd ./build_debug && \
	make clean

.PHONY: create_debug clean_debug build_debug cache_clean_debug

create_release:
	mkdir -p ./build_release && \
	cd ./build_release && \
	cmake ..

clean_release:
	rm -rf ./build_release/*

build_release:
	cd ./build_release && \
	make -j16

cache_clean_release:
	cd ./build_release && \
	make clean

.PHONY: create_release clean_release build_release cache_clean_release