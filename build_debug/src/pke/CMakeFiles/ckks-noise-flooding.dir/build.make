# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/eric070021/openfhe-developmentv1.0.4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/eric070021/openfhe-developmentv1.0.4/build_debug

# Include any dependencies generated for this target.
include src/pke/CMakeFiles/ckks-noise-flooding.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/pke/CMakeFiles/ckks-noise-flooding.dir/compiler_depend.make

# Include the progress variables for this target.
include src/pke/CMakeFiles/ckks-noise-flooding.dir/progress.make

# Include the compile flags for this target's objects.
include src/pke/CMakeFiles/ckks-noise-flooding.dir/flags.make

src/pke/CMakeFiles/ckks-noise-flooding.dir/examples/ckks-noise-flooding.cpp.o: src/pke/CMakeFiles/ckks-noise-flooding.dir/flags.make
src/pke/CMakeFiles/ckks-noise-flooding.dir/examples/ckks-noise-flooding.cpp.o: ../src/pke/examples/ckks-noise-flooding.cpp
src/pke/CMakeFiles/ckks-noise-flooding.dir/examples/ckks-noise-flooding.cpp.o: src/pke/CMakeFiles/ckks-noise-flooding.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/eric070021/openfhe-developmentv1.0.4/build_debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/pke/CMakeFiles/ckks-noise-flooding.dir/examples/ckks-noise-flooding.cpp.o"
	cd /home/eric070021/openfhe-developmentv1.0.4/build_debug/src/pke && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/pke/CMakeFiles/ckks-noise-flooding.dir/examples/ckks-noise-flooding.cpp.o -MF CMakeFiles/ckks-noise-flooding.dir/examples/ckks-noise-flooding.cpp.o.d -o CMakeFiles/ckks-noise-flooding.dir/examples/ckks-noise-flooding.cpp.o -c /home/eric070021/openfhe-developmentv1.0.4/src/pke/examples/ckks-noise-flooding.cpp

src/pke/CMakeFiles/ckks-noise-flooding.dir/examples/ckks-noise-flooding.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ckks-noise-flooding.dir/examples/ckks-noise-flooding.cpp.i"
	cd /home/eric070021/openfhe-developmentv1.0.4/build_debug/src/pke && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/eric070021/openfhe-developmentv1.0.4/src/pke/examples/ckks-noise-flooding.cpp > CMakeFiles/ckks-noise-flooding.dir/examples/ckks-noise-flooding.cpp.i

src/pke/CMakeFiles/ckks-noise-flooding.dir/examples/ckks-noise-flooding.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ckks-noise-flooding.dir/examples/ckks-noise-flooding.cpp.s"
	cd /home/eric070021/openfhe-developmentv1.0.4/build_debug/src/pke && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/eric070021/openfhe-developmentv1.0.4/src/pke/examples/ckks-noise-flooding.cpp -o CMakeFiles/ckks-noise-flooding.dir/examples/ckks-noise-flooding.cpp.s

# Object files for target ckks-noise-flooding
ckks__noise__flooding_OBJECTS = \
"CMakeFiles/ckks-noise-flooding.dir/examples/ckks-noise-flooding.cpp.o"

# External object files for target ckks-noise-flooding
ckks__noise__flooding_EXTERNAL_OBJECTS =

bin/examples/pke/ckks-noise-flooding: src/pke/CMakeFiles/ckks-noise-flooding.dir/examples/ckks-noise-flooding.cpp.o
bin/examples/pke/ckks-noise-flooding: src/pke/CMakeFiles/ckks-noise-flooding.dir/build.make
bin/examples/pke/ckks-noise-flooding: lib/libOPENFHEpke.so.1.0.4
bin/examples/pke/ckks-noise-flooding: lib/libOPENFHEcore.so.1.0.4
bin/examples/pke/ckks-noise-flooding: src/pke/CMakeFiles/ckks-noise-flooding.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/eric070021/openfhe-developmentv1.0.4/build_debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/examples/pke/ckks-noise-flooding"
	cd /home/eric070021/openfhe-developmentv1.0.4/build_debug/src/pke && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ckks-noise-flooding.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/pke/CMakeFiles/ckks-noise-flooding.dir/build: bin/examples/pke/ckks-noise-flooding
.PHONY : src/pke/CMakeFiles/ckks-noise-flooding.dir/build

src/pke/CMakeFiles/ckks-noise-flooding.dir/clean:
	cd /home/eric070021/openfhe-developmentv1.0.4/build_debug/src/pke && $(CMAKE_COMMAND) -P CMakeFiles/ckks-noise-flooding.dir/cmake_clean.cmake
.PHONY : src/pke/CMakeFiles/ckks-noise-flooding.dir/clean

src/pke/CMakeFiles/ckks-noise-flooding.dir/depend:
	cd /home/eric070021/openfhe-developmentv1.0.4/build_debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/eric070021/openfhe-developmentv1.0.4 /home/eric070021/openfhe-developmentv1.0.4/src/pke /home/eric070021/openfhe-developmentv1.0.4/build_debug /home/eric070021/openfhe-developmentv1.0.4/build_debug/src/pke /home/eric070021/openfhe-developmentv1.0.4/build_debug/src/pke/CMakeFiles/ckks-noise-flooding.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/pke/CMakeFiles/ckks-noise-flooding.dir/depend

