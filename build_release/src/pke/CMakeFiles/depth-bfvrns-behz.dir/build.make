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
CMAKE_BINARY_DIR = /home/eric070021/openfhe-developmentv1.0.4/build_release

# Include any dependencies generated for this target.
include src/pke/CMakeFiles/depth-bfvrns-behz.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/pke/CMakeFiles/depth-bfvrns-behz.dir/compiler_depend.make

# Include the progress variables for this target.
include src/pke/CMakeFiles/depth-bfvrns-behz.dir/progress.make

# Include the compile flags for this target's objects.
include src/pke/CMakeFiles/depth-bfvrns-behz.dir/flags.make

src/pke/CMakeFiles/depth-bfvrns-behz.dir/examples/depth-bfvrns-behz.cpp.o: src/pke/CMakeFiles/depth-bfvrns-behz.dir/flags.make
src/pke/CMakeFiles/depth-bfvrns-behz.dir/examples/depth-bfvrns-behz.cpp.o: ../src/pke/examples/depth-bfvrns-behz.cpp
src/pke/CMakeFiles/depth-bfvrns-behz.dir/examples/depth-bfvrns-behz.cpp.o: src/pke/CMakeFiles/depth-bfvrns-behz.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/eric070021/openfhe-developmentv1.0.4/build_release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/pke/CMakeFiles/depth-bfvrns-behz.dir/examples/depth-bfvrns-behz.cpp.o"
	cd /home/eric070021/openfhe-developmentv1.0.4/build_release/src/pke && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/pke/CMakeFiles/depth-bfvrns-behz.dir/examples/depth-bfvrns-behz.cpp.o -MF CMakeFiles/depth-bfvrns-behz.dir/examples/depth-bfvrns-behz.cpp.o.d -o CMakeFiles/depth-bfvrns-behz.dir/examples/depth-bfvrns-behz.cpp.o -c /home/eric070021/openfhe-developmentv1.0.4/src/pke/examples/depth-bfvrns-behz.cpp

src/pke/CMakeFiles/depth-bfvrns-behz.dir/examples/depth-bfvrns-behz.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/depth-bfvrns-behz.dir/examples/depth-bfvrns-behz.cpp.i"
	cd /home/eric070021/openfhe-developmentv1.0.4/build_release/src/pke && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/eric070021/openfhe-developmentv1.0.4/src/pke/examples/depth-bfvrns-behz.cpp > CMakeFiles/depth-bfvrns-behz.dir/examples/depth-bfvrns-behz.cpp.i

src/pke/CMakeFiles/depth-bfvrns-behz.dir/examples/depth-bfvrns-behz.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/depth-bfvrns-behz.dir/examples/depth-bfvrns-behz.cpp.s"
	cd /home/eric070021/openfhe-developmentv1.0.4/build_release/src/pke && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/eric070021/openfhe-developmentv1.0.4/src/pke/examples/depth-bfvrns-behz.cpp -o CMakeFiles/depth-bfvrns-behz.dir/examples/depth-bfvrns-behz.cpp.s

# Object files for target depth-bfvrns-behz
depth__bfvrns__behz_OBJECTS = \
"CMakeFiles/depth-bfvrns-behz.dir/examples/depth-bfvrns-behz.cpp.o"

# External object files for target depth-bfvrns-behz
depth__bfvrns__behz_EXTERNAL_OBJECTS =

bin/examples/pke/depth-bfvrns-behz: src/pke/CMakeFiles/depth-bfvrns-behz.dir/examples/depth-bfvrns-behz.cpp.o
bin/examples/pke/depth-bfvrns-behz: src/pke/CMakeFiles/depth-bfvrns-behz.dir/build.make
bin/examples/pke/depth-bfvrns-behz: lib/libOPENFHEpke.so.1.0.4
bin/examples/pke/depth-bfvrns-behz: lib/libOPENFHEcore.so.1.0.4
bin/examples/pke/depth-bfvrns-behz: src/pke/CMakeFiles/depth-bfvrns-behz.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/eric070021/openfhe-developmentv1.0.4/build_release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/examples/pke/depth-bfvrns-behz"
	cd /home/eric070021/openfhe-developmentv1.0.4/build_release/src/pke && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/depth-bfvrns-behz.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/pke/CMakeFiles/depth-bfvrns-behz.dir/build: bin/examples/pke/depth-bfvrns-behz
.PHONY : src/pke/CMakeFiles/depth-bfvrns-behz.dir/build

src/pke/CMakeFiles/depth-bfvrns-behz.dir/clean:
	cd /home/eric070021/openfhe-developmentv1.0.4/build_release/src/pke && $(CMAKE_COMMAND) -P CMakeFiles/depth-bfvrns-behz.dir/cmake_clean.cmake
.PHONY : src/pke/CMakeFiles/depth-bfvrns-behz.dir/clean

src/pke/CMakeFiles/depth-bfvrns-behz.dir/depend:
	cd /home/eric070021/openfhe-developmentv1.0.4/build_release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/eric070021/openfhe-developmentv1.0.4 /home/eric070021/openfhe-developmentv1.0.4/src/pke /home/eric070021/openfhe-developmentv1.0.4/build_release /home/eric070021/openfhe-developmentv1.0.4/build_release/src/pke /home/eric070021/openfhe-developmentv1.0.4/build_release/src/pke/CMakeFiles/depth-bfvrns-behz.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/pke/CMakeFiles/depth-bfvrns-behz.dir/depend

