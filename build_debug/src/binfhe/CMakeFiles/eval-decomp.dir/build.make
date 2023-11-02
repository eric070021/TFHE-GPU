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
include src/binfhe/CMakeFiles/eval-decomp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/binfhe/CMakeFiles/eval-decomp.dir/compiler_depend.make

# Include the progress variables for this target.
include src/binfhe/CMakeFiles/eval-decomp.dir/progress.make

# Include the compile flags for this target's objects.
include src/binfhe/CMakeFiles/eval-decomp.dir/flags.make

src/binfhe/CMakeFiles/eval-decomp.dir/examples/eval-decomp.cpp.o: src/binfhe/CMakeFiles/eval-decomp.dir/flags.make
src/binfhe/CMakeFiles/eval-decomp.dir/examples/eval-decomp.cpp.o: ../src/binfhe/examples/eval-decomp.cpp
src/binfhe/CMakeFiles/eval-decomp.dir/examples/eval-decomp.cpp.o: src/binfhe/CMakeFiles/eval-decomp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/eric070021/openfhe-developmentv1.0.4/build_debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/binfhe/CMakeFiles/eval-decomp.dir/examples/eval-decomp.cpp.o"
	cd /home/eric070021/openfhe-developmentv1.0.4/build_debug/src/binfhe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/binfhe/CMakeFiles/eval-decomp.dir/examples/eval-decomp.cpp.o -MF CMakeFiles/eval-decomp.dir/examples/eval-decomp.cpp.o.d -o CMakeFiles/eval-decomp.dir/examples/eval-decomp.cpp.o -c /home/eric070021/openfhe-developmentv1.0.4/src/binfhe/examples/eval-decomp.cpp

src/binfhe/CMakeFiles/eval-decomp.dir/examples/eval-decomp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/eval-decomp.dir/examples/eval-decomp.cpp.i"
	cd /home/eric070021/openfhe-developmentv1.0.4/build_debug/src/binfhe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/eric070021/openfhe-developmentv1.0.4/src/binfhe/examples/eval-decomp.cpp > CMakeFiles/eval-decomp.dir/examples/eval-decomp.cpp.i

src/binfhe/CMakeFiles/eval-decomp.dir/examples/eval-decomp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/eval-decomp.dir/examples/eval-decomp.cpp.s"
	cd /home/eric070021/openfhe-developmentv1.0.4/build_debug/src/binfhe && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/eric070021/openfhe-developmentv1.0.4/src/binfhe/examples/eval-decomp.cpp -o CMakeFiles/eval-decomp.dir/examples/eval-decomp.cpp.s

# Object files for target eval-decomp
eval__decomp_OBJECTS = \
"CMakeFiles/eval-decomp.dir/examples/eval-decomp.cpp.o"

# External object files for target eval-decomp
eval__decomp_EXTERNAL_OBJECTS =

bin/examples/binfhe/eval-decomp: src/binfhe/CMakeFiles/eval-decomp.dir/examples/eval-decomp.cpp.o
bin/examples/binfhe/eval-decomp: src/binfhe/CMakeFiles/eval-decomp.dir/build.make
bin/examples/binfhe/eval-decomp: lib/libOPENFHEbinfhe.so.1.0.4
bin/examples/binfhe/eval-decomp: lib/libOPENFHEcore.so.1.0.4
bin/examples/binfhe/eval-decomp: src/binfhe/CMakeFiles/eval-decomp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/eric070021/openfhe-developmentv1.0.4/build_debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/examples/binfhe/eval-decomp"
	cd /home/eric070021/openfhe-developmentv1.0.4/build_debug/src/binfhe && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/eval-decomp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/binfhe/CMakeFiles/eval-decomp.dir/build: bin/examples/binfhe/eval-decomp
.PHONY : src/binfhe/CMakeFiles/eval-decomp.dir/build

src/binfhe/CMakeFiles/eval-decomp.dir/clean:
	cd /home/eric070021/openfhe-developmentv1.0.4/build_debug/src/binfhe && $(CMAKE_COMMAND) -P CMakeFiles/eval-decomp.dir/cmake_clean.cmake
.PHONY : src/binfhe/CMakeFiles/eval-decomp.dir/clean

src/binfhe/CMakeFiles/eval-decomp.dir/depend:
	cd /home/eric070021/openfhe-developmentv1.0.4/build_debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/eric070021/openfhe-developmentv1.0.4 /home/eric070021/openfhe-developmentv1.0.4/src/binfhe /home/eric070021/openfhe-developmentv1.0.4/build_debug /home/eric070021/openfhe-developmentv1.0.4/build_debug/src/binfhe /home/eric070021/openfhe-developmentv1.0.4/build_debug/src/binfhe/CMakeFiles/eval-decomp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/binfhe/CMakeFiles/eval-decomp.dir/depend

