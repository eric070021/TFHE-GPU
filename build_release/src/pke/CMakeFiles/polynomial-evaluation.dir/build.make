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
include src/pke/CMakeFiles/polynomial-evaluation.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/pke/CMakeFiles/polynomial-evaluation.dir/compiler_depend.make

# Include the progress variables for this target.
include src/pke/CMakeFiles/polynomial-evaluation.dir/progress.make

# Include the compile flags for this target's objects.
include src/pke/CMakeFiles/polynomial-evaluation.dir/flags.make

src/pke/CMakeFiles/polynomial-evaluation.dir/examples/polynomial-evaluation.cpp.o: src/pke/CMakeFiles/polynomial-evaluation.dir/flags.make
src/pke/CMakeFiles/polynomial-evaluation.dir/examples/polynomial-evaluation.cpp.o: ../src/pke/examples/polynomial-evaluation.cpp
src/pke/CMakeFiles/polynomial-evaluation.dir/examples/polynomial-evaluation.cpp.o: src/pke/CMakeFiles/polynomial-evaluation.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/eric070021/openfhe-developmentv1.0.4/build_release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/pke/CMakeFiles/polynomial-evaluation.dir/examples/polynomial-evaluation.cpp.o"
	cd /home/eric070021/openfhe-developmentv1.0.4/build_release/src/pke && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/pke/CMakeFiles/polynomial-evaluation.dir/examples/polynomial-evaluation.cpp.o -MF CMakeFiles/polynomial-evaluation.dir/examples/polynomial-evaluation.cpp.o.d -o CMakeFiles/polynomial-evaluation.dir/examples/polynomial-evaluation.cpp.o -c /home/eric070021/openfhe-developmentv1.0.4/src/pke/examples/polynomial-evaluation.cpp

src/pke/CMakeFiles/polynomial-evaluation.dir/examples/polynomial-evaluation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/polynomial-evaluation.dir/examples/polynomial-evaluation.cpp.i"
	cd /home/eric070021/openfhe-developmentv1.0.4/build_release/src/pke && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/eric070021/openfhe-developmentv1.0.4/src/pke/examples/polynomial-evaluation.cpp > CMakeFiles/polynomial-evaluation.dir/examples/polynomial-evaluation.cpp.i

src/pke/CMakeFiles/polynomial-evaluation.dir/examples/polynomial-evaluation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/polynomial-evaluation.dir/examples/polynomial-evaluation.cpp.s"
	cd /home/eric070021/openfhe-developmentv1.0.4/build_release/src/pke && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/eric070021/openfhe-developmentv1.0.4/src/pke/examples/polynomial-evaluation.cpp -o CMakeFiles/polynomial-evaluation.dir/examples/polynomial-evaluation.cpp.s

# Object files for target polynomial-evaluation
polynomial__evaluation_OBJECTS = \
"CMakeFiles/polynomial-evaluation.dir/examples/polynomial-evaluation.cpp.o"

# External object files for target polynomial-evaluation
polynomial__evaluation_EXTERNAL_OBJECTS =

bin/examples/pke/polynomial-evaluation: src/pke/CMakeFiles/polynomial-evaluation.dir/examples/polynomial-evaluation.cpp.o
bin/examples/pke/polynomial-evaluation: src/pke/CMakeFiles/polynomial-evaluation.dir/build.make
bin/examples/pke/polynomial-evaluation: lib/libOPENFHEpke.so.1.0.4
bin/examples/pke/polynomial-evaluation: lib/libOPENFHEcore.so.1.0.4
bin/examples/pke/polynomial-evaluation: src/pke/CMakeFiles/polynomial-evaluation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/eric070021/openfhe-developmentv1.0.4/build_release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/examples/pke/polynomial-evaluation"
	cd /home/eric070021/openfhe-developmentv1.0.4/build_release/src/pke && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/polynomial-evaluation.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/pke/CMakeFiles/polynomial-evaluation.dir/build: bin/examples/pke/polynomial-evaluation
.PHONY : src/pke/CMakeFiles/polynomial-evaluation.dir/build

src/pke/CMakeFiles/polynomial-evaluation.dir/clean:
	cd /home/eric070021/openfhe-developmentv1.0.4/build_release/src/pke && $(CMAKE_COMMAND) -P CMakeFiles/polynomial-evaluation.dir/cmake_clean.cmake
.PHONY : src/pke/CMakeFiles/polynomial-evaluation.dir/clean

src/pke/CMakeFiles/polynomial-evaluation.dir/depend:
	cd /home/eric070021/openfhe-developmentv1.0.4/build_release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/eric070021/openfhe-developmentv1.0.4 /home/eric070021/openfhe-developmentv1.0.4/src/pke /home/eric070021/openfhe-developmentv1.0.4/build_release /home/eric070021/openfhe-developmentv1.0.4/build_release/src/pke /home/eric070021/openfhe-developmentv1.0.4/build_release/src/pke/CMakeFiles/polynomial-evaluation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/pke/CMakeFiles/polynomial-evaluation.dir/depend

