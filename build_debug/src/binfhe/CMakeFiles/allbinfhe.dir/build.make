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

# Utility rule file for allbinfhe.

# Include any custom commands dependencies for this target.
include src/binfhe/CMakeFiles/allbinfhe.dir/compiler_depend.make

# Include the progress variables for this target.
include src/binfhe/CMakeFiles/allbinfhe.dir/progress.make

allbinfhe: src/binfhe/CMakeFiles/allbinfhe.dir/build.make
.PHONY : allbinfhe

# Rule to build all files generated by this target.
src/binfhe/CMakeFiles/allbinfhe.dir/build: allbinfhe
.PHONY : src/binfhe/CMakeFiles/allbinfhe.dir/build

src/binfhe/CMakeFiles/allbinfhe.dir/clean:
	cd /home/eric070021/openfhe-developmentv1.0.4/build_debug/src/binfhe && $(CMAKE_COMMAND) -P CMakeFiles/allbinfhe.dir/cmake_clean.cmake
.PHONY : src/binfhe/CMakeFiles/allbinfhe.dir/clean

src/binfhe/CMakeFiles/allbinfhe.dir/depend:
	cd /home/eric070021/openfhe-developmentv1.0.4/build_debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/eric070021/openfhe-developmentv1.0.4 /home/eric070021/openfhe-developmentv1.0.4/src/binfhe /home/eric070021/openfhe-developmentv1.0.4/build_debug /home/eric070021/openfhe-developmentv1.0.4/build_debug/src/binfhe /home/eric070021/openfhe-developmentv1.0.4/build_debug/src/binfhe/CMakeFiles/allbinfhe.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/binfhe/CMakeFiles/allbinfhe.dir/depend

