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

# Utility rule file for coreinfo.

# Include any custom commands dependencies for this target.
include src/core/CMakeFiles/coreinfo.dir/compiler_depend.make

# Include the progress variables for this target.
include src/core/CMakeFiles/coreinfo.dir/progress.make

src/core/CMakeFiles/coreinfo: src/core/coreinfocmd

src/core/coreinfocmd:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/eric070021/openfhe-developmentv1.0.4/build_release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating coreinfocmd"
	cd /home/eric070021/openfhe-developmentv1.0.4/build_release/src/core && echo Builds OPENFHEcore and these apps: parallel sampling

coreinfo: src/core/CMakeFiles/coreinfo
coreinfo: src/core/coreinfocmd
coreinfo: src/core/CMakeFiles/coreinfo.dir/build.make
.PHONY : coreinfo

# Rule to build all files generated by this target.
src/core/CMakeFiles/coreinfo.dir/build: coreinfo
.PHONY : src/core/CMakeFiles/coreinfo.dir/build

src/core/CMakeFiles/coreinfo.dir/clean:
	cd /home/eric070021/openfhe-developmentv1.0.4/build_release/src/core && $(CMAKE_COMMAND) -P CMakeFiles/coreinfo.dir/cmake_clean.cmake
.PHONY : src/core/CMakeFiles/coreinfo.dir/clean

src/core/CMakeFiles/coreinfo.dir/depend:
	cd /home/eric070021/openfhe-developmentv1.0.4/build_release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/eric070021/openfhe-developmentv1.0.4 /home/eric070021/openfhe-developmentv1.0.4/src/core /home/eric070021/openfhe-developmentv1.0.4/build_release /home/eric070021/openfhe-developmentv1.0.4/build_release/src/core /home/eric070021/openfhe-developmentv1.0.4/build_release/src/core/CMakeFiles/coreinfo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/core/CMakeFiles/coreinfo.dir/depend

