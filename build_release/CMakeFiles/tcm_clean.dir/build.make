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

# Utility rule file for tcm_clean.

# Include any custom commands dependencies for this target.
include CMakeFiles/tcm_clean.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/tcm_clean.dir/progress.make

CMakeFiles/tcm_clean:
	cd /home/eric070021/openfhe-developmentv1.0.4/build_release/third-party && rm -rf include/gperftools include/google lib/libtcmalloc_minimal* lib/pkgconfig/libtcmalloc* lib/pkgconfig/libprofiler.pc share/doc/gperftools

tcm_clean: CMakeFiles/tcm_clean
tcm_clean: CMakeFiles/tcm_clean.dir/build.make
.PHONY : tcm_clean

# Rule to build all files generated by this target.
CMakeFiles/tcm_clean.dir/build: tcm_clean
.PHONY : CMakeFiles/tcm_clean.dir/build

CMakeFiles/tcm_clean.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tcm_clean.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tcm_clean.dir/clean

CMakeFiles/tcm_clean.dir/depend:
	cd /home/eric070021/openfhe-developmentv1.0.4/build_release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/eric070021/openfhe-developmentv1.0.4 /home/eric070021/openfhe-developmentv1.0.4 /home/eric070021/openfhe-developmentv1.0.4/build_release /home/eric070021/openfhe-developmentv1.0.4/build_release /home/eric070021/openfhe-developmentv1.0.4/build_release/CMakeFiles/tcm_clean.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tcm_clean.dir/depend

