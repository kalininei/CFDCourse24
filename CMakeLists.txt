cmake_minimum_required(VERSION 3.0)

Project(cfdcourse24)

set(CMAKE_CXX_STANDARD 17)

# Catch framework
include(FetchContent)
FetchContent_Declare(
  catch
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.10
)

FetchContent_GetProperties(catch)
if(NOT catch_POPULATED)
  FetchContent_Populate(catch)
  add_subdirectory(${catch_SOURCE_DIR} ${catch_BINARY_DIR})
endif()
set(CATCH2_FRAMEWORK "Catch2::Catch2")

# Boost search paths
set(BOOST_ROOT ${CMAKE_SOURCE_DIR}/../)
find_package(Boost REQUIRED)

# ------ amgcl
FetchContent_Declare(
	amgcl
	GIT_REPOSITORY https://github.com/ddemidov/amgcl
	GIT_TAG 1.4.2
)
FetchContent_GetProperties(amgcl)
if(NOT amgcl_POPULATED)
	FetchContent_Populate(amgcl)
endif()
include_directories(${amgcl_SOURCE_DIR})

# ------- ghc
FetchContent_Declare(
	ghc
	GIT_REPOSITORY https://github.com/gulrak/filesystem
	GIT_TAG v1.5.14
)
FetchContent_GetProperties(ghc)
if(NOT ghc_POPULATED)
	FetchContent_Populate(ghc)
endif()
include_directories(${ghc_SOURCE_DIR}/include)

# -------- mpi
option(MPI_TESTS "Enable MPI tests" OFF)
if (MPI_TESTS)
	find_package(MPI REQUIRED)
endif()


# build all binaries to build/bin
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_custom_target(htmldoc
	COMMAND doxygen Doxyfile
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/doc
	COMMENT "Generating API documentation with Doxygen"
	VERBATIM)


include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${Boost_INCLUDE_DIRS})

add_subdirectory(src)
