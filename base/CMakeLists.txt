add_library(base_omp)
add_library(base_cuda)

file(GLOB_RECURSE CppSources CONFIGURE_DEPENDS
    src/*.cpp src/*.hpp src/*.h src/*.inl)

file(GLOB_RECURSE CudaSources CONFIGURE_DEPENDS
    src/*.cu src/*.cuh)

file(GLOB_RECURSE Headers CONFIGURE_DEPENDS
    include/*.hpp include/*.h include/*.inl include/*.cuh)

target_sources(base_omp
    PRIVATE ${CppSources}
    PUBLIC ${Headers})

target_sources(base_cuda
    PRIVATE ${CppSources} ${CudaSources}
    PUBLIC ${Headers})

foreach (target "base_omp" "base_cuda")
    target_include_directories(${target}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/hpc>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include/hpc>)

    target_compile_options(${target}
        PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>;
        $<$<CONFIG:Release>:
        $<$<COMPILE_LANGUAGE:CXX>:-Ofast;-march=native>>;
        $<$<CONFIG:Debug>:-g>)
endforeach ()

find_package(OpenMP)

target_link_libraries(base_omp
    PUBLIC OpenMP::OpenMP_CXX)

find_package(Thrust QUIET)
if (NOT ${Thrust_FOUND})
    add_subdirectory(ext/thrust)
endif ()
thrust_create_target(Thrust)

target_link_libraries(base_cuda
    PUBLIC Thrust)

set_target_properties(base_cuda
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 70)