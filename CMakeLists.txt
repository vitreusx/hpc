cmake_minimum_required(VERSION 3.14)
project(hpc)

enable_language(CUDA)
add_subdirectory(base)

function(create_omp_target target)
    add_executable(${target})

    file(GLOB_RECURSE ${target}_HEADERS CONFIGURE_DEPENDS
        ${target}/*.h ${target}/*.hpp)
    file(GLOB_RECURSE ${target}_SOURCES CONFIGURE_DEPENDS
        ${target}/*.cpp)

    target_sources(${target}
        PRIVATE ${${target}_SOURCES}
        PUBLIC ${${target}_HEADERS})

    target_link_libraries(${target}
        PRIVATE base_omp)

    target_compile_features(${target}
        PRIVATE cxx_std_14)

    target_compile_options(${target}
        PRIVATE
        -Wall;-Wextra
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Release>:-Ofast;-march=native>)
endfunction()

create_omp_target(hello_world)
create_omp_target(mat_mult)
create_omp_target(pi)

function(create_cuda_target target)
    add_executable(${target})

    file(GLOB_RECURSE ${target}_HEADERS CONFIGURE_DEPENDS
        ${target}/*.h ${target}/*.cuh ${target}/*.hpp)
    file(GLOB_RECURSE ${target}_SOURCES CONFIGURE_DEPENDS
        ${target}/*.cpp ${target}/*.cu)

    target_sources(${target}
        PUBLIC ${${target}_HEADERS}
        PRIVATE ${${target}_SOURCES})

    target_link_libraries(${target}
        PRIVATE base_cuda)

    target_compile_features(${target}
        PRIVATE cxx_std_14)

    target_compile_options(${target}
        PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>;
        $<$<CONFIG:Release>:
        $<$<COMPILE_LANGUAGE:CXX>:-Ofast;-march=native>>;
        $<$<CONFIG:Debug>:-g>)

    set_target_properties(${target}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 70)
endfunction()

create_cuda_target(stencil)
create_cuda_target(raytracer)
create_cuda_target(streams)