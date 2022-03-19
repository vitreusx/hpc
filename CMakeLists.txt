cmake_minimum_required(VERSION 3.14)
project(hpc)

enable_language(CUDA)
add_subdirectory(base)

function(create_omp_target target)
    add_executable(${target})

    target_sources(${target}
        PRIVATE ${target}.cpp)

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

add_subdirectory(hello_world)
add_subdirectory(mat_mult)
add_subdirectory(pi)

function(create_cuda_target target)
    add_executable(${target})

    target_sources(${target}
        PRIVATE ${target}.cu)

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

add_subdirectory(stencil)