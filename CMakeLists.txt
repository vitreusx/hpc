cmake_minimum_required(VERSION 3.21)
project(hpc)

find_package(OpenMP)

function(create_omp_target name)
    add_executable(${name})

    target_sources(${name}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp)

    target_link_libraries(${name}
        PRIVATE OpenMP::OpenMP_CXX)

    target_compile_features(${name}
        PRIVATE cxx_std_17)

    target_compile_options(${name}
        PRIVATE
        -Wall;-Wextra
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Release>:-Ofast;-march=native>)
endfunction()

create_omp_target(hello_world)
create_omp_target(mat_mult)
create_omp_target(pi)