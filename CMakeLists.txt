cmake_minimum_required(VERSION 3.0)
project(bertc)

option(BERTC_BUILD_TESTS "Build executable with tests" ON)

add_library(bertc)

target_sources(bertc
        PRIVATE
        bertc.h
)

set_target_properties(bertc PROPERTIES LANGUAGE C)
set_target_properties(bertc PROPERTIES LINKER_LANGUAGE C)
target_compile_features(bertc PUBLIC c_std_17)

target_include_directories(bertc
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
)


if (BERTC_BUILD_TESTS)
    set(test_sources test_bert.c)

    add_executable(bertc-tests)
    target_sources(bertc-tests PRIVATE ${test_sources})
    target_link_libraries(bertc-tests PRIVATE bertc)

    add_executable(bertc-tests-asan)
    target_sources(bertc-tests-asan PRIVATE ${test_sources})
    target_compile_options(bertc-tests-asan PRIVATE -fsanitize=address,undefined)
    target_link_options(bertc-tests-asan PRIVATE -fsanitize=address,undefined)
    target_link_libraries(bertc-tests-asan PRIVATE bertc)

    add_executable(bertc-tests-msan)
    target_sources(bertc-tests-msan PRIVATE ${test_sources})
    target_compile_options(bertc-tests-msan PRIVATE -fsanitize=leak)
    target_link_options(bertc-tests-msan PRIVATE -fsanitize=leak)
    target_link_libraries(bertc-tests-msan PRIVATE bertc)

    add_executable(bertc-tests-tsan)
    target_sources(bertc-tests-tsan PRIVATE ${test_sources})
    target_compile_options(bertc-tests-tsan PRIVATE -fsanitize=thread)
    target_link_options(bertc-tests-tsan PRIVATE -fsanitize=thread)
    target_link_libraries(bertc-tests-tsan PRIVATE bertc)
endif ()


