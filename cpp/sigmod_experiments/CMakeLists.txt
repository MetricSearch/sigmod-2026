find_package(HDF5 REQUIRED COMPONENTS CXX)


add_library(experiment_helpers STATIC run_experiment.cpp)
target_include_directories(experiment_helpers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(experiment_helpers SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(experiment_helpers PRIVATE ${HDF5_CXX_LIBRARIES})
rabitq_enable_warnings(experiment_helpers)

add_executable(run_experiment run_experiment_main.cpp)
target_link_libraries(run_experiment PRIVATE experiment_helpers rabitq_headers)
rabitq_enable_warnings(run_experiment)

add_executable(microbenchmark microbenchmark.cpp
    accutest.h)
target_compile_definitions(microbenchmark PRIVATE ANKERL_NANOBENCH_IMPLEMENT)
target_link_libraries(microbenchmark PRIVATE rabitq_headers)
rabitq_enable_warnings(microbenchmark)
add_test(NAME microbenchmark_evp_check COMMAND microbenchmark --check)
set_tests_properties(microbenchmark_evp_check PROPERTIES SKIP_RETURN_CODE 77)

add_executable(unit_tests_microbenchmark unit_tests_microbenchmark.cpp)
target_link_libraries(unit_tests_microbenchmark PRIVATE rabitq_headers)
rabitq_enable_warnings(unit_tests_microbenchmark)
add_test(NAME unit_tests_microbenchmark COMMAND unit_tests_microbenchmark)

add_executable(run_experiment_test run_experiment_test.cpp)
target_include_directories(run_experiment_test SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(run_experiment_test PRIVATE experiment_helpers ${HDF5_CXX_LIBRARIES})
# The standalone test uses assert, including in Release builds.
target_compile_options(run_experiment_test PRIVATE -UNDEBUG)
rabitq_enable_warnings(run_experiment_test)
add_test(NAME run_experiment_test COMMAND run_experiment_test)
add_test(NAME run_experiment_help COMMAND run_experiment --help)

add_executable(rabit_packed_test rabit_packed_test.cpp)
target_link_libraries(rabit_packed_test PRIVATE rabitq_headers)
target_compile_options(rabit_packed_test PRIVATE -UNDEBUG)
rabitq_enable_warnings(rabit_packed_test)
add_test(NAME rabit_packed_test COMMAND rabit_packed_test)
set_tests_properties(rabit_packed_test PROPERTIES SKIP_RETURN_CODE 77)

add_executable(recall_test recall_test.cpp)
target_link_libraries(recall_test PRIVATE rabitq_headers)
target_compile_options(recall_test PRIVATE -UNDEBUG)
rabitq_enable_warnings(recall_test)
add_test(NAME recall_test COMMAND recall_test)
set_tests_properties(recall_test PROPERTIES SKIP_RETURN_CODE 77)
