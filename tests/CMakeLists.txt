find_package(GTest REQUIRED)

# Добавляем директорию с заголовочными файлами проекта
include_directories(${CMAKE_SOURCE_DIR}/include)

# Создаем исполняемый файл теста, подключая нужные исходники
add_executable(
    unit_tests
    test_func.cpp
    test_population.cpp
    ${CMAKE_SOURCE_DIR}/src/process_block.cpp
    ${CMAKE_SOURCE_DIR}/src/population.cpp
    ${CMAKE_SOURCE_DIR}/src/random_utils.cpp
)

# Линкуем библиотеки
target_link_libraries(
    unit_tests
    PRIVATE
    GTest::GTest
    GTest::Main
    ${ARMADILLO_LIBRARIES}
    ${OpenCV_LIBS}
)

# Добавляем тест в CTest
add_test(NAME GBO_Unit_Tests COMMAND unit_tests)
