
find_library(PTHREAD_LIB pthread)

add_executable(MLPracticalTest
        main.cpp ml.cpp ml.h InputParser.cpp InputParser.h concurrent_set.h)

target_link_libraries(MLPracticalTest "${PTHREAD_LIB}")
