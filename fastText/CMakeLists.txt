cmake_minimum_required(VERSION 3.12)
project(fastText)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

add_executable(fastText
        src/args.cc
        src/args.h
        src/dictionary.cc
        src/dictionary.h
        src/fasttext.cc
        src/fasttext.h
        src/main.cc
        src/matrix.cc
        src/matrix.h
        src/model.cc
        src/model.h
        src/real.h
        src/utils.cc
        src/utils.h
        src/vector.cc
        src/vector.h)
