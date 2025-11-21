cmake_minimum_required(VERSION 3.10)
project(RayTracer)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(main main.cpp raytracer_kernel.cu)

# --- Run target -------------------------------------------------------
add_custom_target(run
    COMMAND $<TARGET_FILE:main> > image.ppm
    DEPENDS main
    COMMENT "Running ray tracer to generate image.ppm"
)

# --- View target ------------------------------------------------------
add_custom_target(view
    COMMAND mpv image.ppm
    DEPENDS run
    COMMENT "Opening image.ppm with mpv"
)

# --- Clean build target ----------------------------------------------
add_custom_target(clean_build
    COMMAND ${CMAKE_COMMAND} -E remove -f $<TARGET_FILE:main> image.ppm
    COMMAND ${CMAKE_COMMAND} --build .
    COMMENT "Clean build: deleted executable and image.ppm, rebuilding..."
)

# --- Clean + view target ---------------------------------------------
add_custom_target(clean_view
    COMMAND ${CMAKE_COMMAND} -E remove -f $<TARGET_FILE:main> image.ppm
    COMMAND ${CMAKE_COMMAND} --build . --target view
    COMMENT "Clean build, regenerate image.ppm, and view with mpv"
)
