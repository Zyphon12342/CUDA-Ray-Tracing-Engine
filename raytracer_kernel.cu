#include "cuda_utils.cuh"
#include <iostream>
#include <vector>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result)
                  << " at " << file << ":" << line << " '" << func << "'\n";
        cudaDeviceReset();
        exit(99);
    }
}

// Finds the nearest sphere hit by ray r in [t_min, t_max].
__device__ bool world_hit(const Sphere* spheres, int num_spheres, const ray& r,
                           double t_min, double t_max,
                           double& t, vec3& p, vec3& normal, Material& mat, bool& front_face) {
    double closest = t_max;
    bool hit_anything = false;

    for (int i = 0; i < num_spheres; i++) {
        double tmp_t;
        vec3 tmp_p, tmp_normal;
        Material tmp_mat;
        bool tmp_front;

        if (spheres[i].hit(r, t_min, closest, tmp_t, tmp_p, tmp_normal, tmp_mat, tmp_front)) {
            hit_anything = true;
            closest = tmp_t;
            t = tmp_t; p = tmp_p; normal = tmp_normal; mat = tmp_mat; front_face = tmp_front;
        }
    }
    return hit_anything;
}

// One CUDA thread per pixel.
// Each thread:
//   1. Computes the ray direction for its pixel.
//   2. Tests intersection against every sphere.
//   3. Shades with simple diffuse lighting (Lambert's law).
//   4. Writes the resulting color to the framebuffer.
__global__ void render_kernel(vec3* fb, int image_width, int image_height,
                               point3 lookfrom, point3 lookat, vec3 vup, double vfov,
                               Sphere* spheres, int num_spheres) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= image_width || j >= image_height) return;

    // ---- Camera / viewport setup ----
    auto theta          = vfov * M_PI / 180.0;
    auto viewport_height = 2.0 * tan(theta / 2.0);
    auto viewport_width  = viewport_height * (double(image_width) / image_height);

    // Orthonormal camera basis
    auto w = unit_vector(lookfrom - lookat); // points toward viewer
    auto u = unit_vector(cross(vup, w));      // points right
    auto v = cross(w, u);                     // points up

    // Pixel step vectors across the viewport
    auto pixel_delta_u = (viewport_width  / image_width)  * u;
    auto pixel_delta_v = (viewport_height / image_height) * (-v);

    // Top-left pixel center
    auto viewport_upper_left = lookfrom - w
                                - (viewport_width  / 2.0) * u
                                + (viewport_height / 2.0) * v;
    auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // ---- Cast ray for this pixel ----
    auto pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
    ray r(lookfrom, pixel_center - lookfrom);

    // ---- Shade ----
    color pixel_color;
    double t;
    vec3 p, normal;
    Material mat;
    bool front_face;

    if (world_hit(spheres, num_spheres, r, 0.001, 1e20, t, p, normal, mat, front_face)) {
        // Lambertian (diffuse) shading: intensity = dot(surface_normal, light_direction)
        vec3 light_dir = unit_vector(vec3(1.0, 2.0, 1.0));
        double diffuse  = fmax(0.0, dot(normal, light_dir));
        pixel_color = mat.albedo * (0.15 + 0.85 * diffuse); // ambient + diffuse
    } else {
        // Sky gradient (white → light blue)
        auto unit_dir = unit_vector(r.direction());
        auto a = 0.5 * (unit_dir.y() + 1.0);
        pixel_color = (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
    }

    fb[j * image_width + i] = pixel_color;
}

// Host wrapper: uploads the scene to the GPU, launches the kernel, downloads the result.
extern "C" void render_cuda(const std::vector<Sphere>& host_spheres,
                            int image_width, int image_height,
                            double vfov, point3 lookfrom, point3 lookat, vec3 vup,
                            std::vector<int>& output_image_data) {

    int num_pixels  = image_width * image_height;
    size_t fb_size  = num_pixels * sizeof(vec3);

    // Allocate device framebuffer
    vec3* fb;
    checkCudaErrors(cudaMalloc((void**)&fb, fb_size));

    // Upload scene to device
    Sphere* d_spheres;
    checkCudaErrors(cudaMalloc((void**)&d_spheres, host_spheres.size() * sizeof(Sphere)));
    checkCudaErrors(cudaMemcpy(d_spheres, host_spheres.data(),
                               host_spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice));

    // Launch: 8×8 thread blocks covering the full image
    dim3 threads(8, 8);
    dim3 blocks(image_width / threads.x + 1, image_height / threads.y + 1);

    std::clog << "Rendering " << image_width << "x" << image_height
              << " with CUDA (" << host_spheres.size() << " spheres)..." << std::endl;

    render_kernel<<<blocks, threads>>>(fb, image_width, image_height,
                                       lookfrom, lookat, vup, vfov,
                                       d_spheres, host_spheres.size());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Download framebuffer
    std::vector<vec3> host_fb(num_pixels);
    checkCudaErrors(cudaMemcpy(host_fb.data(), fb, fb_size, cudaMemcpyDeviceToHost));

    // Convert float [0,1] → int [0,255] with gamma-2 correction
    output_image_data.resize(num_pixels * 3);
    for (int idx = 0; idx < num_pixels; idx++) {
        auto r = sqrt(host_fb[idx].x());
        auto g = sqrt(host_fb[idx].y());
        auto b = sqrt(host_fb[idx].z());
        output_image_data[3*idx + 0] = int(255.99 * r);
        output_image_data[3*idx + 1] = int(255.99 * g);
        output_image_data[3*idx + 2] = int(255.99 * b);
    }

    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_spheres));
}
