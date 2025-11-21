#include "cuda_utils.cuh"
#include <iostream>
#include <vector>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ bool world_hit(const Sphere* spheres, int num_spheres, const ray& r, double t_min, double t_max, double& t, vec3& p, vec3& normal, Material& mat, bool& front_face) {
    double temp_t;
    vec3 temp_p, temp_normal;
    Material temp_mat;
    bool temp_front_face;
    bool hit_anything = false;
    double closest_so_far = t_max;

    for (int i = 0; i < num_spheres; i++) {
        if (spheres[i].hit(r, t_min, closest_so_far, temp_t, temp_p, temp_normal, temp_mat, temp_front_face)) {
            hit_anything = true;
            closest_so_far = temp_t;
            t = temp_t;
            p = temp_p;
            normal = temp_normal;
            mat = temp_mat;
            front_face = temp_front_face;
        }
    }
    return hit_anything;
}

__device__ color ray_color(ray& r, const Sphere* spheres, int num_spheres, curandState *local_rand_state, int max_depth) {
    color cur_attenuation(1.0, 1.0, 1.0);
    
    for (int i = 0; i < max_depth; i++) {
        double t;
        vec3 p, normal;
        Material mat;
        bool front_face;
        
        if (world_hit(spheres, num_spheres, r, 0.001, 1e20, t, p, normal, mat, front_face)) {
            ray scattered;
            vec3 attenuation;
            if (scatter(r, p, normal, front_face, mat, attenuation, scattered, local_rand_state)) {
                cur_attenuation = cur_attenuation * attenuation;
                r = scattered;
            } else {
                return color(0,0,0);
            }
        } else {
            vec3 unit_direction = unit_vector(r.direction());
            auto a = 0.5*(unit_direction.y() + 1.0);
            vec3 background = (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
            return cur_attenuation * background;
        }
    }
    return color(0,0,0); // Exceeded recursion depth
}

__global__ void render_kernel(vec3* fb, int max_x, int max_y, int start_y, int num_rows, int samples_per_pixel, int max_depth, 
                              point3 lookfrom, point3 lookat, vec3 vup, double vfov, double aspect_ratio, double focus_dist, double defocus_angle,
                              Sphere* spheres, int num_spheres) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y + start_y;

    if ((i >= max_x) || (j >= max_y) || (j >= start_y + num_rows)) return;

    int pixel_index = j * max_x + i;
    curandState local_rand_state;
    curand_init(1984 + pixel_index, 0, 0, &local_rand_state);

    // Camera setup (re-calculated per thread to avoid passing too many params or creating a camera struct on device for now)
    // Ideally we should pass a Camera struct.
    // Let's calculate camera params here based on inputs.
    
    auto theta = vfov * M_PI / 180.0;
    auto h = tan(theta/2);
    auto viewport_height = 2.0 * h * focus_dist;
    auto viewport_width = viewport_height * (double(max_x)/double(max_y)); // Using passed aspect ratio might be safer but this is consistent with image dims

    auto w = unit_vector(lookfrom - lookat);
    auto u = unit_vector(cross(vup, w));
    auto v = cross(w, u);

    auto viewport_u = viewport_width * u;
    auto viewport_v = viewport_height * -v;

    auto pixel_delta_u = viewport_u / max_x;
    auto pixel_delta_v = viewport_v / max_y;

    auto viewport_upper_left = lookfrom - (focus_dist * w) - viewport_u/2 - viewport_v/2;
    auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    auto defocus_radius = focus_dist * tan((defocus_angle / 2) * M_PI / 180.0);
    auto defocus_disk_u = u * defocus_radius;
    auto defocus_disk_v = v * defocus_radius;

    color pixel_color(0,0,0);

    for (int s = 0; s < samples_per_pixel; s++) {
        // Get Ray
        auto offset_x = curand_uniform(&local_rand_state) - 0.5;
        auto offset_y = curand_uniform(&local_rand_state) - 0.5;
        auto pixel_sample = pixel00_loc + ((i + offset_x) * pixel_delta_u) + ((j + offset_y) * pixel_delta_v);

        vec3 ray_origin;
        if (defocus_angle <= 0) {
            ray_origin = lookfrom;
        } else {
            // Defocus disk sample
            vec3 p;
            do {
                p = vec3(curand_uniform(&local_rand_state)*2.0-1.0, curand_uniform(&local_rand_state)*2.0-1.0, 0);
            } while (p.length_squared() >= 1);
            ray_origin = lookfrom + (p.x() * defocus_disk_u) + (p.y() * defocus_disk_v);
        }
        
        vec3 ray_direction = pixel_sample - ray_origin;
        ray r(ray_origin, ray_direction);

        pixel_color += ray_color(r, spheres, num_spheres, &local_rand_state, max_depth);
    }

    fb[pixel_index] = pixel_color / samples_per_pixel;
}

// Host wrapper
extern "C" void render_cuda(const std::vector<Sphere>& host_spheres, int image_width, int image_height, 
                            int samples_per_pixel, int max_depth,
                            double vfov, point3 lookfrom, point3 lookat, vec3 vup, 
                            double focus_dist, double defocus_angle,
                            std::vector<int>& output_image_data) {

    int num_pixels = image_width * image_height;
    size_t fb_size = num_pixels * sizeof(vec3);

    // Allocate FB
    vec3* fb;
    checkCudaErrors(cudaMalloc((void**)&fb, fb_size));

    // Allocate Spheres
    Sphere* d_spheres;
    checkCudaErrors(cudaMalloc((void**)&d_spheres, host_spheres.size() * sizeof(Sphere)));
    checkCudaErrors(cudaMemcpy(d_spheres, host_spheres.data(), host_spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice));

    // Block/Grid dims
    int tx = 8;
    int ty = 8;
    
    // Render in chunks to show progress
    int rows_per_chunk = 50;
    int num_chunks = (image_height + rows_per_chunk - 1) / rows_per_chunk;

    std::clog << "Rendering " << image_height << " scanlines..." << std::endl;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int start_y = chunk * rows_per_chunk;
        int end_y = min(start_y + rows_per_chunk, image_height);
        int num_rows = end_y - start_y;

        dim3 blocks(image_width / tx + 1, num_rows / ty + 1);
        dim3 threads(tx, ty);

        render_kernel<<<blocks, threads>>>(fb, image_width, image_height, start_y, num_rows, samples_per_pixel, max_depth,
                                           lookfrom, lookat, vup, vfov, double(image_width)/image_height, focus_dist, defocus_angle,
                                           d_spheres, host_spheres.size());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // Progress bar
        float progress = (float)end_y / image_height;
        int barWidth = 70;
        std::clog << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::clog << "=";
            else if (i == pos) std::clog << ">";
            else std::clog << " ";
        }
        std::clog << "] " << int(progress * 100.0) << " %\r";
        std::clog.flush();
    }
    std::clog << std::endl;

    // Copy back
    std::vector<vec3> host_fb(num_pixels);
    checkCudaErrors(cudaMemcpy(host_fb.data(), fb, fb_size, cudaMemcpyDeviceToHost));

    // Convert to int RGB
    output_image_data.resize(num_pixels * 3);
    for (int i = 0; i < num_pixels; i++) {
        auto r = host_fb[i].x();
        auto g = host_fb[i].y();
        auto b = host_fb[i].z();

        // Linear to gamma transform
        r = sqrt(r);
        g = sqrt(g);
        b = sqrt(b);

        output_image_data[3*i + 0] = int(255.99 * r);
        output_image_data[3*i + 1] = int(255.99 * g);
        output_image_data[3*i + 2] = int(255.99 * b);
    }

    // Cleanup
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_spheres));
}
