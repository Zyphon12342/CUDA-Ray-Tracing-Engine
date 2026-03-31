#include "rtweekend.h"

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"

#include <vector>
#include <iostream>

// CUDA-compatible material: just the surface color (albedo).
// Must match the Material struct layout in cuda_utils.cuh.
struct CUDAMaterial {
    vec3 albedo;
};

struct CUDASphere {
    point3 center;
    double radius;
    CUDAMaterial mat;
};

extern "C" void render_cuda(const std::vector<CUDASphere>& host_spheres,
                            int image_width, int image_height,
                            double vfov, point3 lookfrom, point3 lookat, vec3 vup,
                            std::vector<int>& output_image_data);

int main() {
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    camera cam;

    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width  = 1920;

    cam.vfov     = 20;
    cam.lookfrom = point3(13, 2, 3);
    cam.lookat   = point3(0, 0, 0);
    cam.vup      = vec3(0, 1, 0);

    // Calculate image height
    int image_height = int(cam.image_width / cam.aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // Convert world to CUDA spheres
    std::vector<CUDASphere> cuda_spheres;
    for (const auto& obj : world.objects) {
        auto s = std::dynamic_pointer_cast<sphere>(obj);
        if (s) {
            CUDASphere cs;
            cs.center = s->get_center();
            cs.radius = s->get_radius();
            
            auto mat = s->get_material();
            if (auto l = std::dynamic_pointer_cast<lambertian>(mat)) {
                cs.mat.albedo = l->get_albedo();
            } else if (auto m = std::dynamic_pointer_cast<metal>(mat)) {
                cs.mat.albedo = m->get_albedo();
            } else if (auto d = std::dynamic_pointer_cast<dielectric>(mat)) {
                cs.mat.albedo = vec3(0.95, 0.95, 0.95); // glass: near-white
            }
            cuda_spheres.push_back(cs);
        }
    }

    std::clog << "Rendering with CUDA..." << std::endl;
    std::vector<int> image_data;
    render_cuda(cuda_spheres, cam.image_width, image_height,
                cam.vfov, cam.lookfrom, cam.lookat, cam.vup,
                image_data);

    std::cout << "P3\n" << cam.image_width << " " << image_height << "\n255\n";
    for (size_t i = 0; i < image_data.size() / 3; i++) {
        std::cout << image_data[3*i] << " " << image_data[3*i+1] << " " << image_data[3*i+2] << "\n";
    }
    std::clog << "\rDone.           \n";
}
