#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <iostream>

// --- Vec3 Class (CUDA compatible) ---
class vec3 {
public:
    double e[3];

    __host__ __device__ vec3() : e{0,0,0} {}
    __host__ __device__ vec3(double e0, double e1, double e2) : e{e0,e1,e2} {}

    __host__ __device__ double x() const { return e[0]; }
    __host__ __device__ double y() const { return e[1]; }
    __host__ __device__ double z() const { return e[2]; }

    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ double operator[](int i) const { return e[i]; }
    __host__ __device__ double& operator[](int i) { return e[i]; }

    __host__ __device__ vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ vec3& operator*=(const double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ vec3& operator/=(const double t) {
        return *this *= 1/t;
    }

    __host__ __device__ double length() const {
        return sqrt(length_squared());
    }

    __host__ __device__ double length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }
};

using point3 = vec3;
using color = vec3;

// --- Vec3 Utility Functions ---

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(double t, const vec3& v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, double t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3& v, double t) {
    return (1/t) * v;
}

__host__ __device__ inline double dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

// --- Ray Class ---

class ray {
public:
    point3 orig;
    vec3 dir;

    __host__ __device__ ray() {}
    __host__ __device__ ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {}

    __host__ __device__ point3 origin() const { return orig; }
    __host__ __device__ vec3 direction() const { return dir; }

    __host__ __device__ point3 at(double t) const {
        return orig + t*dir;
    }
};

// --- Random Number Generation ---

#define RANDVEC3 vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))

__device__ inline vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * RANDVEC3 - vec3(1,1,1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ inline vec3 random_unit_vector(curandState *local_rand_state) {
    return unit_vector(random_in_unit_sphere(local_rand_state));
}

__device__ inline vec3 random_on_hemisphere(const vec3& normal, curandState *local_rand_state) {
    vec3 on_unit_sphere = random_unit_vector(local_rand_state);
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

// --- Material Struct (Simplified for CUDA) ---

enum MaterialType { LAMBERTIAN, METAL, DIELECTRIC };

struct Material {
    MaterialType type;
    vec3 albedo;
    double fuzz;
    double ref_idx;
};

// --- Sphere Struct ---

struct Sphere {
    point3 center;
    double radius;
    Material mat;

    __device__ bool hit(const ray& r, double t_min, double t_max, double& t, vec3& p, vec3& normal, Material& material, bool& front_face) const {
        vec3 oc = center - r.origin();
        auto a = r.direction().length_squared();
        auto h = dot(r.direction(), oc);
        auto c = oc.length_squared() - radius*radius;
        auto discriminant = h*h - a*c;

        if (discriminant < 0) return false;
        auto sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;
        if (root <= t_min || t_max <= root) {
            root = (h + sqrtd) / a;
            if (root <= t_min || t_max <= root)
                return false;
        }

        t = root;
        p = r.at(t);
        vec3 outward_normal = (p - center) / radius;
        
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
        material = mat;

        return true;
    }
};

// --- Scatter Functions ---

__device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v,n)*n;
}

__device__ inline vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

__device__ inline double reflectance(double cosine, double ref_idx) {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1-ref_idx) / (1+ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine),5);
}

__device__ inline bool scatter(const ray& r_in, const vec3& p, const vec3& normal, bool front_face, const Material& mat, vec3& attenuation, ray& scattered, curandState *local_rand_state) {
    if (mat.type == LAMBERTIAN) {
        vec3 scatter_direction = normal + random_unit_vector(local_rand_state);
        if (scatter_direction.length_squared() < 1e-8) scatter_direction = normal;
        scattered = ray(p, scatter_direction);
        attenuation = mat.albedo;
        return true;
    } else if (mat.type == METAL) {
        vec3 reflected = reflect(unit_vector(r_in.direction()), normal);
        scattered = ray(p, reflected + mat.fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = mat.albedo;
        return (dot(scattered.direction(), normal) > 0);
    } else if (mat.type == DIELECTRIC) {
        attenuation = vec3(1.0, 1.0, 1.0);
        double refraction_ratio = front_face ? (1.0/mat.ref_idx) : mat.ref_idx;

        vec3 unit_direction = unit_vector(r_in.direction());
        double cos_theta = fmin(dot(-unit_direction, normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(local_rand_state))
            direction = reflect(unit_direction, normal);
        else
            direction = refract(unit_direction, normal, refraction_ratio);

        scattered = ray(p, direction);
        return true;
    }
    return false;
}

#endif
