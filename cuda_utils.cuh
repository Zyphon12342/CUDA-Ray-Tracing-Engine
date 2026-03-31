#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
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
using color  = vec3;

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
    __host__ __device__ ray(const point3& origin, const vec3& direction)
        : orig(origin), dir(direction) {}

    __host__ __device__ point3 origin()    const { return orig; }
    __host__ __device__ vec3   direction() const { return dir; }

    __host__ __device__ point3 at(double t) const {
        return orig + t * dir;
    }
};

// --- Material ---
// Stores only the surface albedo (base color).
// Simple Lambertian shading is computed in the kernel.

struct Material {
    vec3 albedo;
};

// --- Sphere ---

struct Sphere {
    point3 center;
    double radius;
    Material mat;

    // Ray-sphere intersection via the quadratic formula.
    // Returns true and fills hit data if an intersection in [t_min, t_max] exists.
    __device__ bool hit(const ray& r, double t_min, double t_max,
                        double& t, vec3& p, vec3& normal, Material& material,
                        bool& front_face) const {
        vec3 oc = center - r.origin();
        auto a  = r.direction().length_squared();
        auto h  = dot(r.direction(), oc);
        auto c  = oc.length_squared() - radius * radius;
        auto discriminant = h*h - a*c;

        if (discriminant < 0) return false;
        auto sqrtd = sqrt(discriminant);

        // Take the nearest root inside [t_min, t_max]
        auto root = (h - sqrtd) / a;
        if (root <= t_min || root >= t_max) {
            root = (h + sqrtd) / a;
            if (root <= t_min || root >= t_max)
                return false;
        }

        t = root;
        p = r.at(t);
        vec3 outward_normal = (p - center) / radius;
        front_face = dot(r.direction(), outward_normal) < 0;
        normal     = front_face ? outward_normal : -outward_normal;
        material   = mat;
        return true;
    }
};

#endif
