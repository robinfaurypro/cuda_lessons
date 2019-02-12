#pragma once

#include <math.h>
#include <iostream>

class Vector3 {
public:
	__host__ __device__ Vector3() : x_(0.f), y_(0.f), z_(0.f) {}
	__host__ __device__ Vector3(float x, float y, float z) : x_(x), y_(y), z_(z) {}

	__host__ __device__ float x() const { return x_; }
	__host__ __device__ float y() const { return y_; }
	__host__ __device__ float z() const { return z_; }
	__host__ __device__ float r() const { return x_; }
	__host__ __device__ float g() const { return y_; }
	__host__ __device__ float b() const { return z_; }

	__host__ __device__ const Vector3& operator+() const { return *this; }
	__host__ __device__ Vector3 operator-() const { return Vector3(-x_, -y_, -z_); }
	__host__ __device__ float operator[] (int i) const { return i==0?x_:(i==1?y_:z_); }
	__host__ __device__ float& operator[] (int i) { return i==0?x_:(i==1?y_:z_); }

	__host__ __device__ Vector3& operator+=(const Vector3& v) {x_ += v.x_; y_ += v.y_; z_ += v.z_; return *this;}
	__host__ __device__ Vector3& operator-=(const Vector3& v) {x_ -= v.x_; y_ -= v.y_; z_ -= v.z_; return *this;}
	__host__ __device__ Vector3& operator*=(const Vector3& v) {x_ *= v.x_; y_ *= v.y_; z_ *= v.z_; return *this;}
	__host__ __device__ Vector3& operator/=(const Vector3& v) {x_ /= v.x_; y_ /= v.y_; z_ /= v.z_; return *this;}
	__host__ __device__ Vector3& operator*=(const float t) {x_ *= t; y_ *= t; z_ *= t; return *this;}
	__host__ __device__ Vector3& operator/=(const float t) {x_ /= t; y_ /= t; z_ /= t; return *this;}

	__host__ __device__ float norm() const { return std::sqrt(x_*x_+y_*y_+z_*z_); }
	__host__ __device__ float squared_norm() const { return x_*x_+y_*y_+z_*z_; }
	__host__ __device__ void normalize() { float n = 1.f/norm(); x_*=n; y_*=n; z_*=n; }

private:
	float x_;
	float y_;
	float z_;
};

__host__ __device__ Vector3 operator+(const Vector3& v1, const Vector3& v2);
__host__ __device__ Vector3 operator-(const Vector3& v1, const Vector3& v2);
__host__ __device__ Vector3 operator*(const Vector3& v1, const Vector3& v2);
__host__ __device__ Vector3 operator/(const Vector3& v1, const Vector3& v2);
__host__ __device__ Vector3 operator*(const Vector3& v1, float t);
__host__ __device__ Vector3 operator*(float t, const Vector3& v1);
__host__ __device__ Vector3 operator/(const Vector3& v1, float t);

__host__ __device__ float dot(const Vector3& v1, const Vector3& v2);
__host__ __device__ Vector3 cross(const Vector3& v1, const Vector3& v2);