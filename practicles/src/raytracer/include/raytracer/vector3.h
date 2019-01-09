#include <math.h>
#include <iostream>

class Vector3 {
public:
	Vector3() : x_(0.f), y_(0.f), z_(0.f) {}
	Vector3(float x, float y, float z) : x_(x), y_(y), z_(z) {}

	inline float x() const { return x_; }
	inline float y() const { return y_; }
	inline float z() const { return z_; }
	inline float r() const { return x_; }
	inline float g() const { return y_; }
	inline float b() const { return z_; }

	inline const Vector3& operator+() const { return *this; }
	inline Vector3 operator-() const { return Vector3(-x_, -y_, -z_); }
	inline float operator[] (int i) const { return i==0?x_:(i==1?y_:z_); }
	inline float& operator[] (int i) { return i==0?x_:(i==1?y_:z_); }

	inline Vector3& operator+=(const Vector3& v) {x_ += v.x_; y_ += v.y_; z_ += v.z_; return *this;}
	inline Vector3& operator-=(const Vector3& v) {x_ -= v.x_; y_ -= v.y_; z_ -= v.z_; return *this;}
	inline Vector3& operator*=(const Vector3& v) {x_ *= v.x_; y_ *= v.y_; z_ *= v.z_; return *this;}
	inline Vector3& operator/=(const Vector3& v) {x_ /= v.x_; y_ /= v.y_; z_ /= v.z_; return *this;}
	inline Vector3& operator*=(const float t) {x_ *= t; y_ *= t; z_ *= t; return *this;}
	inline Vector3& operator/=(const float t) {x_ /= t; y_ /= t; z_ /= t; return *this;}

	inline float norm() const { return std::sqrt(x_*x_+y_*y_+z_*z_); }
	inline float squared_norm() const { return x_*x_+y_*y_+z_*z_; }
	inline void normelize() { float n = 1.f/norm(); x_*=n; y_*=n; z_*=n; }

private:
	float x_;
	float y_;
	float z_;
};

inline Vector3 operator+(const Vector3& v1, const Vector3& v2);
inline Vector3 operator-(const Vector3& v1, const Vector3& v2);
inline Vector3 operator*(const Vector3& v1, const Vector3& v2);
inline Vector3 operator/(const Vector3& v1, const Vector3& v2);
inline Vector3 operator*(const Vector3& v1, float t);
inline Vector3 operator*(float t, const Vector3& v1);
inline Vector3 operator/(const Vector3& v1, float t);

inline float dot(const Vector3& v1, const Vector3& v2);
inline Vector3 cross(const Vector3& v1, const Vector3& v2);