#include <raytracer/vector3.h>

inline Vector3 operator+(const Vector3& v1, const Vector3& v2) {
	return Vector3(v1.x()+v2.x(), v1.y()+v2.y(), v1.z()+v2.z());
}

inline Vector3 operator-(const Vector3& v1, const Vector3& v2) {
	return Vector3(v1.x()-v2.x(), v1.y()-v2.y(), v1.z()-v2.z());
}

inline Vector3 operator*(const Vector3& v1, const Vector3& v2) {
	return Vector3(v1.x()*v2.x(), v1.y()*v2.y(), v1.z()*v2.z());
}

inline Vector3 operator/(const Vector3& v1, const Vector3& v2) {
	return Vector3(v1.x()/v2.x(), v1.y()/v2.y(), v1.z()/v2.z());
}

inline Vector3 operator*(const Vector3& v1, float t) {
	return Vector3(v1.x()*t, v1.y()*t, v1.z()*t);
}

inline Vector3 operator*(float t, const Vector3& v1) {
	return Vector3(v1.x()*t, v1.y()*t, v1.z()*t);
}

inline Vector3 operator/(const Vector3& v1, float t) {
	return Vector3(v1.x()/t, v1.y()/t, v1.z()/t);
}

inline float dot(const Vector3& v1, const Vector3& v2) {
	return v1.x()*v2.x() + v1.y()*v2.y() + v1.z()*v2.z();
}

inline Vector3 cross(const Vector3& v1, const Vector3& v2) {
	return Vector3(
		v1.y()*v2.z() - v1.z()*v2.y(),
		-(v1.x()*v2.z() - v1.z()*v2.x()),
		v1.x()*v2.y() - v1.y()*v2.x());
}
