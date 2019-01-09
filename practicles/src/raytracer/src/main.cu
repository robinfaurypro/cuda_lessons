#include <string>
#include <fstream>
#include <iostream>
#include <vector>

#include <raytracer/vector3.h>

void writeBufferAsBMP(
	const std::vector<Vector3>& buffer,
	unsigned int w, unsigned int h,
	std::string filename) {

	unsigned char header[54] = {
		'B','M',0,0,0,0,0,0,0,0,54,0,0,0,40,0,0,0,0,0,0,0,0,0,0,0,1,0,
		24,0,0,0,0,0,0,0,0,0,0x13,0x0B,0,0,0x13,0x0B,0,0,0,0,0,0,0,0,0,0};

	const unsigned int padSize  = (4-3*w%4)%4;
	const unsigned int size = w*h*3 + h*padSize;

	auto fillHeader = [&](unsigned int value, unsigned int index) {
		header[index] = (unsigned char)(value);
		header[index+1u] = (unsigned char)(value>>8u);
		header[index+2u] = (unsigned char)(value>>16u);
		header[index+3u] = (unsigned char)(value>>24u);
	};

	fillHeader(size+54, 2);
	fillHeader(w, 18);
	fillHeader(h, 22);
	fillHeader(size, 34);

	std::ofstream imageout(filename, std::ios::out|std::ios::binary);
	imageout.write((char*)header, sizeof(header));

	const char pad[3] = {0,0,0};
	for (unsigned int y=0; y<h; ++y) {
		for (unsigned int x=0; x<w; ++x) {
			unsigned char pixelc[3]{
				unsigned char(255.99f*buffer[x+w*y].b()),
				unsigned char(255.99f*buffer[x+w*y].g()),
				unsigned char(255.99f*buffer[x+w*y].r()),
			};
			imageout.write((char*)pixelc, 3);
		}
		imageout.write(pad, padSize);
	}
}

int main(int argc, char *argv[])
{
	return 0;
}
