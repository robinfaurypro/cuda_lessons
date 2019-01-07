#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

__device__
int mod(int a, int b) {
    return ((a%=b)<0)?a+b:a;
}

__global__
void caesarEncode(int N, char* buffer, char* outBuffer, int shiftValue) {
	int stride = blockDim.x * gridDim.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = index; i<N; i += stride) {
		if ('a'<=buffer[i] && buffer[i]<='z' || 'A'<=buffer[i] && buffer[i]<='Z') {
			outBuffer[i] = buffer[i]<='Z'?
				mod((buffer[i]-'A')+shiftValue, 26)+'A':
				mod((buffer[i]-'a')+shiftValue, 26)+'a';
		} else {
			outBuffer[i] = buffer[i];
		}
	}
}

bool testInvariant() {
	bool test_ok = true;
	int N = 32;

	char *buffer, *outBuffer;
	cudaMallocManaged(&buffer, N*sizeof(char));
	cudaMallocManaged(&outBuffer, N*sizeof(char));

	strcpy(buffer, "abcdefghijklmnopqrstuvwxyzABCDZ");

	caesarEncode<<<1, 32>>>(N, buffer, outBuffer, 3);
	cudaDeviceSynchronize();

	if (strcmp(outBuffer, "defghijklmnopqrstuvwxyzabcDEFGC")!=0) {
		test_ok = false;
	}

	caesarEncode<<<1, 32>>>(N, outBuffer, outBuffer, -3);
	cudaDeviceSynchronize();

	if (strcmp(buffer, outBuffer)!=0) {
		test_ok = false;
	}

	std::cout<<"testInvariant: "<<(test_ok?"pass":"fail")<<std::endl;

	cudaFree(buffer);
	cudaFree(outBuffer);

	return test_ok;
}

bool testNonAlphabeticalChar() {
	bool test_ok = true;
	int N = 32;

	char *buffer;
	cudaMallocManaged(&buffer, N*sizeof(char));

	strcpy(buffer, "aAzZ1234[]/*-+$*! .,| HIJ%42 щи");

	caesarEncode<<<1, 32>>>(N, buffer, buffer, 1);
	cudaDeviceSynchronize();

	if (strcmp(buffer, "bBaA1234[]/*-+$*! .,| IJK%42 щи")!=0) {
		test_ok = false;
	}

	std::cout<<"testNonAlphabeticalChar: "<<(test_ok?"pass":"fail")<<std::endl;

	cudaFree(buffer);

	return test_ok;
}

int main(int argc, char *argv[])
{
	testInvariant();
	testNonAlphabeticalChar();

	if (argc<2) {
		std::cout<<"No file specified."<<std::endl;
		return 0;
	}

	std::ifstream file(argv[1], std::ios::in);
	std::string str(static_cast<std::stringstream const&>(std::stringstream()<<file.rdbuf()).str());

	char *buffer;
	cudaMallocManaged(&buffer, str.size()*sizeof(char));

	memcpy(buffer, str.c_str(), str.size());

	caesarEncode<<<32, 256>>>(str.size(), buffer, buffer, 3);

	cudaDeviceSynchronize();

	std::ofstream fileout(std::string(argv[1]) + std::string(".out.txt"), std::ios::out);
	fileout.write(buffer, str.size());

	cudaFree(buffer);

	return 0;
}
