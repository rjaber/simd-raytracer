# simd-raytracer
Multithreaded sphere ray tracer optimized with SIMD intrinsics

- Ray tracing divided into tasks submitted to a thread pool making full use of  multicore systems
- Ray tracing algorithm is entirely programmed using x86 SSE SIMD intrinsics
- Includes custom memory allocator, memory alignment, and buffer overflow protection
- Multiplatform: Supports Windows and Linux (x86-32 & x86-64)
- Technologies: C++11, STL, SSE SIMD, Multithreading, Custom Memory Allocator, Git

About
=====
This is a sphere ray tracer mostly done for educational purposes. The main focus is on performance
while keeping the ray tracer simple (no algorithmic optimizations related to spatial data structures).
This raytracer makes heavy use of SIMD intrinsic instructions as well as a thread pool to distribute
the ray tracing work evenly over multicores in modern processors.

Screenshots
===========
![Alt text](./docs/screenshots/spheres.jpg?raw=true "Spheres")

Usage
=====
The raytracer application takes two arguments: 
arg1: ascii art file name w/ extention (i.e image.txt) following the format described below or "d" for the default image.
arg2: resolution factor ( 1 <= resolution <= 4 ). If missing, the factor is set to "1" by default, the higher the factor
	  the more rays are shot at the scene yielding a more crispy look.
 
 Examples:
./raytracer d	<-- (resolution is "1" by default)
./raytracer ascii_art.txt 2 
Create a text file and insert a "1" wherever you wish to place a sphere and a space (NOT a tab!) to separate the "1"s.
Also, ensure that the last line is a carriage return. For example, to render a raytraced image with the word "C++",
create a text file containing the following:
```bash
111111  
1           1          1  
1           1          1  
1        1111111    1111111  
1           1          1  
1           1          1  
111111  
```
Note: The ray tracer cannot take more than 255 spheres. Maximum frame size is 64x12 spheres.

Building Instructions
======================
- For a single-threaded ray tracer add the option "-DRT_SINGLE_THREADED" when compiling.
- For buffer overflow detection add the option `-DRT_DEBUG_BUFFER_OVERFLOW".

- Windows MSVC (Tested with MSVC 2013)  
x86:  
```bash
cl -O2 -Oi -Ot -EHsc -nologo raytracer.cpp
```

- Linux GCC (Tested with gcc 4.8) 

x86-32:  
```bash
g++ -m32 -O3 -msse4 -std=c++11 -pthread -DNDEBUG -o raytracer raytracer.cpp 
```

x86-64:  
```bash
g++ -O3 -msse4 -std=c++11 -pthread -DNDEBUG -o raytracer raytracer.cpp 
```
