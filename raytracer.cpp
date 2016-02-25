/*
The MIT License (MIT)

Copyright (c) 2016 Rami Jaber 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
About
-----
This is a sphere ray tracer mostly done for educational purposes. The main focus is on performance
while keeping the ray tracer simple (no algorithmic optimizations related to spatial data structures).
This raytracer makes heavy use of SIMD intrinsic instructions as well as a thread pool to distribute
the ray tracing work evenly over multicores in modern processors.

Usage
-----
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

111111
1           1          1
1           1          1
1        1111111    1111111
1           1          1
1           1          1
111111

Note: The ray tracer cannot take more than 255 spheres. Maximum frame size is 64x12 spheres.

Compiling Instructions
----------------------
For a single-threaded ray tracer add the option "-DRT_SINGLE_THREADED" when compiling.
For buffer overflow detection add the option `-DRT_DEBUG_BUFFER_OVERFLOW".

Windows MSVC (Tested with MSVC 2013)
x86: 
>>cl -O2 -Oi -Ot -EHsc -nologo raytracer.cpp

Linux GCC (Tested with gcc 4.8)
x86-32:
>>g++ -m32 -O3 -msse4 -std=c++11 -pthread -DNDEBUG -o raytracer raytracer.cpp 

x86-64:
>>g++ -O3 -msse4 -std=c++11 -pthread -DNDEBUG -o raytracer raytracer.cpp 
*/

#include <assert.h>
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <stdint.h>
#include <string>
#include <cstring>
#include <thread>
#include <vector>

#if defined(_MSC_VER)

#include <intrin.h>

#define VECTOR_CC __vectorcall
#define FORCE_INLINE __forceinline
#define DEBUG_BREAK __debugbreak();

#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))

#include <x86intrin.h>

#define VECTOR_CC 
#define FORCE_INLINE __attribute__((__always_inline__))
#define DEBUG_BREAK __builtin_trap();

#endif

//#define RT_DEBUG_BUFFER_OVERFLOW
#define PI 3.141592653589793238462643
#define SSE_ALIGNMENT 16
#define TRACER_DEPTH 2 

typedef int32_t int32;
typedef uint32_t uint32;
typedef float real;
typedef unsigned char uchar;

typedef uint32 x;
typedef uint32 y;

#if defined(RT_DEBUG_BUFFER_OVERFLOW)
struct SsePos
{
	SsePos(size_t p) : pos(p) {}
	size_t pos;
};
#else
typedef size_t SsePos;
#endif

//DebugIterator is for debugging purposes only, used to check for buffer overflows
template <typename T>
class DebugIterator 
{
public:
	DebugIterator() :
		m_begin(nullptr),
		m_size(0)
	{
	}

	DebugIterator(T* buffer, size_t buffer_size)
	{
		m_begin = buffer;
		assert(buffer_size % sizeof(T) == 0);
		m_size = buffer_size / sizeof(T);
	}

	DebugIterator(const DebugIterator&) = default;
	DebugIterator& operator=(const DebugIterator& o) = default;

	T& operator[](size_t pos)
	{
		//Check if pos is within bounds
		if (pos < 0 || pos > m_size - 1)
			DEBUG_BREAK

		return *(m_begin + pos);
	}

#ifdef RT_DEBUG_BUFFER_OVERFLOW
	T& operator[](SsePos s)
	{
		//Check if pos is within bounds for an sse read (i.e loading/storing 128 bits at a time)
		if (s.pos < 0 || s.pos > m_size - 3)
			DEBUG_BREAK	

		return *(m_begin + s.pos);
	}
#endif

private:
	T *m_begin; //points to first element
	size_t m_size; //size of buffer owned by this Iterator
};

#if defined(RT_DEBUG_BUFFER_OVERFLOW)
template <typename T>
using Iterator = DebugIterator<T>;
#else
template <typename T>
using Iterator = T*;
#endif

static void save_bmp(uint32 width, uint32 height, Iterator<real> pixmap) {
	uint32 pixmapsize = width * height * 4;
	uint32 filesize = 54 + pixmapsize;
	
	uint32 print_resolution = static_cast<uint32>(72 * 39.375f); //72 DPI x factor
	
	uchar bmpfileheader[14] = {'B','M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
	uchar bmpinfoheader[40] = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0};
	
	bmpfileheader[ 2] = (uchar)(filesize);
	bmpfileheader[ 3] = (uchar)(filesize >> 8);
	bmpfileheader[ 4] = (uchar)(filesize >> 16);
	bmpfileheader[ 5] = (uchar)(filesize >> 24);
	
	bmpinfoheader[ 4] = (uchar)(width);
	bmpinfoheader[ 5] = (uchar)(width >> 8);
	bmpinfoheader[ 6] = (uchar)(width >> 16);
	bmpinfoheader[ 7] = (uchar)(width >> 24);
	
	bmpinfoheader[ 8] = (uchar)(height);
	bmpinfoheader[ 9] = (uchar)(height >> 8);
	bmpinfoheader[10] = (uchar)(height >> 16);
	bmpinfoheader[11] = (uchar)(height >> 24);
	
	bmpinfoheader[21] = (uchar)(pixmapsize);
	bmpinfoheader[22] = (uchar)(pixmapsize >> 8);
	bmpinfoheader[23] = (uchar)(pixmapsize >> 16);
	bmpinfoheader[24] = (uchar)(pixmapsize >> 24);
	
	bmpinfoheader[25] = (uchar)(print_resolution);
	bmpinfoheader[26] = (uchar)(print_resolution >> 8);
	bmpinfoheader[27] = (uchar)(print_resolution >> 16);
	bmpinfoheader[28] = (uchar)(print_resolution >> 24);
	
	bmpinfoheader[29] = (uchar)(print_resolution);
	bmpinfoheader[30] = (uchar)(print_resolution >> 8);
	bmpinfoheader[31] = (uchar)(print_resolution >> 16);
	bmpinfoheader[32] = (uchar)(print_resolution >> 24);

	std::ofstream ofs("./image.bmp", std::ios::out | std::ios::binary);

	ofs.write((char*)bmpfileheader, 14);
	ofs.write((char*)bmpinfoheader, 40);

	uint32 disp;
	uint32 disp_adjusted = width * height * 3;

	for (uint32 y = height; y > 0; --y) {
		disp_adjusted -= (width * 3);
		disp = disp_adjusted;
		for (uint32 x = 0; x < width - 3; x += 4) {
			ofs << (uchar)(std::min(1.0f, pixmap[disp + 8]) * 255) << 
				(uchar)(std::min(1.0f, pixmap[disp + 4]) * 255) <<
				(uchar)(std::min(1.0f, pixmap[disp + 0]) * 255) <<

				(uchar)(std::min(1.0f, pixmap[disp + 9]) * 255) <<
				(uchar)(std::min(1.0f, pixmap[disp + 5]) * 255) <<
				(uchar)(std::min(1.0f, pixmap[disp + 1]) * 255) <<

				(uchar)(std::min(1.0f, pixmap[disp + 10]) * 255) <<
				(uchar)(std::min(1.0f, pixmap[disp + 6]) * 255) <<
				(uchar)(std::min(1.0f, pixmap[disp + 2]) * 255) <<

				(uchar)(std::min(1.0f, pixmap[disp + 11]) * 255) <<
				(uchar)(std::min(1.0f, pixmap[disp + 7]) * 255) <<
				(uchar)(std::min(1.0f, pixmap[disp + 3]) * 255);

			disp += 12;
		}
	}

	ofs.close();
}

static void save_ppm(uint32 width, uint32 height, Iterator<real> pixmap)
{
	std::ofstream ofs("./image.ppm", std::ios::out | std::ios::binary);
	ofs << "P6\n" << width << " " << height << "\n255\n";

	int disp = 0;

	for (uint32 y = 0; y < height; ++y) {
		for (uint32 x = 0; x < width - 3; x += 4) {
			ofs << (uchar)(std::min(1.0f, pixmap[disp + 0]) * 255) << 
				(uchar)(std::min(1.0f, pixmap[disp + 4]) * 255) <<
				(uchar)(std::min(1.0f, pixmap[disp + 8]) * 255) <<

				(uchar)(std::min(1.0f, pixmap[disp + 1]) * 255) <<
				(uchar)(std::min(1.0f, pixmap[disp + 5]) * 255) <<
				(uchar)(std::min(1.0f, pixmap[disp + 9]) * 255) <<

				(uchar)(std::min(1.0f, pixmap[disp + 2]) * 255) <<
				(uchar)(std::min(1.0f, pixmap[disp + 6]) * 255) <<
				(uchar)(std::min(1.0f, pixmap[disp + 10]) * 255) <<

				(uchar)(std::min(1.0f, pixmap[disp + 3]) * 255) <<
				(uchar)(std::min(1.0f, pixmap[disp + 7]) * 255) <<
				(uchar)(std::min(1.0f, pixmap[disp + 11]) * 255);

			disp += 12;
		}
	}

	ofs.close();
}

//Used for profiling via instrumentation
class Instrument {
public:
	explicit Instrument(const char *c, bool block_mode = false) :
		m_block_name(c), 
		m_counter(0), 
		m_block_mode(block_mode)
	{
		m_tlast = std::chrono::high_resolution_clock::now();
	}

	Instrument(const Instrument&) = delete;
	Instrument& operator=(const Instrument&) = delete;

	void now(const char* c)
	{
		std::cout << m_block_name << "(" << m_counter << ")" << "[ " << c << "]: " << 
			std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - m_tlast).count() << " s" << std::endl;
		++m_counter;
		m_tlast = std::chrono::high_resolution_clock::now();
	}

	~Instrument()
	{
		if (m_block_mode) {
			std::cout << m_block_name << "[END]: " << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - m_tlast).count() << " s" << std::endl;
		}
	}

private:
	const char *m_block_name;
	uint32 m_counter;
	bool m_block_mode;
	std::chrono::high_resolution_clock::time_point m_tlast;
};

class Task {
public:
	Task(std::function<void(void*)> f, void *args)
	{
		m_task = std::move(f);
		m_args = args;
	}

	void run()
	{
		m_task(m_args);
	}

private:
	std::function<void(void*)> m_task;
	void *m_args;
};

class ThreadPool {
public:
	explicit ThreadPool(uint32 tasks_size, uint32 thread_num = 0) :
		m_tasks_size(tasks_size)
	{
		m_tasks_completed.store(0);
		uint32 pool_size = thread_num == 0 ? std::thread::hardware_concurrency() : thread_num;

		for (uint32 i = 0; i < pool_size; ++i) {
			worker_threads.push_back(std::thread(&ThreadPool::worker_thread, this));
		}
	}

	~ThreadPool()
	{
	}

	void join()
	{
		std::for_each(worker_threads.begin(), worker_threads.end(), std::mem_fn(&std::thread::join));
	}

	void worker_thread()
	{
		while (++m_tasks_completed <= m_tasks_size) {
			Task* task = nullptr;
			wait_and_dequeue_task(&task);
			task->run();
		}
	}

	void enqueue_task(Task *t)
	{
		std::lock_guard<std::mutex> guard(queue_mutex);
		task_queue.push(t);
		queue_cond.notify_one();
	}

	bool wait_and_dequeue_task(Task **t)
	{
		std::unique_lock<std::mutex> guard(queue_mutex);
		if (task_queue.empty())
			queue_cond.wait(guard, [this] {return !task_queue.empty(); });

		*t = task_queue.front();
		task_queue.pop();
		return true;
	}

private:
	std::vector<std::thread> worker_threads;
	std::queue<Task*> task_queue;
	std::mutex queue_mutex;
	std::condition_variable queue_cond;
	std::atomic<int32> m_tasks_completed;
	int32 m_tasks_size;
};

class LinearAllocator
{
public:
	LinearAllocator() :
	m_clear_on_destruction(false),
	m_size(0),
	m_bottom(0),
	m_top(0),
	m_allocated_memory(0),
	m_num_allocations(0)
	{
	}

	LinearAllocator(const LinearAllocator&) = delete;
	LinearAllocator& operator=(const LinearAllocator&) = delete;

	~LinearAllocator()
	{
		if (m_clear_on_destruction) {
			delete[] reinterpret_cast<uchar*>(m_top);
		}
	}

	void init(size_t size, bool initialize_to_zero = false)
	{
		assert(size != 0);
		m_size = size;
		m_bottom = m_top = static_cast<void*>(new uchar[size]);

		if (initialize_to_zero) {
			memset(m_top, 0, size);
		}
	}

	void* allocate(size_t size, uchar alignment = 0)
	{
		uchar alignment_padding = 0;
		void *aligned_address = m_top;

		//Align address if necessary 
		if (alignment != 0 && (reinterpret_cast<size_t>(aligned_address) & (alignment - 1))) {
			aligned_address = reinterpret_cast<void*>((reinterpret_cast<size_t>(aligned_address) + (alignment - 1)) & ~(alignment - 1));
			alignment_padding = static_cast<uchar>((size_t)aligned_address - (size_t)m_top);
		}

		m_allocated_memory += size + alignment_padding;
		++m_num_allocations;
		m_top = static_cast<void*>(static_cast<uchar*>(m_top) + size + alignment_padding);
		assert((size_t)m_top - (size_t)m_bottom <= m_size);

		return aligned_address;
	}

private:
	void *m_bottom;
	void *m_top;
	size_t m_size;
	size_t m_allocated_memory;
	size_t m_num_allocations;
	bool m_clear_on_destruction;
};

template <typename T>
FORCE_INLINE Iterator<T> init_itr(T* mem, size_t allocated_size)
{
#ifdef RT_DEBUG_BUFFER_OVERFLOW
	Iterator<T> itr(mem, allocated_size);
	return itr;
#else
	return mem;
#endif
}

//Ray Tracer Memory 
struct 
{
	LinearAllocator allocator;
	Iterator<real> pixmap;

	struct {
		Iterator<real> x;
		Iterator<real> y;
		Iterator<real> z;
	} cam_ray;

	struct {
		Iterator<real> x;
		Iterator<real> y;
		Iterator<real> z;

		Iterator<real> radius;

		Iterator<real> r;
		Iterator<real> g;
		Iterator<real> b;
	} spheres;

	struct lights {
		Iterator<real> x;
		Iterator<real> y;
		Iterator<real> z;
		Iterator<real> strength;
	} lights;

	struct {
		real r;
		real g;
		real b;
	} background_color;

	real global_ambiance;
	uint32 width;
	uint32 height;
	uint32 num_of_spheres;
	uint32 num_of_lights;
	 
	void init()
	{
		const int bytes_per_rgb_pixel = 3 * sizeof(real);
		const uint32 pixels_heap_size = (width * height * bytes_per_rgb_pixel);

		const int cam_xyz_buffers = sizeof(cam_ray) / sizeof(real);
		const uint32 cam_rays_heap_size = ((width * height) * cam_xyz_buffers) * sizeof(real);
		const uint32 cam_rays_buffer_size = cam_rays_heap_size / cam_xyz_buffers;

		const int sphere_buffers = sizeof(spheres) / sizeof(real);
		const uint32 sphere_coords_size = (num_of_spheres) * sphere_buffers * sizeof(real);
		const uint32 sphere_buffer_size = sphere_coords_size / sphere_buffers;

		const int lights_components = sizeof(lights) / sizeof(real);
		const uint32 lights_size = num_of_lights * lights_components * sizeof(real);
		const uint32 lights_component_size = lights_size / lights_components;

		const size_t main_arena_size = pixels_heap_size + cam_rays_heap_size + sphere_coords_size + lights_size;
		const uint32 ALIGNMENT_PADDING = 1024 * 1024 * 10;
		allocator.init(main_arena_size + ALIGNMENT_PADDING, true);

		//Allign ptrs to 16 bytes for easy access by SIMD load/store instructions
		pixmap = init_itr<real>((real*)allocator.allocate(pixels_heap_size, SSE_ALIGNMENT), pixels_heap_size);

		cam_ray.x = init_itr<real>((real*)allocator.allocate(cam_rays_buffer_size, SSE_ALIGNMENT), cam_rays_buffer_size);
		cam_ray.y = init_itr<real>((real*)allocator.allocate(cam_rays_buffer_size, SSE_ALIGNMENT), cam_rays_buffer_size);
		cam_ray.z = init_itr<real>((real*)allocator.allocate(cam_rays_buffer_size, SSE_ALIGNMENT), cam_rays_buffer_size);

		spheres.x = init_itr<real>((real*)allocator.allocate(sphere_buffer_size, SSE_ALIGNMENT), sphere_buffer_size);
		spheres.y = init_itr<real>((real*)allocator.allocate(sphere_buffer_size, SSE_ALIGNMENT), sphere_buffer_size);
		spheres.z = init_itr<real>((real*)allocator.allocate(sphere_buffer_size, SSE_ALIGNMENT), sphere_buffer_size);
		spheres.radius = init_itr<real>((real*)allocator.allocate(sphere_buffer_size), sphere_buffer_size);
		spheres.r = init_itr<real>((real*)allocator.allocate(sphere_buffer_size), sphere_buffer_size);
		spheres.g = init_itr<real>((real*)allocator.allocate(sphere_buffer_size), sphere_buffer_size);
		spheres.b = init_itr<real>((real*)allocator.allocate(sphere_buffer_size), sphere_buffer_size);

		lights.x = init_itr<real>((real*)allocator.allocate(lights_component_size), lights_component_size);
		lights.y = init_itr<real>((real*)allocator.allocate(lights_component_size), lights_component_size);
		lights.z = init_itr<real>((real*)allocator.allocate(lights_component_size), lights_component_size);
		lights.strength = init_itr<real>((real*)allocator.allocate(lights_component_size), lights_component_size);
	}
} rtm;

struct sse_pixel
{
	__m128 r;
	__m128 g;
	__m128 b;
};

struct sse_ray
{
	__m128 x;
	__m128 y;
	__m128 z;
};

struct ray_sphere_interesection
{
	__m128 tpoint;
	__m128 index;
};

static inline  __m128 VECTOR_CC dot(sse_ray a, sse_ray b)
{
	__m128 ret;
	sse_ray tmp;

	tmp.x = _mm_mul_ps(a.x, b.x);
	tmp.y = _mm_mul_ps(a.y, b.y);
	tmp.z = _mm_mul_ps(a.z, b.z);
	ret = _mm_add_ps(tmp.x, tmp.y);
	ret = _mm_add_ps(ret, tmp.z);

	return ret;
}

static inline __m128 VECTOR_CC length(sse_ray r)
{
	__m128 x2 = _mm_mul_ps(r.x, r.x);
	__m128 y2 = _mm_mul_ps(r.y, r.y);
	__m128 z2 = _mm_mul_ps(r.z, r.z);

	__m128 len = _mm_add_ps(x2, y2);
	len = _mm_add_ps(len, z2);
	len = _mm_sqrt_ps(len);

	return len;
}

static inline void normalize(sse_ray &r)
{
	__m128 x2 = _mm_mul_ps(r.x, r.x);
	__m128 y2 = _mm_mul_ps(r.y, r.y);
	__m128 z2 = _mm_mul_ps(r.z, r.z);

	__m128 len = _mm_add_ps(x2, y2);
	len = _mm_add_ps(len, z2);
	//len = _mm_rsqrt_ps(len); //Relative error too high causing image artifacts, will opt for _mm_sqrt_ps() and _mm_div_ps()  
	len = _mm_sqrt_ps(len);

	r.x = _mm_div_ps(r.x, len);
	r.y = _mm_div_ps(r.y, len);
	r.z = _mm_div_ps(r.z, len);
}

static ray_sphere_interesection VECTOR_CC intersect_ray_spheres(sse_ray rorigin, sse_ray rdir)
{
	ray_sphere_interesection rs;
	rs.tpoint = _mm_set_ps1(FLT_MAX);
	rs.index = _mm_set_ps1(-1.0f);

	sse_ray rlength;
	__m128 no_intersection;

	for (uint32 i = 0; i < rtm.num_of_spheres; ++i) {
		sse_ray sphere_pos;
		sphere_pos.x = _mm_set_ps1(rtm.spheres.x[i]);
		sphere_pos.y = _mm_set_ps1(rtm.spheres.y[i]);
		sphere_pos.z = _mm_set_ps1(rtm.spheres.z[i]);

		__m128 radius2 = _mm_set_ps1(rtm.spheres.radius[i]);
		radius2 = _mm_mul_ps(radius2, radius2);
	
		rlength.x = _mm_sub_ps(sphere_pos.x, rorigin.x);
		rlength.y = _mm_sub_ps(sphere_pos.y, rorigin.y);
		rlength.z = _mm_sub_ps(sphere_pos.z, rorigin.z);

		__m128 lendotdir = dot(rlength, rdir);
		no_intersection = _mm_cmplt_ps(lendotdir, _mm_set_ps1(0.0f));


		__m128 d2 = _mm_sub_ps(dot(rlength, rlength), _mm_mul_ps(lendotdir, lendotdir));
		no_intersection = _mm_or_ps(no_intersection, _mm_cmpgt_ps(d2, radius2));

		if (_mm_test_all_ones(_mm_castps_si128(no_intersection)))
			continue;

		__m128 dprojrdir = _mm_sub_ps(radius2, d2);
		dprojrdir = _mm_sqrt_ps(dprojrdir);

		__m128 tnear = _mm_sub_ps(lendotdir, dprojrdir);
		__m128 tfar = _mm_add_ps(lendotdir, dprojrdir);

		__m128 tpoint = _mm_cmpgt_ps(tnear, _mm_set_ps1(0.0f));
		tnear = _mm_blendv_ps(tfar, tnear, tpoint);

		tpoint = _mm_cmplt_ps(tnear, rs.tpoint);
		__m128 add_intersection = _mm_andnot_ps(no_intersection, tpoint);

		rs.tpoint = _mm_blendv_ps(rs.tpoint, tnear, add_intersection);
		rs.index = _mm_blendv_ps(rs.index, _mm_set_ps1(static_cast<float>(i)), add_intersection);
	}

	return rs;
}

static sse_pixel VECTOR_CC trace_ray(sse_ray rorigin, sse_ray rdir, int32 depth)
{
	sse_pixel pixel;
	ray_sphere_interesection irs = intersect_ray_spheres(rorigin, rdir);

	__m128i test_intersection = _mm_castps_si128(_mm_cmpeq_ps(irs.index, _mm_set_ps1(-1.0f)));
	int32 no_intersections = _mm_test_all_ones(test_intersection);
	if (no_intersections) {
		pixel.r = _mm_set_ps1(rtm.background_color.r);
		pixel.g = _mm_set_ps1(rtm.background_color.g);
		pixel.b = _mm_set_ps1(rtm.background_color.b);
	} else {
		pixel.r = _mm_set_ps1(0.0f);
		pixel.g = _mm_set_ps1(0.0f);
		pixel.b = _mm_set_ps1(0.0f);

		sse_ray phit; // point of intersection
		phit.x = _mm_add_ps(rorigin.x, _mm_mul_ps(rdir.x, irs.tpoint));
		phit.y = _mm_add_ps(rorigin.y, _mm_mul_ps(rdir.y, irs.tpoint));
		phit.z = _mm_add_ps(rorigin.z, _mm_mul_ps(rdir.z, irs.tpoint));

		uint32 sphere_idx = 0;
		__m128i idxi = _mm_cvtps_epi32(irs.index);
		idxi = _mm_andnot_si128(test_intersection, idxi);
		uint32 *ptr = (uint32*)&idxi;
		sphere_idx = (ptr[0] & 0xff) | ((ptr[1] & 0xff) << 8) | ((ptr[2] & 0xff) << 16) | ((ptr[3] & 0xff) << 24);

		sse_ray sphere;
		Iterator<real> sphx_ptr = rtm.spheres.x;

		real x1 = sphx_ptr[(sphere_idx & 0xff)];
		real x2 = sphx_ptr[(sphere_idx >> 8) & 0xff];		
		real x3 = sphx_ptr[(sphere_idx >> 16) & 0xff];
		real x4 = sphx_ptr[(sphere_idx >> 24) & 0xff];

		sphere.x = _mm_set_ps(x4, x3, x2, x1);

		Iterator<real> sphy_ptr = rtm.spheres.y;

		real y1 = sphy_ptr[(sphere_idx & 0xff)];
		real y2 = sphy_ptr[(sphere_idx >> 8) & 0xff];		
		real y3 = sphy_ptr[(sphere_idx >> 16) & 0xff];
		real y4 = sphy_ptr[(sphere_idx >> 24) & 0xff];

		sphere.y = _mm_set_ps(y4, y3, y2, y1);

		Iterator<real> sphz_ptr = rtm.spheres.z;

		real z1 = sphz_ptr[(sphere_idx & 0xff)];
		real z2 = sphz_ptr[(sphere_idx >> 8) & 0xff];		
		real z3 = sphz_ptr[(sphere_idx >> 16) & 0xff];
		real z4 = sphz_ptr[(sphere_idx >> 24) & 0xff];

		sphere.z = _mm_set_ps(z4, z3, z2, z1);

		sse_ray nhit; // normal at the intersection point
		nhit.x = _mm_sub_ps(phit.x, sphere.x);
		nhit.y = _mm_sub_ps(phit.y, sphere.y);
		nhit.z = _mm_sub_ps(phit.z, sphere.z);

		normalize(nhit);
		
		//Flip normal if it has the same direction as the ray
		__m128 negate_mask = _mm_cmpgt_ps(dot(rdir, nhit), _mm_set_ps1(0.0f));
		negate_mask = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(negate_mask), 31));
		nhit.x = _mm_xor_ps(nhit.x, negate_mask);
		nhit.y = _mm_xor_ps(nhit.y, negate_mask);
		nhit.z = _mm_xor_ps(nhit.z, negate_mask);

		sse_pixel sphere_color;

		for (uint32 i = 0; i < rtm.num_of_lights; ++i) {
			sse_ray lightpos;
			lightpos.x = _mm_set_ps1(rtm.lights.x[i]);
			lightpos.y = _mm_set_ps1(rtm.lights.y[i]);
			lightpos.z = _mm_set_ps1(rtm.lights.z[i]);

			sse_ray lightdir;
			lightdir.x = _mm_sub_ps(lightpos.x, phit.x);
			lightdir.y = _mm_sub_ps(lightpos.y, phit.y);
			lightdir.z = _mm_sub_ps(lightpos.z, phit.z);
			
			__m128 lightdist = length(lightdir);

			normalize(lightdir);

			for (uint32 j = 0; j < rtm.num_of_spheres; j += 4) {
				sse_ray rlight;
				rlight.x = phit.x;
				rlight.y = phit.y;
				rlight.z = phit.z;

				ray_sphere_interesection lsi = intersect_ray_spheres(rlight, lightdir);

				__m128i light_intersection = _mm_castps_si128(_mm_and_ps(_mm_cmpneq_ps(lsi.index, _mm_set_ps1(-1.0f)), _mm_cmplt_ps(lsi.tpoint, lightdist)));
				__m128 lightcos = _mm_set_ps1(0.0f);
				lightcos = _mm_max_ps(lightcos, dot(nhit, lightdir));

				__m128 lightval = _mm_mul_ps(lightcos, _mm_set_ps1(rtm.lights.strength[i]));
				lightval = _mm_andnot_ps(_mm_castsi128_ps(light_intersection), lightval);
				lightval = _mm_andnot_ps(_mm_castsi128_ps(test_intersection), lightval);
				
				real r1 = rtm.spheres.r[(sphere_idx & 0xff)];
				real r2 = rtm.spheres.r[(sphere_idx >> 8) & 0xff];		
				real r3 = rtm.spheres.r[(sphere_idx >> 16) & 0xff];
				real r4 = rtm.spheres.r[(sphere_idx >> 24) & 0xff];

				sphere_color.r = _mm_set_ps(r4, r3, r2, r1);
				sphere_color.r = _mm_andnot_ps(_mm_castsi128_ps(test_intersection), sphere_color.r);
      
				real g1 = rtm.spheres.g[(sphere_idx & 0xff)];
				real g2 = rtm.spheres.g[(sphere_idx >> 8) & 0xff];		
				real g3 = rtm.spheres.g[(sphere_idx >> 16) & 0xff];
				real g4 = rtm.spheres.g[(sphere_idx >> 24) & 0xff];

				sphere_color.g = _mm_set_ps(g4, g3, g2, g1);
				sphere_color.g = _mm_andnot_ps(_mm_castsi128_ps(test_intersection), sphere_color.g);

				real b1 = rtm.spheres.b[(sphere_idx & 0xff)];
				real b2 = rtm.spheres.b[(sphere_idx >> 8) & 0xff];		
				real b3 = rtm.spheres.b[(sphere_idx >> 16) & 0xff];
				real b4 = rtm.spheres.b[(sphere_idx >> 24) & 0xff];

				sphere_color.b = _mm_set_ps(b4, b3, b2, b1);
				sphere_color.b = _mm_andnot_ps(_mm_castsi128_ps(test_intersection), sphere_color.b);

				//Add Ambience Effect
				pixel.r = _mm_mul_ps(sphere_color.r, _mm_set_ps1(rtm.global_ambiance));
				pixel.g = _mm_mul_ps(sphere_color.g, _mm_set_ps1(rtm.global_ambiance));
				pixel.b = _mm_mul_ps(sphere_color.b, _mm_set_ps1(rtm.global_ambiance));

				pixel.r = _mm_add_ps(_mm_mul_ps(sphere_color.r, lightval), pixel.r);
				pixel.g = _mm_add_ps(_mm_mul_ps(sphere_color.g, lightval), pixel.g);
				pixel.b = _mm_add_ps(_mm_mul_ps(sphere_color.b, lightval), pixel.b);
			}
		}

		__m128 bg_mask = _mm_and_ps(_mm_castsi128_ps(test_intersection), _mm_set_ps1(1.0f));
		pixel.r = _mm_or_ps(_mm_and_ps(_mm_set_ps1(rtm.background_color.r), bg_mask), pixel.r);
		pixel.g = _mm_or_ps(_mm_and_ps(_mm_set_ps1(rtm.background_color.g), bg_mask), pixel.g);
		pixel.b = _mm_or_ps(_mm_and_ps(_mm_set_ps1(rtm.background_color.b), bg_mask), pixel.b);
		
		if (depth < TRACER_DEPTH) {
			__m128 facingratio = dot(rdir, nhit);
			facingratio = _mm_mul_ps(facingratio, _mm_set_ps1(-1.0f));

			__m128 fresneleffect = _mm_sub_ps(_mm_set_ps1(1.0f), facingratio);
			fresneleffect = _mm_mul_ps(fresneleffect, fresneleffect);
			fresneleffect = _mm_mul_ps(fresneleffect, fresneleffect);
			const real mixing_value = 0.2f;
			fresneleffect = _mm_mul_ps(fresneleffect, _mm_set_ps1(1.0f - 0.2f));
			fresneleffect = _mm_add_ps(fresneleffect, _mm_set_ps1(mixing_value));

			sse_ray refldir;
			__m128 rdirdotnhit = dot(rdir, nhit);
			refldir.x = _mm_sub_ps(rdir.x, _mm_mul_ps(_mm_mul_ps(nhit.x, rdirdotnhit), _mm_set_ps1(2.0f)));
			refldir.y = _mm_sub_ps(rdir.y, _mm_mul_ps(_mm_mul_ps(nhit.y, rdirdotnhit), _mm_set_ps1(2.0f)));
			refldir.z = _mm_sub_ps(rdir.z, _mm_mul_ps(_mm_mul_ps(nhit.z, rdirdotnhit), _mm_set_ps1(2.0f)));

			normalize(refldir);

			sse_pixel reflpixel = trace_ray(phit, refldir, depth + 1);

			__m128 fresnel_r = _mm_mul_ps(reflpixel.r, fresneleffect);
			__m128 fresnel_g = _mm_mul_ps(reflpixel.g, fresneleffect);
			__m128 fresnel_b = _mm_mul_ps(reflpixel.b, fresneleffect);

			fresnel_r = _mm_mul_ps(sphere_color.r, _mm_andnot_ps(_mm_castsi128_ps(test_intersection), fresnel_r));
			fresnel_g = _mm_mul_ps(sphere_color.g, _mm_andnot_ps(_mm_castsi128_ps(test_intersection), fresnel_g));
			fresnel_b = _mm_mul_ps(sphere_color.b, _mm_andnot_ps(_mm_castsi128_ps(test_intersection), fresnel_b));

			pixel.r = _mm_add_ps(pixel.r, fresnel_r);
			pixel.g = _mm_add_ps(pixel.g, fresnel_g);
			pixel.b = _mm_add_ps(pixel.b, fresnel_b);
		}
	}

	return pixel;
}

static void VECTOR_CC trace_section(sse_ray rorigin, uint32 width, uint32 height, int ystart)
{
	sse_ray rdir;
	sse_pixel pixel;
	uint32 disp = width*ystart*3;

	for (uint32 y = ystart; y < height; ++y) {
		for (uint32 x = 0; x < width - 3; x += 4) {
			rdir.x = _mm_load_ps(&rtm.cam_ray.x[SsePos(y*width + x)]);
			rdir.y = _mm_load_ps(&rtm.cam_ray.y[SsePos(y*width + x)]);
			rdir.z = _mm_load_ps(&rtm.cam_ray.z[SsePos(y*width + x)]);

			pixel = trace_ray(rorigin, rdir, 0);

			_mm_store_ps(&rtm.pixmap[SsePos(disp)], pixel.r);
			disp += 4; 
			_mm_store_ps(&rtm.pixmap[SsePos(disp)], pixel.g);
			disp += 4;
			_mm_store_ps(&rtm.pixmap[SsePos(disp)], pixel.b);
			disp += 4;
		}
	}
}

struct trace_section_data {
	sse_ray rorigin;
	uint32 width;
	uint32 height;
	uint32 ystart;
};

static void trace_section_task(void *args)
{
	const trace_section_data *d = static_cast<trace_section_data*>(args);
	sse_ray rdir;
	sse_pixel pixel;
	uint32 disp = d->width * d->ystart * 3;

	for (uint32 y = d->ystart; y < d->height; ++y) {
		for (uint32 x = 0; x < d->width - 3; x += 4) {
			rdir.x = _mm_load_ps(&rtm.cam_ray.x[SsePos(y * d->width + x)]);
			rdir.y = _mm_load_ps(&rtm.cam_ray.y[SsePos(y * d->width + x)]);
			rdir.z = _mm_load_ps(&rtm.cam_ray.z[SsePos(y * d->width + x)]);

			pixel = trace_ray(d->rorigin, rdir, 0);

			_mm_store_ps(&rtm.pixmap[SsePos(disp)], pixel.r);
			disp += 4;
			_mm_store_ps(&rtm.pixmap[SsePos(disp)], pixel.g);
			disp += 4;
			_mm_store_ps(&rtm.pixmap[SsePos(disp)], pixel.b);
			disp += 4;
		}
	}
}

int main(int argc, char *argv[])
{
	std::string ascii_filename;
	bool trace_default = false;
	uint32 xresolution = 1;

	if (argc == 1) {
		std::cout << "Hello there! What is the name of the ascii file you wish to ray trace?" << std::endl;
		getline(std::cin, ascii_filename);
	} else if (argc > 1) {
		if (*(argv[1]) == 'd') {
			trace_default = true;
		} else {
			ascii_filename = argv[1];
		}

		if (argc > 2) {
			int32 xres = atoi(argv[2]);
			xresolution = xres > 1 ? xres : 1;

			assert(xresolution <= 4);
			if (xresolution > 4) {
				std::cout << "The maximum resolution is 9" << std::endl;
				return 0;
			}
		}
	}

	std::ifstream in;
	if (!trace_default) {
		in.open(ascii_filename);
		if (/*!in && */in.fail()) {
			char answer;
			std::cout << "Sorry, failed to read given file. Please make sure the file name is spelled correctly along with the file extention." << std::endl;
			std::cout << "Generate default image [y/n]? ";
			std::cin >> answer;
			if (answer == 'n') {
				return 0;
			} else {
				trace_default = true;
			}
		}
	}

	const int32 EXTRA_SPHERES = 2; //Background sphere and floor sphere
	rtm.width = 1920 * xresolution;
	rtm.height = 1020 * xresolution;
	rtm.num_of_lights = 1;
	uint32 num_of_spheres = 0;
	uint32 idx = 0;

	if (!trace_default) {
		std::string sphere_layout((std::istreambuf_iterator<char>(in)), (std::istreambuf_iterator<char>()));
		uint32 canvas_width = 0;
		uint32 canvas_height = 0;
		int32 width_counter = 0;
		int32 height_counter = 0;
		std::vector<std::pair<x, y>> sphere_coords;

		std::for_each(sphere_layout.begin(), sphere_layout.end(), [&canvas_height](char c) { if (c == '\n') ++canvas_height; });
		std::for_each(sphere_layout.begin(), sphere_layout.end(), [&](char c)
		{
			if (c == '1' || c == '0' || c == ' ' || c == '\n') {
				if (c == '1') {
					sphere_coords.push_back(std::make_pair(width_counter, canvas_height - height_counter));
					++num_of_spheres;
				} else if (c == '\n' && width_counter != 0) {
					++height_counter;
					canvas_width = std::max((uint32)(width_counter), canvas_width);
					width_counter = 0;
					return;
				}
			}

			++width_counter;
		});

		assert(num_of_spheres < 256);
		if (num_of_spheres > 256) {
			std::cout << "Cannot generate more than 255 spheres, please decrease the number of spheres to meet this limit" << std::endl;
			return 0;
		}

		rtm.num_of_spheres = num_of_spheres + EXTRA_SPHERES;
		rtm.init();

		const real xleft = -((canvas_width + 1) / 2.0f);
		const real xright = -xleft;
		const real zdepth = xleft;

		for (auto itr = sphere_coords.begin(); itr != sphere_coords.end(); ++idx, ++itr) {
			rtm.spheres.x[idx] = xleft + ((itr->second*canvas_width + itr->first) % canvas_width) * ((xright - xleft) / canvas_width);
			rtm.spheres.y[idx] = real(itr->second);
			rtm.spheres.z[idx] = zdepth;

			rtm.spheres.radius[idx] = 0.5f;

			if (itr->second % 2) {
				rtm.spheres.r[idx] = 1.0f;
				rtm.spheres.g[idx] = 1.0f;
				rtm.spheres.b[idx] = 1.0f;
			}
			else {
				rtm.spheres.r[idx] = 1.0f;
				rtm.spheres.g[idx] = 1.0f;
				rtm.spheres.b[idx] = 1.0f;
			}
		}
	} else {
		num_of_spheres = 9;
		rtm.num_of_spheres = num_of_spheres + EXTRA_SPHERES;
		rtm.init();

		rtm.spheres.x[idx] = 16.0f;
		rtm.spheres.y[idx] = 0.0f;
		rtm.spheres.z[idx] = -15.0f;
		rtm.spheres.radius[idx] = 4.0;
		rtm.spheres.r[idx] = 0.2f;
		rtm.spheres.g[idx] = 1.0f;
		rtm.spheres.b[idx++] = 0.2f;

		rtm.spheres.x[idx] = 12.0f;
		rtm.spheres.y[idx] = -2.0f;
		rtm.spheres.z[idx] = -5.0f;
		rtm.spheres.radius[idx] = 2.0;
		rtm.spheres.r[idx] = 0.75f;
		rtm.spheres.g[idx] = 0.0f;
		rtm.spheres.b[idx++] = 0.0f;

		rtm.spheres.x[idx] = -5.0f;
		rtm.spheres.y[idx] = -2.0f;
		rtm.spheres.z[idx] = -2.0f;
		rtm.spheres.radius[idx] = 2.0;
		rtm.spheres.r[idx] = 0.0f;
		rtm.spheres.g[idx] = 0.0f;
		rtm.spheres.b[idx++] = 1.0f;

		rtm.spheres.x[idx] = 3.0f;
		rtm.spheres.y[idx] = -2.0f;
		rtm.spheres.z[idx] = -1.0f;
		rtm.spheres.radius[idx] = 2.0;
		rtm.spheres.r[idx] = 0.9f;
		rtm.spheres.g[idx] = 0.76f;
		rtm.spheres.b[idx++] = 0.46f;

		rtm.spheres.x[idx] = 2.0f;
		rtm.spheres.y[idx] = -2.0f;
		rtm.spheres.z[idx] = -10.0f;
		rtm.spheres.radius[idx] = 2.0;
		rtm.spheres.r[idx] = 1.0f;
		rtm.spheres.g[idx] = 0.0f;
		rtm.spheres.b[idx++] = 0.0f;

		rtm.spheres.x[idx] = -4.0f;
		rtm.spheres.y[idx] = -2.0f;
		rtm.spheres.z[idx] = -10.0f;
		rtm.spheres.radius[idx] = 2.0;
		rtm.spheres.r[idx] = 1.0f;
		rtm.spheres.g[idx] = 1.0f;
		rtm.spheres.b[idx++] = 1.0f;

		rtm.spheres.x[idx] = -10.0f;
		rtm.spheres.y[idx] = 4.0f;
		rtm.spheres.z[idx] = -10.0f;
		rtm.spheres.radius[idx] = 2.0;
		rtm.spheres.r[idx] = 1.0f;
		rtm.spheres.g[idx] = 0.0f;
		rtm.spheres.b[idx++] = 0.0f;

		rtm.spheres.x[idx] = -16.0f;
		rtm.spheres.y[idx] = 2.0f;
		rtm.spheres.z[idx] = -15.0f;
		rtm.spheres.radius[idx] = 4.0;
		rtm.spheres.r[idx] = 1.0f;
		rtm.spheres.g[idx] = 0.0f;
		rtm.spheres.b[idx++] = 0.0f;

		rtm.spheres.x[idx] = 15.0f;
		rtm.spheres.y[idx] = 10.0f;
		rtm.spheres.z[idx] = -20.0f;
		rtm.spheres.radius[idx] = 5.0;
		rtm.spheres.r[idx] = 1.0f;
		rtm.spheres.g[idx] = 1.0f;
		rtm.spheres.b[idx++] = 1.0f;
	}

	//Spehere representing the floor
	rtm.spheres.x[idx] = 0.0f;
	rtm.spheres.y[idx] = -10004.0f;
	rtm.spheres.z[idx] = -20.0f;
	rtm.spheres.radius[idx] = 10000.0;
	rtm.spheres.r[idx] = 0.8f;
	rtm.spheres.g[idx] = 0.8f;
	rtm.spheres.b[idx++] = 0.8f;

	//Background sphere in the center
	rtm.spheres.x[idx] = 0.0f;
	rtm.spheres.y[idx] = 10.0f;
	rtm.spheres.z[idx] = -55.0f;
	rtm.spheres.radius[idx] = 15.0;
	rtm.spheres.r[idx] = 106.0f/255 * 1.0f;
	rtm.spheres.g[idx] = 90.0f/255 * 1.0f;
	rtm.spheres.b[idx++] = 205.0f/255 * 1.0f;

	//Light pos and strength
	rtm.lights.x[0] = 20.0f;
	rtm.lights.y[0] = 50.0f;
	rtm.lights.z[0] = 50.0f;
	rtm.lights.strength[0] = 1.0f;

	uint32 width = rtm.width;
	uint32 height = rtm.height;

	std::cout << "RayTracing Started" << std::endl;
	std::cout << "Width: " << width << " rays" << std::endl;
	std::cout << "Height: " << height << " rays" << std::endl;
	std::cout << "Tracing " << num_of_spheres << " spheres" << std::endl;

	//Generate camera rays and lay them out for SIMD processing
	real fov = 60.0f;
	real ar = static_cast<real>(width) / height;
	real fov_rad = real(fov * PI / 180.0f);
	real theta = tan(fov_rad / 2.0f);

	real kx = theta * ar;
	__m128 x_factor = _mm_set_ps1((2.0f * kx) / width);
	__m128 x_add = _mm_set_ps1((kx*(2.0f - width)) / width);

	real ky = theta;
	__m128 y_factor = _mm_set_ps1((-2.0f * ky) / height);
	__m128 y_add = _mm_set_ps1((ky*(1.0f + height)) / height);

	//Camera ray data
	__m128 rx, ry;
	__m128 rz = _mm_set_ps1(-1.0f);

	Instrument ins("Ray Tracer");

	for (uint32 y = 0; y < height; ++y) {
		for (uint32 x = 0; x < width - 3; x += 4) {
			rx = _mm_set_ps(real(x + 3), real(x + 2), real(x + 1), real(x));
			rx = _mm_mul_ps(rx, x_factor);
			rx = _mm_add_ps(rx, x_add);

			ry = _mm_set_ps1(real(y));
			ry = _mm_mul_ps(ry, y_factor);
			ry = _mm_add_ps(ry, y_add);

			sse_ray rdirnorm;
			rdirnorm.x = rx;
			rdirnorm.y = _mm_sub_ps(ry, _mm_set_ps1(0.0f));
			rdirnorm.z = rz;
			normalize(rdirnorm);

			_mm_store_ps(&rtm.cam_ray.x[SsePos(y * width + x)], rdirnorm.x);
			_mm_store_ps(&rtm.cam_ray.y[SsePos(y * width + x)], rdirnorm.y);
			_mm_store_ps(&rtm.cam_ray.z[SsePos(y * width + x)], rdirnorm.z);
		}
	}

	ins.now("Generated Camera Rays");

	sse_ray rorigin;
	rorigin.x = _mm_set_ps1(0.0f);
	rorigin.y = _mm_set_ps1(0.0f);
	rorigin.z = _mm_set_ps1(8.0f);

#if defined(RT_SINGLE_THREADED)
	trace_section(rorigin, width, height, 0);
#else
	uint32 num_of_tasks = 32;
	const uint32 height_section = height / num_of_tasks;
	const uint32 last_section = height % num_of_tasks ? 1 : 0;
	num_of_tasks += last_section;

	ThreadPool pool(num_of_tasks);
	trace_section_data *thread_pool_data = reinterpret_cast<trace_section_data*>(rtm.allocator.allocate(num_of_tasks * sizeof(trace_section_data), SSE_ALIGNMENT));
	Task *thread_pool_tasks = reinterpret_cast<Task*>(rtm.allocator.allocate(num_of_tasks * sizeof(Task)));

	for (uint32 i = 0; i < num_of_tasks; ++i) {
		uint32 actual_height_section = i == (num_of_tasks - 1) && last_section ? i * height_section + (height % (num_of_tasks - 1)) : (i + 1) * height_section;
		
		thread_pool_data[i] = (trace_section_data{ rorigin, width, actual_height_section, i*height_section });
		thread_pool_tasks[i] = (Task(trace_section_task, &thread_pool_data[i]));
		pool.enqueue_task(&thread_pool_tasks[i]);
	}

	pool.join();
#endif

	ins.now("Raytracing Complete");

	save_bmp(width, height, rtm.pixmap);
	save_ppm(width, height, rtm.pixmap);

	ins.now("BMP and PPM Images Saved to Disk");

	std::cout << "Have a nice day!\n\n";
	return 0;
}