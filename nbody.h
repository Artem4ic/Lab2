#ifndef NBODY_H
#define NBODY_H

#include <stddef.h>
#include <stdio.h>

struct vector_s {
   double x, y, z;
};

struct body_s {
   double mass;
   struct vector_s position;
   struct vector_s velocity;
};

struct input_s {
   double end_time;
   double time_step;
   size_t thread_count;
   const char* input_name;
   const char* output_name;
   int output_3d;
};

void* Allocate(size_t count, size_t size);
void Get_options(int argc, char* argv[], int parallel, struct input_s* input);
struct body_s* Read_bodies(const char* file_name, size_t* body_count);
FILE* Open_output(const struct input_s* input, size_t body_count);
void Write_state(FILE* file, double time, const struct body_s bodies[],
                 size_t body_count, int output_3d);
void Close_output(FILE* file);
struct vector_s Pair_force(const struct body_s* first,
                           const struct body_s* second);
void Add_force(struct vector_s* sum, struct vector_s force);
void Subtract_force(struct vector_s* sum, struct vector_s force);
void Update_body(struct body_s* body, struct vector_s force, double step);
double Next_time(double time, const struct input_s* input);

#endif
