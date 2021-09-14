# sesegpu
Repository for SeSE course GPU programming for Machine Learning and Data Processing (version given in June 2020)

This course is a primer in using accelerators, specifically GPUs. In most laptop CPUs these days, the integrated GPU is consuming more silicon space than the rest of the chip, and definitely more than the actual CPU cores (excluding caches). For deep learning-style computations, accelerators like GPUs beat CPU-based implementations hands down, allowing for faster iteration of possible model concepts.

We will consider several ways to access the power of GPUs, both with ready-made frameworks for deep learning, and with code that gives more freedom in expressing what operations you want to perform, from Python and C++. Cursory familiarity with these languages is expected, but we will specifically try to focus on techniques that allow the programmer to focus on semantics, rather than arcane syntax and hardware details. The intended scope of the course is to sample a number of libraries and technologies related to GPU programming, for students with no previous familiarity with their usage, or exclusive familiarity with a single framework.

Together with the strong focus on the accelerator programming for machine learning algorithms, the course will also cover aspects of large-scale machine learning. In particular the needs for continuous analysis of large data volumes. It requires understanding of available infrastructures, tools and technologies, and strategies for efficient data access and management. The course will cover model serving and many-task-computing model for different machine learning algorithms.

Zoom room details will only be announced in the Slack channel in the sesegpu workspace.

## Approximate schedule (in June 2020)

### Monday
8.45 Lectures start. GPU vs CPU, deep learning, TensorFlow and Tensorflow contrasted to other frameworks. Ends no later than 12.00 (with breaks).

13.15 Lab 1

### Tuesday
8.45 Lectures start. The history of "GPGPU" programming. Current frameworks, focusing on Cuda and OpenMP Target and contrasting those against alternatives. Ends no later than 12.00 (with breaks).

13.15 Lab 2

### Wednesday
09.15 First session

10.15 Second session

11.15 Third session

12.00 Lunch break

13.30 Lab (-17)

### Thursday
Jim Dowling will give the guest lecture.

9.15 First session

10.15 Second session

11.15 Third session

12.00 Lunch Break

13:30 Lab (-17)

### Friday
10.15 Profiling, debugging, how to approach the project. Ends no later than 12.00.

13.15 Lab time, for exploring project ideas.

15.15 Common brainstorming of project ideas, with feedback from teachers regarding what seems feasible and important aspects to consider.



