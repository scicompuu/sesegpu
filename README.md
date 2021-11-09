# sesegpu
Repository for SeSE course GPU programming for Machine Learning and Data Processing (version given in November 2021)

This course is a primer in using accelerators, specifically GPUs. In most laptop CPUs these days, the integrated GPU is consuming more silicon space than the rest of the chip, and definitely more than the actual CPU cores (excluding caches). For deep learning-style computations, accelerators like GPUs beat CPU-based implementations hands down, allowing for faster iteration of possible model concepts.

We will consider several ways to access the power of GPUs, both with ready-made frameworks for deep learning, and with code that gives more freedom in expressing what operations you want to perform, from Python and C++. Cursory familiarity with these languages is expected, but we will specifically try to focus on techniques that allow the programmer to focus on semantics, rather than arcane syntax and hardware details. The intended scope of the course is to sample a number of libraries and technologies related to GPU programming, for students with no previous familiarity with their usage, or exclusive familiarity with a single framework.

Together with the strong focus on the accelerator programming for machine learning algorithms, the course will also cover aspects of large-scale machine learning. In particular the needs for continuous analysis of large data volumes. It requires understanding of available infrastructures, tools and technologies, and strategies for efficient data access and management. The course will cover model serving and many-task-computing model for different machine learning algorithms.

Zoom room details will only be announced in the Slack channel in the sesegpu workspace.

## Approximate schedule

### Monday
10.15 Lecture 1: Introduction to distributed computing infrastructures

11.15 Lecture 2: Frameworks and strategies for scalable deployments

12.00 Lunch break

13.30 Lab 1: Implement first cloud service (ends at 16:30)

### Tuesday

9.15 Lab 2:     Large-scale model training and serving using Ansible and Kubernetes.

12.00 Lunch Break

13:30 Lab 2 (ends at 16:30)


### Wednesday
9.15 Lectures: GPU vs CPU, deep learning, TensorFlow and Tensorflow contrasted to other frameworks. Ends no later than 12.00 (with breaks).

13.15 Lab 3: Using TensorFlow for optimizing two simple models.

### Thursday

9.15 Lecture: The history of "GPGPU" programming. Current frameworks, focusing on Cuda and OpenMP Target and contrasting those against alternatives. Ends no later than 12.00 (with breaks).

13.15 Lab 4: Work on a selected problem (based on examples or prelab) and a framework



### Friday
10.15 Lecture: Profiling, debugging, how to approach the project. Ends no later than 12.00.

13.15 Lab 5: Explore project ideas.

15.15 Lecture: Common brainstorming of project ideas, with feedback from teachers regarding what seems feasible and important aspects to consider.
