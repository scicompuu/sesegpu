# Pre-lab 3

The purpose of this lab is to expose you to some of the prevailing ways to write efficient GPU code these days. We are focusing on Nvidia technologies, but considering very hardware-dependent and less tightly coupled alternatives.

## Setup
1. We assume that you have completed pre-lab 2, so your UPPMAX account etc are in order.
2. Specifically, we will be using the MNIST dataset in this lab as well. If you are not sure if that was actually downloaded to your home directory, you can run the following command on the login node:

       singularity run /proj/g2020014/nobackup/private/container.sif -c ./downloadmnist.py
Like in the previous lab, we are using Singularity to provide a consistent software environment. You might get warnings about your home directory being shadowed and not being able to load GPU libraries. This is OK, the directory mapping is working and since we are not running this on a GPU node (and without the Singularity `--nv` flag enabling GPU support), that is fully natural.

## Exploring the original code
First of all, we are going to look at and time a code version using plain C++, with no additional frills. This is short enough that you can run it directly on the login node, you do not have to run a separate job.
First of all, have a look at [data.h](https://github.com/scicompuu/sesegpu/blob/master/data.h) and [plaincpp.cpp](https://github.com/scicompuu/sesegpu/blob/master/plaincpp.cpp). You can do that online or in the local version of the repository that you pulled in the previous lab. You will be compiling the local version soon.

1. Do you understand what the code is doing? It is using two third-party open-source libraries, [cnpy](https://github.com/rogersce/cnpy) and [mdspan](https://github.com/kokkos/mdspan), the latter of which is an implementation of a proposed addition to the C++ standard.
2. Compile the code:

       singularity run /proj/g2020014/nobackup/private/cppgpu.sif clang++ plaincpp.cpp -march=sandybridge -O3 -lcnpy -o plaincpp
3. Run the code directly on the login node:

       singularity run /proj/g2020014/nobackup/private/cppgpu.sif ./plaincpp
4. Note the time usage. We also want to run this as a computational job, since our other versions will be running on the Snowy cluster nodes.

       sbatch ./runnogpu.sh cppgpu.sif ./plaincpp
5. This helper script prints the job number. You can check the status for the job by running the following command (insert your job number).

       scontrol show job 5473794 -M snowy
       JobId=5473794 JobName=runnogpu.sh
          UserId=martinkr(43077) GroupId=nobody(43077) MCS_label=N/A
          Priority=100000 Nice=0 Account=g2021027 QOS=normal WCKey=*
          JobState=COMPLETED Reason=None Dependency=(null)
          Requeue=0 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0
          RunTime=00:00:26 TimeLimit=00:59:00 TimeMin=N/A
          SubmitTime=2021-11-09T19:50:28 EligibleTime=2021-11-09T19:50:28
          AccrueTime=2021-11-09T19:50:28
          StartTime=2021-11-09T19:50:29 EndTime=2021-11-09T19:50:58 Deadline=N/A
          SuspendTime=None SecsPreSuspend=0 LastSchedEval=2021-11-09T19:50:29
          Partition=devcore AllocNode:Sid=rackham3:33723
          ReqNodeList=(null) ExcNodeList=(null)
          NodeList=s1
          BatchHost=s1
          NumNodes=1 NumCPUs=4 NumTasks=4 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
          TRES=cpu=4,mem=32000M,node=1,billing=4
          Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
          MinCPUsNode=1 MinMemoryCPU=8000M MinTmpDiskNode=0
          Features=(null) DelayBoot=00:00:00
          OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null)
          Command=./runnogpu.sh cppgpu.sif ./plaincpp
          WorkDir=/domus/h1/martinkr/teaching/sesegpu
          StdErr=/domus/h1/martinkr/teaching/sesegpu/slurm-5473794.out
          StdIn=/dev/null
          StdOut=/domus/h1/martinkr/teaching/sesegpu/slurm-5473794.out
          Power=

6. The output from the job is thus found in `slurm-5473794.out` (since the job state was `COMPLETED`).

          cat slurm-5473794.out
          Loading file /home/martinkr/.keras/datasets/mnist.npz
          ...

7. Note this time usage. You can try what happens if you change the optimization level from `-O3` to `-O0` or other similar changes. This CPU implementation is far from perfect!

## OpenMP Target
OpenMP is a standardized way to express parallelism in code in C/C++/Fortran. The main idea is to rely on the compiler to convert a serial code with loops and other constructs, into a properly distributed code. Originally, OpenMP was designed for homogenous shared-memory systems, i.e., multicore and multi-processor CPU-based machines where all compute cores are essentially equivalent and have access to all memory.

OpenMP Target changes that. One can add `target` blocks inside the code. Those are possibly compiled to a completely different compute architecture (a different target), which does not necessarily fully share memory with the host. OpenMP Target is thus a reasonable way to write something close to sequential code, and still have part of it executed on the GPU.

1. Read the two different OpenMP versions of the code [openmptarget.cpp](https://github.com/scicompuu/sesegpu/blob/master/openmptarget.cpp) and [openmptarget2.cpp](https://github.com/scicompuu/sesegpu/blob/master/openmptarget2.cpp). What differences do you see?
2. Compile the code. We want to compile it for generating GPU code the Tesla T4 cards at UPPMAX, which are Nvidia SM generation 75:

       singularity run /proj/g2020014/nobackup/private/cppgpu.sif clang++ openmptarget.cpp -march=sandybridge -O3 -lcnpy -o openmptarget -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_75
3. The nice thing is that this binary contains GPU code, but if no GPU is provided, it will run parallel on CPUs as well. Let us test this, and also the GPU version. Start three  jobs and collect their results.

        sbatch ./runnogpu.sh cppgpu.sif ./openmptarget
        sbatch -n 16 ./runnogpu.sh cppgpu.sif ./openmptarget
        sbatch ./runongpu.sh cppgpu.sif ./openmptarget

4. The default `runnogpu.sh` uses 4 cores. What performance are you getting for the CPU versions? What about the GPU version? Which job is the fastest?
5. Repeat this for `openmptarget2.cpp`. Do your results differ?

The same codebase *can* be used for CPU and GPU, but it is not necessarily optimal for both.

6. If you find yourself waiting a long time to get a GPU job, you can get a single interactive job on the GPU as well and run your GPU codes within that one:

        srun -A g2021027 -t 0:59:00 -p core -n 4 -M snowy --gres=gpu:t4:1 --pty bash
        ...
        singularity run --nv /proj/g2020014/nobackup/private/cppgpu.sif ./openmptarget

   Note that you need to specify `--nv` to make the GPU available within the container, as well as specifying the correct `--gres` flag to `srun`.
## Thrust
[Thrust]([https://github.com/thrust/thrust](https://github.com/thrust/thrust)) is a library relying on C++ templates for expressing parallel operations. The point is to (mostly) avoid specifying *how* to do stuff, but rather express in more high-level terms *what* to accomplish. For example, there are ready-made functions for searching, sorting, and other operations that can be hard to express in a high-performance way on an extremely parallel architecture. If one pays care, the same Thrust code can also be compiled for parallel CPU usage, although this will not be the case in this example.

We will now use another container, `cuda.sif`, which includes a proper library setup for compiling our depencies using the Nvidia compiler `nvcc`.
1. Compile `thrust.cu`.

       singularity run /proj/g2020014/nobackup/private/cuda.sif nvcc thrust.cu -lcnpy -O3 -std=c++14 -arch=sm_75 -o thrust --expt-relaxed-constexpr -rdc=true -lcudadevrt
2. Test this new version, either using `sbatch runongpu.sh cuda.sif ./thrust` or by launching an interactive job with `srun`.
3. Do the same to `thrustmax.cu`. Which version performs better? How do they compare to the other versions we have seen?

## CUDA
CUDA is the "native" way to do GPU programming on Nvidia GPUs. It is a set of extensons to C++ that allow you to more directly see that these are single instruction multiple thread architectures, where the normal mode of operation is that all threads in a group (called a warp) execute the very same instructions. Warps are ordered into blocks. Each block is then a member of a grid. Even here, we will be using a specific library, [CUB](https://github.com/thrust/cub) to implement some logic for us. CUB is more low-level than Thrust, but has the same idea of helping in synchronizing work, but at a level which more clearly makes blocks, grids and warps transparent to the developer, for better and worse.
1. Compile cuda.cu:

        singularity run /proj/g2020014/nobackup/private/cuda.sif nvcc cuda.cu -lcnpy -O3 -std=c++14 -arch=sm_75 -o cuda --expt-relaxed-constexpr -rdc=true -lcudadevrt
2. Run this version as well and time it.
## Reading materials
You can read more in the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html), although it is quite overwhelming. For OpenMP, the official examples and especially the reference guide found on the [OpenMP website](https://www.openmp.org/specifications/) can be useful (as well as other collections of existing resources). For Thrust you can start with the [Quick Start Guide](https://github.com/thrust/thrust/wiki/Quick-Start-Guide). More [CUB](https://nvlabs.github.io/cub/) information is available as well.

## numpy
There is also an ipython notebook `prelab3.ipynb`. During Lab 4 you will try to make this code go faster. Explore why you think it is slow. Since it is currently not using a gpu, you can launch `notebook.py --gres=gpu:t4:0 -p devcore` to get a running job faster.

When you have explored this code, you can decide whether you want to try to implenent that one faster during Lab2, or if you want to explore the C++ based libraries, or if you have some other computation-intensive Python code that you want to try to make faster using GPU-based acceleration. The point is that you should have made up your mind for what code/algorithm you want to explore when Lab2 starts. You need to know the current state of the code, what is making it slow on the CPU and what parts you believe should/could be implemented on a GPU.

## Discussion
Do the code versions do the same thing? What differences are there? Can you think of any further experiments you would like to test? What is the performance difference between the original CPU version and the fastest version of the code?
