# Pre-lab 2

The purpose of this lab is to get your user environment set up, train a simple deep learning model based on a demo and compare its performance in different operating modes.

## Setup
1. First, you need to apply for membership for the UPPMAX HPC course project g2021027  (this is different from the cloud project). If you are not registered in SUPR/SNIC, do so now. Then go to [https://supr.snic.se/project/](https://supr.snic.se/project/) and search for the project code g2021027. Apply to join this project. Approval is manual, so this will not happen instantly. Once you are part of the project g2021027, you need to additionally apply for an account on UPPMAX. There is a guide at [https://www.uppmax.uu.se/support/getting-started/applying-for-a-user-account/](https://www.uppmax.uu.se/support/getting-started/applying-for-a-user-account/) for applying for project membership as well as a user account, if you don't already have one. Note that step 4 is the crucial one to complete before you can log in to UPPMAX.
2. Log in to your UPPMAX account on Rackham. In this lab, we will use the Snowy cluster, but it is accessed through the Rackham login nodes. Instructions can be found at [https://www.uppmax.uu.se/support/user-guides/guide--first-login-to-uppmax/](https://www.uppmax.uu.se/support/user-guides/guide--first-login-to-uppmax/).
3. Clone the contents of the course repository using git. This puts a subfolder `sesegpu` into your current directory:

       git clone https://github.com/scicompuu/sesegpu.git
4. Some large files (container images) associated to the course are found in a different directory. Verify that you can access them (TODO: check):

        ls /proj/g2020014/nobackup/private
5. An HPC environment is a shared resource. You are not supposed to run large computations on the login node where you are currently placed. Rather, you request to run jobs, either executing bash scripts or interactive jobs. We will use both within the course.
6. We recommend using text-based ssh connections to the login nodes. Some of the work will be done in a web browser anyway. If you want to, you can explore using the graphical [ThinLinc]([https://www.uppmax.uu.se/support-sv/user-guides/thinlinc-graphical-connection-guide/](https://www.uppmax.uu.se/support-sv/user-guides/thinlinc-graphical-connection-guide/)) client to UPPMAX as well, but the ThinLinc resources are limited.
## Opening Jupyter
[Jupyter](https://jupyter.org/) is a browser-based environment for running code, including Python. We have provided some scripts that should make it somewhat simple to host a Jupyter environment inside a cluster job.
Let's test that.
1. If you are currently in the `sesegpu`subdirectory, it's enough to write:

       ./notebook.py
2. The script adds a job into a queue. Hopefullly, it will start soon. Keep waiting, or do something else while keeping an eye on when the job starts. If you read up, you can also request that the job should start at a specific time, and the queuing system will try to service that request.
 3. Now, this script starts a server on a compute node, and it creates a network tunnel. If you are running in e.g. ThinLinc, the script prints a URL you can open directly within that session. Otherwise, you'll need to create a second ssh session. The scripts suggests a command, that you might need to modify slightly (e.g. add your user name) depending on your configuration. Note that the port number (45788) is random and will change every time. You need to match the output from the script.

        ssh rackham3.uppmax.uu.se -L 127.0.0.1:45788:127.0.0.1:45788
4. The scripts tells you what URL to open. Try it. In this example, it was http://127.0.0.1:45788/?token=341c60678a7c41d8368ef6b70a9ed9df92ef97a54d75729d
5. While both ssh windows are still open, you can now freely use Jupyter in the browser. Create a new Python3 notebook to try it out.
6. A prompt is shown, write `print('Hello')` and press Shift + Enter.
7. The output from the command is visible. In a notebook, you can combine snippets of code and run them zero, one, or many times. You can also add other instructions.
8. When you're done, press Enter in the original ssh session (where you started `notebook.py`). This ends the session.
9. (extra) The job will also end automatically after 59 minutes. This is a setting in the job. If you are familiar with Slurm flags, you can add any conventional `sbatch`flag on top of the `./notebook.py`command. For example, you can add `-t 2:00:00`to request a two-hour job instead. If you know when you want to run, you can start queuing a 3-hour job at 2 PM for starting at 3 PM with `-t 3:00:00 -b 15:00`. Note that the ssh tunneling requires the same window to stay open until the job quits.

## Some actual deep learning
Not, it's time to do some deep learning from within Jupyter using [TensorFlow](TensorFlow). You will explore a simple example. It comes from [here]([https://www.tensorflow.org/tutorials/quickstart/beginner](https://www.tensorflow.org/tutorials/quickstart/beginner)), but all the things you need can actually be found inside the Jupyter notebook.
1. Start a new notebook session with `notebook.py`. This requires you to set up the ssh sessions like we did in he previous section
2. When you have opened Jupyter successfully in the browser, navigate to the file `beginner.ipynb` in the sidebar to the left.
3. This is a Jupyter notebook with some introduction text and code snippets. You can run the code sections one by one and watch the results, or you can select Run -> Run All Cells in the menu.

Right now, we'll just conclude that TensorFlow allows us to formulate very flexible models including a lot of numerical computations in a succinct manner.
## Time to think about time
We are concerned with performance in this course. The single reason to use GPUs rather than other computational resources, is to gain their performance. Performance can mean that things go faster, or simply that you waste less electrical power getting them done. Or both...
1. Open `beginner.ipynb`in Jupyter again (start a new notebook session, if you don't already have one running).
2. There's a single code section which is actually time consuming here. Even though this model is small, it takes several seconds to train, and if one would want to improve the model, it would take far longer. Change the model fitting line to read:

        %time model.fit(x_train, y_train, epochs=5)
3. Note these timing results. You will be comparing it to other options.
4. In the model definition, two lines (Dense and Dropout) are repeated before and after a comment about them being added by UPPMAX. Comment these out. Does the timing change? What does that tell you about the overhead to run the model, relative to the actual computations?
5. If everything worked correctly (check the output from the first code section), these calculations were done on a Snowy GPU. We will now try doing them in CPU mode instead.
6. Close your existing notebook and related ssh forwarding sessions.
7. Start a new notebook with the command `./notebook.py -n 4 -p devcore --gres=gpu:t4:0 -- tf-mkl`. This will launch the notebook on a node with shorter job queues (unless they are full, you can try dropping `-p devcore`), but most importantly NOT requesting a GPU. In addition, we will use a build of tensorflow using the [Intel Math Kernel Library](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html), which is supposed to optimize the core linear algebra operations a lot. Do these tests with and without the commented out extra layers.
8. Run the timings with this setting. Are they similar to what you got before?
9. In the example above, we asked for 4 core (`-n 4`). Redo the full thing with 8 or even 16 cores instead. This gives you 2 or 4 times the computational power. Do the results improve accordingly? (Note, you can enqueue multiple notebook jobs with different settings in different windows.)
10. Redo with just one core (`-n 1`).
11. (extra) There is also a default setup of TensorFlow available. Launch it using `./notebook.py -n 4 -p devcore --gres=gpu:t4:0 -- tf-pip`. How do these results compare to those on the GPU, and those with MKL? Is the scaling to more cores than 4 similar?

Evidently, this model is very small, so at least a moderately strong GPU has a hard time matching CPU. For many real-life models, there are thousands or millions of activation values to be evaluated in the inner layers, rather than the puny 128 here. However, even in this case, running compute on the GPU means potentially freeing up CPU cores for other tasks -- and there are _no_ programming changes involved in switching between GPU and CPU.

## Reading materials
There is far more reference material, tutorials, and examples on the general [TensorFlow](https://www.tensorflow.org/) website. Like any popular programming model, Q&A on StackOverflow and other Internet resources are also popular. Note that there are quite significant differences in the API and programming model between TensorFlow 1 and TensorFlow 2, so be a bit wary if you cannot determine clearly what version you're finding information for -- especially if it is more than a year old. (TF2 was released in September 2019, but the pre-releases were pretty popular before that.) You might want to try out a more complex tutorial and time that one as well.

Some very concrete examples for writing your own numerical calculations in TensorFlow can be found on [https://mlfromscratch.com/tensorflow-2/#/](https://mlfromscratch.com/tensorflow-2/#/). Rather than following the insatllation instructions there, you can keep on using Jupyter.

## Under the hood  (extra)
Explore the files `notebook.py`and `notebookjob.sh`to see what's really going on. We're using the tool [Singularity](https://sylabs.io/docs/) to pack all the software needed for these three environments in a single file. If you want to, you can copy the file `container.sif` to your local machine and setup Singularity there as well.
If you would just like to start an interactive command line job inside the software environment on a GPU node, you could write:

    srun -A g2021027 -t 0:59:00 -p core -n 4 -M snowy --gres=gpu:t4:1 singularity run --nv /proj/g2020014/nobackup/private/container.sif
