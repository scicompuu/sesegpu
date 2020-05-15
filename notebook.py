#!/usr/bin/env python3

import socket, errno
import os
import sys
import time
import re
from contextlib import closing
from subprocess import Popen, PIPE

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', 0))
        return s.getsockname()[1]

ports = [str(find_free_port()) for _ in range(4)]

if sys.argv.contains("--"):
    firstargs = sys.argv[1:]
    secondargs = []
else:
    index = sys.argv.index("--")
    firstargs = sys.argv[1:index]
    secondargs = sys.argv[index + 1:]

with Popen(['sbatch'] + firstargs + ['notebookjob.sh'] + ports + secondargs, stdin=PIPE, stdout=PIPE, stderr=PIPE) as proc:
    jobline = proc.stdout.readline().decode()
    jobline = jobline.strip()
    vals = jobline.split(';')
    jobid = int(vals[0])
    joblist = []
    if len(vals) > 1:
        jobcluster = vals[1]
        joblist = ['-M', jobcluster]

print(f'Waiting for job ID {jobid} to start')

url = ""
while url == "":
    try:
        with open(f'slurm-{jobid}.out') as file:
            for line in file:
                index = line.find("http://")
                if index >= 0:
                    url = line[index:]
            if url == "":
                print("Job started, valid URL not yet provided by Jupyter. This should be pretty fast.")
    except FileNotFoundError:
        print("Output file not yet found. Waiting for job to start.")
    time.sleep(10)

port = int(re.search(':([0-9]+)/', url).group(1))
with Popen(['squeue', '-o', '%N', '-j', str(jobid)] + joblist, stdin=PIPE, stdout=PIPE, stderr=PIPE) as proc:
    lines = proc.stdout.readlines()
    node = lines[-1].decode().strip()

with Popen(['ssh', node, '-L', f'127.0.0.1:{port}:127.0.0.1:{port}', '-T'], stdin=PIPE, stdout=PIPE, stderr=PIPE) as proc:
    print('If you can open a web browser on the login node using e.g. Thinlinc, you can now open this URL:')
    print(url)
    print('If not, run the following ssh command locally and leave that secondary ssh shell open:')
    print(f'ssh {socket.gethostname()} -L 127.0.0.1:{port}:127.0.0.1:{port}')
    print()
    print("When you've done this, you can open the url in a normal web browser.")
    print("Press Enter in this terminal when you're done. The job has a default time limit of 59 minutes.")
    input()
    proc.terminate()

with Popen(['scancel', str(jobid)] + joblist, stdin=PIPE, stdout=PIPE, stderr=PIPE) as proc:
    pass

