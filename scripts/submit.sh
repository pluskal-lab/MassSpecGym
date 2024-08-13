#!/bin/bash

# Submit a single-node job to a SLURM cluster.

# Usage example:
# Run for 4 hours on 8 GPUs:
# ./submit.sh run.sh 12:00 8

# 1st argument: submission script name
if [ -z "$1" ]
then
  echo "Error: No submission script name provided."
  exit 1
else
  script_name="$1"
fi

# 2nd argument: time limit in hours
if [ -z "$2" ]
then
  time="48:00:00"
else
  time="${2}:00"
fi

# 3rd argument: number of GPUs
if [ -z "$3" ]
then
  gpus="8"
else
  gpus="${3}"
fi
# Set the queue variable based on the value of gpus
if [ "$gpus" -eq 1 ]
then
  # queue="small-g"
  queue="standard-g"
else
  queue="standard-g"
fi

# Init logging dir and common file
train_dir="${WORK}/ms/MassSpecGym/scripts"
outdir="${train_dir}/submissions"
mkdir -p "${outdir}"
outfile="${outdir}/submissions.csv"

# Generate random key for a job
job_key=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 10 ; echo '')

# Submit and capture the job ID
job_id=$(sbatch \
  --account="${PROJECTID}" \
  --partition="${queue}" \
  --nodes=1 \
  --gpus-per-node="${gpus}" \
  --ntasks-per-node="${gpus}" \
  --time="${time}" \
  --output="${outdir}/${job_key}"_stdout.txt \
  --error="${outdir}/${job_key}"_errout.txt \
  --job-name="${job_key}" \
  --export=job_key="${job_key}" \
  "${train_dir}/${script_name}"
)

# Extract job ID from the output
job_id=$(echo "$job_id" | awk '{print $4}')

# Log
submission="$(date),${job_id},${job_key}"
echo "${submission}" >> "${outfile}"
echo "${submission}"
