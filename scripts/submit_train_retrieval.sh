#!/bin/bash

# Get submission running time as first positional argument 
# Expected format: HH:MM
if [ -z "$1" ]
then
  time="48:00:00"
else
  time="${1}:00"
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
  --partition=standard-g \
  --nodes=1 \
  --gpus-per-node=8 \
  --ntasks-per-node=8 \
  --time="${time}" \
  --output="${outdir}/${job_key}"_stdout.txt \
  --error="${outdir}/${job_key}"_errout.txt \
  --job-name="${job_key}" \
  --export=job_key="${job_key}" \
  "${train_dir}/train_retrieval.sh"
)

# Extract job ID from the output
job_id=$(echo "$job_id" | awk '{print $4}')

# Log
submission="$(date),${job_id},${job_key}"
echo "${submission}" >> "${outfile}"
echo "${submission}"
