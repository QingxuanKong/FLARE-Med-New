#!/bin/bash

max_retries=999 # Maximum number of retries before giving up
retries=0
wait_time=5 # Seconds to wait before retrying

while true; do
  echo "Attempting to run ./run_benchmark.sh (Attempt $((retries + 1)))..."
  # Ensure run_benchmark.sh is executable or run it with bash/sh
  # If run_benchmark.sh is in the current directory and executable:
  ./run_benchmark.sh
  # If it's not executable or you want to be explicit:
  # bash ./run_benchmark.sh

  exit_status=$?

  if [ $exit_status -ne 0 ]; then
    echo "run_benchmark.sh failed with exit status $exit_status."
    retries=$((retries + 1))

    if [ $retries -ge $max_retries ]; then
      echo "Maximum retries ($max_retries) reached. Aborting."
      exit 1 # Exit the wrapper script with an error status
    fi

    echo "Waiting $wait_time seconds before retrying..."
    sleep $wait_time
  else
    echo "run_benchmark.sh completed successfully."
    break # Exit the loop on success
  fi
done

echo "Wrapper script finished successfully."
exit 0 # Exit the wrapper script with a success status 