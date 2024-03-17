#!/bin/bash
# .bash script to automate iterating over arguments to test with


entropy=(1.0 0.1 0.01)
decay=(1.0 0.99 0.95 0.9)

# Loop over the arguments and run the Python script
for entropy_value in "${entropy[@]}"; do
  for decay_value in "${decay[@]}"; do
      python3 testing_script.py "$entropy_value" "$decay_value"
  done
done
