# TinyCodeExecutor

This repository contains scripts for conducting  experiments related to the Code Execution with Tiny Language Models Project. The primary goal of this repository is to ensure experiments can be easily reproduced or redone in the future. By running a few key scripts, you can generate all the results necessary for the associated paper.

## Repository Structure

- **Data-Scaling/**: Contains scripts for experiments related to scaling data size. The experiments in this directory focus on understanding how different amounts of training data affect the performance of the models.
  
- **Epochs-Scaling/**: Includes scripts for experiments where the number of training epochs is varied. These experiments aim to analyze the impact of training duration on model accuracy and performance.
  
- **Models-Scaling/**: Contains scripts for experiments where the model size is scaled. This includes varying the number of embedding dimensions (`n_embed`) to observe how model capacity influences performance.

- **Final-Model/**: This directory contains the final model scripts. Use these scripts to load, test, and evaluate the final model with the best model, data and epoch sizes after all experiments are complete.


## Data Access and Updates

The datasets required for these experiments will be shared through NYU HPC. If there are any modifications to the data or if new datasets are introduced, you will need to update the data paths within the scripts accordingly. Once the paths are updated, you can re-run the experiments to generate new results.

## How to Run Experiments

1. **Clone the Repository**:
   ```
   git clone https://github.com/MarwaNair/TinyCodeExecutor.git
   cd TinyCodeExecutor
   ```

2. **Set Up the Environment**:
   - Ensure you have Python 3.11.7+ installed.
   - Install the necessary Python packages.

3. **Update Data Paths**:
   - Before running any experiment, make sure the data paths in the scripts point to the correct location of your datasets on the NYU HPC.

4. **Running Experiments**:
   - For example, navigate to the `Data-Scaling` directory.
   - Use the provided scripts to conduct data scaling experiments:
     ```
     cd Data-Scaling
     python train.py
     python evaluate.py # once the training is done for all models
     ```
