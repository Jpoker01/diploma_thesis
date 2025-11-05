# Authorship verification using chosen artificial intelligence methods

This project folder includes all experimentation code needed to train an optimal model for the Master's thesis "Authorship verification using chosen artificial intelligence methods"

## Project structure

- **requirements.txt** - The requirements for reproducing the implementation environment
- **.gitignore** - To ignore files that should not be committed
- **/conf** - Space for configurations
  - **/base** - Shared configuration like parameters
  - **/local** - Local configurations such as credentials
- **/data** - For saving any data worked with in the project
  - **/01_processed** - Data used to develop models
  - **/02_models** - Trained models
  - **/03_model_output** - The output of models
  - **/04_reporting** - Reports
- **/notebooks** - Jupyter notebooks - naming convention "YYYYMMDD, developer initials, -, description"
- **/results** - Final analysis documents
- **/src** - Source code for use in production code
  - **__init__.py** - For making the folder a module
  - **/d00_utils** - Functions used across the whole project
  - **/d01_data** - Functions for reading and writing data 
  - **/d02_intermediate** - Functions to transform data from raw to intermediate
  - **/d03_processing** - Functions to turn intermediate data into the input of models
  - **/d04_modelling** - Functions to train models & to use the models for inference
  - **/d05_model_evaluation** - Functions that evaluate model results
  - **/d06_reporting** - Functions to produce reports
  - **/d07_visualization** - Functions for visualizations
