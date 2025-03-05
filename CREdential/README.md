# CREdential

This package takes processed inferred log ZsGreen values for each synCRE 
and performs linear modelling, as specified in the Materials and Methods. 

## Model code
The package acts through a pipeline with the following steps. These are defined in the following sub-packages 
- [Assembler](CREdential/assembler.py): Takes raw output from (STAN) inference of processed inferred log ZsGreen by synCRE. Calculates all theoretical features (fragments, fragment-pairs, syntax (i.e. fragment-pair + relative spacing)), and assigns these to a set of feature matrices by synCRE entry. These are all compiled into a single dictionary for subsequent usage. 
- [ProcessFitData](CREdential/process_fit_data.py): Compiles the above feature matrices a global feature matrix *Z*. Additionally compiles log ZsGreen values and optionally weights, ready for regression.
- [Model](CREdential/model.py): Wrapper class of any potential linear model. Performs fitting, summary statistic generation, bootstrapping. 
- [LogMultiplicativeModel](CREdential/models/log_multiplicative_model.py): Specifies the log-multiplicative model. Uses *sklearn.linear_model.Ridge* to perform ridge regression
- [HyperparameterTuning](CREdential/hyperparameter_tuning.py):     K-fold cross validation method to choose regularisation strength

## Pipeline code
Additionally, there are scripts to run the full pipeline 
- [PipelineModular](CREdential/pipeline.py): Runs the fitting pipeline on the simpler linear model using only fragment counts
- [PipelineSynergy](CREdential/pipeline_synergy.py): Runs fitting on the extended linear model incorporating pairwise contributions. 
