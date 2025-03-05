# Predictable Engineering of Signal-Dependent Cis-Regulatory Elements

### Cornwall-Scoones et al. 

This is the code-base related to the paper "Predictable Engineering of Signal-Dependent Cis-Regulatory Elements"
describing the method and application of the technology Nested Modular Expressible Cis Screening (NeMECiS). 

Below is a summary of the code, in order of analysis

## Code-base
- [Alignment](alignment): Conversion of raw .fastq files to read-counts by sample. 
- [Inference](inference): Conversion of read-counts to inferred ZsGreen levels
- [CREdential](CREdential): Linear modelling of inferred ZsGreen levels
- [Motifs](motifs): Motif analysis of NeMECiS data

Additionally [Reference](reference) contains the relevant auxilliary data plus various key post-processing results. 
