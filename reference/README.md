# Reference data

This directory has the following files:

- [aliases.json](reference/aliases.json): JSON file of gene name aliases
- [amplicons.fa](reference/amplicons.fa): Fasta file of synCRE barcode concatemer amplicons
- [archetypes.csv](reference/archetypes.csv): CSV file recording the motif clustering results of JASPAR2024 TFBSs
- [count_matrix.npy](reference/count_matrix.npy): Compact representation of read-counts by replicate, SAG and gate (+ unsorted), used in some analysis
- [esp_gblocks.csv](reference/esp_gblocks.csv): CSV file used to order IDT gblocks to generate the subcloning vectors for NeMECiS. Contains the sequences of each of the fragments and their associated barcodes (for Esp/Tetracycline subcloning vectors)
- [expressed_genes.txt](reference/expressed_genes.txt): A list of expressed genes from Delas et al., 2023 (Dev Cell). A processed version of *RNA_normCounts_filter1.csv*. 
- [fasta.zip](reference/fasta.zip): Zipped directory of fasta files for each of the fragments + the Olig2 CRE. 
- [JASPAR2024_CORE_PROCESSED.zip](reference/JASPAR2024_CORE_PROCESSED.zip): Zipped directory of all .pfm files used in motif alignment. Originates from https://jaspar.elixir.no/
- [out_bounds.npy](reference/out_bounds.npy): Fitted sort-gate boundaries used in Bayesian inference of logZsGreen levels by synCRE. 
- [pairwise_comparisons.tab](pairwise_comparisons.tab): Originates from https://jaspar.elixir.no/
- [RNA_normCounts_filter1.csv](reference/RNA_normCounts_filter1.csv): Originates from Delas et al., 2023 (Dev Cell)
- [SampleSheetUsed.csv](reference/SampleSheetUsed.csv): NextSeq 500 sample sheet used in demultiplexing
- [sorted_cells_statistics.csv](reference/sorted_cells_statistics.csv): CSV file of number of cells sorted by gate by replicate by condition. 
- [stan_values.csv](reference/stan_values.csv): Bayesian inference of logZsGreen levels, compiled into a single csv file, used in downstream analysis (e.g. linear modelling)
- [zsgreen_levels.zip](reference/zsgreen_levels.zip): Zipped directory of purity-check flow cytometry .fcs files, used in calibrating flow-sort gates. 