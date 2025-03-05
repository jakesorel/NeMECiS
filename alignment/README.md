# Alignment

In this package, raw fastq files are aligned to a white-list of barcode-trimer amplicons to determine read-counts by synCRE by experimental (sequencing) sample 

## Code
The code is run in the following order
- [Run alignment](alignment/run_alignment): Uses *bowtie2* to map fastq reads to a whitelist of amplicon sequences, saved in *reference/amplicons.fa*. The output is a folder of .sam files. This is done in a by-sequencing-lane manner.
- [Process sam files](alignment/process_sam_files.py): Filters alignments in the .sam files by excluding all reads that do not perfectly align to the sequenced region between Nextera adaptors (CIGAR=96M). Exports a csv file with N_synCRE rows, returning read-counts by synCRE. Additionally, includes information about the identities of each fragment in each position. 

