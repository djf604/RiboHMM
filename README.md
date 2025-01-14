# RiboHMM

## Installation
The RiboHMM package can be installed through pip from the Github repository:

```pip install git+https://github.com/djf604/RiboHMM```

This will install two executables:
* ``ribohmm``, which is the interface for learning the model and inferring translated sequences
* ``ribohmm-utils``, which is used to pre-process the input data and compute mappability

## Pre-processing the Sample Data
Aligned Ribo-seq and RNA-seq data (as BAM files) must first be converted to a tabix-indexed tabular 
count format. The ``ribohmm-utils`` executable has a sub-program called ``bam-to-counts`` which automates 
much of this process. Given multiple BAM files, ``bam-to-counts`` will output a single tabix-indexed 
counts file which has aggregated counts from all input. For example:

```
ribohmm-utils bam-to-counts --bams sample1.riboseq.bam sample2.riboseq.bam --bam-type riboseq \
--output-prefix example_output
```

The above will read in ``sample1.riboseq.bam`` and ``sample2.riboseq.bam`` and produce two files:
* ``example_output.ribo.counts.bed.gz``, the tabular counts file
* ``example_output.ribo.counts.bed.gz.tbi``, the tabix index of the above file

The sample BAMs can also be input directly into the main ``ribohmm`` executable and will be converted 
as above on-the-fly.

For more detailed documentation refer to the wiki (coming soon).

## Computing Mappability for Ribo-seq Data
Since ribosome footprints are typically short (28-31 base pairs), footprints originating from many 
positions in the transcriptome are likely to not be uniquely mappable. Thus, with standard parameters 
for mapping ribosome footprint sequencing data, a large fraction of the transcriptome will have no 
footprints mapping to them due to mappability issues. While RiboHMM can be used without accounting 
for missing data due to mappability, we have observed the results to be substantially more accurate 
when mappability is properly handled.

Given a GTF file that contains the transcriptome, mappability information (i.e., whether each position 
in the transcriptome can produce a uniquely mappable ribosome footprint or not) can be obtained in 
3 steps:

1. For each desired footprint length (default 28-31bp), build a FASTQ with all footprints that could 
originate from the given transcriptome. This is done with the ``ribohmm-utils`` sub-program 
``mappability-generate``.

    ```
    ribohmm-utils mappability-generate --gtf-file transcriptome.gtf --fasta-reference genome.fa \
    --footprint-lengths 28 29 30 31 --output-fastq-stub example_output
    ```

    The above will produces four files:
    * ``example_output_footprint28.fq.gz``
    * ``example_output_footprint29.fq.gz``
    * ``example_output_footprint30.fq.gz``
    * ``example_output_footprint31.fq.gz``

2. Align the created synthetic FASTQs, using the same mapping strategy used for the original ribosome 
footprint profiling data. The BAM alignments will be the input into the next step.

3. For each desired footprint length, build a tabix-indexed mappability file from the BAM produced in 
step 2. This marks whether a footprint originating from a given position uniquely mapped back to the 
same place. This is done with the ``ribohmm-utils`` sub-program ``mappability-compute``.

    ```
    ribohmm-utils mappability-compute --mappability-bam example_output_length28.bam \
    --output-tabix example_output_mappability_28.bed 
    ```
    
    The above produces two files:
    * ``example_output_mappability_28.bed.gz``
    * ``example_output_mappability_28.bed.gz.tbi``

For more detailed documentation refer to the wiki (coming soon).

## Running the RiboHMM Algorithm
The main interface to RiboHMM is through the ``ribohmm`` executable. By default, both the parameter 
learning and inference steps are run in sequence. Some flags can be given to change that behavior:
* ``--learn-only`` causes the program to learn the model parameters, save them to a model parameters 
JSON, and exit
* ``--infer-only`` cause the program to skip the model learning step, instead accepting a model 
parameters JSON and moving directly to inference

For more detailed documentation refer to the wiki (coming soon).

### Learning the Model
In general the necessary inputs to learn the model parameters are:
* Reference genome in FASTA format
* Transcriptome in GTF format
* Ribo-seq BAM(s) or tabix-indexed counts file (created with ``ribohmm-utils bam-to-counts``)

Optional but helpful inputs include:
* Corresponding RNA-seq BAM(s) or tabix-indexed counts file (created with ``ribohmm-utils bam-to-counts``)
* Mappability tabix-indexed counts files (created with ``ribohmm-utils mappability-generate,mappability-compute``)

An example:
```
ribohmm --learn-only --reference ref/genome.fa --transcriptome ref/transcriptome.gtf \
    --riboseq-counts data/example_output.ribo.counts.bed.gz \
    --rnaseq-counts data/example_output.rna.counts.bed.gz \
    --mappability-tabix-prefix data/mappability/example_output_mappability \
    --output run001 --batch-size 10
```

This produces a directory called ``run001`` which has a single file ``model_parameters.json``, which 
can be passed into future ``--infer-only`` runs of RiboHMM.

### Inferring Translated Sequences
Unless ``ribohmm`` is run with the ``--learn-only`` flag, this inference step is automatically run 
following the learning step. This step can also be run directly by giving the ``--infer-only`` flag and 
providing a model parameters JSON.

If run directly, this step generally needs as input the same set as the learning step, with the addition 
of the model parameters JSON, which is often called ``model_parameters.json``.

An example:
```
ribohmm --infer-only --reference ref/genome.fa --transcriptome ref/transcriptome.gtf \
    --riboseq-counts data/example_output.ribo.counts.bed.gz \
    --rnaseq-counts data/example_output.rna.counts.bed.gz \
    --mappability-tabix-prefix data/mappability/example_output_mappability \
    --model-parameters run001/model_parameters.json \
    --output run001
```

This produces inside of the ``run001`` directory a file called ``inferred_CDS.bed``, which contains 
the inferred translated sequences.

## Providing a Custom Kozak Model
Though a Kozak model is included with RiboHMM and is suitable for use in humans, a custom Kozak model can be supplied 
using the `--kozak-model` argument.

The custom Kozak model is expected to be in the `.npz` format, which is a numpy compressed file. It is expected to have 
two arrays, one called `freq` and one called `altfreq`. Each must be a 2-dimensional array of shape (4, 13) and must be 
position weight matrices where the sum of each column is 1. The first row represents the proportion of the A base, the 
second row the proportion of the U base, the third row the G base, and the fourth row the C base.

The `freq` array is the observed frequencies of the Kozak model, and `altfreq` is the background frequencies.

## Support
If errors are encountered, please open an issue on this repository with a detailed bug report.