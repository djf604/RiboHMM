The slurm script, to be run from `/home1/08246/dfitzger`:
```shell
#!/bin/bash

#SBATCH --time=2:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -p development

cd /home1/08246/dfitzger

ribohmm --infer-only --reference riboHMM_chr11_example_YRI_Data/GRCh37.p13.genome.fa   \
--transcriptome riboHMM_chr11_example_YRI_Data/example.339.chr11.CCDS.Stringtie.gtf   \
--riboseq-counts riboHMM_chr11_example_YRI_Data/YRI_all_uniquely_mapped_chr11_reads.ribo.counts.bed.gz   \
--rnaseq-counts riboHMM_chr11_example_YRI_Data/RNAseqGeuvadis_STAR_combined.chr11.rna.counts.bed.gz   \
--mappability-tabix-prefix riboHMM_chr11_example_YRI_Data/RNAseqGeuvadis_STAR_combined_mappability   \
--model-parameters riboHMM_chr11_example_YRI_Data/model_parameters.json   --output /work/08246/dfitzger/ls6/run001 \
--infer-algorithm ${algo} --n-procs 4 --dev-output-debug-data dev_output_$(date +"%Y%m%d_%H%M%S").json \
--dev-restrict-transcripts-to ${n_transcripts}
```

The command to run: 
```shell
sbatch --export=algo=discovery,n_transcripts=20 run_infer_skx_unit_tests.sh
```

The output should be compared to the file at `/work/08246/dfitzger/ls6/run001/inferred_CDS_baseline.bed`

Where the results should match:
```

```