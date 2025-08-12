# Description of RMSE Algorithm
The RMSE algorithm is run over each ORF for every Transcript.

Before each ORF is processed, a map is created for all triplets that are fully mappable. This is done for each 
(frame, footprint_length, triplet) in the Transcript. A triplet is considered fully mappable if _every_ one of 
its base positions is mappable according to the mappability file provided to RiboHMM.

Then, for each ORF:
1. The size of the ORF is computed as the number of triplets
2. The state diagram is computed, where the only thing that varies is the number of elogation states
    1. States `0` (5'UTS), `1` (5'UTS+), and `8` (3'UTS) are not considered here

Within each ORF, for each footprint length:
1. The footprint's expected proportions are pulled from the emissions parameters. There is one set of expected proportions 
for each state.
2. The ORF frame index, start triplet, and stop triplet are stored.
3. The raw pileups (Riboseq reads) for each triplet are stored. Any extra base pairs at the end of the transcript (for a given frame index) 
that don't fit into a triplet are dropped.
4. Any triplets which are not fully mappable are dropped from the calculation.
5. Pileup proportions are computed within each triplet. If the result is `NaN` (such as when there are no reads, leading to a 
division by 0 error) then the proportion is set to a uniform `(1/3, 1/3, 1/3)`.
6. Each proportion value is individually subtracted from a corresponding expected value (at this point the triplet structure 
is not considered, only the individual proportions associated with a base position). Each value is then squared.
7. If the function argument `normalize_tes` is set to `True`, then that operation is applied here (see below for details).
8. The mean of all remaining values is taken, and that the square root of that value is taken.
9. The resulting value is stored as the RMSE for this (ORF_index, footprint_length).

The result of the above is four RMSE values, one for each footprint legnth. The final RMSE value for the ORF is taken as 
the mean of the footprint legnth RMSE values.

## Normalize TES
If the function is directed to normalize the TES region (by setting `normalize_tes=True`), then the following process is 
applied:
1. The TES states are taken as all states denoted as `4`, the size of which is fixed at `n_states - 5`.
2. The mean of all TES state squared errors is calculated, respecting the position within the triplet.
    - For example, given the following TES region of 4 triplets:
    ```
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [4, 3, 2]
    ]
    ```
    The result of step 2 would be:
    ```
    >>> [mean([1, 4, 7, 4]), mean([2, 5, 8, 3]), mean([3, 6, 9, 2])]
    >>> [4.0, 4.5, 5.0]
    ```
3. The result from step 2 is inserted into the state diagram in place of all TES states. The resulting size of the state 
diagram is always 6. 

When this setting is on, the resulting metric is called **SSRMSE**.