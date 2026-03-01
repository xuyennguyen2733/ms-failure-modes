
This file explains the structure of the Shift Benchmark on Multiple Sclerosis lesion segmentation - see https://arxiv.org/pdf/2206.15407.pdf . The dataset is split into two archives. This archive contains data from multiple providers shared under a public CC BY NC SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/) . A second archive pertains to the MSSEG data provided by OFSEP is available under an OFSEP data usage agreement (DUA).

----------------------------------------------------------------
----------------- FULL DATASET DESCRIPTION ---------------------
----------------------------------------------------------------

There FULL dataset consists of separate directories, split either by the location of the medical centre (Best and Ljubljana), or the name of the original dataset (MSSEG):
- msseg (consists of multiple locations including lyon, bordeaux and rennes)
- best
- ljubljana

Within each directory, the data is split according to the canonical partitioning for the Shifts Benchmark. Splits must be one of the following:
- train
- dev_in
- dev_out
- eval_in

An additional split called 'unsupervised' is provided that consists of FLAIR images of MS patients from the MSSEG-2 dataset that have not been labelled.

Within each of the canonical splits (train / dev_in / dev_out / eval_in) , the following data is provided - note, not all locations have all the modalities:
- flair: FLAIR modality input images 
- t1: T1 modality input images
- pd: Proton Density (PD) weighted input images
- t2: T2 modality input images
- t1ce: T1 contrast enhanced modality (enhancement due to Gadolinium injection) 
- gt: The 'ground-truth' concensus masks to identify locations of lesion voxels
- fg_mask: The foreground masks to identify the brain area (i.e. necessary for calculation error retention curves in only the brain area)

All input images have the following preprocessing applied:
- Denoising
- Registration to the FLAIR space
- Skull stripping (brain mask calculated from the T1 images)
- Bias field correction
- Interpolation to the isovoxel space (input images are linearly interpolated while all masks are interpolated using nearest neighbour)

Additionally, the msseg directory offers the annotations by 7 individual annotators (these were used to generate the ground-truths using a consensus by majority vote):
The best directory offers annotations for 2 individual annotators (ground-truths are selected as the second annotator who is more experienced)
The Ljubljana directory does not have the annotations available for the individual annotators

----------------------------------------------------------------
--------------- HOW TO COMBINE Part 1 and Part2 ----------------
----------------------------------------------------------------

Please download both archives, for each split (train / dev_in / dev_out / eval_in), for each data folder (flair, t1, pd, t2, etc...), please combine data from each location (MSSEG, best, Ljubljana). Note, MSSEG and best correspond to train, dev_in and eval_in data, while Ljubljana corresponds to dev_out data. 

Such that the resulting strutting is as follows:

-unsupervised 
  ---flair
      --- data from MSSEG unsupervised-flair

-train
  ---flair
      --- data from MSSEG train-flair
             .
             .
             .
      --- data from best train-flair
             .
             .
             .

  ---t1
      --- data from MSSEG train-t1
             .
             .
             .
      --- data from best train-t1
             .
             .
             .
  ---pd 
   .
   .
   .

-dev_in
  ---flair
      --- data from MSSEG dev_in-flair
             .
             .
             .
      --- data from best dev_in-flair
             .
             .
             .

  ---t1
      --- data from MSSEG dev_in-t1
             .
             .
             .
      --- data from best dev_in-t1
             .
             .
             .
  ---pd 
     .
     .
     .

-eval_in
  ---flair
      --- data from MSSEG eval_in-flair
             .
             .
             .
      --- data from Best eval_in-flair
             .
             .
             .
  ---pd 
     .
     .
     .

-dev_out
  ---flair
      --- data from Ljubljana dev_out-flair
             .
             .
             .
  ---t1
      --- data from Ljubljana dev_out-t1
             .
             .
             .
  ---pd
      --- data from Ljubljana dev_out-pd
             .
             .
             .
  ---t2
     .
     .
     .
          
