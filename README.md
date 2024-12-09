# MicrobioRel

This repo presents the associated corpora to the article "MicrobioRel: A Manually Annotated Dataset for Microbiome Relation Extraction". 

- docs includes annotation scheme as well as the decision tree.
- data/export_data.json contains raw extracted annotations from Label Studio.
- data/RE_classif_data.csv represents the used dataset for our study, after applying all the preprocessing steps, including the adding of "None" Class.
  
Following several rounds of annotation, a decision tree has been implemented. Its main objective is to streamline the process, particularly in examples where there is ambiguity regarding the relation type. It serves as a supporting item to the annotation scheme.
The scripts corresponding to experiments are to be added to this repo.

