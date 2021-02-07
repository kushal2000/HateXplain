# :mag_right: Few-Shot Learning from HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection 

## Objective of Work

HateXplain is a rationales annotated dataset which is trained on the rationales using attention losses. This branch transfers the HateXplain model on another benchmark Dataset - Davidson with same class labels (without rationales). The use of rationales helps in better generalization.

***WARNING: The repository contains content that are offensive and/or hateful in nature.***

## Reference

~~~bibtex
@article{mathew2020hatexplain,
  title={HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection},
  author={Mathew, Binny and Saha, Punyajoy and Yimam, Seid Muhie and Biemann, Chris and Goyal, Pawan and Mukherjee, Animesh},
  journal={arXiv preprint arXiv:2012.10289},
  year={2020}
}

~~~

------------------------------------------
***Folder Description*** :open_file_folder:	
------------------------------------------
~~~

./Data                --> Contains the dataset related files including HateXplain and Davidson
./Models              --> Contains the codes for all the classifiers used
./Preprocess  	      --> Contains the codes for preprocessing the dataset	
./best_model_json     --> Contains the parameter values for the best models

~~~

------------------------------------------
***Usage instructions*** 
------------------------------------------
Install the required libraries using the following command (preferably inside an environemt)
~~~
pip install -r requirements.txt
~~~
#### Training
First generate a pre-trained model on HateXplain by running '''manual_inference.py'''
Then walk through '''Run_Davidson.ipnyb'' to get a few-shot transferred model on Davidson benchmark dataset.
'''Plotting.ipynb''' will generate visualizations related to the results
~~~
