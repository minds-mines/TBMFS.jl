# Task Balanced Multimodal Feature Selection Implementation

This code is designed to support the submission BIBE 2020 titled: 

"Task Balanced Multimodal Feature Selection to Predict the Progression of
Alzheimer’s Disease"

Implementation details of the Task Balanced Multimodal Feature Selection
method can be found in the `TBMFS.jl` file in this folder. Usage can be found
in the file titled `example.jl`. If you find this code useful please consider
citing the following:

```bibtex
@inproceedings{brand2020task,
  title={Task Balanced Multimodal Feature Selection to Predict the Progression of Alzheimer’s Disease},
  author={Brand, Lodewijk and O’Callaghan, Braedon and Sun, Anthony and Wang, Hua},
  booktitle={2020 IEEE 20th International Conference on Bioinformatics and Bioengineering (BIBE)},
  pages={196--203},
  year={2020},
  organization={IEEE}
}
```

## Running the code

First, download a recent distribution of the Julia software package from:
https://julialang.org/. 

Then, run the following from the command line:
```
julia --project
julia> include("example.jl") 
```

Tested on Julia Version 1.3.1

## Algorithm Convegence

![TBMFS Convergence](/images/convergence.png)`
