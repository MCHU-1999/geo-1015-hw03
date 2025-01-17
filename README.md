# geo-1015-hw03
A repo for geo-1015 Digital terrain modeling homework 3 file syncing and storing.

### Requirements
The packages we used (and thrie dependents) are in the [`requirements.txt`](requirements.txt)  

### The 3 Algorithms
- RANSAC: [`code/ransac.py`](code/ransac.py)  
- Region Growing: [`code/regiongrowing.py`](code/regiongrowing.py)  
- Hough Transform: [`code/houghtransform.py`](code/houghtransform.py)  

### Other Functions
- distance clusterin for RANSAC: [`code/clustering.py`](code/clustering.py)  
- basic RANSAC from terrain-book: [`code/ransac_simple.py`](code/ransac_simple.py)  
- plotting function using matplotlib: [`code/plot.py`](code/plot.py)  

### parameters
Create a configuration file `params.json` in [`data/`](data/) according to the format below before running the program.   
```json
{
  "input_file": "bk.laz",
  "RANSAC": {
    "k": 1000,
    "min_score": 200,
    "epsilon": 0.1,
    "multiplier": 5,
    "delta": 3.5,
    "min_samples": 20
  },
  "RegionGrowing": {
    "k": 25,
    "max_angle": 30,
    "min_seeds": 20000,
    "min_region_size": 10
  },
  "HoughTransform": {
    "alpha": 50,
    "epsilon": 0.15,
    "theta segment size": 10,
    "phi segment size": 10,
    "rho segment size": 0.1,
    "chunk size": 25,
    "acceleration factor": 10,
    "reprocessing": true
  }
}
```

### Students
Ming-Chieh Hu, 6186416\ 
Daan Schlosser, 5726042\ 
Neelabh Singh, 6052045\ 
Lars van Blokland, 4667778

### Folder Structure
```
├── report
│   └── report.pdf
├── data
│   └── out_bk.ply 
│   └── out_bk_subset1.ply 
├── code
│   └── geo1015_hw03.py
│   └── ransac.py
│   └── regiongrowing.py
│   └── houghtransform.py
│   └── ...
└── README.md
```

