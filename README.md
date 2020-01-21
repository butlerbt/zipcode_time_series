### National Zip Code Real Estate Analysis

### `zipcode` conda environment

This project relies on you using the [`environment.yml`](environment.yml) file to recreate the `zipcode` conda environment. To do so, please run the following commands:

```bash
# create the zipcode conda environment
conda env create -f environment.yml

# activate the oy-env conda environment
conda activate zipcode

# make oy-env available to you as a kernel in jupyter
python -m ipykernel install --user --name zipcode --display-name "Python (zipcode)"
```

