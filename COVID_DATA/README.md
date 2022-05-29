# Dataset used.



## [COVID-US](https://github.com/nrc-cnrc/COVID-US)

To generate this dataset, use the create_COVIDxUS.ipynb in the notebook to extract the ultrasound videos from multiple sources and integrate them in the COVIDx-US dataset. Make sure to modify the file paths in the code to your own paths.
    * __Note:__ The `covid_us_data.py` file is used to create the folders used.

    Check the script and adjust the code to your paths
 

## [covid19 ultrasound](https://github.com/jannisborn/covid19_ultrasound)

To generate this dataset, follow the instructions in the [readme here](https://github.com/jannisborn/covid19_ultrasound/tree/master/pocovidnet) and you can stop when you have a new folder `image_dataset`
with folders `covid`, `pneumonia`, `regular`. 


### Summary

Each of these dataset should have  `covid`, `pneumonia`, `regular` as we would only focus on these 3 targets.