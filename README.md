# GuanjinQiuyuML
## 1. Some tools for dataset preparation and production
### 1.1  Calculate the reaction rate constant k.
        cd data_preparation
        Start fitting the reaction rate constant k.bat

Click this file will bring up a web. The function of this web is to fit the reaction rate constant k.  



### 1.2  Calculate the mass ratio of elements in the catalyst(mass inputs).
            cd data_preparation
            Start the Atomic Ratio Calculator and enter g.bat
Click this bat file will bring up a web. The function of this web is to calculate the mass ratio of elements in the catalyst.



### 1.3  Calculate the mass ratio of elements in the catalyst(mol inputs).
       cd data_preparation
       Start the Atomic Ratio Calculator and enter mol.bat
Click this bat file will bring up a web. The function of this web is to calculate the mass ratio of elements in the catalyst.


### 1.4  Calculate the mass ratio of elements in the catalyst(XPS data inputs).
       cd data_preparation
       XPS atomic number ratio conversion to atomic mass ratio.bat
Click this bat file will bring up a web. The function of this web is to calculate the mass ratio of elements in the catalyst.

## 2. Data pre-processing
### 2.1 
        cd data_preprocessing
        python fill_elements_0.py
Run this python file can fill 0s in the elements area.
### 2.2
        cd data_preprocessing
        python fill_ions_0.py
Run this python file can fill 0s in the ions area.
### 2.3
         cd data_preprocessing
         python k_fitting_dataset.py
Run this python file can fitting k value.
### 2.4
          cd data_preprocessing
          python log2k.py
Run this python file can compute log2k.



  



                        