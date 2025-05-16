# GuanjinQiuyuML
This project contains 2 subsections. First is predict k value. Second is predict KTBA/k and KMEOH/k.
## K Prediction
        cd k_prediction
### 1. Some tools for dataset preparation and creation
        cd data_preparation
#### 1.1  Calculate the reaction rate constant k.

        Start fitting the reaction rate constant k.bat

Click this file will bring up a web. The function of this web is to fit the reaction rate constant k.  



#### 1.2  Calculate the mass ratio of elements in the catalyst(mass inputs).
            
        Start the Atomic Ratio Calculator and enter g.bat
Click this bat file will bring up a web. The function of this web is to calculate the mass ratio of elements in the catalyst.



#### 1.3  Calculate the mass ratio of elements in the catalyst(mol inputs).
       
       Start the Atomic Ratio Calculator and enter mol.bat
Click this bat file will bring up a web. The function of this web is to calculate the mass ratio of elements in the catalyst.


#### 1.4  Calculate the mass ratio of elements in the catalyst(XPS data inputs).
       
       XPS atomic number ratio conversion to atomic mass ratio.bat
Click this bat file will bring up a web. The function of this web is to calculate the mass ratio of elements in the catalyst.

### 2. Data pre-processing: Batching in dataset.
        cd data_preprocessing
#### 2.1 
        
        python fill_elements_0.py
Run this python file to fill 0s in the elements area.
#### 2.2
        
        python fill_ions_0.py
Run this python file to fill 0s in the ions area.
#### 2.3
         
         python k_fitting_dataset.py
Run this python file to fit k value.
#### 2.4
          
          python log2k.py
Run this python file to compute log2k.



  



                        