echo "# SRDRN_Cedar" >> README.md

SRDRN: a DL approach, namely, Super Resolution Deep Residual Network (SRDRN). 
The brief introduction of my ML code is  here. 
1. it is used to do downscaling data from coarse resolution (13 x 16) to the fine resolution (156 x 192). The temporal resolution is one-hour data, so the matrix is  very large, for the training, n = 121466.  
2. the maximum epochs is 150.

###References: Wang, F., Tian, D., & Carroll, M. (2023). Customized deep learning for precipitation bias correction and downscaling. Geoscientific Model Development, 16(2), 535-556.

Hisotry:
1. testing the DL SRDRN on computecanada
2. codes can work using python 3.10 + tensorflow 2.8 on cedar
3. on cedar, ~/scratch/SRDRN is the repository
4. push from cedar.computecanada.ca to github



##git init
##git add README.md
##git commit -m "first commit"
##git branch -M main
##git remote add origin https://github.com/honghong2023/SRDRN_Cedar.git
##git push -u origin main
