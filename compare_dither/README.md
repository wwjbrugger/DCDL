# compare_dither 
Scripts for experiment with different dither algorithms and how they change the performance of the neural net. 

### Algorithms: 
- floyd-steinberg
  
- atkinson

- jarvis-judice-ninke
  
- stucki

- burkes
  
- sierra-2-4a
  
- stevenson-arce
  
- sierra3
  
- sierra2
  
- atkinson,
  
- jarvis-judice-ninke
  
- stucki
  
- burkes,
  
- sierra-2-4a
  
- stevenson-arce

### Difference to other experiments 
Compared to the dither procedure used in NN_DCDL_SLS_Blackbox_comparison and visualize_rules_found, these implementations are self-written. 
Since dithering is slower with these self-written procedures, the already dithered data are stored. 
So they only have to be dithered once. 

For this reason the script for loading the data is quite different from the other two experiments. 