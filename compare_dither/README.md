# compare_dither 
Scripts for experiment with different dither algorithms and how they change the performance of the neurel net. 

### Algorithms: 
[ 'floyd-steinberg', 'atkinson', 'jarvis-judice-ninke', 'stucki', 'burkes',  'sierra-2-4a', 'stevenson-arce', 'sierra3',  'sierra2', 'atkinson', 'jarvis-judice-ninke', 'stucki', 'burkes',  'sierra-2-4a', 'stevenson-arce'

### Difference to other experiments 
Compared to the dither procedure used in comparision_DCDL_vs_SLS and visualize_rules_found, these implementations are self-written. 
Since dithering is slower with these self-written procedures, the already dithered data is stored. Because of this they only have to be dithered once. 

The script for loading the data is quite different from the other two experiments. 