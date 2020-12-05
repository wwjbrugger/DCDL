Scripts to explore the visualization of the rules found by the SLS algorithm. 

Entry point is the script start.py where the hyperparameters can be changed. 
It can be changed: 

`one_against_all_array` Which label should be tested against all others. 

`repetitions_of_sls` How often the extraction of the logical rules should be calculated to ensure a statistically safe statement

`Number_of_disjuntion_term_in_SLS` How many disjunctions may be used in the SLS. A small number provides general/discriminative rules and 
a big k  specific/characteristic rules 
