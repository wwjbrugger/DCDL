## SLS Rule Extraction
The SLS algorithm call has the follwing parameters.

```python
   clauses_n,  # k
   maxSteps,  # # of Updates
   p_g1,  # Prob of rand term in candidate
   p_g2,  # Prob of rand literal in candidate
   p_s,  # Prob of rand term in candidate
   data,  # Training set
   label,  # Training label
   data_val,  # Validation set
   label_val,  # Validation label
   data_test,  # Test set
   label_test,  # Test label
   pos_neg,  # Storage for candidates
   on_off,  # Mask for candidates
   pos_neg_to_store,  # Storage for current BEST candidate
   on_off_to_store,  # Mask for current BEST candidate
   vector_n,  # of training instances !!!!NEEDS TO BE BIGGER THEN BATCH_SIZE!!!!
   vector_n_val,  # # of validation instances !!!!NEEDS TO BE BIGGER THEN BATCH_SIZE!!!!
   vector_n_test,  # # of data instances !!!!NEEDS TO BE BIGGER THEN BATCH_SIZE!!!!
   features_n,  # # of Features
   batch,  # If score calculation should be done batchwise
   cold_restart,  # Restart if stuck in bad local minimum
   decay,  # Decay factor, could be zero. Up to min_prob
   min_prob,  # No decay below this threshold
   zero_init  # Alternativ initialization
```

The batch size as well as the trailing stop need to be set as Makros in the
DCDL/parallel_sls/src/sls_mult_core.cpp
file.

Recomplie files with calling `make` in Terminal 