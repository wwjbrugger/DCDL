#include "sls_multi_core.h"
extern "C" {
uint32_t sls(uint32_t clauses_n,              // # of DNFs
         uint32_t maxSteps,               // # of Updates
         float p_g1,                      // Prob of rand term in H
         float p_g2,                      // Prob of rand literal in H
         float p_s,                       // Prob of rand term in H
         features_t* data,                // Data input
         bool* label,                     // Label input
         features_t* pos_neg,             // Positive or engative for formula
         features_t* on_off,              // Mask for formula
         features_t* pos_neg_to_store,    // Positive or engative for formula
         features_t* on_off_to_store,     // Mask for formula
         uint32_t vector_n,               // # of data vectors
         uint32_t features_n,             // # of Features
         const bool batch,                // If score calculation should be done batchwise for given clause
         const bool cold_restart,         // Restar if stuck in bad local minimum
         const float decay,               // Decay factor, could be zero. Up to min_prob
         const float min_prob,            // Not decay below this threshold
         const bool zero_init             // Wether to go bigger steps in case of no sucess
         );

uint32_t sls_val(uint32_t clauses_n,                        // # of DNFs
             uint32_t maxSteps,                            // # of Updates
             float p_g1,                                   // Prob of rand term in H
             float p_g2,                                   // Prob of rand term in H
             float p_s,                                    // Prob of rand literal in H
             features_t* data,                             // Data input
             bool* label,                                  // Label input
             features_t* data_val,                         // Data input
             bool* label_val,                              // Label input
             features_t* pos_neg,                          // Positive or engative for formula
             features_t* on_off,                           // Mask for formula
             features_t* pos_neg_to_store,                 // Positive or engative for formula
             features_t* on_off_to_store,                  // Mask for formula
             uint32_t vector_n,                            // # of data vectors
             uint32_t vector_n_val,                        // # of data vectors
             uint32_t features_n,                          // # of Features
             const bool batch,                             // If score calculation should be done batchwise for given clause
             const bool cold_restart = false,              // Restar if stuck in bad local minimum
             const float decay = 0,                        // Decay factor, could be zero. Up to min_prob
             const float min_prob = 0,                     // Not decay below this threshold
             const bool zero_init = false                  // Wether to go bigger steps in case of no sucess
             );

uint32_t sls_test(uint32_t clauses_n,                                     // # of DNFs
              uint32_t maxSteps,                                        // # of Updates
              float p_g1,                                               // Prob of rand term in H
              float p_g2,                                               // Prob of rand term in H
              float p_s,                                                // Prob of rand literal in H
              features_t* data,                                         // Data input
              bool* label,                                              // Label input
              features_t* data_val,                                     // Data input
              bool* label_val,                                          // Label input
              features_t* data_test,                                     // Data input
              bool* label_test,                                          // Label input
              features_t* pos_neg,                                      // Positive or engative for formula
              features_t* on_off,                                       // Mask for formula
              features_t* pos_neg_to_store,                             // Positive or engative for formula
              features_t* on_off_to_store,                              // Mask for formula
              uint32_t vector_n,                                        // # of data vectors
              uint32_t vector_n_val,                                    // # of data vectors
              uint32_t vector_n_test,                                                 // # of data vectors
              uint32_t features_n,                                      // # of Features
              const bool batch,                                         // If score calculation should be done batchwise for given clause
              const bool cold_restart = false,                          // Restar if stuck in bad local minimum
              const float decay = 0,                                    // Decay factor, could be zero. Up to min_prob
              const float min_prob = 0,                                 // Not decay below this threshold
              const bool zero_init = false                              // Wether to go bigger steps in case of no sucess
              );
}
