#include "../inc/python_wrapper.h"
extern "C" {
uint32_t sls(uint32_t clauses_n,                 // # of DNFs
         uint32_t maxSteps,                  // # of Updates
         float p_g1,                         // Prob of rand term in H
         float p_g2,                         // Prob of rand term in H
         float p_s,                          // Prob of rand literal in H
         features_t* data,                   // Data input
         bool* label,                        // Label input
         features_t* pos_neg,                // Positive or engative for formula
         features_t* on_off,                 // Mask for formula
         features_t* pos_neg_to_store,       // Positive or engative for formula
         features_t* on_off_to_store,        // Mask for formula
         uint32_t vector_n,                  // # of data vectors
         uint32_t features_n,                // # of Features
         const bool batch,                   // If score calculation should be done batchwise for given clause
         const bool cold_restart,            // Restar if stuck in bad local minimum
         const float decay,                  // Decay factor, could be zero. Up to min_prob
         const float min_prob,               // Not decay below this threshold
         const bool zero_init                // Wether to go bigger steps in case of no sucess
         )
{
      return  sls_val(clauses_n, maxSteps, p_g1, p_g2, p_s, data, label, data, label, pos_neg, on_off, pos_neg_to_store, on_off_to_store,
                vector_n, vector_n, features_n, batch, cold_restart, decay, min_prob, zero_init);
}

uint32_t sls_val(uint32_t clauses_n,                          // # of DNFs
             uint32_t maxSteps,                       // # of Updates
             float p_g1,                              // Prob of rand term in H
             float p_g2,                              // Prob of rand term in H
             float p_s,                               // Prob of rand literal in H
             features_t* data,                        // Data input
             bool* label,                             // Label input
             features_t* data_val,                    // Data input
             bool* label_val,                         // Label input
             features_t* pos_neg,                     // Positive or engative for formula
             features_t* on_off,                      // Mask for formula
             features_t* pos_neg_to_store,            // Positive or engative for formula
             features_t* on_off_to_store,             // Mask for formula
             uint32_t vector_n,                       // # of data vectors
             uint32_t vector_n_val,                   // # of data vectors
             uint32_t features_n,                     // # of Features
             const bool batch,                        // If score calculation should be done batchwise for given clause
             const bool cold_restart,                 // Restar if stuck in bad local minimum
             const float decay,                       // Decay factor, could be zero. Up to min_prob
             const float min_prob,                    // Not decay below this threshold
             const bool zero_init                     // Wether to go bigger steps in case of no sucess
             ){
       return sls_test(clauses_n, maxSteps, p_g1, p_g2, p_s, data, label, data_val, label_val, data_val, label_val,pos_neg, on_off, pos_neg_to_store, on_off_to_store,
                 vector_n, vector_n_val, vector_n_val, features_n, batch, cold_restart, decay, min_prob, zero_init);
}

uint32_t sls_test(uint32_t clauses_n,                          // # of DNFs
              uint32_t maxSteps,                      // # of Updates
              float p_g1,                             // Prob of rand term in H
              float p_g2,                             // Prob of rand term in H
              float p_s,                              // Prob of rand literal in H
              features_t* data,                       // Data input
              bool* label,                            // Label input
              features_t* data_val,                   // Data input
              bool* label_val,                        // Label input
              features_t* data_test,                   // Data input
              bool* label_test,                        // Label input
              features_t* pos_neg,                    // Positive or engative for formula
              features_t* on_off,                     // Mask for formula
              features_t* pos_neg_to_store,           // Positive or engative for formula
              features_t* on_off_to_store,            // Mask for formula
              uint32_t vector_n,                      // # of data vectors
              uint32_t vector_n_val,                  // # of data vectors
              uint32_t vector_n_test,                 // # of data vectors
              uint32_t features_n,                    // # of Features
              const bool batch,                       // If score calculation should be done batchwise for given clause
              const bool cold_restart,                // Restar if stuck in bad local minimum
              const float decay,                      // Decay factor, could be zero. Up to min_prob
              const float min_prob,                   // Not decay below this threshold
              const bool zero_init                    // Wether to go bigger steps in case of no sucess
              )
{
        uint32_t vars_per_vector = SDIV(features_n, sizeof(features_t) * 8);

        TIMERSTART(MULTI_CORE_SLS)
        uint32_t total_error = multi_core::sls_test(clauses_n, /*Max Steps*/ maxSteps, /*p_g1*/ p_g1, /*p_g2*/ p_g2, /*p_s*/ p_s,
                             data, label, data_val, label_val, data_test, label_test,pos_neg, on_off, pos_neg_to_store, on_off_to_store, vector_n, vector_n_val, vector_n_test,features_n, /*batch*/ batch, /*Cold restart*/ cold_restart,
                             decay, min_prob, zero_init);
        TIMERSTOP(MULTI_CORE_SLS)
        //std::cout << "#Bits set in total " << multi_core::count_set_bits(on_off_to_store, vars_per_vector, clauses_n) << std::endl;

        return total_error;
        std::cout << std::endl;
        std::cout << "RETURNED" << std::endl;
        std::cout << "pos_neg" << std::endl;
        for(uint32_t index = 0; index < (vars_per_vector * clauses_n); index++) {
                std::cout << unsigned(pos_neg[index]) << " ";
        }
        std::cout << std::endl;
        std::cout << "RETURNED" << std::endl;
        std::cout << "on_off" << std::endl;
        for(uint32_t index = 0; index < (vars_per_vector * clauses_n); index++) {
                std::cout << unsigned(on_off[index]) << " ";
        }
        std::cout << std::endl;

}

void calc_prediction(
        features_t *data,             // input data
        bool *prediction_label,       // space to store prediction
        features_t *pos_neg_to_store,          //  part 1 from logic formula which variables should be negated?
        features_t *on_off_to_store,           // part 2 from logic formula which variables are relevant
        uint32_t vector_n,            // # of data vectors
        uint32_t clauses_n,          // # of DNFs
        uint32_t features_n              // # of Features
)
{
    multi_core::calc_prediction(data, prediction_label, pos_neg_to_store, on_off_to_store, vector_n, clauses_n, features_n);
}
}
