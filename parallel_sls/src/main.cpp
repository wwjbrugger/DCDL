#include <iostream>

#include "../inc/sls_multi_core.h"

int main(int argc, char const *argv[]) {

    std::cout << "Hallo" << std::endl;
    features_t data [5] = { 16, 2, 77, 40, 2 };
    features_t pos_neg [1] = { 16};
    features_t on_off [1] = { 16 };
    features_t pos_neg_to_store [1] = { 16 };
    features_t on_off_to_store [1] = { 16 };
    bool label [5] = {true, true, false, false, true};
    bool prediction_label [5] = {false, false, false, false, true};
    uint32_t vector_n =  5;
    uint32_t features_n = 8;
    uint32_t clauses_n = 3;



   int total_error =  multi_core::sls(clauses_n,               // # of DNFs
                    10,                // # of Updates
                    0.5,                       // Prob of rand term in H
                    0.5,                       // Prob of rand term in H
                    0.5,                        // Prob of rand literal in H
                    //nullptr,                 // Data input
                    data,
                    //nullptr,                      // Label input
                    label,
                    //nullptr,              // Positive or negative for formula
                    pos_neg,
                    //nullptr,               // Mask for formula
                    on_off,
                    //nullptr,     // Positive or negative for formula
                    pos_neg_to_store,
                    //nullptr,      // Mask for formula
                    on_off_to_store,
                    vector_n,                // # of data vectors
                    features_n,              // # of Features
                    false,                 // If score calculation should be done batchwise for given clause
                    false,          // Restart if stuck in bad local minimum
                    0,                // Decay factor, could be zero. Up to min_prob
                    0,             // Not decay below this threshold
                    false              // Wether to go bigger steps in case of no sucess
                    );

   std::cout << "Total_Error " << total_error << std::endl;

   multi_core::calc_prediction(data, prediction_label, pos_neg_to_store, on_off_to_store, vector_n, clauses_n, features_n);

    return 0;
}
