#pragma once

#include "stdint.h"
#include "../inc/cuda_helpers.cuh"
#include <random>
#include <vector>
#include <limits>
#include <bitset>
#include <stdlib.h>
using features_t = uint8_t;
using label_t = uint8_t;
namespace multi_core {
// Allows for assuming padded feature vectors (only small size to pad)
// Avoids overhead for calculating right positions (in comparison to only small overhead in memory)
using features_t = uint8_t;

// Additionally optimized by particular score function for one clause and random drawing of wrong example avoided

void random_dnf(
        features_t* pos_neg,       // Positive or engative for formula  *is mark for pointer
        features_t* on_off,        // Mask for formula
        uint32_t dnf_s
        );

void zero_dnf(
        features_t* pos_neg,       // Positive or uint32_t engative for formula
        features_t* on_off,        // Mask for formula
        uint32_t dnf_s
        );

uint32_t sls(uint32_t clauses_n,               // # of DNFs
         uint32_t maxSteps,                // # of Updates
         float p_g1,                       // Prob of rand term in H
         float p_g2,                       // Prob of rand term in H
         float p_s,                        // Prob of rand literal in H
         features_t* data,                 // Data input
         bool* label,                      // Label input
         features_t* pos_neg,              // Positive or engative for formula
         features_t* on_off,               // Mask for formula
         features_t* pos_neg_to_store,     // Positive or engative for formula
         features_t* on_off_to_store,      // Mask for formula
         uint32_t vector_n,                // # of data vectors
         uint32_t features_n,              // # of Features
         const bool batch,                 // If score calculation should be done batchwise for given clause
         const bool cold_restart = false,  // Restar if stuck in bad local minimum
         const float decay = 0,            // Decay factor, could be zero. Up to min_prob
         const float min_prob = 0,         // Not decay below this threshold
         const bool zero_init = false // Wether to go bigger steps in case of no sucess
         );

uint32_t sls_val(uint32_t clauses_n,                        // # of DNFs
             uint32_t maxSteps,                     // # of Updates
             float p_g1,                            // Prob of rand term in H
             float p_g2,                            // Prob of rand term in H
             float p_s,                             // Prob of rand literal in H
             features_t* data,                      // Data input
             bool* label,                           // Label input
             features_t* data_val,                  // Data input
             bool* label_val,                       // Label input
             features_t* pos_neg,                   // Positive or engative for formula
             features_t* on_off,                    // Mask for formula
             features_t* pos_neg_to_store,          // Positive or engative for formula
             features_t* on_off_to_store,           // Mask for formula
             uint32_t vector_n,                     // # of data vectors
             uint32_t vector_n_val,                 // # of data vectors
             uint32_t features_n,                   // # of Features
             const bool batch,                      // If score calculation should be done batchwise for given clause
             const bool cold_restart = false,       // Restar if stuck in bad local minimum
             const float decay = 0,                 // Decay factor, could be zero. Up to min_prob
             const float min_prob = 0,              // Not decay below this threshold
             const bool zero_init = false      // Wether to go bigger steps in case of no sucess
             );

uint32_t sls_test(uint32_t clauses_n,                                     // # of DNFs
              uint32_t maxSteps,                                 // # of Updates
              float p_g1,                                        // Prob of rand term in H
              float p_g2,                                        // Prob of rand term in H
              float p_s,                                         // Prob of rand literal in H
              features_t* data,                                  // Data input
              bool* label,                                       // Label input
              features_t* data_val,                              // Data input
              bool* label_val,                                   // Label input
              features_t* data_test,                              // Data input
              bool* label_test,
              features_t* pos_neg,                               // Positive or engative for formula
              features_t* on_off,                                // Mask for formula
              features_t* pos_neg_to_store,                      // Positive or engative for formula
              features_t* on_off_to_store,                       // Mask for formula
              uint32_t vector_n,                                 // # of data vectors
              uint32_t vector_n_val,                             // # of data vectors
              uint32_t vector_n_test,                              // # of data vectors
              uint32_t features_n,                               // # of Features
              const bool batch,                                  // If score calculation should be done batchwise for given clause
              const bool cold_restart = false,                   // Restar if stuck in bad local minimum
              const float decay = 0,                             // Decay factor, could be zero. Up to min_prob
              const float min_prob = 0,                          // Not decay below this threshold
              const bool zero_init = false                  // Wether to go bigger steps in case of no sucess
              );

// Adaption of original algorithm: draw the first missclassified example
void draw_missed(
        features_t* data,
        bool* label,
        features_t* pos_neg,
        features_t* on_off,
        uint32_t vector_n,
        uint32_t vars_per_vector,
        uint32_t clauses_n,
        uint32_t &missed_ex,
        uint32_t &wrongly_hit_clause
        );

uint32_t calc_score(
        const features_t* data,
        const bool* label,
        const features_t* pos_neg,
        const features_t* on_off,
        const uint32_t vector_n,
        const uint32_t vars_per_vector,
        const uint32_t clauses_n,
        uint32_t &wrongly_negative,
        uint32_t &wrongly_positive
        );

uint32_t calc_score_batch(
        const features_t* data,
        const bool* label,
        const features_t* pos_neg,
        const features_t* on_off,
        const uint32_t vector_n,
        const uint32_t vars_per_vector,
        const uint32_t clauses_n,
        const uint32_t batch_size
        );


uint32_t calc_score_given_clause(
        features_t* data,
        bool* label,
        features_t* pos_neg,
        features_t* on_off,
        uint32_t vector_n,
        uint32_t vars_per_vector,
        uint32_t clauses_n,
        uint32_t given_clause
        );

uint32_t calc_score_given_clause_batch(
        features_t* data,
        bool* label,
        features_t* pos_neg,
        features_t* on_off,
        uint32_t vector_n,
        uint32_t vars_per_vector,
        uint32_t clauses_n,
        uint32_t given_clause,
        uint32_t batch_size
        );

uint32_t count_set_bits(const features_t* to_count,
                        const uint32_t vars_per_vector,
                        const uint32_t clauses_n
                        );



// SAME FORE GIVEN CLAUSE
void calc_prediction(
        const features_t* data,             // input data
        bool* prediction_label,       // space to store prediction
        const features_t* pos_neg,          //  part 1 from logic formula which variables should be negated?
        const features_t* on_off,           // part 2 from logic formula which variables are relevant
        const uint32_t vector_n,            // # of data vectors
        const uint32_t clauses_n,          // # of DNFs
        uint32_t features_n             // # of Features
);
}