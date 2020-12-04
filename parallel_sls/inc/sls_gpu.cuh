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
namespace gpu {
// Allows for assuming padded feature vectors (only small size to pad)
// Avoids overhead for calculating right positions (in comparison to only small overhead in memory)


// Additionally optimized by particular score function for one clause and random drawing of wrong example avoided

void random_dnf(
        std::vector<features_t> &pos_neg, // Positive or negative literal
        std::vector<features_t> &on_off,  // If literal is needed
        features_t dnf_s
        );

void sls(uint16_t clauses_n,                      // # of DNFs
         uint32_t maxSteps,                       // # of Updates
         float p_g1,                              // Prob of rand term in H
         float p_g2,                              // Prob of rand term in H
         float p_s,                               // Prob of rand literal in H
         features_t* data,                        // Data input
         label_t* label,                             // Label input
         std::vector<features_t> &pos_neg,         // Positive or engative for formula
         std::vector<features_t> &on_off,          // Mask for formula
         uint32_t vector_n,                       // # of data vectors
         uint32_t features_n,                     // # of Features
         bool parallel                            // If gpu should be used
         );

// Adaption of original algorithm: draw the first missclassified example
void draw_missed(
        features_t* data,
        label_t* label,
        features_t* pos_neg,
        features_t* on_off,
        uint32_t vector_n,
        uint32_t vars_per_vector,
        uint16_t clauses_n,
        uint32_t labels_per_label_t,
        uint32_t &missed_ex,
        uint16_t &wrongly_hit_clause
        );


GLOBALQUALIFIER
void calc_score_kernel(
        features_t* data,
        label_t* label,
        features_t* pos_neg,
        features_t* on_off,
        uint32_t vector_n,
        uint32_t vars_per_vector,
        uint16_t clauses_n,
        uint32_t labels_per_label_t,
        uint32_t* global_score
        );



GLOBALQUALIFIER
void calc_score_given_clause_kernel(
        features_t* data,
        label_t* label,
        features_t* pos_neg,
        features_t* on_off,
        uint32_t vector_n,
        uint32_t vars_per_vector,
        uint16_t clauses_n,
        uint32_t labels_per_label_t,
        uint16_t given_clause,
        uint32_t* global_score
        );
}
