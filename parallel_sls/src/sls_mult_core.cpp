#include "../inc/sls_multi_core.h"
#include "omp.h"
#include <sstream>

#define BATCH_SIZE 1024
#define RESTART_ITER 600
#define PRINT_EVERY 100
#define VERBOSITY 1

namespace multi_core {

    std::random_device rd;    //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());   //Standard mersenne_twister_engine seeded with rd()

    std::uniform_real_distribution<> dis(.0, 1.);

    uint32_t sls(uint32_t clauses_n,               // # of DNFs
                 uint32_t maxSteps,                // # of Updates
                 float p_g1,                       // Prob of rand term in H
                 float p_g2,                       // Prob of rand term in H
                 float p_s,                        // Prob of rand literal in H
                 features_t *data,                 // Data input
                 bool *label,                      // Label input
                 features_t *pos_neg,              // Positive or negative for formula
                 features_t *on_off,               // Mask for formula
                 features_t *pos_neg_to_store,     // Positive or negative for formula
                 features_t *on_off_to_store,      // Mask for formula
                 uint32_t vector_n,                // # of data vectors
                 uint32_t features_n,              // # of Features
                 const bool batch,                 // If score calculation should be done batchwise for given clause
                 const bool cold_restart,          // Restart if stuck in bad local minimum
                 const float decay,                // Decay factor, could be zero. Up to min_prob
                 const float min_prob,             // Not decay below this threshold
                 const bool zero_init              // Wether to go bigger steps in case of no sucess
    ) {
        return sls_val(clauses_n, maxSteps, p_g1, p_g2, p_s, data, label, data, label, pos_neg, on_off,
                       pos_neg_to_store, on_off_to_store,
                       vector_n, vector_n, features_n, batch, cold_restart, decay, min_prob, zero_init);
    }

    uint32_t sls_val(uint32_t clauses_n,                        // # of DNFs
                     uint32_t maxSteps,                     // # of Updates
                     float p_g1,                            // Prob of rand term in H
                     float p_g2,                            // Prob of rand term in H
                     float p_s,                             // Prob of rand literal in H
                     features_t *data,                      // Data input
                     bool *label,                           // Label input
                     features_t *data_val,                  // Data input
                     bool *label_val,                       // Label input
                     features_t *pos_neg,                   // Positive or engative for formula
                     features_t *on_off,                    // Mask for formula
                     features_t *pos_neg_to_store,          // Positive or engative for formula
                     features_t *on_off_to_store,           // Mask for formula
                     uint32_t vector_n,                     // # of data vectors
                     uint32_t vector_n_val,                 // # of data vectors
                     uint32_t features_n,                   // # of Features
                     const bool batch,                      // If score calculation should be done batchwise for given clause
                     const bool cold_restart,               // Restar if stuck in bad local minimum
                     const float decay,                     // Decay factor, could be zero. Up to min_prob
                     const float min_prob,                  // Not decay below this threshold
                     const bool zero_init                   // Wether to go bigger steps in case of no sucess
    ) {
        return sls_test(clauses_n, maxSteps, p_g1, p_g2, p_s, data, label, data_val, label_val, data_val, label_val,
                        pos_neg, on_off, pos_neg_to_store, on_off_to_store,
                        vector_n, vector_n_val, vector_n_val, features_n, batch, cold_restart, decay, min_prob,
                        zero_init);
    }

    uint32_t sls_test(uint32_t clauses_n,                        // # of DNFs
                      uint32_t maxSteps,                    // # of Updates
                      float p_g1,                           // Prob of rand term in H
                      float p_g2,                           // Prob of rand term in H
                      float p_s,                            // Prob of rand literal in H
                      features_t *data,                     // Data input
                      bool *label,                          // Label input
                      features_t *data_val,                 // Data input
                      bool *label_val,                      // Label input
                      features_t *data_test,                 // Data input
                      bool *label_test,                      // Label input
                      features_t *pos_neg,                  // Positive or negative for formula
                      features_t *on_off,                   // Mask for formula
                      features_t *pos_neg_to_store,         // Positive or engative for formula
                      features_t *on_off_to_store,          // Mask for formula
                      uint32_t vector_n,                    // # of data vectors
                      uint32_t vector_n_val,                // # of data vectors
                      uint32_t vector_n_test,                // # of data vectors
                      uint32_t features_n,                  // # of Features
                      const bool batch,                     // If score calculation should be done batchwise for given clause
                      const bool cold_restart,              // Restart if stuck in bad local minimum
                      const float decay,                    // Decay factor, could be zero. Up to min_prob
                      const float min_prob,                 // Not decay below this threshold
                      const bool zero_init                  // Wether to go bigger steps in case of no sucess
    ) {

        std::cout << std::endl<< std::endl<< "++++++++++++++++++++++++++++++++++++++++"<< std::endl
                  << "Started SLS using "<< omp_get_max_threads () << " Threads with Params: " << std::endl
                  << "++++++++++++++++++++++++++++++++++++++++"<< std::endl
                  << std::endl
                  << "Number of Clauses        " << clauses_n << std::endl
                  << "Maximum Steps            " << maxSteps << std::endl
                  << "P_g1                     " << p_g1 << std::endl
                  << "P_g2                     " << p_g2 << std::endl
                  << "P_s                      " << p_s << std::endl
                  << "++++++++++++++++++++++++++++++++++++++++"<< std::endl
                  << "Number of Features       " << features_n << std::endl
                  << "Training Set Size        " << vector_n << std::endl
                  << "Validation Set Size      " << vector_n_val << std::endl
                  << "Test Set Size            " << vector_n_test << std::endl
                  << "++++++++++++++++++++++++++++++++++++++++"<< std::endl

                  << "Decay of Probs           " << decay << std::endl
                  << "Minimum  Probs           " << min_prob << std::endl
                  << "++++++++++++++++++++++++++++++++++++++++"<< std::endl
                  << "Batchwise Calculation    " << batch << std::endl
                  << "Cold Restarts            " << batch << std::endl
                  << "Zero Init                " << zero_init << std::endl
                  << "++++++++++++++++++++++++++++++++++++++++"<< std::endl<<std::endl;

        //How many vars are needed for one instance
        uint32_t vars_per_vector = SDIV(features_n, sizeof(features_t) * 8);

        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        // Generate starting points for DNF search
        ////////////////////////////////////////////////////////////////////////

        if (!zero_init) // Create random dnfs
            random_dnf(pos_neg, on_off, vars_per_vector * clauses_n);
        else // Zero init/home/jannis
            zero_dnf(pos_neg, on_off, vars_per_vector * clauses_n);

        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        // Algorithm
        ////////////////////////////////////////////////////////////////////////

        // Initialize global vars
        uint32_t step = 0; // count interations
        uint32_t steps_unchanged = 0;         // checks iterations since last improvement
        // set to maximal value
        uint32_t wrongly_negative = 4294967295,   wrongly_positive = 4294967295; // get score per class
        uint32_t score = calc_score(data_val, label_val, pos_neg, on_off, vector_n_val, vars_per_vector, clauses_n,
                                    wrongly_negative, wrongly_positive); // calc initial score

        uint32_t min_score = UINT32_MAX, min_unchanged = UINT32_MAX, min_wrongly_negative = 0, min_wrongly_positive = 0, min_score_since_last_printed = UINT32_MAX; // set all tracked scores to init values
        // std::cout << " score  before algorithm start:" << score  <<std::endl;
        // Stop if score is zero and max num of iterations not reached
        while (score > 0 &&
               step < maxSteps) {

                    // Print intermediate results if needed (different styles)
                    if(step % PRINT_EVERY == 1)
                            if(VERBOSITY) {
                                    //std::cout << '\t'<< "step: " << step ;// << " Min Score "  << min_score << " Wrongly classified as negative " << min_wrongly_negative << " Wrongly classified as positive " << min_wrongly_positive; // << std::endl;
                            }
                            else{
                                    std::cout << min_score << " " << min_score_since_last_printed << std::endl;
                                    min_score_since_last_printed = UINT32_MAX;
                            }

            // Calculate validation score
            score = calc_score(data_val, label_val, pos_neg, on_off, vector_n_val, vars_per_vector, clauses_n,
                               wrongly_negative, wrongly_positive);

            // Track if improving...
            if (cold_restart) {
                if (score < min_score) {
                    steps_unchanged = 0;
                } else {
                    steps_unchanged++;
                    if (steps_unchanged > RESTART_ITER) {
                        random_dnf(pos_neg, on_off, vars_per_vector * clauses_n);
                        steps_unchanged = 0;
                        //break;
                    }
                }
            }

            // Update minimum score
            if (score < min_score) {
                min_score = score;
                min_wrongly_negative = wrongly_negative;
                min_wrongly_positive = wrongly_positive;

                //Update and store best formula so far
#pragma omp parallel for
                for (uint32_t ind = 0; ind < (clauses_n * vars_per_vector); ind++) {
                    pos_neg_to_store[ind] = pos_neg[ind];
                    on_off_to_store[ind] = on_off[ind];
                }

            }

            // Increment iteration if not "breaked" above
            step++;
            // currently not needed
            if (score < min_score_since_last_printed) min_score_since_last_printed = score;

            // Get a random misclassified example
            uint32_t number_drawn;
            uint32_t clause_drawn;
            draw_missed(data, label, pos_neg, on_off, vector_n, vars_per_vector, clauses_n, number_drawn, clause_drawn);


            // Update positive example
            if (label[number_drawn]) {
                uint32_t clause_to_change;
                uint32_t literal_to_change;
                uint32_t diff_min = UINT32_MAX;
                const features_t *current_data = &data[number_drawn * vars_per_vector];
                features_t check = 0;

                // Get a clause
                if (dis(gen) < p_g1) { // get random clause
                    clause_to_change = rand() % clauses_n;
                    const features_t *current_pos_neg = &pos_neg[clause_to_change * vars_per_vector];
                    const features_t *current_on_off = &on_off[clause_to_change * vars_per_vector];
                } else { // Get clause that differs least
                    uint32_t min_clause = 0;
                    for (uint32_t current_clause = 0; current_clause < clauses_n; current_clause++) {

                        const features_t *current_pos_neg = &pos_neg[current_clause * vars_per_vector];
                        const features_t *current_on_off = &on_off[current_clause * vars_per_vector];

                        uint32_t diff = 0;
                        //calculate difference of the complete clause
                        for (uint32_t current_var = 0; current_var < vars_per_vector; current_var++) {
                            check = ((current_data[current_var] ^ current_pos_neg[current_var])
                                     & current_on_off[current_var]);

                            std::bitset<sizeof(features_t) * 8> bits(check);
                            diff = diff + bits.count();
                        }

                        if (diff < diff_min) {
                            diff_min = diff;
                            min_clause = current_clause;
                        }

                    }
                    clause_to_change = min_clause;

                }

                // Get a literal
                if (dis(gen) < p_g2) { // Gets a random activated literal
                    features_t already_activated = 0;
                    uint32_t current_var;
                    uint32_t current_shift;
                    while (!already_activated) {
                        literal_to_change = rand() % features_n;
                        current_var = literal_to_change / (sizeof(features_t) * 8);
                        current_shift = literal_to_change % (sizeof(features_t) * 8);
                        already_activated =
                                on_off[vars_per_vector * clause_to_change + vars_per_vector - 1 - current_var] &
                                (features_t) (1 << current_shift);
                    }

                    // Do update
                    on_off[vars_per_vector * clause_to_change + vars_per_vector - 1 - current_var] &= (features_t) (~(1
                            << current_shift));

                } else { // Corrects all differing literals
                    uint32_t score_min = UINT32_MAX;
                    uint32_t min_literal = 0;

                    for (uint32_t current_literal = 0; current_literal < features_n; current_literal++) {
                        uint32_t current_var = current_literal / (sizeof(features_t) * 8);
                        uint32_t current_shift = current_literal % (sizeof(features_t) * 8);

                        features_t already_activated =
                                on_off[vars_per_vector * clause_to_change + vars_per_vector - 1 - current_var] &
                                (features_t) (1 << current_shift);
                        if (!already_activated) continue;
                        if (!((pos_neg[vars_per_vector * clause_to_change + vars_per_vector - 1 - current_var] &
                               (features_t) (1 << current_shift))
                              ^ (data[number_drawn * vars_per_vector + vars_per_vector - 1 - current_var] &
                                 (features_t) (1 << current_shift))))
                            continue;

                        // Do update
                        on_off[vars_per_vector * clause_to_change + vars_per_vector - 1 -
                               current_var] &= (features_t) (~(1 << current_shift));

                    }

                }

            }


                // Update negative example
            else {
                // Get a clause
                // Already done at the drawing of an wrong instance
                const uint32_t clause_to_change = clause_drawn;
                uint32_t literal_to_change;


                // Get a literal
                if (dis(gen) < p_s) { // Gets literal by chance
                    literal_to_change = rand() % features_n;
                } else {  // Get most decreasing literal

                    uint32_t score_min = UINT32_MAX;;
                    uint32_t min_literal = 0;

                    for (uint32_t current_literal = 0; current_literal < features_n; current_literal++) {
                        uint32_t current_var = current_literal / (sizeof(features_t) * 8);
                        uint32_t current_shift = current_literal % (sizeof(features_t) * 8);

                        const features_t already_activated =
                                on_off[vars_per_vector * clause_to_change + vars_per_vector - 1 - current_var] &
                                (features_t) (1 << current_shift);

                        // Get bit of example to negate in new literal added
                        const features_t to_add =
                                data[number_drawn * vars_per_vector + vars_per_vector - 1 - current_var] &
                                (features_t) (1 << current_shift);


                        // Do temporary changes on formula for evaluating
                        if (!already_activated)
                            on_off[vars_per_vector * clause_to_change + vars_per_vector - 1 -
                                   current_var] |= (features_t) (1 << current_shift);

                        if (!to_add)
                            pos_neg[vars_per_vector * clause_to_change + vars_per_vector - 1 -
                                    current_var] |= (features_t) (1 << current_shift);
                        else
                            pos_neg[vars_per_vector * clause_to_change + vars_per_vector - 1 -
                                    current_var] &= (features_t) (~(1 << current_shift));

                        // Calculate new score including changes
                        uint32_t diff;

                        if (!batch) {
                            //  std::cout << "calc_score_given_clause is called" <<std::endl;
                            diff = calc_score_given_clause(data, label, pos_neg, on_off, vector_n, vars_per_vector,
                                                           clauses_n, clause_to_change);
                        } else {    // wird aufgerufen
                            //  std::cout << "calc_score_given_clause_batch is called" <<std::endl;
                            diff = calc_score_given_clause_batch(data, label, pos_neg, on_off, vector_n,
                                                                 vars_per_vector, clauses_n, clause_to_change,
                                                                 BATCH_SIZE);
                        }
                        // Undo temporary changes
                        if (!already_activated)
                            on_off[vars_per_vector * clause_to_change + vars_per_vector - 1 -
                                   current_var] &= (features_t) (~(1 << current_shift));
                        if (!to_add)
                            pos_neg[vars_per_vector * clause_to_change + vars_per_vector - 1 -
                                    current_var] &= (features_t) (~(1 << current_shift));
                        else
                            pos_neg[vars_per_vector * clause_to_change + vars_per_vector - 1 -
                                    current_var] |= (features_t) ((1 << current_shift));

                        if (diff < score_min) {
                            score_min = diff;
                            min_literal = current_literal;
                        }

                    }
                    literal_to_change = min_literal;
                }

                // Do update
                uint32_t current_var = literal_to_change / (sizeof(features_t) * 8);
                uint32_t current_shift = literal_to_change % (sizeof(features_t) * 8);
                // Get bit of example to negate in new literal added
                features_t to_add = data[number_drawn * vars_per_vector + vars_per_vector - 1 - current_var] &
                                    (features_t) ((1 << current_shift));
                on_off[vars_per_vector * clause_to_change + vars_per_vector - 1 - current_var] |= (features_t) ((1
                        << current_shift));
                if (!to_add)
                    pos_neg[vars_per_vector * clause_to_change + vars_per_vector - 1 - current_var] |= (features_t) ((1
                            << current_shift));
                else
                    pos_neg[vars_per_vector * clause_to_change + vars_per_vector - 1 - current_var] &= (features_t) (~(1
                            << current_shift));
            }

            // Decay probs
            p_g1 -= decay * p_g1;
            p_s -= decay * p_s;
            p_g1 = std::max(p_g1, min_prob);
            p_s = std::max(p_s, min_prob);

        }
        std::cout << '\t' << "step: " << step << " Min Score " << min_score << " Wrongly classified as negative "
                  << min_wrongly_negative << " Wrongly classified as positive " << min_wrongly_positive << std::endl;
        return min_score;


    }


    void draw_missed(
            features_t *data,
            bool *label,
            features_t *pos_neg,
            features_t *on_off,
            uint32_t vector_n,
            uint32_t vars_per_vector,
            uint32_t clauses_n,
            uint32_t &missed_ex,
            uint32_t &wrongly_hit_clause
    ) {
        // Generate random starting point and return the first drawn misclassified instance
        std::uniform_int_distribution <int32_t> uniform_dist(0, vector_n);
        int32_t start = uniform_dist(gen);

        features_t check = 0;
        // Iterate through all data
        for (int32_t i = 0; i < vector_n; i++) {
            // Assure data boundaries
            int32_t pos_within_data = (start + i) % vector_n;
            bool covered_by_any_clause = 0;
            // Check if covered by any claus
            for (uint32_t pos_within_clauses = 0; pos_within_clauses < clauses_n; pos_within_clauses++) {
                bool covered_by_clause = 1;
                // Check if covered by given clause
                for (uint32_t pos_within_vector = 0; pos_within_vector < vars_per_vector; pos_within_vector++) {

                    // If diff between formula and data return 1, if part of formula (1) and different(1) return 1
                    check = ((data[pos_within_data * vars_per_vector + pos_within_vector] ^
                              pos_neg[pos_within_clauses * vars_per_vector + pos_within_vector])
                             & on_off[pos_within_clauses * vars_per_vector + pos_within_vector]);
                    if (check != 0) {
                        covered_by_clause = 0;
                        break;
                    }
                }
                // In case wrongly covered, store which clause wrongly covered example
                if (covered_by_clause) {
                    covered_by_any_clause = 1;
                    wrongly_hit_clause = pos_within_clauses;
                    break;
                }
            }

            // Ehm, yeah. actually not needed currently.
            if (label[pos_within_data]) {
                if (!covered_by_any_clause) {
                    missed_ex = pos_within_data;
                    return;
                }
            } else {
                if (covered_by_any_clause) {
                    missed_ex = pos_within_data;
                    return;
                }
            }

        }

        // Ensure that anything is returned
        missed_ex = 0;
        wrongly_hit_clause = 0;
    }


// SAME FORE GIVEN CLAUSE
    uint32_t calc_score(
            const features_t *data,
            const bool *label,
            const features_t *pos_neg,
            const features_t *on_off,
            const uint32_t vector_n,
            const uint32_t vars_per_vector,
            const uint32_t clauses_n,
            uint32_t &wrongly_negative,
            uint32_t &wrongly_positive
    ) {

        wrongly_negative = 0; // count wrongly classified
        wrongly_positive = 0; // count wrongly classified

        // delete me
        uint32_t true_negative = 0;
        uint32_t true_posetive = 0;

        // delete me

        features_t check = 0; // global auxilliary var
        bool covered_by_any_clause = 0;
        bool covered_by_clause = 1;
        //std::cout <<" reached  score calculation " <<std::endl;


        // iterate through data completly
#pragma omp parallel for private(check, covered_by_any_clause, covered_by_clause) reduction(+: wrongly_negative) reduction(+: wrongly_positive)
        for (int32_t pos_within_data = 0; pos_within_data < vector_n; pos_within_data++) {
            //std::cout << " reached first for loop in scorefunction: " << std::endl;
            covered_by_any_clause = 0;
            // check if covered by any clause
            for (uint32_t pos_within_clauses = 0; pos_within_clauses < clauses_n; pos_within_clauses++) {
                covered_by_clause = 1;
                // check if covered by given clause
                for (uint32_t pos_within_vector = 0; pos_within_vector < vars_per_vector; pos_within_vector++) {
                    // If diff between formula and data return 1, if part of formula (1) and different(1) return 1
                    // ^ is a bitwise xor command values must be diffrent for 1
                    // bitwise and
                    // int position = pos_within_data * vars_per_vector + pos_within_vector;
                    //std::cout << " position :        " << position << std::endl;

                    check = ((data[pos_within_data * vars_per_vector + pos_within_vector] ^
                              pos_neg[pos_within_clauses * vars_per_vector + pos_within_vector])
                             & on_off[pos_within_clauses * vars_per_vector + pos_within_vector]);
                    if (check != 0) {
                        covered_by_clause = 0;
                        break;
                    }
                }
                if (covered_by_clause) {
                    covered_by_any_clause = 1;
                    break;
                }
            }

            // Count class of failure
            if (label[pos_within_data]) {
                if (!covered_by_any_clause) wrongly_negative++;
            } else {
                if (covered_by_any_clause) wrongly_positive++;
            }

        }

        // return total of misclassified
        return wrongly_negative + wrongly_positive;
    }

    uint32_t calc_score_batch(
            const features_t *data,
            const bool *label,
            const features_t *pos_neg,
            const features_t *on_off,
            const uint32_t vector_n,
            const uint32_t vars_per_vector,
            const uint32_t clauses_n,
            const uint32_t batch_size
    ) {
        uint32_t score = 0;
        features_t check;

        // In contrast to method above only calculation over random batch
        // No split up into class of failure needed

        // Get random starting point for batch
        std::mt19937_64 geni(rd());
        std::uniform_int_distribution <int32_t> uniform_dist(0, vector_n - batch_size);
        const int32_t start = uniform_dist(geni);

#pragma omp parallel
        {       // iterate through data batchwise
#pragma omp for private(check) reduction(+: score)
            for (int32_t pos_within_data_out = 0; pos_within_data_out < (batch_size); pos_within_data_out++) {

                int32_t pos_within_data = (start + pos_within_data_out) % vector_n;
                bool covered_by_any_clause = 0;
                // check if covered by any clause
                for (uint32_t pos_within_clauses = 0; pos_within_clauses < clauses_n; pos_within_clauses++) {
                    bool covered_by_clause = 1;
                    // check if covered by given clause
                    for (uint32_t pos_within_vector = 0; pos_within_vector < vars_per_vector; pos_within_vector++) {

                        // If diff between formula and data return 1, if part of formula (1) and different(1) return 1
                        check = ((data[pos_within_data * vars_per_vector + pos_within_vector] ^
                                  pos_neg[pos_within_clauses * vars_per_vector + pos_within_vector])
                                 & on_off[pos_within_clauses * vars_per_vector + pos_within_vector]);
                        if (check != 0) {
                            covered_by_clause = 0;
                            break;
                        }
                    }
                    if (covered_by_clause) {
                        covered_by_any_clause = 1;
                        break;
                    }
                }

                if (label[pos_within_data]) {
                    if (!covered_by_any_clause) score++;
                } else {
                    if (covered_by_any_clause) score++;
                }

            }
        }

        return score;
    }


    uint32_t calc_score_given_clause(
            features_t *data,
            bool *label,
            features_t *pos_neg,
            features_t *on_off,
            uint32_t vector_n,
            uint32_t vars_per_vector,
            uint32_t clauses_n,
            uint32_t given_clause
    ) {
        uint32_t score = 0;
        features_t check;
        const uint32_t pos_within_clauses = given_clause;
#pragma omp parallel for private(check) reduction(+: score)
        for (int32_t pos_within_data = 0; pos_within_data < vector_n; pos_within_data++) {
            bool covered_by_clause = 1;
            for (uint32_t pos_within_vector = 0; pos_within_vector < vars_per_vector; pos_within_vector++) {
                // If diff between formula and data return 1, if part of formula (1) and different(1) return 1
                check = ((data[pos_within_data * vars_per_vector + pos_within_vector] ^
                          pos_neg[pos_within_clauses * vars_per_vector + pos_within_vector])
                         & on_off[pos_within_clauses * vars_per_vector + pos_within_vector]);
                if (check != 0) {
                    covered_by_clause = 0;
                    break;
                }
            }

            if (label[pos_within_data]) {
                if (!covered_by_clause) score++;
            } else {
                if (covered_by_clause) score++;
            }

        }

        return score;
    }


    uint32_t calc_score_given_clause_batch(
            features_t *data,
            bool *label,
            features_t *pos_neg,
            features_t *on_off,
            uint32_t vector_n,
            uint32_t vars_per_vector,
            uint32_t clauses_n,
            uint32_t given_clause,
            uint32_t batch_size
    ) {
        uint32_t score = 0;
        features_t check;
        const uint32_t pos_within_clauses = given_clause;
        std::mt19937_64 geni(rd());
        std::uniform_int_distribution <int32_t> uniform_dist(0, vector_n - batch_size);

        const int32_t start = uniform_dist(geni);

#pragma omp parallel
        {
#pragma omp for private(check) reduction(+: score)
            for (int32_t pos_within_data_out = 0; pos_within_data_out < (batch_size); pos_within_data_out++) {

                int32_t pos_within_data = (start + pos_within_data_out) % vector_n;

                bool covered_by_clause = 1;
                for (uint32_t pos_within_vector = 0; pos_within_vector < vars_per_vector; pos_within_vector++) {
                    // If diff between formula and data return 1, if part of formula (1) and different(1) return 1
                    check = ((data[pos_within_data * vars_per_vector + pos_within_vector] ^
                              pos_neg[pos_within_clauses * vars_per_vector + pos_within_vector])
                             & on_off[pos_within_clauses * vars_per_vector + pos_within_vector]);
                    if (check != 0) {
                        covered_by_clause = 0;
                        break;
                    }
                }

                if (label[pos_within_data]) {
                    if (!covered_by_clause) score++;
                } else {
                    if (covered_by_clause) score++;
                }
            }

        }
        return score;
    }

    uint32_t count_set_bits(const features_t *to_count,
                            const uint32_t vars_per_vector,
                            const uint32_t clauses_n
    ) {
        uint32_t count = 0;
        for (uint32_t current_clause = 0; current_clause < clauses_n; current_clause++) {
            const features_t *current_to_count = &to_count[current_clause * vars_per_vector];
            for (uint32_t current_var = 0; current_var < vars_per_vector; current_var++) {
                std::bitset<sizeof(features_t) * 8> bits(current_to_count[current_var]);
                count += bits.count();
            }
        }
        return count;
    }

    void random_dnf(
            features_t *pos_neg,       // Positive or nengative for formula
            features_t *on_off,        // Mask for formula
            uint32_t dnf_s
    ) {

        std::mt19937_64 geni(rd());
        std::uniform_int_distribution <int8_t> uniform_dist(0, 255);
        // std::cout << "random seed " << rd() << std::endl;
        //std::stringstream ss ;
        // std::cout << "random_dnf values of pos_neg: " << std::endl;
#pragma omp parallel for
        for (uint32_t i = 0; i < (uint32_t) dnf_s; i++) {
            pos_neg[i] = uniform_dist(geni);
            on_off[i] = uniform_dist(geni);

        }


    }

    void zero_dnf(
            features_t *pos_neg,       // Positive or negative for formula
            features_t *on_off,        // Mask for formula
            uint32_t dnf_s
    ) {

        std::mt19937_64 geni(rd());
        std::uniform_int_distribution <int8_t> uniform_dist(0, 255);

#pragma omp parallel for
        for (uint32_t i = 0; i < (uint32_t) dnf_s; i++) {
            pos_neg[i] = uniform_dist(geni);
            on_off[i] = 0;
        }


    }


// SAME FORE GIVEN CLAUSE
    void calc_prediction(
            const features_t *data,             // input data
            bool *prediction_label,       // space to store prediction
            const features_t *pos_neg,          //  part 1 from logic formula which variables should be negated?
            const features_t *on_off,           // part 2 from logic formula which variables are relevant
            const uint32_t vector_n,            // # of data vectors
            const uint32_t clauses_n,          // # of DNFs
            uint32_t features_n              // # of Features
    ) {
        //How many vars are needed for one instance
        uint32_t vars_per_vector = SDIV(features_n, sizeof(features_t) * 8);

        features_t check = 0; // global auxilliary var
        bool covered_by_any_clause = 0;
        bool covered_by_clause = 1;
        //std::cout <<" reached  score calculation " <<std::endl;
        int clause_covered[clauses_n] = {0};
        // iterate through data completly
//#pragma omp parallel for private(check, covered_by_any_clause, covered_by_clause)
        for (int32_t pos_within_data = 0;
             pos_within_data < vector_n; pos_within_data++) {   // iteration through dataset
            //std::cout << " reached first for loop in scorefunction: " << std::endl;
            covered_by_any_clause = 0;
            // check if covered by any clause
            for (uint32_t pos_within_clauses = 0; pos_within_clauses < clauses_n; pos_within_clauses++) {
                covered_by_clause = 1;
                // check if covered by given clause
                for (uint32_t pos_within_vector = 0; pos_within_vector < vars_per_vector; pos_within_vector++) {
                    // If diff between formula and data return 1, if part of formula (1) and different(1) return 1
                    // ^ is a bitwise xor command values must be diffrent for 1
                    // bitwise and
                    // int position = pos_within_data * vars_per_vector + pos_within_vector;
                    //std::cout << " position :        " << position << std::endl;

                    check = ((data[pos_within_data * vars_per_vector + pos_within_vector] ^
                              pos_neg[pos_within_clauses * vars_per_vector + pos_within_vector])
                             & on_off[pos_within_clauses * vars_per_vector + pos_within_vector]);
                    if (check != 0) {
                        covered_by_clause = 0;

                        break;
                    }
                }
                if (covered_by_clause) {
                    clause_covered[pos_within_clauses] +=1;
                    covered_by_any_clause = 1;
                    break;
                }
            }

            prediction_label[pos_within_data] = covered_by_any_clause;

        }

        std::cout  <<  std::endl;

    }

}

