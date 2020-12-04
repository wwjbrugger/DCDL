#include "../inc/sls_gpu.cuh"
#define CLAUSES_N 10
#define VARS_PER_VECTOR 8
namespace gpu {
template <typename T>
DEVICEQUALIFIER
T warp_shfl_down(T var, unsigned int delta) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
        return __shfl_down_sync(0xFFFFFFFF, var, delta, 32);
#else
        return __shfl_down(var, delta, 32);
#endif
}

void sls(uint16_t clauses_n,              // # of DNFs
         uint32_t maxSteps,               // # of Updates
         float p_g1,                      // Prob of rand term in H
         float p_g2,                      // Prob of rand term in H
         float p_s,                       // Prob of rand literal in H
         features_t* data,                // Data input
         label_t* label,                  // Label input
         std::vector<features_t> &pos_neg,// Positive or engative for formula
         std::vector<features_t> &on_off, // Mask for formula
         uint32_t vector_n,               // # of data vectors
         uint32_t features_n,             // # of Features
         bool parallel                    // If gpu should be used
         )
{
        const auto labels_per_label_t = (sizeof(label_t) * 8);
        const label_t one_left = (label_t) (1 << (sizeof(label_t) * 8));
        const uint32_t vars_per_vector = SDIV(features_n, sizeof(features_t) * 8);
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(.0, 1.);

        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        // GPU Init in case of parallel
        ////////////////////////////////////////////////////////////////////////
        features_t* data_D; label_t* label_D;
        uint32_t* global_score_D;
        features_t* pos_neg_D, *on_off_D;

        cudaMalloc(&data_D, sizeof(features_t) * vars_per_vector * vector_n);
        cudaMalloc(&label_D, sizeof(label_t) * (vector_n/labels_per_label_t));
        cudaMalloc(&global_score_D, sizeof(uint32_t));
        cudaMalloc(&pos_neg_D, sizeof(features_t) * vars_per_vector * clauses_n);
        cudaMalloc(&on_off_D, sizeof(features_t) * vars_per_vector * clauses_n);

        cudaMemcpy(data_D,data,sizeof(features_t) * vars_per_vector * vector_n, H2D);
        cudaMemcpy(label_D,label,sizeof(label_t) * (vector_n/labels_per_label_t), H2D);



        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        // Generate starting points for dnf search
        ////////////////////////////////////////////////////////////////////////

        // Create random dnfs
        random_dnf(pos_neg, on_off, vars_per_vector * clauses_n);

        ////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////
        // Algorithm
        ////////////////////////////////////////////////////////////////////////

        // Init score with max
        uint32_t step = 0;
        uint32_t score = 3;

        ////////////////////////////////////////////////////////////////////////
        // Stop if score is almost zero and max num of iterations not reached
        while(//score > 0.0001 &&
                step < maxSteps)
        {
                TIMERSTART(ITER_GPU)
                printf("%d\n", step);
                cudaMemset(global_score_D, 0, sizeof(uint32_t)); CUERR;
                cudaMemcpy(on_off_D,on_off.data(),sizeof(features_t) * vars_per_vector * clauses_n, H2D); CUERR;
                cudaMemcpy(pos_neg_D,pos_neg.data(),sizeof(features_t) * vars_per_vector * clauses_n, H2D); CUERR;
                calc_score_kernel<<<1024, 1024>>>(data_D, label_D, pos_neg_D,on_off_D, vector_n, vars_per_vector, clauses_n, sizeof(label_t) * 8,global_score_D);
                cudaMemcpy(&score,global_score_D,sizeof(uint32_t), D2H); CUERR;

                //std::cout << "Score " << score <<  std::endl;

                ////////////////////////////////////////////////////////////////
                // Get a random missclassified example
                step++;

                uint32_t number_drawn;
                uint16_t clause_drawn;


                draw_missed(data, label, pos_neg.data(), on_off.data(), vector_n, vars_per_vector, clauses_n, sizeof(label_t) * 8,number_drawn, clause_drawn);
                //std::cout<< number_drawn <<std::endl;



                ////////////////////////////////////////////////////////////////
                // Update positive example
                auto pos_within_label_array = number_drawn / (sizeof(label_t) * 8);
                auto pos_within_label = number_drawn % (sizeof(label_t) * 8);
                if(label[pos_within_label_array] & (label_t)(one_left >> pos_within_label))  {
                        uint16_t clause_to_change;
                        uint32_t literal_to_change;

                        ////////////////////////////////////////////////////////
                        // Get a clause
                        if(dis(gen) < p_g1) {
                                clause_to_change = rand() % clauses_n;
                        } else {
                                features_t check = 0;
                                uint32_t diff_min = UINT32_MAX;;
                                uint16_t min_clause = 0;

                                features_t* current_data = &data[number_drawn * vars_per_vector];

                                for(uint16_t current_clause = 0; current_clause < clauses_n; current_clause++) {

                                        features_t* current_pos_neg = &pos_neg.data()[current_clause * vars_per_vector];
                                        features_t* current_on_off = &on_off.data()[current_clause * vars_per_vector];

                                        uint32_t diff = 0;
                                        for(uint32_t current_var = 0; current_var < vars_per_vector; current_var++) {
                                                check = ((current_data[current_var] ^ current_pos_neg[current_var])
                                                         & current_on_off[current_var]);

                                                std::bitset<sizeof(features_t) * 8> bits(check);
                                                diff = diff +  bits.count();
                                        }

                                        if(diff < diff_min) {
                                                diff_min = diff;
                                                min_clause = current_clause;
                                        }

                                }
                                clause_to_change = min_clause;
                        }

                        ////////////////////////////////////////////////////////
                        // Get a literal

                        if(dis(gen) < p_g2) {
                                features_t already_activated = 0;
                                while(!already_activated) {
                                        literal_to_change = rand() % features_n;
                                        uint32_t current_var = literal_to_change / (sizeof(features_t) * 8);
                                        uint32_t current_shift = literal_to_change % (sizeof(features_t) * 8);
                                        already_activated = on_off.data()[vars_per_vector * clause_to_change + vars_per_vector - 1 - current_var] & (features_t)(1 << current_shift);
                                }
                        } else {

                                // More efficient than previous implementation, only slices needed are considered
                                std::vector<features_t>::const_iterator first = on_off.begin() + (vars_per_vector * clause_to_change);
                                std::vector<features_t>::const_iterator last = on_off.begin() + (vars_per_vector * (clause_to_change + 1));
                                std::vector<features_t> on_off_copy(first, last);
                                first = pos_neg.begin() + (vars_per_vector * clause_to_change);
                                last = pos_neg.begin() + (vars_per_vector * (clause_to_change + 1));
                                std::vector<features_t> pos_neg_copy(first, last);

                                uint32_t score_min = UINT32_MAX;;
                                uint16_t min_literal = 0;

                                for(uint32_t current_literal = 0; current_literal < features_n; current_literal++) {
                                        uint32_t current_var = current_literal / (sizeof(features_t) * 8);
                                        uint32_t current_shift = current_literal % (sizeof(features_t) * 8);

                                        features_t already_activated = on_off_copy.data()[vars_per_vector - 1 - current_var] & (features_t)(1 << current_shift);
                                        if(!already_activated) continue; // Not done in prev implementation

                                        on_off_copy.data()[vars_per_vector - 1 - current_var] &= (features_t)(~(1 << current_shift));

                                        uint32_t diff;


                                        cudaMemset(global_score_D, 0, sizeof(uint32_t)); CUERR;
                                        cudaMemcpy(on_off_D,on_off_copy.data(),sizeof(features_t) * vars_per_vector, H2D); CUERR;
                                        cudaMemcpy(pos_neg_D,pos_neg_copy.data(),sizeof(features_t) * vars_per_vector, H2D); CUERR;
                                        calc_score_given_clause_kernel<<<1024, 1024>>>(data_D, label_D, pos_neg_D,on_off_D, vector_n, vars_per_vector, clauses_n, sizeof(label_t) * 8,0, global_score_D);
                                        cudaMemcpy(&diff,global_score_D,sizeof(uint32_t), D2H); CUERR;


                                        on_off_copy.data()[vars_per_vector - 1 - current_var] |= (features_t)((1 << current_shift));

                                        if(diff < score_min) {
                                                score_min = diff;
                                                min_literal = current_literal;
                                        }

                                }
                                literal_to_change = min_literal;

                        }

                        ////////////////////////////////////////////////////////
                        // Do update
                        uint32_t current_var = literal_to_change / (sizeof(features_t) * 8);
                        uint32_t current_shift = literal_to_change % (sizeof(features_t) * 8);

                        on_off.data()[vars_per_vector * clause_to_change + vars_per_vector - 1 - current_var] &= (features_t)(~(1 << current_shift));

                }

                ////////////////////////////////////////////////////////////////
                // Update negative example
                else {
                        uint16_t clause_to_change = clause_drawn;
                        uint32_t literal_to_change;

                        ////////////////////////////////////////////////////////
                        // Get a literal

                        if(dis(gen) < p_s) {
                                features_t already_activated = 1;
                                while(already_activated) {
                                        literal_to_change = rand() % features_n;
                                        uint32_t current_var = literal_to_change / (sizeof(features_t) * 8);
                                        uint32_t current_shift = literal_to_change % (sizeof(features_t) * 8);
                                        already_activated = on_off.data()[vars_per_vector * clause_to_change + vars_per_vector - 1 - current_var] & (features_t)(1 << current_shift);
                                }
                        } else {


                                // More efficient than previous implementation, only slices needed are considered
                                std::vector<features_t>::const_iterator first = on_off.begin() + (vars_per_vector * clause_to_change);
                                std::vector<features_t>::const_iterator last = on_off.begin() + (vars_per_vector * (clause_to_change + 1));
                                std::vector<features_t> on_off_copy(first, last);
                                first = pos_neg.begin() + (vars_per_vector * clause_to_change);
                                last = pos_neg.begin() + (vars_per_vector * (clause_to_change + 1));
                                std::vector<features_t> pos_neg_copy(first, last);

                                uint32_t score_min = UINT32_MAX;;
                                uint16_t min_literal = 0;

                                for(uint32_t current_literal = 0; current_literal < features_n; current_literal++) {
                                        uint32_t current_var = current_literal / (sizeof(features_t) * 8);
                                        uint32_t current_shift = current_literal % (sizeof(features_t) * 8);

                                        features_t already_activated = on_off_copy.data()[vars_per_vector - 1 - current_var] & (features_t)(1 << current_shift);
                                        if(already_activated) continue; // Not done in prev implementation

                                        // Get bit of example to negate in new literal added
                                        features_t to_add = data[number_drawn * vars_per_vector + vars_per_vector - 1 - current_var] & (features_t)(1 << current_shift);
                                        on_off_copy.data()[vars_per_vector - 1 - current_var] |= (features_t)(1 << current_shift);
                                        if(!to_add)
                                                pos_neg_copy.data()[vars_per_vector - 1 - current_var] |= (features_t)(1 << current_shift);
                                        else
                                                pos_neg_copy.data()[vars_per_vector - 1 - current_var] &= (features_t)(~(1 << current_shift));

                                        uint32_t diff;


                                        cudaMemset(global_score_D, 0, sizeof(uint32_t)); CUERR;
                                        cudaMemcpy(on_off_D,on_off_copy.data(),sizeof(features_t) * vars_per_vector, H2D); CUERR;
                                        cudaMemcpy(pos_neg_D,pos_neg_copy.data(),sizeof(features_t) * vars_per_vector, H2D); CUERR;

                                        calc_score_given_clause_kernel<<<1024, 1024>>>(data_D, label_D, pos_neg_D,on_off_D, vector_n, vars_per_vector, clauses_n, sizeof(label_t) * 8,0, global_score_D);
                                        cudaMemcpy(&diff,global_score_D,sizeof(uint32_t), D2H); CUERR;


                                        if(!to_add)
                                                pos_neg_copy.data()[vars_per_vector - 1 - current_var] &= (features_t)(~(1 << current_shift));
                                        else
                                                pos_neg_copy.data()[vars_per_vector - 1 - current_var] |= (features_t)((1 << current_shift));

                                        if(diff < score_min) {
                                                score_min = diff;
                                                min_literal = current_literal;
                                        }

                                }
                                literal_to_change = min_literal;



                        }

                        ////////////////////////////////////////////////////////
                        // Do update
                        uint32_t current_var = literal_to_change / (sizeof(features_t) * 8);
                        uint32_t current_shift = literal_to_change % (sizeof(features_t) * 8);

                        // Get bit of example to negate in new literal added
                        features_t to_add = data[number_drawn * vars_per_vector + vars_per_vector - 1 - current_var] & (features_t)((1 << current_shift));

                        on_off.data()[vars_per_vector * clause_to_change + vars_per_vector - 1 - current_var] |= (features_t)((1 << current_shift));
                        if(!to_add)
                                pos_neg.data()[vars_per_vector * clause_to_change + vars_per_vector - 1 - current_var] |= (features_t)((1 << current_shift));
                        else
                                pos_neg.data()[vars_per_vector * clause_to_change + vars_per_vector - 1 - current_var] &= (features_t)(~(1 << current_shift));


                }
                TIMERSTOP(ITER_GPU)

        }



        cudaFree(data_D); cudaFree(label_D); cudaFree(global_score_D);
        cudaFree(on_off_D); cudaFree(pos_neg_D);



}

// Adaption of original algorithm: draw the first missclassified example
// Lot of speed up.
// WRONG....in case wrongly as 1 it is sufficient if the first var is worngly as 1
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
        )
{
        const label_t one_left = (label_t) (1 << (sizeof(label_t) * 8));
        features_t check = 0;
        for(int32_t pos_within_data = 0; pos_within_data < vector_n; pos_within_data++) {

                bool covered_by_any_clause = 0;
                for(uint16_t pos_within_clauses = 0; pos_within_clauses < clauses_n; pos_within_clauses++) {
                        bool covered_by_clause = 1;
                        for(uint32_t pos_within_vector = 0; pos_within_vector < vars_per_vector; pos_within_vector++) {

                                // If diff between formula and data return 1, if part of formula (1) and different(1) return 1
                                check = ((data[pos_within_data * vars_per_vector + pos_within_vector] ^ pos_neg[pos_within_clauses * vars_per_vector + pos_within_vector])
                                         & on_off[pos_within_clauses * vars_per_vector + pos_within_vector]);
                                if(check != 0) covered_by_clause = 0; break;
                        }
                        if(covered_by_clause) {covered_by_any_clause = 1; wrongly_hit_clause = pos_within_clauses; break;}
                }

                auto pos_within_label_array = pos_within_data / labels_per_label_t;
                auto pos_within_label = pos_within_data % labels_per_label_t;
                if(label[pos_within_label_array] & (label_t)(one_left >> pos_within_label)) {
                        if(!covered_by_any_clause) {missed_ex = pos_within_data; return;}
                }
                else{
                        if(covered_by_any_clause) {missed_ex = pos_within_data; return;}
                }

        }
}


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
        )
{
        const auto local_thread_id = threadIdx.x;
        const auto block_stride = blockDim.x;
        //const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        const auto lane_id   = threadIdx.x % 32;
        const auto var_of_vector_id = lane_id % vars_per_vector;

        const label_t one_left = (label_t) (1 << (sizeof(label_t) * 8));

        __shared__ features_t local_pos_neg[CLAUSES_N * VARS_PER_VECTOR];
        __shared__ features_t local_on_off[CLAUSES_N * VARS_PER_VECTOR];

        // Load formulas into shared memory
        for(uint32_t i = local_thread_id; i < (clauses_n * vars_per_vector); i += block_stride) {
                local_pos_neg[i] = pos_neg[i];
                local_on_off[i]  = on_off[i];
        }
        __syncthreads();

        uint32_t score = 0;
        features_t check = 0;
        bool covered_by_any_clause = 0;

        for(int32_t pos_within_data = local_thread_id; pos_within_data < (vector_n * vars_per_vector); pos_within_data += block_stride) {
                covered_by_any_clause = 0;
                for(uint16_t pos_within_clauses = 0; pos_within_clauses < clauses_n; pos_within_clauses++) {
                        features_t current_data = data[pos_within_data];
                        // If diff between formula and data return 1, if part of formula (1) and different(1) return 1
                        check = ((current_data ^ local_pos_neg[pos_within_clauses * vars_per_vector + var_of_vector_id])
                                 & local_on_off[pos_within_clauses * vars_per_vector + var_of_vector_id]);

                        // Could be done faster with butterfly reduction and breaks
                        for (uint8_t offset = vars_per_vector / 2; offset > 0; offset >>= 1) check |= warp_shfl_down(check, offset);

                        if(var_of_vector_id == 0 && !check) {covered_by_any_clause = 1;}
                }

                if(var_of_vector_id == 0) {
                        auto pos_within_data_array = (pos_within_data/vars_per_vector);
                        auto pos_within_label_array = pos_within_data_array / labels_per_label_t;
                        auto pos_within_label = pos_within_data_array % labels_per_label_t;


                        if(label[pos_within_label_array] & (label_t)(one_left >> pos_within_label) ) {
                                if(!covered_by_any_clause) score++;
                        }
                        else{
                                if(covered_by_any_clause) score++;
                        }
                }
        }

        // Could be optimized by smalle warp reduction i.e. by only using var_of_vector_id == 0
        for (uint8_t offset = 32 / 2; offset > 0; offset >>= 1) score += warp_shfl_down(score, offset);
        if(lane_id == 0) atomicAdd(global_score, score);
}


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
        )
{
        const auto local_thread_id = threadIdx.x;
        const auto block_stride = blockDim.x;
        //const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        const auto lane_id   = threadIdx.x % 32;
        const auto var_of_vector_id = lane_id % vars_per_vector;

        const label_t one_left = (label_t) (1 << (sizeof(label_t) * 8));

        __shared__ features_t local_pos_neg[VARS_PER_VECTOR];
        __shared__ features_t local_on_off[VARS_PER_VECTOR];

        // Load formulas into shared memory
        for(uint32_t i = local_thread_id; i < (clauses_n * vars_per_vector); i += block_stride) {
                local_pos_neg[i] = pos_neg[i];
                local_on_off[i]  = on_off[i];
        }
        __syncthreads();

        uint32_t score = 0;
        features_t check = 0;
        bool covered_by_any_clause = 0;
        uint16_t pos_within_clauses = given_clause;

        for(int32_t pos_within_data = local_thread_id; pos_within_data < (vector_n * vars_per_vector); pos_within_data += block_stride) {
                covered_by_any_clause = 0;

                features_t current_data = data[pos_within_data];
                // If diff between formula and data return 1, if part of formula (1) and different(1) return 1
                check = ((current_data ^ local_pos_neg[pos_within_clauses * vars_per_vector + var_of_vector_id])
                         & local_on_off[pos_within_clauses * vars_per_vector + var_of_vector_id]);

                // Could be done faster with butterfly reduction and breaks
                for (uint8_t offset = vars_per_vector / 2; offset > 0; offset >>= 1) check |= warp_shfl_down(check, offset);

                if(var_of_vector_id == 0) {
                        if(!check) covered_by_any_clause = 1;
                        auto pos_within_data_array = (pos_within_data/vars_per_vector);
                        auto pos_within_label_array = pos_within_data_array / labels_per_label_t;
                        auto pos_within_label = pos_within_data_array % labels_per_label_t;


                        if(label[pos_within_label_array] & (label_t)(one_left >> pos_within_label) ) {
                                if(!covered_by_any_clause) score++;
                        }
                        else{
                                if(covered_by_any_clause) score++;
                        }
                }
        }

        // Could be optimized by smalle warp reduction i.e. by only using var_of_vector_id == 0
        for (uint8_t offset = 32 / 2; offset > 0; offset >>= 1) score += warp_shfl_down(score, offset);
        if(lane_id == 0) atomicAdd(global_score, score);
}

void random_dnf(
        std::vector<features_t> &pos_neg, // Positive or negative literal
        std::vector<features_t> &on_off,  // If literal is needed
        features_t dnf_s
        )
{
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<features_t> dis;

        for(size_t i = 0; i < dnf_s; i++)
        {
                pos_neg.push_back(dis(gen)); on_off.push_back(dis(gen));
        }
}
}
