#pragma once

#include "hy_mt2/decoder_layer.h"
#include "hy_mt2/tensor.h"
#include "hy_mt2/weights.h"

#include <acl/acl.h>

#include <cstdint>
#include <string>
#include <vector>

namespace hy_mt2 {

struct LanguageModelConfig {
    int64_t hidden_size{2048};
    int64_t intermediate_size{6144};
    int64_t num_q_heads{16};
    int64_t num_kv_heads{4};
    int64_t head_dim{128};
    int64_t num_layers{32};
    int64_t vocab_size{120818};
    int64_t max_position_embeddings{262144};
    double rope_theta{10000.0};
    double rope_scaling_factor{1.0};
    double rms_epsilon{1e-5};
    int64_t bos_token_id{120000};
    int64_t eos_token_id{120020};
};

LanguageModelConfig default_hy_mt2_config();

struct LayerWeights {
    Tensor input_norm_w;
    Tensor post_norm_w;
    Tensor q_w;
    Tensor k_w;
    Tensor v_w;
    Tensor o_w;
    Tensor q_norm_w;
    Tensor k_norm_w;
    Tensor gate_w;
    Tensor up_w;
    Tensor down_w;
};

struct LmHeadChunk {
    int64_t start_vocab{0};
    int64_t valid_vocab{0};
    Tensor weight;  // [hidden, padded_vocab_chunk]
};

struct LanguageModelWeights {
    Tensor embed;
    Tensor final_norm_w;
    std::vector<LayerWeights> layers;
    std::vector<LmHeadChunk> lm_head_chunks;
};

struct DecodeState {
    int64_t max_seq_len{0};
    int64_t seq_len{0};
    std::vector<LayerCache> layers;
    DecoderStepScratch scratch;
};

LanguageModelWeights load_language_model_weights(WeightsIndex& index,
                                                 const LanguageModelConfig& cfg);

DecodeState make_decode_state(int64_t max_seq_len,
                              const LanguageModelConfig& cfg,
                              aclrtStream stream);

void build_rope_tables(int64_t max_seq_len,
                       const LanguageModelConfig& cfg,
                       Tensor& cos_table,
                       Tensor& sin_table);

Tensor prefill(const std::vector<int32_t>& token_ids,
               const LanguageModelWeights& weights,
               const LanguageModelConfig& cfg,
               const Tensor& cos_table,
               const Tensor& sin_table,
               DecodeState& state,
               aclrtStream stream);

int64_t lm_head_greedy(const Tensor& last_hidden_1xH,
                       const LanguageModelWeights& weights,
                       const LanguageModelConfig& cfg,
                       aclrtStream stream);

int64_t decode_step_greedy(int32_t token_id,
                           const LanguageModelWeights& weights,
                           const LanguageModelConfig& cfg,
                           const Tensor& cos_table,
                           const Tensor& sin_table,
                           DecodeState& state,
                           aclrtStream stream);

bool is_eos(int64_t token_id, const LanguageModelConfig& cfg);

}  // namespace hy_mt2
