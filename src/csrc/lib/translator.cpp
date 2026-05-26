#include "hy_mt2/translator.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hy_mt2 {
namespace {

std::string safetensors_path_for(const std::string& model_dir) {
    const std::filesystem::path dir(model_dir);
    const auto single = dir / "model.safetensors";
    if (std::filesystem::exists(single)) return single.string();
    throw std::runtime_error("model.safetensors not found under " + model_dir);
}

std::string tokenizer_path_for(const std::string& model_dir) {
    const std::filesystem::path path = std::filesystem::path(model_dir) / "tokenizer.json";
    if (std::filesystem::exists(path)) return path.string();
    throw std::runtime_error("tokenizer.json not found under " + model_dir);
}

}  // namespace

Translator::Translator(const std::string& model_dir, TranslatorOptions options)
    : ctx_(0),
      cfg_(default_hy_mt2_config()),
      options_(std::move(options)),
      tokenizer_(tokenizer_path_for(model_dir)),
      weights_index_(safetensors_path_for(model_dir)),
      weights_(load_language_model_weights(weights_index_, cfg_)) {
    if (options_.max_seq_len <= 0 || options_.max_new_tokens <= 0) {
        throw std::runtime_error("TranslatorOptions max_seq_len/max_new_tokens must be positive");
    }
    build_rope_tables(options_.max_seq_len, cfg_, cos_table_, sin_table_);
}

std::string Translator::translate(const std::string& text,
                                  const std::string& target_lang) {
    std::string out;
    translate_stream(text, target_lang, [&](const std::string& piece) { out += piece; });
    return out;
}

void Translator::translate_stream(const std::string& text,
                                  const std::string& target_lang,
                                  const std::function<void(const std::string&)>& on_text) {
    const std::string prompt = tokenizer_.format_translation_prompt(text, target_lang);
    const std::vector<int32_t> prompt_ids = tokenizer_.encode(prompt);
    if (prompt_ids.empty()) throw std::runtime_error("translation prompt encoded to zero tokens");
    if (static_cast<int64_t>(prompt_ids.size()) >= options_.max_seq_len) {
        throw std::runtime_error("translation prompt exceeds max_seq_len");
    }

    DecodeState state = make_decode_state(options_.max_seq_len, cfg_, ctx_.stream());
    Tensor last_hidden = prefill(prompt_ids, weights_, cfg_, cos_table_, sin_table_, state, ctx_.stream());
    int32_t token = static_cast<int32_t>(lm_head_greedy(last_hidden, weights_, cfg_, ctx_.stream()));
    std::vector<int32_t> generated;
    generated.reserve(static_cast<size_t>(options_.max_new_tokens));
    std::string emitted;

    for (int64_t step = 0; step < options_.max_new_tokens; ++step) {
        if (is_eos(token, cfg_)) break;
        generated.push_back(token);
        const std::string decoded = tokenizer_.decode(generated);
        if (decoded.size() > emitted.size()) {
            on_text(decoded.substr(emitted.size()));
            emitted = decoded;
        }
        if (state.seq_len >= options_.max_seq_len) break;
        token = static_cast<int32_t>(decode_step_greedy(token, weights_, cfg_, cos_table_, sin_table_, state, ctx_.stream()));
    }
}

}  // namespace hy_mt2
