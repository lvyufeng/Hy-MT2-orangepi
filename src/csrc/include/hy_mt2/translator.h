#pragma once

#include "hy_mt2/acl_context.h"
#include "hy_mt2/language_model.h"
#include "hy_mt2/tokenizer.h"
#include "hy_mt2/weights.h"

#include <functional>
#include <string>

namespace hy_mt2 {

struct TranslatorOptions {
    int64_t max_seq_len{2048};
    int64_t max_new_tokens{1024};
};

class Translator {
public:
    explicit Translator(const std::string& model_dir,
                        TranslatorOptions options = TranslatorOptions{});

    std::string translate(const std::string& text,
                          const std::string& target_lang);

    void translate_stream(const std::string& text,
                          const std::string& target_lang,
                          const std::function<void(const std::string&)>& on_text);

private:
    AclContext ctx_;
    LanguageModelConfig cfg_;
    TranslatorOptions options_;
    Tokenizer tokenizer_;
    WeightsIndex weights_index_;
    LanguageModelWeights weights_;
    Tensor cos_table_;
    Tensor sin_table_;
};

}  // namespace hy_mt2
