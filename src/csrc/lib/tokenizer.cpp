#include "hy_mt2/tokenizer.h"
#include "hy_mt2/weights.h"

#include <tokenizers_cpp.h>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace hy_mt2 {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open tokenizer file: " + path);
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

}  // namespace

Tokenizer::Tokenizer(const std::string& tokenizer_json_path) {
    impl_ = tokenizers::Tokenizer::FromBlobJSON(read_file(tokenizer_json_path));
    if (!impl_) {
        throw std::runtime_error("failed to construct tokenizer from " + tokenizer_json_path);
    }
}

Tokenizer::~Tokenizer() = default;

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    return impl_->Encode(text);
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    return impl_->Decode(ids);
}

size_t Tokenizer::vocab_size() const {
    return impl_->GetVocabSize();
}

int32_t Tokenizer::token_to_id(const std::string& token) const {
    return impl_->TokenToId(token);
}

std::string Tokenizer::id_to_token(int32_t token_id) const {
    return impl_->IdToToken(token_id);
}

std::string Tokenizer::format_translation_prompt(const std::string& text,
                                                 const std::string& target_lang) const {
    return hy_mt2::format_translation_prompt(text, target_lang);
}

std::string default_tokenizer_json_path() {
    return default_model_dir() + "/tokenizer.json";
}

std::string format_translation_prompt(const std::string& text, const std::string& target_lang) {
    return std::string("<｜hy_begin▁of▁sentence｜><｜hy_User｜>") +
           "Translate the following text into " + target_lang +
           ". Note that you should only output the translated result without any additional explanation:\n\n" +
           text + "<｜hy_Assistant｜>";
}

}  // namespace hy_mt2
