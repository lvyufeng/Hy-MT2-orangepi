#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tokenizers {
class Tokenizer;
}

namespace hy_mt2 {

class Tokenizer {
public:
    explicit Tokenizer(const std::string& tokenizer_json_path);
    ~Tokenizer();

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    std::vector<int32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int32_t>& ids) const;
    size_t vocab_size() const;
    int32_t token_to_id(const std::string& token) const;
    std::string id_to_token(int32_t token_id) const;

    std::string format_translation_prompt(const std::string& text,
                                          const std::string& target_lang) const;

private:
    std::unique_ptr<tokenizers::Tokenizer> impl_;
};

std::string default_tokenizer_json_path();
std::string format_translation_prompt(const std::string& text, const std::string& target_lang);

}  // namespace hy_mt2
