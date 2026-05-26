#include "hy_mt2/tokenizer.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::string path = hy_mt2::default_tokenizer_json_path();
    if (const char* env = std::getenv("HY_MT2_TOKENIZER_JSON")) {
        if (*env) path = env;
    }
    if (!std::filesystem::exists(path)) {
        std::cout << "[skip] tokenizer.json not found at " << path
                  << " (set HY_MT2_TOKENIZER_JSON or run scripts/fetch_model.sh)\n";
        return 0;
    }

    hy_mt2::Tokenizer tok(path);
    if (tok.vocab_size() != 120818) {
        std::cerr << "unexpected vocab size: " << tok.vocab_size() << '\n';
        return 1;
    }
    if (tok.token_to_id("<｜hy_begin▁of▁sentence｜>") != 120000 ||
        tok.token_to_id("<｜hy_User｜>") != 120006 ||
        tok.token_to_id("<｜hy_Assistant｜>") != 120007) {
        std::cerr << "special token ids mismatch\n";
        return 1;
    }

    const std::string prompt = tok.format_translation_prompt("我叫沃尔夫冈，我住在柏林。", "English");
    const auto ids = tok.encode(prompt);
    if (ids.empty() || ids.front() != 120000) {
        std::cerr << "encoded prompt missing BOS\n";
        return 1;
    }
    bool saw_user = false;
    bool saw_assistant = false;
    for (int32_t id : ids) {
        saw_user = saw_user || id == 120006;
        saw_assistant = saw_assistant || id == 120007;
    }
    if (!saw_user || !saw_assistant) {
        std::cerr << "encoded prompt missing chat marker tokens\n";
        return 1;
    }

    const std::vector<int32_t> simple = {120000, 120006, 120007};
    const std::string decoded = tok.decode(simple);
    if (decoded.find("<｜hy_User｜>") == std::string::npos ||
        decoded.find("<｜hy_Assistant｜>") == std::string::npos) {
        std::cerr << "decode special token mismatch: " << decoded << '\n';
        return 1;
    }

    std::cout << "[ok] tokenizer loads Hy-MT2 tokenizer.json and encodes chat prompt ("
              << ids.size() << " tokens)\n";
    return 0;
}
