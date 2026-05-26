#include "hy_mt2/translator.h"
#include "hy_mt2/weights.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Args {
    std::string model_dir = hy_mt2::default_model_dir();
    std::string target_lang = "English";
    int64_t max_seq_len = 2048;
    int64_t max_new_tokens = 1024;
    bool stream = false;
};

void usage(const char* argv0) {
    std::cerr << "usage: " << argv0
              << " --model <dir> --tgt <language> [--max-seq N] [--max-new N] [--stream]\n"
              << "Reads one source sentence per stdin line and writes one translation per line.\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (key == "--model") args.model_dir = need_value("--model");
        else if (key == "--tgt") args.target_lang = need_value("--tgt");
        else if (key == "--max-seq") args.max_seq_len = std::stoll(need_value("--max-seq"));
        else if (key == "--max-new") args.max_new_tokens = std::stoll(need_value("--max-new"));
        else if (key == "--stream") args.stream = true;
        else if (key == "--help" || key == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else if (key == "--src") {
            (void)need_value("--src");
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        hy_mt2::TranslatorOptions options;
        options.max_seq_len = args.max_seq_len;
        options.max_new_tokens = args.max_new_tokens;
        hy_mt2::Translator translator(args.model_dir, options);

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) {
                std::cout << '\n';
                continue;
            }
            if (args.stream) {
                translator.translate_stream(line, args.target_lang, [](const std::string& piece) {
                    std::cout << piece << std::flush;
                });
                std::cout << '\n';
            } else {
                std::cout << translator.translate(line, args.target_lang) << '\n';
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        usage(argv[0]);
        return 1;
    }
    return 0;
}
