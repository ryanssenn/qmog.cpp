// Hand-maintained expected values for tokenizer encode/decode tests.

const std::vector<std::pair<std::string, std::vector<uint32_t>>> kEncodeDecodeCases = {
    {"Hello", {9707}},
    {"The quick brown fox", {785, 3974, 13876, 38835}},
    {"Hello, world!", {9707, 11, 1879, 0}},
    {"hello\nworld", {14990, 198, 14615}},
    {" emojis 😊 and 中文", {99066, 26525, 232, 323, 72858, 16744}},
    {"<|endoftext|>", {151643}},
    {"<|im_start|>user<|im_end|>", {151644, 872, 151645}},
    {"  leading and  double spaces", {220, 6388, 323, 220, 1990, 12621}},
    {"", {}},
    {"\tindent", {197, 32840}},
    {"line\r\nnext", {1056, 319, 3600}},
    {"<|im_start|><|im_end|>", {151644, 151645}},
    {"!!!???", {12069, 33015}},
    {"café naïve résumé", {924, 58858, 94880, 586, 9333, 1242, 963}},
    {"👩‍💻 works", {145233, 378, 235, 145851, 4278}},
};
