// Hand-maintained expected values for pre-tokenize tests.

const std::vector<std::pair<std::string, std::vector<std::string>>> kSplitCases = {
    {"Hello, world!", {"Hello", ",", " world", "!"}},
    {"don't", {"don", "'t"}},
    {"  spaces", {" ", " spaces"}},
    {"hello\nworld", {"hello", "\n", "world"}},
    {"café", {"café"}},
    {"中文", {"中文"}},
};

const std::vector<std::string> kRoundtripInputs = {
    "The quick brown fox",
    " emojis 😊 and 中文",
    "café naïve résumé",
    "\tindent",
    "line\r\nnext",
    "!!!???",
    "hello world",
};
