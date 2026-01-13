using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Tokenization.Core;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.Tokenization.CodeTokenization
{
    /// <summary>
    /// Code-aware tokenizer that handles programming language constructs.
    /// Supports identifier splitting, keyword recognition, and language-specific patterns.
    /// </summary>
    public class CodeTokenizer : TokenizerBase
    {
        private readonly HashSet<string> _keywords;
        private readonly ITokenizer _baseTokenizer;
        private readonly bool _splitIdentifiers;
        private readonly ProgrammingLanguage _language;

        /// <summary>
        /// Creates a new code tokenizer.
        /// </summary>
        /// <param name="baseTokenizer">The base tokenizer to use for subword tokenization.</param>
        /// <param name="language">The programming language.</param>
        /// <param name="splitIdentifiers">Whether to split identifiers (camelCase, snake_case).</param>
        public CodeTokenizer(
            ITokenizer baseTokenizer,
            ProgrammingLanguage language = ProgrammingLanguage.Generic,
            bool splitIdentifiers = true)
            : base(baseTokenizer?.Vocabulary ?? throw new ArgumentNullException(nameof(baseTokenizer)),
                   baseTokenizer.SpecialTokens)
        {
            _baseTokenizer = baseTokenizer;
            _language = language;
            _splitIdentifiers = splitIdentifiers;
            _keywords = GetLanguageKeywords(language);
        }

        /// <summary>
        /// Gets keywords for a programming language.
        /// </summary>
        private static HashSet<string> GetLanguageKeywords(ProgrammingLanguage language)
        {
            return language switch
            {
                ProgrammingLanguage.CSharp => new HashSet<string>
                {
                    "abstract", "as", "base", "bool", "break", "byte", "case", "catch", "char", "checked",
                    "class", "const", "continue", "decimal", "default", "delegate", "do", "double", "else",
                    "enum", "event", "explicit", "extern", "false", "finally", "fixed", "float", "for",
                    "foreach", "goto", "if", "implicit", "in", "int", "interface", "internal", "is", "lock",
                    "long", "namespace", "new", "null", "object", "operator", "out", "override", "params",
                    "private", "protected", "public", "readonly", "ref", "return", "sbyte", "sealed", "short",
                    "sizeof", "stackalloc", "static", "string", "struct", "switch", "this", "throw", "true",
                    "try", "typeof", "uint", "ulong", "unchecked", "unsafe", "ushort", "using", "virtual",
                    "void", "volatile", "while", "async", "await", "var", "dynamic"
                },
                ProgrammingLanguage.Python => new HashSet<string>
                {
                    "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del",
                    "elif", "else", "except", "False", "finally", "for", "from", "global", "if", "import",
                    "in", "is", "lambda", "None", "nonlocal", "not", "or", "pass", "raise", "return",
                    "True", "try", "while", "with", "yield"
                },
                ProgrammingLanguage.Java => new HashSet<string>
                {
                    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class",
                    "const", "continue", "default", "do", "double", "else", "enum", "extends", "final",
                    "finally", "float", "for", "goto", "if", "implements", "import", "instanceof", "int",
                    "interface", "long", "native", "new", "package", "private", "protected", "public",
                    "return", "short", "static", "strictfp", "super", "switch", "synchronized", "this",
                    "throw", "throws", "transient", "try", "void", "volatile", "while"
                },
                ProgrammingLanguage.JavaScript => new HashSet<string>
                {
                    "async", "await", "break", "case", "catch", "class", "const", "continue", "debugger",
                    "default", "delete", "do", "else", "export", "extends", "false", "finally", "for",
                    "function", "if", "import", "in", "instanceof", "let", "new", "null", "return", "super",
                    "switch", "this", "throw", "true", "try", "typeof", "var", "void", "while", "with", "yield"
                },
                ProgrammingLanguage.TypeScript => new HashSet<string>
                {
                    // JavaScript keywords
                    "async", "await", "break", "case", "catch", "class", "const", "continue", "debugger",
                    "default", "delete", "do", "else", "export", "extends", "false", "finally", "for",
                    "function", "if", "import", "in", "instanceof", "let", "new", "null", "return", "super",
                    "switch", "this", "throw", "true", "try", "typeof", "var", "void", "while", "with", "yield",
                    // TypeScript-specific keywords
                    "abstract", "any", "as", "asserts", "bigint", "boolean", "declare", "enum", "implements",
                    "infer", "interface", "is", "keyof", "module", "namespace", "never", "number", "object",
                    "override", "private", "protected", "public", "readonly", "require", "static", "string",
                     "symbol", "type", "undefined", "unique", "unknown"
                },
                ProgrammingLanguage.C => new HashSet<string>
                {
                    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else",
                    "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long", "register",
                    "restrict", "return", "short", "signed", "sizeof", "static", "struct", "switch",
                    "typedef", "union", "unsigned", "void", "volatile", "while", "_Bool", "_Complex", "_Imaginary"
                },
                ProgrammingLanguage.Cpp => new HashSet<string>
                {
                    "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor", "bool", "break",
                    "case", "catch", "char", "char16_t", "char32_t", "class", "compl", "const", "constexpr",
                    "const_cast", "continue", "decltype", "default", "delete", "do", "double", "dynamic_cast",
                    "else", "enum", "explicit", "export", "extern", "false", "float", "for", "friend", "goto",
                    "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq",
                    "nullptr", "operator", "or", "or_eq", "private", "protected", "public", "register",
                    "reinterpret_cast", "return", "short", "signed", "sizeof", "static", "static_assert",
                    "static_cast", "struct", "switch", "template", "this", "thread_local", "throw", "true",
                    "try", "typedef", "typeid", "typename", "union", "unsigned", "using", "virtual", "void",
                    "volatile", "wchar_t", "while", "xor", "xor_eq"
                },
                ProgrammingLanguage.Go => new HashSet<string>
                {
                    "break", "default", "func", "interface", "select", "case", "defer", "go", "map", "struct",
                    "chan", "else", "goto", "package", "switch", "const", "fallthrough", "if", "range", "type",
                    "continue", "for", "import", "return", "var"
                },
                ProgrammingLanguage.Rust => new HashSet<string>
                {
                    "as", "break", "const", "continue", "crate", "else", "enum", "extern", "false", "fn",
                    "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref",
                    "return", "self", "Self", "static", "struct", "super", "trait", "true", "type", "unsafe",
                    "use", "where", "while", "async", "await", "dyn", "union"
                },
                ProgrammingLanguage.SQL => new HashSet<string>(StringComparer.OrdinalIgnoreCase)
                {
                    "select", "from", "where", "join", "inner", "left", "right", "full", "outer", "cross", "on",
                    "group", "by", "order", "having", "limit", "offset", "fetch", "first", "rows", "only",
                    "insert", "into", "values", "update", "set", "delete",
                    "create", "table", "alter", "drop", "index", "view",
                    "distinct", "as", "and", "or", "not", "null", "is", "in", "exists", "between", "like",
                    "union", "all", "case", "when", "then", "else", "end",
                    "count", "sum", "avg", "min", "max", "group_concat"
                },
                _ => new HashSet<string>()
            };
        }

        /// <summary>
        /// Encodes code into a tokenization result with best-effort character offsets.
        /// </summary>
        public override TokenizationResult Encode(string text, EncodingOptions? options = null)
        {
            if (string.IsNullOrEmpty(text))
                return new TokenizationResult();

            options ??= new EncodingOptions();

            var tokens = new List<string>();
            var offsets = new List<(int Start, int End)>();

            // Pre-process code: split by code structure (with offsets)
            var codeTokens = PreTokenizeCodeWithOffsets(text);

            foreach (var (codeToken, start, end) in codeTokens)
            {
                if (_keywords.Contains(codeToken))
                {
                    tokens.Add(codeToken);
                    offsets.Add((start, end));
                    continue;
                }

                if (_splitIdentifiers && IsIdentifier(codeToken))
                {
                    var parts = SplitIdentifierWithOffsets(codeToken, start);
                    foreach (var (partToken, partStart, partEnd) in parts)
                    {
                        var subTokens = _baseTokenizer.Tokenize(partToken);
                        foreach (var subToken in subTokens)
                        {
                            tokens.Add(subToken);
                            offsets.Add((partStart, partEnd));
                        }
                    }

                    continue;
                }

                var baseTokens = _baseTokenizer.Tokenize(codeToken);
                foreach (var baseToken in baseTokens)
                {
                    tokens.Add(baseToken);
                    offsets.Add((start, end));
                }
            }

            // Truncate BEFORE adding special tokens to preserve them
            if (options.Truncation && options.MaxLength.HasValue)
            {
                var reservedSpace = options.AddSpecialTokens ? 2 : 0; // [CLS] and [SEP]
                var maxContentLength = options.MaxLength.Value - reservedSpace;

                if (tokens.Count > maxContentLength)
                {
                    if (options.TruncationSide == "left")
                    {
                        tokens = tokens.Skip(tokens.Count - maxContentLength).ToList();
                        offsets = offsets.Skip(offsets.Count - maxContentLength).ToList();
                    }
                    else
                    {
                        tokens = tokens.Take(maxContentLength).ToList();
                        offsets = offsets.Take(maxContentLength).ToList();
                    }
                }
            }

            // Add special tokens if requested (after truncation to preserve them)
            if (options.AddSpecialTokens)
            {
                if (!string.IsNullOrEmpty(SpecialTokens.ClsToken))
                {
                    tokens.Insert(0, SpecialTokens.ClsToken);
                    offsets.Insert(0, (0, 0));
                }

                if (!string.IsNullOrEmpty(SpecialTokens.SepToken))
                {
                    tokens.Add(SpecialTokens.SepToken);
                    offsets.Add((0, 0));
                }
            }

            var tokenIds = ConvertTokensToIds(tokens);
            var attentionMask = Enumerable.Repeat(1, tokenIds.Count).ToList();

            // Pad if necessary
            if (options.Padding && options.MaxLength.HasValue)
            {
                var paddingLength = options.MaxLength.Value - tokenIds.Count;
                if (paddingLength > 0)
                {
                    var padTokenId = Vocabulary.GetTokenId(SpecialTokens.PadToken);
                    var padding = Enumerable.Repeat(padTokenId, paddingLength).ToList();
                    var paddingTokens = Enumerable.Repeat(SpecialTokens.PadToken, paddingLength).ToList();
                    var paddingMask = Enumerable.Repeat(0, paddingLength).ToList();
                    var paddingOffsets = Enumerable.Repeat((0, 0), paddingLength).ToList();

                    if (options.PaddingSide == "right")
                    {
                        tokenIds.AddRange(padding);
                        tokens.AddRange(paddingTokens);
                        attentionMask.AddRange(paddingMask);
                        offsets.AddRange(paddingOffsets);
                    }
                    else
                    {
                        tokenIds.InsertRange(0, padding);
                        tokens.InsertRange(0, paddingTokens);
                        attentionMask.InsertRange(0, paddingMask);
                        offsets.InsertRange(0, paddingOffsets);
                    }
                }
            }

            var result = new TokenizationResult
            {
                Tokens = tokens,
                TokenIds = tokenIds,
                AttentionMask = options.ReturnAttentionMask ? attentionMask : new List<int>(),
                Offsets = offsets
            };

            if (options.ReturnTokenTypeIds)
            {
                result.TokenTypeIds = Enumerable.Repeat(0, tokenIds.Count).ToList();
            }

            if (options.ReturnPositionIds)
            {
                result.PositionIds = Enumerable.Range(0, tokenIds.Count).ToList();
            }

            return result;
        }

        /// <summary>
        /// Tokenizes code with language-aware handling.
        /// </summary>
        public override List<string> Tokenize(string text)
        {
            if (string.IsNullOrEmpty(text))
                return new List<string>();

            var tokens = new List<string>();

            // Pre-process code: split by code structure
            var codeTokens = PreTokenizeCode(text);

            foreach (var codeToken in codeTokens)
            {
                // Check if it's a keyword
                if (_keywords.Contains(codeToken))
                {
                    tokens.Add(codeToken);
                }
                // Check if it's an identifier that should be split
                else if (_splitIdentifiers && IsIdentifier(codeToken))
                {
                    var splitTokens = SplitIdentifier(codeToken);
                    foreach (var splitToken in splitTokens)
                    {
                        // Apply base tokenizer to each part
                        tokens.AddRange(_baseTokenizer.Tokenize(splitToken));
                    }
                }
                else
                {
                    // Use base tokenizer for other tokens
                    tokens.AddRange(_baseTokenizer.Tokenize(codeToken));
                }
            }

            return tokens;
        }

        private List<(string Token, int Start, int End)> PreTokenizeCodeWithOffsets(string code)
        {
            var tokens = new List<(string, int, int)>();

            // Pattern for code tokenization (strings, comments, identifiers, operators, etc.)
            // Supports multiple languages: C#, Python, JavaScript, etc.
            var pattern = @"
                @""(?:""""|[^""])*""|           # C# verbatim strings (@""..."")
                \$""(?:\\.|[^""\\])*""|         # C# interpolated strings ($""..."")
                [rf]?""(?:\\.|[^""\\])*""|      # Python raw/f-strings and double-quoted strings
                '(?:\\.|[^'\\])*'|              # Single-quoted strings
                \#[^\n]*|                       # Python-style single-line comments
                //[^\n]*|                       # C-style single-line comments
                /\*[\s\S]*?\*/|                 # Multi-line comments
                \b[a-zA-Z_][a-zA-Z0-9_]*\b|     # Identifiers
                \b\d+\.?\d*\b|                  # Numbers
                [+\-*/%=<>!&|^~]+|              # Operators
                [{}()\[\];,.]|                  # Delimiters
                \s+                             # Whitespace
            ";

            var regex = RegexHelper.Create(pattern, RegexOptions.IgnorePatternWhitespace);
            var matches = regex.Matches(code);

            foreach (Match match in matches)
            {
                var value = match.Value;
                if (string.IsNullOrWhiteSpace(value))
                {
                    continue;
                }

                var trimmed = value.Trim();
                if (trimmed.Length == 0)
                {
                    continue;
                }

                int leading = value.Length - value.TrimStart().Length;
                int trailing = value.Length - value.TrimEnd().Length;

                int start = match.Index + leading;
                int end = match.Index + value.Length - trailing;

                tokens.Add((trimmed, start, end));
            }

            return tokens;
        }

        private List<(string Token, int Start, int End)> SplitIdentifierWithOffsets(string identifier, int absoluteStart)
        {
            var parts = new List<(string, int, int)>();

            // Handle snake_case
            if (identifier.Contains('_'))
            {
                int cursor = 0;
                foreach (var part in identifier.Split(new[] { '_' }, StringSplitOptions.RemoveEmptyEntries))
                {
                    int partIndex = identifier.IndexOf(part, cursor, StringComparison.Ordinal);
                    if (partIndex < 0)
                    {
                        partIndex = cursor;
                    }

                    int start = absoluteStart + partIndex;
                    int end = start + part.Length;
                    parts.Add((part, start, end));
                    cursor = partIndex + part.Length;
                }

                return parts;
            }

            // Handle camelCase and PascalCase
            var pattern = @"([A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b))";
            var matches = RegexHelper.Matches(identifier, pattern, RegexOptions.None);

            if (matches.Count > 0)
            {
                foreach (Match match in matches)
                {
                    if (string.IsNullOrWhiteSpace(match.Value))
                    {
                        continue;
                    }

                    int start = absoluteStart + match.Index;
                    int end = start + match.Length;
                    parts.Add((match.Value, start, end));
                }

                return parts;
            }

            // If no pattern matched, return the original identifier
            parts.Add((identifier, absoluteStart, absoluteStart + identifier.Length));
            return parts;
        }

        /// <summary>
        /// Pre-tokenizes code by splitting on whitespace and operators while preserving strings and comments.
        /// </summary>
        private List<string> PreTokenizeCode(string code)
        {
            var tokens = new List<string>();

            // Pattern for code tokenization (strings, comments, identifiers, operators, etc.)
            // Supports multiple languages: C#, Python, JavaScript, etc.
            var pattern = @"
                @""(?:""""|[^""])*""|           # C# verbatim strings (@""..."")
                \$""(?:\\.|[^""\\])*""|         # C# interpolated strings ($""..."")
                [rf]?""(?:\\.|[^""\\])*""|      # Python raw/f-strings and double-quoted strings
                '(?:\\.|[^'\\])*'|              # Single-quoted strings
                \#[^\n]*|                       # Python-style single-line comments
                //[^\n]*|                       # C-style single-line comments
                /\*[\s\S]*?\*/|                 # Multi-line comments
                \b[a-zA-Z_][a-zA-Z0-9_]*\b|     # Identifiers
                \b\d+\.?\d*\b|                  # Numbers
                [+\-*/%=<>!&|^~]+|              # Operators
                [{}()\[\];,.]|                  # Delimiters
                \s+                             # Whitespace
            ";

            var regex = RegexHelper.Create(pattern, RegexOptions.IgnorePatternWhitespace);
            var matches = regex.Matches(code);

            tokens.AddRange(matches.Cast<Match>()
                .Select(m => m.Value)
                .Where(token => !string.IsNullOrWhiteSpace(token))
                .Select(token => token.Trim()));

            return tokens;
        }

        /// <summary>
        /// Checks if a token is an identifier.
        /// </summary>
        private bool IsIdentifier(string token)
        {
            return RegexHelper.IsMatch(token, @"^[a-zA-Z_][a-zA-Z0-9_]*$", RegexOptions.None);
        }

        /// <summary>
        /// Splits an identifier by camelCase, PascalCase, or snake_case.
        /// </summary>
        private List<string> SplitIdentifier(string identifier)
        {
            var parts = new List<string>();

            // Handle snake_case
            if (identifier.Contains('_'))
            {
                parts.AddRange(identifier.Split(new[] { '_' }, StringSplitOptions.RemoveEmptyEntries));
                return parts;
            }

            // Handle camelCase and PascalCase
            var pattern = @"([A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b))";
            var matches = RegexHelper.Matches(identifier, pattern, RegexOptions.None);

            if (matches.Count > 0)
            {
                parts.AddRange(matches.Cast<Match>()
                    .Where(m => !string.IsNullOrWhiteSpace(m.Value))
                    .Select(m => m.Value));
                return parts;
            }

            // If no pattern matched, return the original identifier
            return new List<string> { identifier };
        }

        /// <summary>
        /// Cleans up tokens and converts them back to code.
        /// </summary>
        protected override string CleanupTokens(List<string> tokens)
        {
            return _baseTokenizer.Decode(_baseTokenizer.ConvertTokensToIds(tokens), skipSpecialTokens: true);
        }
    }
}



