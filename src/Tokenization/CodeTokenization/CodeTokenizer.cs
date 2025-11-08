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
        /// Programming languages supported by the code tokenizer.
        /// </summary>
        public enum ProgrammingLanguage
        {
            CSharp,
            Python,
            Java,
            JavaScript,
            TypeScript,
            Generic
        }

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
            : base(baseTokenizer.Vocabulary, baseTokenizer.SpecialTokens)
        {
            _baseTokenizer = baseTokenizer ?? throw new ArgumentNullException(nameof(baseTokenizer));
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
                _ => new HashSet<string>()
            };
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

        /// <summary>
        /// Pre-tokenizes code by splitting on whitespace and operators while preserving strings and comments.
        /// </summary>
        private List<string> PreTokenizeCode(string code)
        {
            var tokens = new List<string>();

            // Pattern for code tokenization (strings, comments, identifiers, operators, etc.)
            var pattern = @"
                ""(?:\\.|[^""\\])*""|           # Double-quoted strings
                '(?:\\.|[^'\\])*'|              # Single-quoted strings
                //[^\n]*|                       # Single-line comments
                /\*[\s\S]*?\*/|                 # Multi-line comments
                \b[a-zA-Z_][a-zA-Z0-9_]*\b|     # Identifiers
                \b\d+\.?\d*\b|                  # Numbers
                [+\-*/%=<>!&|^~]+|              # Operators
                [{}()\[\];,.]|                  # Delimiters
                \s+                             # Whitespace
            ";

            var regex = new Regex(pattern, RegexOptions.IgnorePatternWhitespace);
            var matches = regex.Matches(code);

            foreach (Match match in matches)
            {
                var token = match.Value;
                if (!string.IsNullOrWhiteSpace(token))
                {
                    tokens.Add(token.Trim());
                }
            }

            return tokens;
        }

        /// <summary>
        /// Checks if a token is an identifier.
        /// </summary>
        private bool IsIdentifier(string token)
        {
            return Regex.IsMatch(token, @"^[a-zA-Z_][a-zA-Z0-9_]*$");
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
                parts.AddRange(identifier.Split('_', StringSplitOptions.RemoveEmptyEntries));
                return parts;
            }

            // Handle camelCase and PascalCase
            var pattern = @"([A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b))";
            var matches = Regex.Matches(identifier, pattern);

            if (matches.Count > 0)
            {
                foreach (Match match in matches)
                {
                    if (!string.IsNullOrWhiteSpace(match.Value))
                    {
                        parts.Add(match.Value);
                    }
                }
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
