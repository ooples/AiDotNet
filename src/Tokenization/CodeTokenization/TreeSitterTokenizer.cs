using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tokenization.Core;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;
using TreeSitter;

namespace AiDotNet.Tokenization.CodeTokenization
{
    /// <summary>
    /// AST-aware tokenizer using Tree-sitter for parsing source code into syntax trees.
    /// Provides structure-aware tokenization that understands programming language grammar.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Tree-sitter is an incremental parsing library that builds concrete syntax trees for source code.
    /// Unlike simple regex-based tokenizers, Tree-sitter understands the actual structure of code,
    /// enabling more intelligent tokenization that preserves semantic meaning.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this tokenizer as a code-reading expert that truly
    /// understands programming languages. While simple tokenizers just split text on spaces and
    /// punctuation (like cutting a sentence into individual words), Tree-sitter actually reads
    /// and understands the code's structure.
    ///
    /// For example, when parsing "function add(a, b) { return a + b; }":
    /// - A simple tokenizer sees: ["function", "add", "(", "a", ",", "b", ")", "{", ...]
    /// - Tree-sitter sees: A function declaration named "add" with parameters "a" and "b",
    ///   containing a return statement with a binary expression.
    ///
    /// This deeper understanding helps machine learning models learn code patterns more effectively,
    /// because tokens are grouped by their semantic role (function names, variable names, operators, etc.)
    /// rather than just their text content.
    /// </para>
    /// </remarks>
    public sealed class TreeSitterTokenizer : TokenizerBase, IDisposable
    {
        private readonly (string LibraryName, string FunctionName) _languageSpec;
        private Language? _language;
        private Parser? _parser;
        private readonly ITokenizer _baseTokenizer;
        private readonly TreeSitterLanguage _languageType;
        private readonly bool _includeNodeTypes;
        private readonly bool _flattenTree;
        private bool _treeSitterAvailable = true;
        private bool _disposed;

        /// <summary>
        /// Creates a new Tree-sitter tokenizer for the specified programming language.
        /// </summary>
        /// <param name="baseTokenizer">The base tokenizer to use for subword tokenization of identifiers and literals.</param>
        /// <param name="language">The programming language to parse.</param>
        /// <param name="includeNodeTypes">Whether to include AST node types as prefix tokens (e.g., "[FUNC]", "[VAR]").</param>
        /// <param name="flattenTree">Whether to flatten the AST into a sequence or preserve tree structure markers.</param>
        /// <remarks>
        /// <para><b>For Beginners:</b> The base tokenizer handles breaking down individual code elements
        /// (like variable names) into smaller pieces, while Tree-sitter handles understanding the
        /// overall code structure.
        ///
        /// - Set includeNodeTypes=true if you want tokens prefixed with their syntactic role
        ///   (helps the model understand what each token represents).
        /// - Set flattenTree=true for a simple sequence of tokens, or false to include tree
        ///   structure markers like "[BEGIN_FUNC]" and "[END_FUNC]".
        /// </para>
        /// </remarks>
        public TreeSitterTokenizer(
            ITokenizer baseTokenizer,
            TreeSitterLanguage language = TreeSitterLanguage.Python,
            bool includeNodeTypes = true,
            bool flattenTree = true)
            : base(baseTokenizer?.Vocabulary ?? throw new ArgumentNullException(nameof(baseTokenizer)),
                   baseTokenizer.SpecialTokens)
        {
            _baseTokenizer = baseTokenizer;
            _languageType = language;
            _includeNodeTypes = includeNodeTypes;
            _flattenTree = flattenTree;

            _languageSpec = GetTreeSitterLanguageSpec(language);
        }

        /// <summary>
        /// Gets the Tree-sitter language library/function spec for a language enum value.
        /// </summary>
        /// <param name="language">The language enum value.</param>
        /// <returns>The Tree-sitter language library and function names.</returns>
        private static (string LibraryName, string FunctionName) GetTreeSitterLanguageSpec(TreeSitterLanguage language)
        {
            return language switch
            {
                TreeSitterLanguage.CSharp => ("tree-sitter-c-sharp", "tree_sitter_c_sharp"),
                TreeSitterLanguage.Python => ("tree-sitter-python", "tree_sitter_python"),
                TreeSitterLanguage.JavaScript => ("tree-sitter-javascript", "tree_sitter_javascript"),
                TreeSitterLanguage.TypeScript => ("tree-sitter-typescript", "tree_sitter_typescript"),
                TreeSitterLanguage.Java => ("tree-sitter-java", "tree_sitter_java"),
                TreeSitterLanguage.C => ("tree-sitter-c", "tree_sitter_c"),
                TreeSitterLanguage.Cpp => ("tree-sitter-cpp", "tree_sitter_cpp"),
                TreeSitterLanguage.Go => ("tree-sitter-go", "tree_sitter_go"),
                TreeSitterLanguage.Rust => ("tree-sitter-rust", "tree_sitter_rust"),
                TreeSitterLanguage.Ruby => ("tree-sitter-ruby", "tree_sitter_ruby"),
                TreeSitterLanguage.Json => ("tree-sitter-json", "tree_sitter_json"),
                TreeSitterLanguage.Html => ("tree-sitter-html", "tree_sitter_html"),
                TreeSitterLanguage.Css => ("tree-sitter-css", "tree_sitter_css"),
                _ => ("tree-sitter-python", "tree_sitter_python")
            };
        }

        /// <summary>
        /// Tokenizes source code using AST-aware parsing.
        /// </summary>
        /// <param name="text">The source code to tokenize.</param>
        /// <returns>A list of tokens representing the code structure.</returns>
        /// <remarks>
        /// <para>
        /// The tokenization process:
        /// 1. Parse the source code into an AST using Tree-sitter
        /// 2. Traverse the AST to extract meaningful nodes (identifiers, literals, keywords, operators)
        /// 3. Optionally prefix each token with its AST node type
        /// 4. Apply the base tokenizer to break down complex tokens into subwords
        /// </para>
        /// <para><b>For Beginners:</b> This method reads your code like a compiler would,
        /// building a tree structure that represents the code's meaning. Then it walks through
        /// that tree, collecting tokens in a way that preserves the semantic relationships.
        ///
        /// For example, the code "x = 5 + 3" might produce:
        /// - With includeNodeTypes=true: ["[IDENTIFIER]", "x", "[OPERATOR]", "=", "[NUMBER]", "5", ...]
        /// - With includeNodeTypes=false: ["x", "=", "5", "+", "3"]
        /// </para>
        /// </remarks>
        public override List<string> Tokenize(string text)
        {
            if (string.IsNullOrEmpty(text))
                return new List<string>();

            var tokens = new List<string>();

            if (!TryEnsureParser())
            {
                return _baseTokenizer.Tokenize(text).ToList();
            }

            try
            {
                using var tree = _parser!.Parse(text);
                if (tree is not null)
                {
                    var rootNode = tree.RootNode;
                    ExtractTokensWithQueries(rootNode, tokens);

                    // If no tokens were extracted, fall back to base tokenizer
                    if (tokens.Count == 0)
                    {
                        return _baseTokenizer.Tokenize(text);
                    }
                }
                else
                {
                    // Parsing returned null, fall back to base tokenizer
                    return _baseTokenizer.Tokenize(text);
                }
            }
            catch (InvalidOperationException)
            {
                // If Tree-sitter parsing fails, fall back to base tokenizer
                return _baseTokenizer.Tokenize(text);
            }
            catch (ArgumentException)
            {
                // If argument is invalid, fall back to base tokenizer
                return _baseTokenizer.Tokenize(text);
            }

            return tokens;
        }

        private bool TryEnsureParser()
        {
            if (_parser is not null)
            {
                return true;
            }

            if (!_treeSitterAvailable)
            {
                return false;
            }

            try
            {
                var language = new Language(_languageSpec.LibraryName, _languageSpec.FunctionName);
                try
                {
                    _parser = new Parser(language);
                    _language = language;
                }
                catch
                {
                    language.Dispose();
                    throw;
                }
                return true;
            }
            catch (DllNotFoundException)
            {
                _treeSitterAvailable = false;
                return false;
            }
            catch (BadImageFormatException)
            {
                _treeSitterAvailable = false;
                return false;
            }
            catch (InvalidOperationException)
            {
                _treeSitterAvailable = false;
                return false;
            }
            catch (ArgumentException)
            {
                _treeSitterAvailable = false;
                return false;
            }
        }

        /// <summary>
        /// Extracts tokens from the AST using Tree-sitter queries.
        /// </summary>
        /// <param name="rootNode">The root node of the AST.</param>
        /// <param name="tokens">The list to add tokens to.</param>
        private void ExtractTokensWithQueries(Node rootNode, List<string> tokens)
        {
            // Query pattern to capture all identifiers and literals
            var queryPattern = GetQueryPattern();

            try
            {
                if (_language is null)
                {
                    return;
                }

                using var query = new Query(_language, queryPattern);
                var queryResult = query.Execute(rootNode);

                foreach (var capture in queryResult.Captures)
                {
                    var nodeText = capture.Node.Text;
                    var captureNodeType = capture.Name;

                    if (!string.IsNullOrWhiteSpace(nodeText))
                    {
                        if (!_flattenTree)
                        {
                            tokens.Add($"[BEGIN_{NormalizeNodeType(captureNodeType)}]");
                        }

                        if (_includeNodeTypes)
                        {
                            tokens.Add($"[{NormalizeNodeType(captureNodeType)}]");
                        }

                        // Apply base tokenizer to the node text for subword tokenization
                        var subTokens = _baseTokenizer.Tokenize(nodeText);
                        tokens.AddRange(subTokens);

                        if (!_flattenTree)
                        {
                            tokens.Add($"[END_{NormalizeNodeType(captureNodeType)}]");
                        }
                    }
                }
            }
            catch (InvalidOperationException)
            {
                // If query fails, fall back to simple text tokenization
                // This can happen with invalid query patterns for certain languages
            }
            catch (ArgumentException)
            {
                // If argument is invalid for query execution
            }
        }

        /// <summary>
        /// Gets the Tree-sitter query pattern for the current language.
        /// </summary>
        /// <returns>A query pattern string that captures relevant AST nodes.</returns>
        private string GetQueryPattern()
        {
            // Common patterns that work across many languages
            return _languageType switch
            {
                TreeSitterLanguage.Python => @"
                    (identifier) @identifier
                    (string) @string
                    (integer) @number
                    (float) @number
                    (true) @boolean
                    (false) @boolean
                    (none) @null
                    (comment) @comment
                ",
                TreeSitterLanguage.JavaScript or TreeSitterLanguage.TypeScript => @"
                    (identifier) @identifier
                    (property_identifier) @property
                    (string) @string
                    (number) @number
                    (true) @boolean
                    (false) @boolean
                    (null) @null
                    (comment) @comment
                ",
                TreeSitterLanguage.CSharp => @"
                    (identifier) @identifier
                    (string_literal) @string
                    (integer_literal) @number
                    (real_literal) @number
                    (boolean_literal) @boolean
                    (null_literal) @null
                    (comment) @comment
                ",
                TreeSitterLanguage.Java => @"
                    (identifier) @identifier
                    (string_literal) @string
                    (decimal_integer_literal) @number
                    (decimal_floating_point_literal) @number
                    (true) @boolean
                    (false) @boolean
                    (null_literal) @null
                    (comment) @comment
                ",
                _ => @"
                    (identifier) @identifier
                    (string) @string
                    (number) @number
                    (comment) @comment
                "
            };
        }

        /// <summary>
        /// Normalizes an AST node type to a consistent format for token prefixes.
        /// </summary>
        /// <param name="nodeType">The AST node type.</param>
        /// <returns>The normalized node type string.</returns>
        private static string NormalizeNodeType(string nodeType)
        {
            // Convert to uppercase and replace special characters
            return nodeType.ToUpperInvariant()
                .Replace("_", "")
                .Replace("-", "");
        }

        /// <summary>
        /// Cleans up tokens and converts them back to text.
        /// </summary>
        /// <param name="tokens">The tokens to clean up.</param>
        /// <returns>The reconstructed text.</returns>
        protected override string CleanupTokens(List<string> tokens)
        {
            if (tokens == null || tokens.Count == 0)
                return string.Empty;

            // Remove node type markers if present
            var cleanTokens = tokens
                .Where(t => !t.StartsWith("[") || !t.EndsWith("]"))
                .ToList();

            return string.Join(" ", cleanTokens);
        }

        /// <summary>
        /// Creates a Tree-sitter tokenizer for Python code.
        /// </summary>
        /// <param name="baseTokenizer">The base tokenizer for subword tokenization.</param>
        /// <param name="includeNodeTypes">Whether to include AST node type markers.</param>
        /// <returns>A new TreeSitterTokenizer configured for Python.</returns>
        /// <remarks>
        /// <para><b>For Beginners:</b> Use this factory method to quickly create a tokenizer
        /// for Python source code. Python is commonly used in data science and machine learning,
        /// so this is often a good default choice.
        /// </para>
        /// </remarks>
        public static TreeSitterTokenizer CreatePython(ITokenizer baseTokenizer, bool includeNodeTypes = true)
        {
            return new TreeSitterTokenizer(baseTokenizer, TreeSitterLanguage.Python, includeNodeTypes);
        }

        /// <summary>
        /// Creates a Tree-sitter tokenizer for JavaScript code.
        /// </summary>
        /// <param name="baseTokenizer">The base tokenizer for subword tokenization.</param>
        /// <param name="includeNodeTypes">Whether to include AST node type markers.</param>
        /// <returns>A new TreeSitterTokenizer configured for JavaScript.</returns>
        public static TreeSitterTokenizer CreateJavaScript(ITokenizer baseTokenizer, bool includeNodeTypes = true)
        {
            return new TreeSitterTokenizer(baseTokenizer, TreeSitterLanguage.JavaScript, includeNodeTypes);
        }

        /// <summary>
        /// Creates a Tree-sitter tokenizer for C# code.
        /// </summary>
        /// <param name="baseTokenizer">The base tokenizer for subword tokenization.</param>
        /// <param name="includeNodeTypes">Whether to include AST node type markers.</param>
        /// <returns>A new TreeSitterTokenizer configured for C#.</returns>
        public static TreeSitterTokenizer CreateCSharp(ITokenizer baseTokenizer, bool includeNodeTypes = true)
        {
            return new TreeSitterTokenizer(baseTokenizer, TreeSitterLanguage.CSharp, includeNodeTypes);
        }

        /// <summary>
        /// Creates a Tree-sitter tokenizer for Java code.
        /// </summary>
        /// <param name="baseTokenizer">The base tokenizer for subword tokenization.</param>
        /// <param name="includeNodeTypes">Whether to include AST node type markers.</param>
        /// <returns>A new TreeSitterTokenizer configured for Java.</returns>
        public static TreeSitterTokenizer CreateJava(ITokenizer baseTokenizer, bool includeNodeTypes = true)
        {
            return new TreeSitterTokenizer(baseTokenizer, TreeSitterLanguage.Java, includeNodeTypes);
        }

        /// <summary>
        /// Releases the resources used by the Tree-sitter parser.
        /// </summary>
        /// <remarks>
        /// <para><b>For Beginners:</b> Always dispose of this tokenizer when you're done using it.
        /// The Tree-sitter parser uses native memory that needs to be freed. The best practice
        /// is to use a "using" statement: using var tokenizer = TreeSitterTokenizer.CreatePython(baseTokenizer);
        /// </para>
        /// </remarks>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Releases the resources used by the Tree-sitter parser.
        /// </summary>
        /// <param name="disposing">True if called from Dispose(), false if called from finalizer.</param>
        private void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _parser?.Dispose();
                    _language?.Dispose();
                }
                _disposed = true;
            }
        }

        /// <summary>
        /// Finalizer to ensure resources are released.
        /// </summary>
        ~TreeSitterTokenizer()
        {
            Dispose(false);
        }
    }
}
