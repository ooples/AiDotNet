namespace AiDotNet.Tokenization.CodeTokenization
{
    /// <summary>
    /// Supported programming languages for Tree-sitter parsing.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each language has its own grammar rules that Tree-sitter
    /// uses to understand the code structure. Choose the language that matches your source code.
    /// </para>
    /// </remarks>
    public enum TreeSitterLanguage
    {
        /// <summary>C# programming language.</summary>
        CSharp,
        /// <summary>Python programming language.</summary>
        Python,
        /// <summary>JavaScript programming language.</summary>
        JavaScript,
        /// <summary>TypeScript programming language.</summary>
        TypeScript,
        /// <summary>Java programming language.</summary>
        Java,
        /// <summary>C programming language.</summary>
        C,
        /// <summary>C++ programming language.</summary>
        Cpp,
        /// <summary>Go programming language.</summary>
        Go,
        /// <summary>Rust programming language.</summary>
        Rust,
        /// <summary>Ruby programming language.</summary>
        Ruby,
        /// <summary>JSON data format.</summary>
        Json,
        /// <summary>HTML markup language.</summary>
        Html,
        /// <summary>CSS stylesheet language.</summary>
        Css
    }
}
