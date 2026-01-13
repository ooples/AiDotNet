using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;
using AiDotNet.Reasoning.Strategies;
using AiDotNet.Reasoning.Verification;

namespace AiDotNet.Reasoning.DomainSpecific;

/// <summary>
/// Specialized reasoner for code-related problems including generation, debugging, and explanation.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> CodeReasoner is like a programming tutor that thinks through
/// code problems step-by-step, writes code, and can even test if it works.
///
/// **What it does:**
/// - Breaks down programming problems into steps
/// - Generates code with explanations
/// - Can debug and fix errors
/// - Explains existing code
/// - Uses Tree-of-Thoughts for complex algorithms
///
/// **Example workflow:**
/// Problem: "Write a function to check if a number is prime"
///
/// Step 1: "Define the problem: Check divisibility from 2 to sqrt(n)"
/// Step 2: "Handle edge cases: numbers <= 1 are not prime"
/// Step 3: "Implement the algorithm"
/// ```python
/// def is_prime(n):
///     if n <= 1:
///         return False
///     for i in range(2, int(n**0.5) + 1):
///         if n % i == 0:
///             return False
///     return True
/// ```
/// Step 4: "Test with examples: is_prime(7) = True, is_prime(4) = False"
///
/// **Used for benchmarks:**
/// - HumanEval (Python code generation)
/// - MBPP (Python programming problems)
/// - Code debugging tasks
/// - Code explanation tasks
/// </para>
/// </remarks>
public class CodeReasoner<T>
{
    private readonly IChatModel<T> _chatModel;
    private readonly ChainOfThoughtStrategy<T> _cotStrategy;
    private readonly TreeOfThoughtsStrategy<T> _totStrategy;
    private readonly CriticModel<T> _criticModel;
    private readonly SelfRefinementEngine<T> _refinementEngine;

    /// <summary>
    /// Initializes a new instance of the <see cref="CodeReasoner{T}"/> class.
    /// </summary>
    /// <param name="chatModel">The chat model used for reasoning.</param>
    /// <param name="tools">Optional code-related tools (code execution, linters, etc.).</param>
    public CodeReasoner(IChatModel<T> chatModel, IEnumerable<ITool>? tools = null)
    {
        _chatModel = chatModel ?? throw new ArgumentNullException(nameof(chatModel));
        _cotStrategy = new ChainOfThoughtStrategy<T>(chatModel, tools);
        _totStrategy = new TreeOfThoughtsStrategy<T>(chatModel, tools);
        _criticModel = new CriticModel<T>(chatModel);
        _refinementEngine = new SelfRefinementEngine<T>(chatModel);
    }

    /// <summary>
    /// Solves a code-related problem using reasoning.
    /// </summary>
    /// <param name="problem">The coding problem or task.</param>
    /// <param name="config">Reasoning configuration.</param>
    /// <param name="useTreeSearch">Whether to use Tree-of-Thoughts for complex problems.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Reasoning result with code solution.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Main method for solving coding problems.
    /// - Simple problems: Use Chain-of-Thought (linear reasoning)
    /// - Complex problems: Use Tree-of-Thoughts (explore multiple approaches)
    /// </para>
    /// </remarks>
    public async Task<ReasoningResult<T>> SolveAsync(
        string problem,
        ReasoningConfig? config = null,
        bool useTreeSearch = false,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(problem))
            throw new ArgumentException("Problem cannot be null or empty", nameof(problem));

        config ??= new ReasoningConfig();

        // Configure for code reasoning
        config.EnableVerification = true;
        config.Temperature = 0.2; // Lower temperature for more deterministic code generation

        ReasoningResult<T> result;

        // Choose strategy
        if (useTreeSearch)
        {
            config.ExplorationDepth = Math.Max(config.ExplorationDepth, 3);
            config.BranchingFactor = Math.Max(config.BranchingFactor, 2);
            result = await _totStrategy.ReasonAsync(problem, config, cancellationToken);
        }
        else
        {
            result = await _cotStrategy.ReasonAsync(problem, config, cancellationToken);
        }

        // Extract code from the result
        var extractedCode = ExtractCode(result.FinalAnswer);
        if (extractedCode is not null && !string.IsNullOrEmpty(extractedCode))
        {
            result.Metadata["extracted_code"] = extractedCode;
        }

        // Add domain metadata
        result.Metadata["domain"] = "code";
        result.Metadata["tree_search_used"] = useTreeSearch;
        result.Metadata["language"] = string.IsNullOrEmpty(result.FinalAnswer)
            ? "unknown"
            : DetectProgrammingLanguage(result.FinalAnswer);

        return result;
    }

    /// <summary>
    /// Generates code with step-by-step explanation.
    /// </summary>
    /// <param name="specification">The code specification or requirements.</param>
    /// <param name="language">Programming language (e.g., "python", "javascript", "csharp").</param>
    /// <param name="config">Reasoning configuration.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Result with generated code and explanation.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Generates code for a specific task with explanations
    /// of each step in the implementation.
    /// </para>
    /// </remarks>
    public async Task<ReasoningResult<T>> GenerateCodeAsync(
        string specification,
        string language = "python",
        ReasoningConfig? config = null,
        CancellationToken cancellationToken = default)
    {
        string problem = $@"Write a {language} function or program that: {specification}

Requirements:
1. Break down the problem into clear steps
2. Explain your approach
3. Write clean, well-commented code
4. Consider edge cases
5. Provide example usage

Language: {language}";

        return await SolveAsync(problem, config, useTreeSearch: false, cancellationToken);
    }

    /// <summary>
    /// Debugs code by analyzing errors and suggesting fixes.
    /// </summary>
    /// <param name="code">The code with errors.</param>
    /// <param name="error">The error message or description.</param>
    /// <param name="config">Reasoning configuration.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Result with debugging analysis and fixed code.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Helps debug code by reasoning about what went wrong
    /// and how to fix it.
    /// </para>
    /// </remarks>
    public async Task<ReasoningResult<T>> DebugCodeAsync(
        string code,
        string error,
        ReasoningConfig? config = null,
        CancellationToken cancellationToken = default)
    {
        string problem = $@"Debug this code that produces an error:

```
{code}
```

Error: {error}

Task:
1. Identify what's causing the error
2. Explain why it's happening
3. Provide the corrected code
4. Explain what was fixed";

        return await SolveAsync(problem, config, useTreeSearch: false, cancellationToken);
    }

    /// <summary>
    /// Explains how existing code works.
    /// </summary>
    /// <param name="code">The code to explain.</param>
    /// <param name="config">Reasoning configuration.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Result with step-by-step code explanation.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Provides a step-by-step explanation of how code works.
    /// </para>
    /// </remarks>
    public async Task<ReasoningResult<T>> ExplainCodeAsync(
        string code,
        ReasoningConfig? config = null,
        CancellationToken cancellationToken = default)
    {
        string problem = $@"Explain how this code works step by step:

```
{code}
```

Provide:
1. Overall purpose
2. Step-by-step breakdown
3. Key concepts used
4. Time/space complexity if applicable";

        return await SolveAsync(problem, config, useTreeSearch: false, cancellationToken);
    }

    /// <summary>
    /// Extracts code blocks from markdown-formatted text.
    /// </summary>
    /// <param name="text">Text potentially containing code blocks.</param>
    /// <returns>Extracted code, or null if none found.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> LLMs often return code wrapped in markdown code blocks
    /// (```language ... ```). This extracts just the code part.
    /// </para>
    /// </remarks>
    public string? ExtractCode(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return null;

        // Try to extract code from markdown code blocks
        var match = System.Text.RegularExpressions.Regex.Match(
            text,
            @"```(?:[\w]+)?\s*\n([\s\S]*?)\n```",
            System.Text.RegularExpressions.RegexOptions.Multiline,
            TimeSpan.FromSeconds(1));

        if (match.Success)
        {
            return match.Groups[1].Value.Trim();
        }

        // If no code block, check if the entire text looks like code
        if (LooksLikeCode(text))
        {
            return text.Trim();
        }

        return null;
    }

    /// <summary>
    /// Detects the programming language from text.
    /// </summary>
    private string DetectProgrammingLanguage(string text)
    {
        string lower = text.ToLowerInvariant();

        // Check for language indicators
        if (lower.Contains("def ") || lower.Contains("import ") || lower.Contains("python"))
            return "python";

        if (lower.Contains("function ") || lower.Contains("const ") || lower.Contains("javascript"))
            return "javascript";

        if (lower.Contains("public class") || lower.Contains("namespace") || lower.Contains("c#"))
            return "csharp";

        if (lower.Contains("fn ") || lower.Contains("rust"))
            return "rust";

        if (lower.Contains("func ") || lower.Contains("golang") || lower.Contains("go"))
            return "go";

        return "unknown";
    }

    /// <summary>
    /// Heuristic check if text looks like code.
    /// </summary>
    private bool LooksLikeCode(string text)
    {
        // Simple heuristics
        int lines = text.Split('\n').Length;
        int braces = text.Count(c => c == '{' || c == '}');
        int parens = text.Count(c => c == '(' || c == ')');
        int semicolons = text.Count(c => c == ';');

        // If it has structural elements typical of code
        return (braces > 2 || (parens > 4 && semicolons > 0) || text.Contains("def ") || text.Contains("function "));
    }
}
