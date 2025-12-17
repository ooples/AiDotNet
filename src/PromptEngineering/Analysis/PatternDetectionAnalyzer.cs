using System.Text.RegularExpressions;

namespace AiDotNet.PromptEngineering.Analysis;

/// <summary>
/// Analyzer that specializes in detecting prompt patterns and categorizing prompts.
/// </summary>
/// <remarks>
/// <para>
/// This analyzer identifies what type of prompt is being used and what patterns
/// are present. It can detect few-shot prompts, chain-of-thought patterns,
/// system prompts, and various task types.
/// </para>
/// <para><b>For Beginners:</b> Figures out what kind of prompt you're using.
///
/// Example:
/// <code>
/// var analyzer = new PatternDetectionAnalyzer();
/// var metrics = analyzer.Analyze("Let's think step by step. First, we need to...");
///
/// Console.WriteLine(string.Join(", ", metrics.DetectedPatterns));
/// // Output: "chain-of-thought, instruction"
///
/// // This tells you:
/// // - It's using chain-of-thought reasoning
/// // - It contains instructions
/// </code>
///
/// Common patterns detected:
/// - few-shot: Contains examples
/// - chain-of-thought: Step-by-step reasoning
/// - role-playing: Sets up a persona
/// - template: Contains variables
/// - question, summarization, translation, etc.
/// </para>
/// </remarks>
public class PatternDetectionAnalyzer : PromptAnalyzerBase
{
    /// <summary>
    /// Regex timeout to prevent ReDoS attacks.
    /// </summary>
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);

    /// <summary>
    /// Initializes a new instance of the PatternDetectionAnalyzer class.
    /// </summary>
    public PatternDetectionAnalyzer()
        : base("PatternDetectionAnalyzer", "general", 0.0m)
    {
    }

    /// <summary>
    /// Enhanced pattern detection with more detailed analysis.
    /// </summary>
    protected override IReadOnlyList<string> DetectPatterns(string prompt)
    {
        var patterns = new List<string>();
        var lowerPrompt = prompt.ToLowerInvariant();

        // Core task type detection
        DetectTaskTypes(lowerPrompt, patterns);

        // Prompt engineering technique detection
        DetectTechniques(prompt, lowerPrompt, patterns);

        // Format and structure detection
        DetectStructure(prompt, patterns);

        // Output format detection
        DetectOutputFormats(lowerPrompt, patterns);

        if (patterns.Count == 0)
        {
            patterns.Add("general");
        }

        return patterns.Distinct().ToList().AsReadOnly();
    }

    /// <summary>
    /// Detects task types from the prompt.
    /// </summary>
    private static void DetectTaskTypes(string lowerPrompt, List<string> patterns)
    {
        var taskPatterns = new Dictionary<string, string[]>
        {
            { "question", new[] { @"\?(?:\s|$)", @"\b(what|who|where|when|why|how|which|can you|could you|do you)\b" } },
            { "generation", new[] { @"\b(write|create|generate|produce|compose|draft|author)\b" } },
            { "summarization", new[] { @"\b(summarize|summarise|summary|tldr|condense|brief|overview)\b" } },
            { "translation", new[] { @"\b(translate|translation|convert.*to|from.*to.*language)\b" } },
            { "analysis", new[] { @"\b(analyze|analyse|analysis|examine|evaluate|assess|review)\b" } },
            { "extraction", new[] { @"\b(extract|identify|find|list|get|retrieve|pull out)\b" } },
            { "classification", new[] { @"\b(classify|categorize|categorise|label|tag|sort)\b" } },
            { "comparison", new[] { @"\b(compare|contrast|difference|similar|versus|vs\.?)\b" } },
            { "explanation", new[] { @"\b(explain|describe|elaborate|clarify|define)\b" } },
            { "code-generation", new[] { @"\b(code|function|class|method|script|program|implement)\b" } },
            { "editing", new[] { @"\b(edit|revise|improve|rewrite|fix|correct|polish)\b" } },
            { "reasoning", new[] { @"\b(reason|logic|deduce|infer|conclude|prove)\b" } }
        };

        foreach (var kvp in taskPatterns)
        {
            foreach (var pattern in kvp.Value)
            {
                if (Regex.IsMatch(lowerPrompt, pattern, RegexOptions.None, RegexTimeout))
                {
                    patterns.Add(kvp.Key);
                    break;
                }
            }
        }
    }

    /// <summary>
    /// Detects prompt engineering techniques.
    /// </summary>
    private void DetectTechniques(string prompt, string lowerPrompt, List<string> patterns)
    {
        // Chain-of-thought
        if (Regex.IsMatch(lowerPrompt, @"\b(step.?by.?step|let'?s think|thinking through|reasoning|think about)\b", RegexOptions.None, RegexTimeout) ||
            Regex.IsMatch(lowerPrompt, @"\bfirst.*then.*finally\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("chain-of-thought");
        }

        // Zero-shot chain-of-thought
        if (Regex.IsMatch(lowerPrompt, @"\blet'?s\s+think\s+(about\s+this\s+)?step\s+by\s+step\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("zero-shot-cot");
        }

        // Few-shot
        var exampleCount = CountExamples(prompt);
        if (exampleCount > 0)
        {
            patterns.Add("few-shot");
            if (exampleCount >= 3)
            {
                patterns.Add("many-shot");
            }
        }

        // Role-playing / persona
        if (Regex.IsMatch(lowerPrompt, @"\b(you are|act as|pretend|role|persona|imagine you're)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("role-playing");
        }

        // System prompt indicators
        if (Regex.IsMatch(lowerPrompt, @"\b(your (task|goal|role) is|you will|you should always|never|always respond)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("system-prompt");
        }

        // Self-consistency
        if (Regex.IsMatch(lowerPrompt, @"\b(multiple (ways|approaches|solutions)|different perspectives?)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("self-consistency");
        }

        // Reflexion / self-reflection
        if (Regex.IsMatch(lowerPrompt, @"\b(reflect|self.?check|verify your|double.?check)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("reflexion");
        }

        // Tree of thoughts
        if (Regex.IsMatch(lowerPrompt, @"\b(explore (multiple|different)|branch|evaluate paths?)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("tree-of-thoughts");
        }

        // Prompt injection defense
        if (Regex.IsMatch(lowerPrompt, @"\b(ignore (any )?instructions|user input|untrusted)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("injection-defense");
        }

        // Meta-prompting
        if (Regex.IsMatch(lowerPrompt, @"\b(generate a prompt|create a prompt|write a prompt)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("meta-prompting");
        }
    }

    /// <summary>
    /// Detects structural patterns in the prompt.
    /// </summary>
    private static void DetectStructure(string prompt, List<string> patterns)
    {
        // Template detection
        if (Regex.IsMatch(prompt, @"\{[^}]+\}", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("template");
        }

        // Markdown structure
        if (Regex.IsMatch(prompt, @"^#{1,6}\s", RegexOptions.Multiline, RegexTimeout) ||
            Regex.IsMatch(prompt, @"^\s*[-*]\s", RegexOptions.Multiline, RegexTimeout))
        {
            patterns.Add("markdown-structured");
        }

        // Numbered lists
        if (Regex.IsMatch(prompt, @"^\s*\d+\.\s", RegexOptions.Multiline, RegexTimeout))
        {
            patterns.Add("numbered-list");
        }

        // XML/Tag structure
        if (Regex.IsMatch(prompt, @"<[^>]+>.*?</[^>]+>", RegexOptions.Singleline, RegexTimeout))
        {
            patterns.Add("xml-structured");
        }

        // Code blocks
        if (Regex.IsMatch(prompt, @"```[\s\S]*?```", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("contains-code");
        }

        // Multi-turn conversation
        if (Regex.IsMatch(prompt, @"\b(user|human|assistant|system)\s*:", RegexOptions.IgnoreCase, RegexTimeout))
        {
            patterns.Add("multi-turn");
        }

        // Delimiter-separated sections
        if (Regex.IsMatch(prompt, @"---+|===+|\*\*\*+", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("section-delimited");
        }
    }

    /// <summary>
    /// Detects expected output format specifications.
    /// </summary>
    private static void DetectOutputFormats(string lowerPrompt, List<string> patterns)
    {
        // JSON output
        if (Regex.IsMatch(lowerPrompt, @"\b(json|json format|output.*json|respond.*json)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("json-output");
        }

        // YAML output
        if (Regex.IsMatch(lowerPrompt, @"\b(yaml|yml)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("yaml-output");
        }

        // Structured data
        if (Regex.IsMatch(lowerPrompt, @"\b(table|csv|structured|format.*as|output.*as)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("structured-output");
        }

        // Bullet points
        if (Regex.IsMatch(lowerPrompt, @"\b(bullet|bullets|bullet point|bulleted)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("bullet-output");
        }

        // Brief/concise
        if (Regex.IsMatch(lowerPrompt, @"\b(brief|concise|short|one.?liner|one sentence)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("concise-output");
        }

        // Detailed
        if (Regex.IsMatch(lowerPrompt, @"\b(detailed|comprehensive|thorough|in.?depth|elaborate)\b", RegexOptions.None, RegexTimeout))
        {
            patterns.Add("detailed-output");
        }
    }

    /// <summary>
    /// Counts few-shot examples with more sophisticated detection.
    /// </summary>
    protected override int CountExamples(string prompt)
    {
        var count = 0;

        // Explicit example markers
        count += Regex.Matches(prompt, @"Example\s*\d*\s*:", RegexOptions.IgnoreCase, RegexTimeout).Count;

        // Input/Output pairs
        var inputOutputPairs = Regex.Matches(prompt, @"Input:\s*.*\s*Output:", RegexOptions.IgnoreCase | RegexOptions.Singleline, RegexTimeout).Count;
        count += inputOutputPairs;

        // Q/A pairs
        count += Regex.Matches(prompt, @"Q:\s*.*?\s*A:", RegexOptions.IgnoreCase | RegexOptions.Singleline, RegexTimeout).Count;

        // Arrow patterns (common in few-shot)
        count += Regex.Matches(prompt, @"[^\n]+\s*(?:->|=>|→)\s*[^\n]+", RegexOptions.None, RegexTimeout).Count;

        // User/Assistant pairs
        count += Regex.Matches(prompt, @"User:\s*.*?\s*Assistant:", RegexOptions.IgnoreCase | RegexOptions.Singleline, RegexTimeout).Count;

        return count;
    }
}
