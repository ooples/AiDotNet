using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using AiDotNet.Validation;

namespace AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns;

/// <summary>
/// Multi-step reasoning retriever that breaks down complex queries into sequential steps.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This advanced retrieval pattern orchestrates multi-step reasoning where each step
/// builds upon the results of previous steps. Unlike Chain-of-Thought which plans all
/// steps upfront, multi-step reasoning adapts each step based on what was learned from
/// previous steps, enabling dynamic problem-solving.
/// </para>
/// <para><b>For Beginners:</b> Think of this like solving a mystery by following clues.
///
/// Chain-of-Thought (plan everything first):
/// - Question: "Who invented the transistor and how did it impact computing?"
/// - Plans: [Find inventor, Find invention date, Find early applications, Find impact]
/// - Executes all steps
///
/// Multi-Step Reasoning (adapt as you learn):
/// - Question: "Who invented the transistor and how did it impact computing?"
/// - Step 1: Search "transistor inventor" → Learn about Bell Labs team
/// - Step 2: Based on Bell Labs finding, search "Bell Labs transistor computing applications"
/// - Step 3: Based on applications found, search "transistor revolution computer architecture"
/// - Each step informed by previous discoveries
///
/// This is useful when:
/// - The answer requires building knowledge progressively
/// - Later steps depend on findings from earlier steps
/// - You need to adapt the search strategy based on what you find
/// </para>
/// <para><b>Example Usage:</b>
/// <code>
/// var generator = new StubGenerator&lt;double&gt;();
/// var baseRetriever = new DenseRetriever&lt;double&gt;(embeddingModel, documentStore);
///
/// var multiStepRetriever = new MultiStepReasoningRetriever&lt;double&gt;(
///     generator,
///     baseRetriever,
///     maxSteps: 5  // Allow up to 5 reasoning steps
/// );
///
/// var result = multiStepRetriever.RetrieveMultiStep(
///     "What are the environmental and economic impacts of solar energy adoption?",
///     topK: 15
/// );
///
/// // result.Documents: Final aggregated documents
/// // result.ReasoningTrace: Shows the progression of reasoning steps
/// // result.StepResults: Detailed results from each step
/// </code>
/// </para>
/// </remarks>
public class MultiStepReasoningRetriever<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly IGenerator<T> _generator;
    private readonly RetrieverBase<T> _baseRetriever;
    private readonly int _maxSteps;

    /// <summary>
    /// Represents a single reasoning step in the multi-step process.
    /// </summary>
    public class ReasoningStepResult
    {
        /// <summary>
        /// The query or reasoning focus for this step.
        /// </summary>
        public string StepQuery { get; set; } = string.Empty;

        /// <summary>
        /// Documents retrieved in this step.
        /// </summary>
        public List<Document<T>> Documents { get; set; } = new List<Document<T>>();

        /// <summary>
        /// Summary of findings from this step.
        /// </summary>
        public string StepSummary { get; set; } = string.Empty;

        /// <summary>
        /// Whether this step yielded useful information.
        /// </summary>
        public bool IsSuccessful { get; set; }

        /// <summary>
        /// The step number in the sequence.
        /// </summary>
        public int StepNumber { get; set; }
    }

    /// <summary>
    /// Result of multi-step reasoning retrieval.
    /// </summary>
    public class MultiStepReasoningResult
    {
        /// <summary>
        /// All documents retrieved across all steps.
        /// </summary>
        public IEnumerable<Document<T>> Documents { get; set; } = new List<Document<T>>();

        /// <summary>
        /// Detailed results from each reasoning step.
        /// </summary>
        public IReadOnlyList<ReasoningStepResult> StepResults { get; set; } = new List<ReasoningStepResult>();

        /// <summary>
        /// Trace of the reasoning progression.
        /// </summary>
        public string ReasoningTrace { get; set; } = string.Empty;

        /// <summary>
        /// Total number of steps executed.
        /// </summary>
        public int TotalSteps { get; set; }

        /// <summary>
        /// Whether the reasoning converged to a solution.
        /// </summary>
        public bool Converged { get; set; }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="MultiStepReasoningRetriever{T}"/> class.
    /// </summary>
    /// <param name="generator">The LLM generator for reasoning.</param>
    /// <param name="baseRetriever">The underlying retriever to use.</param>
    /// <param name="maxSteps">Maximum number of reasoning steps (default: 5).</param>
    public MultiStepReasoningRetriever(
        IGenerator<T> generator,
        RetrieverBase<T> baseRetriever,
        int maxSteps = 5)
    {
        Guard.NotNull(generator);
        _generator = generator;
        Guard.NotNull(baseRetriever);
        _baseRetriever = baseRetriever;

        if (maxSteps <= 0 || maxSteps > 20)
            throw new ArgumentOutOfRangeException(nameof(maxSteps), "maxSteps must be between 1 and 20");

        _maxSteps = maxSteps;
    }

    /// <summary>
    /// Retrieves documents using adaptive multi-step reasoning.
    /// </summary>
    /// <param name="query">The complex query to answer.</param>
    /// <param name="topK">Maximum number of documents to return.</param>
    /// <param name="metadataFilters">Metadata filters to apply during retrieval.</param>
    /// <returns>Multi-step reasoning result with documents and reasoning trace.</returns>
    public MultiStepReasoningResult RetrieveMultiStep(
        string query,
        int topK,
        Dictionary<string, object>? metadataFilters = null)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or whitespace", nameof(query));

        if (topK < 1)
            throw new ArgumentOutOfRangeException(nameof(topK), "topK must be positive");

        metadataFilters ??= new Dictionary<string, object>();

        var stepResults = new List<ReasoningStepResult>();
        var allDocuments = new Dictionary<string, Document<T>>();
        var knowledgeAccumulated = new StringBuilder();
        bool converged = false;

        // Initial context
        knowledgeAccumulated.AppendLine($"Original Query: {query}");
        knowledgeAccumulated.AppendLine();

        for (int step = 1; step <= _maxSteps; step++)
        {
            // Determine next step based on accumulated knowledge
            var nextStep = DetermineNextStep(query, knowledgeAccumulated.ToString(), step);

            if (string.IsNullOrWhiteSpace(nextStep.query))
            {
                // No more steps needed
                converged = true;
                break;
            }

            // Execute the step
            var stepResult = ExecuteStep(nextStep.query, step, metadataFilters);
            stepResults.Add(stepResult);

            // Accumulate documents
            foreach (var doc in stepResult.Documents.Where(d => !allDocuments.ContainsKey(d.Id)))
            {
                allDocuments[doc.Id] = doc;
            }

            // Update accumulated knowledge
            knowledgeAccumulated.AppendLine($"Step {step}: {stepResult.StepQuery}");
            knowledgeAccumulated.AppendLine($"Findings: {stepResult.StepSummary}");
            knowledgeAccumulated.AppendLine($"Documents found: {stepResult.Documents.Count}");
            knowledgeAccumulated.AppendLine();

            // Check if we should stop
            if (nextStep.isFinalStep || !stepResult.IsSuccessful)
            {
                converged = nextStep.isFinalStep;
                break;
            }
        }

        // Return results
        var finalDocuments = allDocuments.Values
            .OrderByDescending(d => d.HasRelevanceScore ? d.RelevanceScore : default(T))
            .Take(topK);

        return new MultiStepReasoningResult
        {
            Documents = finalDocuments,
            StepResults = stepResults.AsReadOnly(),
            ReasoningTrace = knowledgeAccumulated.ToString(),
            TotalSteps = stepResults.Count,
            Converged = converged
        };
    }

    /// <summary>
    /// Determines the next reasoning step based on accumulated knowledge.
    /// </summary>
    private (string query, bool isFinalStep) DetermineNextStep(
        string originalQuery,
        string accumulatedKnowledge,
        int stepNumber)
    {
        var prompt = $@"{accumulatedKnowledge}

Based on the original query and what we've learned so far, what should be the next step?

Guidelines:
- If we have sufficient information to answer the query comprehensively, respond with: COMPLETE
- Otherwise, provide a specific search query for the next step
- Focus on gaps in our current knowledge
- Build upon what we've already learned

Next step:";

        var response = _generator.Generate(prompt).Trim();

        bool isFinalStep = response.IndexOf("COMPLETE", StringComparison.OrdinalIgnoreCase) >= 0;

        if (isFinalStep)
        {
            return (string.Empty, true);
        }

        return (response, false);
    }

    /// <summary>
    /// Executes a single reasoning step.
    /// </summary>
    private ReasoningStepResult ExecuteStep(
        string stepQuery,
        int stepNumber,
        Dictionary<string, object> metadataFilters)
    {
        // Retrieve documents for this step
        var documents = _baseRetriever.Retrieve(stepQuery, topK: 5, metadataFilters).ToList();

        // Summarize findings
        var summary = SummarizeStepFindings(stepQuery, documents);

        return new ReasoningStepResult
        {
            StepQuery = stepQuery,
            Documents = documents,
            StepSummary = summary,
            IsSuccessful = documents.Count > 0,
            StepNumber = stepNumber
        };
    }

    /// <summary>
    /// Summarizes the findings from a reasoning step.
    /// </summary>
    private string SummarizeStepFindings(string stepQuery, List<Document<T>> documents)
    {
        if (documents.Count == 0)
        {
            return "No relevant documents found.";
        }

        var topDocsContent = string.Join("\n",
            documents.Take(3).Select((d, i) =>
                $"[{i + 1}] {d.Content.Substring(0, Math.Min(150, d.Content.Length))}..."));

        var summaryPrompt = $@"Query: {stepQuery}

Top documents found:
{topDocsContent}

Provide a brief 1-2 sentence summary of the key findings from these documents:";

        var summary = _generator.Generate(summaryPrompt);
        return summary.Trim();
    }
}

/// <summary>
/// Tool-augmented reasoning retriever that can use external tools during reasoning.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This pattern extends multi-step reasoning by incorporating external tools such as
/// calculators, code interpreters, or specialized APIs. The system can recognize when
/// a tool is needed, invoke it, and incorporate the results into the reasoning process.
/// </para>
/// <para><b>For Beginners:</b> Think of this like a researcher with access to specialized equipment.
///
/// Without tools:
/// - "What is the compound annual growth rate of solar installations from 2015 to 2023?"
/// - Can only retrieve documents about growth rates
///
/// With tools:
/// - Retrieves data: 2015: 50 GW, 2023: 400 GW
/// - Recognizes calculation needed
/// - Uses calculator tool: CAGR = (400/50)^(1/8) - 1 = 29.4%
/// - Incorporates calculation into answer
///
/// Supported tool types:
/// - Calculator: Mathematical computations
/// - Code: Execute code for data processing
/// - Custom: User-defined tools
/// </para>
/// </remarks>
public class ToolAugmentedReasoningRetriever<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly IGenerator<T> _generator;
    private readonly RetrieverBase<T> _baseRetriever;
    private readonly Dictionary<string, Func<string, string>> _tools;

    /// <summary>
    /// Represents a tool invocation during reasoning.
    /// </summary>
    public class ToolInvocation
    {
        public string ToolName { get; set; } = string.Empty;
        public string Input { get; set; } = string.Empty;
        public string Output { get; set; } = string.Empty;
        public bool Success { get; set; }
    }

    /// <summary>
    /// Result of tool-augmented reasoning.
    /// </summary>
    public class ToolAugmentedResult
    {
        public IEnumerable<Document<T>> Documents { get; set; } = new List<Document<T>>();
        public IReadOnlyList<ToolInvocation> ToolInvocations { get; set; } = new List<ToolInvocation>();
        public string ReasoningTrace { get; set; } = string.Empty;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ToolAugmentedReasoningRetriever{T}"/> class.
    /// </summary>
    public ToolAugmentedReasoningRetriever(
        IGenerator<T> generator,
        RetrieverBase<T> baseRetriever)
    {
        Guard.NotNull(generator);
        _generator = generator;
        Guard.NotNull(baseRetriever);
        _baseRetriever = baseRetriever;
        _tools = new Dictionary<string, Func<string, string>>();

        // Register default tools
        RegisterDefaultTools();
    }

    /// <summary>
    /// Registers a custom tool for use during reasoning.
    /// </summary>
    /// <param name="toolName">Name of the tool.</param>
    /// <param name="toolFunction">Function that takes input string and returns output string.</param>
    public void RegisterTool(string toolName, Func<string, string> toolFunction)
    {
        if (string.IsNullOrWhiteSpace(toolName))
            throw new ArgumentException("Tool name cannot be null or whitespace", nameof(toolName));

        if (toolFunction == null)
            throw new ArgumentNullException(nameof(toolFunction));

        _tools[toolName] = toolFunction;
    }

    /// <summary>
    /// Retrieves documents using tool-augmented reasoning.
    /// </summary>
    public ToolAugmentedResult RetrieveWithTools(
        string query,
        int topK,
        Dictionary<string, object>? metadataFilters = null)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or whitespace", nameof(query));

        metadataFilters ??= new Dictionary<string, object>();

        var toolInvocations = new List<ToolInvocation>();
        var trace = new StringBuilder();
        var allDocuments = new Dictionary<string, Document<T>>();

        trace.AppendLine($"Query: {query}");
        trace.AppendLine();

        // Step 1: Initial retrieval
        var initialDocs = _baseRetriever.Retrieve(query, topK: 5, metadataFilters).ToList();
        foreach (var doc in initialDocs)
        {
            allDocuments[doc.Id] = doc;
        }

        trace.AppendLine($"Initial retrieval: {initialDocs.Count} documents");
        trace.AppendLine();

        // Step 2: Analyze if tools are needed
        var toolAnalysis = AnalyzeToolNeeds(query, initialDocs);
        trace.AppendLine($"Tool analysis: {toolAnalysis.reasoning}");
        trace.AppendLine();

        // Step 3: Execute tools if needed
        if (toolAnalysis.needsTools)
        {
            foreach (var toolCall in toolAnalysis.toolCalls)
            {
                var invocation = ExecuteTool(toolCall.toolName, toolCall.input);
                toolInvocations.Add(invocation);

                trace.AppendLine($"Tool: {invocation.ToolName}");
                trace.AppendLine($"Input: {invocation.Input}");
                trace.AppendLine($"Output: {invocation.Output}");
                trace.AppendLine();

                // Retrieve additional documents based on tool output
                if (invocation.Success && !string.IsNullOrWhiteSpace(invocation.Output))
                {
                    var toolDocs = _baseRetriever.Retrieve(invocation.Output, topK: 3, metadataFilters).ToList();
                    foreach (var doc in toolDocs.Where(d => !allDocuments.ContainsKey(d.Id)))
                    {
                        allDocuments[doc.Id] = doc;
                    }
                }
            }
        }

        var finalDocuments = allDocuments.Values
            .OrderByDescending(d => d.HasRelevanceScore ? d.RelevanceScore : default(T))
            .Take(topK);

        return new ToolAugmentedResult
        {
            Documents = finalDocuments,
            ToolInvocations = toolInvocations.AsReadOnly(),
            ReasoningTrace = trace.ToString()
        };
    }

    /// <summary>
    /// Registers default tools (calculator, string operations).
    /// </summary>
    private void RegisterDefaultTools()
    {
        // Calculator tool
        RegisterTool("calculator", input =>
        {
            try
            {
                // Simple expression evaluation (in production, use a proper math parser)
                var result = EvaluateSimpleMathExpression(input);
                return result.ToString();
            }
            catch (Exception ex)
            {
                return $"Error: {ex.Message}";
            }
        });

        // String manipulation tool
        RegisterTool("text_analyzer", input =>
        {
            return $"Length: {input.Length} characters, Words: {input.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Length}";
        });
    }

    /// <summary>
    /// Simple math expression evaluator (basic implementation).
    /// </summary>
    private double EvaluateSimpleMathExpression(string expression)
    {
        // This is a simplified implementation
        // In production, use a proper expression parser like NCalc or similar
        expression = expression.Replace(" ", "");

        // Try simple parsing
        if (double.TryParse(expression, out double result))
            return result;

        // Handle basic operations (very limited)
        if (expression.Contains('+'))
        {
            var parts = expression.Split('+');
            return double.Parse(parts[0]) + double.Parse(parts[1]);
        }

        throw new NotSupportedException("Complex expressions not supported in this simple implementation");
    }

    /// <summary>
    /// Analyzes whether tools are needed for the query.
    /// </summary>
    private (bool needsTools, string reasoning, List<(string toolName, string input)> toolCalls) AnalyzeToolNeeds(
        string query,
        List<Document<T>> initialDocs)
    {
        var availableTools = string.Join(", ", _tools.Keys);

        var analysisPrompt = $@"Query: {query}

Available tools: {availableTools}

Does this query require using any tools? Consider:
- calculator: for mathematical computations
- text_analyzer: for analyzing text properties

Respond in format:
NEEDS_TOOLS: yes/no
REASONING: [brief explanation]
TOOL_CALLS: [if yes, list as 'toolname:input' separated by newlines]";

        var response = _generator.Generate(analysisPrompt);

        bool needsTools = response.IndexOf("NEEDS_TOOLS: yes", StringComparison.OrdinalIgnoreCase) >= 0;
        string reasoning = ExtractSection(response, "REASONING:");
        var toolCalls = ParseToolCalls(response);

        return (needsTools, reasoning, toolCalls);
    }

    private string ExtractSection(string text, string sectionName)
    {
        var match = System.Text.RegularExpressions.Regex.Match(
            text,
            $@"{sectionName}\s*(.+?)(?=\n[A-Z_]+:|$)",
            System.Text.RegularExpressions.RegexOptions.Singleline | System.Text.RegularExpressions.RegexOptions.IgnoreCase,
            RegexTimeout
        );

        return match.Success ? match.Groups[1].Value.Trim() : string.Empty;
    }

    private List<(string toolName, string input)> ParseToolCalls(string response)
    {
        var section = ExtractSection(response, "TOOL_CALLS:");

        if (string.IsNullOrWhiteSpace(section))
            return new List<(string, string)>();

        var calls = section.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(line => line.Split(new[] { ':' }, 2))
            .Where(parts => parts.Length == 2)
            .Select(parts => (parts[0].Trim(), parts[1].Trim()))
            .ToList();

        return calls;
    }

    private ToolInvocation ExecuteTool(string toolName, string input)
    {
        if (!_tools.TryGetValue(toolName, out var tool))
        {
            return new ToolInvocation
            {
                ToolName = toolName,
                Input = input,
                Output = $"Tool '{toolName}' not found",
                Success = false
            };
        }

        try
        {
            var output = tool(input);
            return new ToolInvocation
            {
                ToolName = toolName,
                Input = input,
                Output = output,
                Success = true
            };
        }
        catch (Exception ex)
        {
            return new ToolInvocation
            {
                ToolName = toolName,
                Input = input,
                Output = $"Error: {ex.Message}",
                Success = false
            };
        }
    }
}
