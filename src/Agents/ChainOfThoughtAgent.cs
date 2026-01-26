using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Agents;

/// <summary>
/// Implements a Chain of Thought (CoT) agent that solves problems through explicit step-by-step reasoning.
/// Unlike ReAct agents that alternate between thinking and acting, CoT agents break down complex problems
/// into a series of logical reasoning steps before arriving at the final answer.
/// </summary>
/// <typeparam name="T">The numeric type used for model parameters and operations (e.g., double, float).</typeparam>
/// <remarks>
/// For Beginners:
/// Chain of Thought (CoT) is a prompting technique that improves reasoning by having the AI show its work
/// step by step, similar to how a student solves a math problem.
///
/// **When to use ChainOfThoughtAgent vs Agent (ReAct):**
///
/// Use ChainOfThoughtAgent when:
/// - Problem requires logical deduction or mathematical reasoning
/// - Answer can be derived through pure thinking (minimal tool use)
/// - You want to see detailed reasoning process
/// - Problem involves multiple interconnected steps
///
/// Use Agent (ReAct) when:
/// - Problem requires multiple external tool calls
/// - Information needs to be gathered from various sources
/// - Actions depend on observations from previous actions
///
/// **How it works:**
/// <code>
/// User Query: "If a train travels 60 mph for 2.5 hours, how far does it go?"
///
/// Step 1: Understand the problem
/// - We have speed: 60 mph
/// - We have time: 2.5 hours
/// - We need to find distance
///
/// Step 2: Recall the relevant formula
/// - Distance = Speed × Time
///
/// Step 3: Calculate the result
/// - Distance = 60 mph × 2.5 hours = 150 miles
///
/// Final Answer: The train travels 150 miles.
/// </code>
///
/// **Benefits:**
/// - More interpretable reasoning (you can see why it arrived at the answer)
/// - Better performance on complex reasoning tasks
/// - Helps catch logical errors in the reasoning process
/// - Works well for problems that humans solve with explicit steps
///
/// **Research background:**
/// Based on "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022)
/// which showed that prompting models to show their reasoning steps significantly improves performance
/// on arithmetic, commonsense, and symbolic reasoning tasks.
/// </remarks>
public class ChainOfThoughtAgent<T> : AgentBase<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly bool _allowTools;

    /// <summary>
    /// Initializes a new instance of the <see cref="ChainOfThoughtAgent{T}"/> class.
    /// </summary>
    /// <param name="chatModel">The chat model used for reasoning and decision-making.</param>
    /// <param name="tools">The collection of tools available to the agent (optional for pure reasoning).</param>
    /// <param name="allowTools">Whether to allow tool use during reasoning (default: true). Set to false for pure CoT reasoning.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="chatModel"/> is null.</exception>
    /// <remarks>
    /// For Beginners:
    /// Creates a Chain of Thought agent that reasons through problems step by step.
    ///
    /// **Parameters:**
    /// - chatModel: The language model that will do the reasoning
    /// - tools: Optional tools the agent can use (Calculator, Search, etc.)
    /// - allowTools: If false, agent will only use reasoning without tools (pure CoT)
    ///
    /// **Pure CoT vs CoT with Tools:**
    /// - Pure CoT (allowTools=false): Agent solves everything through logical reasoning
    ///   Best for: Math problems, logic puzzles, deduction tasks
    ///
    /// - CoT with Tools (allowTools=true): Agent can use tools when reasoning alone isn't enough
    ///   Best for: Complex problems requiring both reasoning AND external information
    ///
    /// Example:
    /// <code>
    /// // Pure reasoning agent
    /// var pureCoT = new ChainOfThoughtAgent&lt;double&gt;(chatModel, allowTools: false);
    ///
    /// // CoT agent with calculator for complex math
    /// var calculator = new CalculatorTool();
    /// var cotWithTools = new ChainOfThoughtAgent&lt;double&gt;(chatModel, new[] { calculator });
    /// </code>
    /// </remarks>
    public ChainOfThoughtAgent(IChatModel<T> chatModel, IEnumerable<ITool>? tools = null, bool allowTools = true)
        : base(chatModel, tools)
    {
        _allowTools = allowTools;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Note: For ChainOfThoughtAgent, the maxIterations parameter controls the maximum number
    /// of reasoning steps the LLM should generate (not iteration cycles). The agent makes
    /// one initial LLM call and at most one refinement call if tools are used. The parameter
    /// is used to: (1) guide the LLM on how many reasoning steps to include, and (2) truncate
    /// the response if too many steps are generated. This differs from iterative agents like
    /// ReAct which loop multiple times.
    /// </remarks>
    public override async Task<string> RunAsync(string query, int maxIterations = 5)
    {
        ValidateMaxIterations(maxIterations);

        if (string.IsNullOrWhiteSpace(query))
        {
            throw new ArgumentException("Query cannot be null or whitespace.", nameof(query));
        }

        ClearScratchpad();
        AppendToScratchpad($"Query: {query}\n");

        // Step 1: Generate the chain of thought
        AppendToScratchpad("=== Generating Chain of Thought ===");

        string prompt = BuildChainOfThoughtPrompt(query, maxIterations);
        string llmResponse;

        try
        {
            llmResponse = await ChatModel.GenerateResponseAsync(prompt);
        }
        catch (System.Net.Http.HttpRequestException ex)
        {
            string errorMsg = $"Error communicating with chat model: {ex.Message}";
            AppendToScratchpad($"ERROR: {errorMsg}");
            return $"I encountered an error while reasoning: {ex.Message}";
        }
        catch (System.IO.IOException ex)
        {
            string errorMsg = $"IO error communicating with chat model: {ex.Message}";
            AppendToScratchpad($"ERROR: {errorMsg}");
            return $"I encountered an IO error while reasoning: {ex.Message}";
        }
        catch (TaskCanceledException ex)
        {
            string errorMsg = $"Timeout communicating with chat model: {ex.Message}";
            AppendToScratchpad($"ERROR: {errorMsg}");
            return $"I encountered a timeout while reasoning: {ex.Message}";
        }

        AppendToScratchpad($"LLM Response:\n{llmResponse}\n");

        // Step 2: Parse the chain of thought
        var parsed = ParseChainOfThought(llmResponse);

        // Enforce maxIterations limit on reasoning steps
        if (parsed.ReasoningSteps.Count > maxIterations)
        {
            AppendToScratchpad($"WARNING: LLM generated {parsed.ReasoningSteps.Count} steps, truncating to {maxIterations}.");
            parsed.ReasoningSteps = parsed.ReasoningSteps.Take(maxIterations).ToList();
        }

        // Step 3: Record the reasoning steps
        if (parsed.ReasoningSteps.Count > 0)
        {
            AppendToScratchpad("\n=== Reasoning Steps ===");
            for (int i = 0; i < parsed.ReasoningSteps.Count; i++)
            {
                AppendToScratchpad($"Step {i + 1}: {parsed.ReasoningSteps[i]}");
            }
        }

        // Step 4: Execute any required tool calls
        if (_allowTools && parsed.ToolCalls.Count > 0)
        {
            AppendToScratchpad("\n=== Tool Execution ===");

            foreach (var toolCall in parsed.ToolCalls)
            {
                AppendToScratchpad($"Tool: {toolCall.ToolName}");
                AppendToScratchpad($"Input: {toolCall.Input}");

                string observation = ExecuteTool(toolCall.ToolName, toolCall.Input);
                AppendToScratchpad($"Result: {observation}");

                // Add observation to context for potential refinement
                toolCall.Result = observation;
            }

            // Step 5: If tools were used, ask LLM to incorporate results into final answer
            if (string.IsNullOrWhiteSpace(parsed.FinalAnswer))
            {
                var refinementPrompt = BuildRefinementPrompt(query, parsed);
                try
                {
                    var refinedResponse = await ChatModel.GenerateResponseAsync(refinementPrompt);
                    AppendToScratchpad($"\n=== Refined Answer ===");
                    AppendToScratchpad(refinedResponse);

                    var refinedParsed = ParseChainOfThought(refinedResponse);
                    if (!string.IsNullOrWhiteSpace(refinedParsed.FinalAnswer))
                    {
                        parsed.FinalAnswer = refinedParsed.FinalAnswer;
                    }
                    else
                    {
                        parsed.FinalAnswer = refinedResponse;
                    }
                }
                catch (Exception ex) when (ex is System.Net.Http.HttpRequestException || ex is System.IO.IOException || ex is TaskCanceledException)
                {
                    AppendToScratchpad($"Warning: Could not refine answer: {ex.Message}");
                }
            }
        }

        // Step 6: Return the final answer
        string finalAnswer = parsed.FinalAnswer ?? "I was unable to determine a final answer.";
        AppendToScratchpad($"\n=== Final Answer ===");
        AppendToScratchpad(finalAnswer);

        return finalAnswer;
    }

    /// <summary>
    /// Builds the Chain of Thought prompt that encourages step-by-step reasoning.
    /// </summary>
    /// <param name="query">The user's query.</param>
    /// <param name="maxSteps">Maximum number of reasoning steps.</param>
    /// <returns>A formatted prompt for CoT reasoning.</returns>
    /// <remarks>
    /// For Beginners:
    /// This creates special instructions that tell the language model to "show its work"
    /// like a student solving a problem. The prompt explicitly asks for:
    /// - Breaking down the problem into steps
    /// - Explaining each step clearly
    /// - Showing intermediate calculations or reasoning
    /// - Arriving at a final answer
    ///
    /// Research shows this significantly improves accuracy on complex problems.
    /// </remarks>
    private string BuildChainOfThoughtPrompt(string query, int maxSteps)
    {
        string toolInfo = "";
        if (_allowTools && Tools.Count > 0)
        {
            toolInfo = $@"
You have access to the following tools:
{GetToolDescriptions()}

If you need to use a tool, include it in your response like this:
{{
  ""tool_name"": ""ToolName"",
  ""tool_input"": ""input for the tool""
}}
";
        }

        var prompt = $@"You are a helpful AI assistant that solves problems using step-by-step reasoning.

Query: {query}
{toolInfo}
Please solve this problem by:
1. Breaking it down into clear, logical steps
2. Explaining your reasoning for each step
3. Showing any calculations or deductions
4. Arriving at a final answer

Use the following JSON format:
{{
  ""reasoning_steps"": [
    ""Step 1: [your first reasoning step]"",
    ""Step 2: [your second reasoning step]"",
    ...
  ],
  ""tool_calls"": [
    {{
      ""tool_name"": ""ToolName"",
      ""tool_input"": ""input""
    }}
  ],
  ""final_answer"": ""your final answer to the query""
}}

Important:
- Be explicit about your reasoning in each step
- If you need to use a tool (like Calculator), include it in tool_calls
- Only provide final_answer when you have fully reasoned through the problem
- Maximum {maxSteps} reasoning steps

Think step by step and respond in JSON format:";

        return prompt;
    }

    /// <summary>
    /// Builds a refinement prompt to incorporate tool results into the final answer.
    /// </summary>
    private string BuildRefinementPrompt(string query, ChainOfThoughtResponse parsed)
    {
        var toolResults = new System.Text.StringBuilder();
        foreach (var toolCall in parsed.ToolCalls)
        {
            toolResults.AppendLine($"Tool: {toolCall.ToolName}");
            toolResults.AppendLine($"Input: {toolCall.Input}");
            toolResults.AppendLine($"Result: {toolCall.Result}");
            toolResults.AppendLine();
        }

        return $@"Based on your previous reasoning and the tool results below, provide the final answer.

Original Query: {query}

Your Reasoning Steps:
{string.Join("\n", parsed.ReasoningSteps)}

Tool Results:
{toolResults}

Provide your final answer in this format:
{{
  ""final_answer"": ""your complete answer incorporating the tool results""
}}

Final answer:";
    }

    /// <summary>
    /// Parses the Chain of Thought response from the language model.
    /// </summary>
    private ChainOfThoughtResponse ParseChainOfThought(string response)
    {
        var result = new ChainOfThoughtResponse();

        if (string.IsNullOrWhiteSpace(response))
        {
            return result;
        }

        // Extract JSON from response
        string jsonContent = ExtractJsonFromResponse(response);

        try
        {
            var root = JObject.Parse(jsonContent);

            // Parse reasoning steps
            if (root["reasoning_steps"] is JArray steps)
            {
                foreach (var step in steps)
                {
                    var stepText = step.Value<string>();
                    if (stepText != null && stepText.Trim().Length > 0)
                    {
                        result.ReasoningSteps.Add(stepText);
                    }
                }
            }

            // Parse tool calls
            if (root["tool_calls"] is JArray tools)
            {
                foreach (var tool in tools)
                {
                    string? toolName = tool["tool_name"]?.Value<string>();
                    string? toolInput = tool["tool_input"]?.Value<string>();

                    if (toolName != null && toolName.Trim().Length > 0)
                    {
                        result.ToolCalls.Add(new ToolCall
                        {
                            ToolName = toolName,
                            Input = toolInput ?? ""
                        });
                    }
                }
            }

            // Parse final answer
            if (root["final_answer"] != null)
            {
                var answerText = root["final_answer"]?.Value<string>();
                if (!string.IsNullOrWhiteSpace(answerText))
                {
                    result.FinalAnswer = answerText;
                }
            }
        }
        catch (Newtonsoft.Json.JsonException)
        {
            // Fallback: try to extract with regex
            result = ParseWithRegex(response);
        }

        return result;
    }

    /// <summary>
    /// Extracts JSON content from markdown code blocks.
    /// </summary>
    private string ExtractJsonFromResponse(string response)
    {
        // Remove markdown code block markers if present
        var jsonMatch = Regex.Match(response, @"```(?:json)?\s*(\{[\s\S]*\})\s*```", RegexOptions.Multiline, RegexTimeout);
        if (jsonMatch.Success)
        {
            return jsonMatch.Groups[1].Value;
        }

        // Try to find JSON object without code blocks using brace balancing
        // This handles nested objects correctly
        int startIndex = response.IndexOf('{');
        if (startIndex >= 0)
        {
            int braceCount = 0;
            bool inString = false;
            bool escapeNext = false;

            for (int i = startIndex; i < response.Length; i++)
            {
                char c = response[i];

                if (escapeNext)
                {
                    escapeNext = false;
                    continue;
                }

                if (c == '\\' && inString)
                {
                    escapeNext = true;
                    continue;
                }

                if (c == '"')
                {
                    inString = !inString;
                    continue;
                }

                if (!inString)
                {
                    if (c == '{')
                    {
                        braceCount++;
                    }
                    else if (c == '}')
                    {
                        braceCount--;
                        if (braceCount == 0)
                        {
                            return response.Substring(startIndex, i - startIndex + 1);
                        }
                    }
                }
            }
        }

        return response;
    }

    /// <summary>
    /// Fallback regex parser for non-JSON responses.
    /// </summary>
    private ChainOfThoughtResponse ParseWithRegex(string response)
    {
        var result = new ChainOfThoughtResponse();

        // Try to extract steps
        var stepMatches = Regex.Matches(response, @"(?:Step\s*\d+:|^\d+\.)\s*(.+?)(?=(?:Step\s*\d+:|\d+\.|Final Answer:|$))",
            RegexOptions.IgnoreCase | RegexOptions.Multiline | RegexOptions.Singleline, RegexTimeout);

        foreach (Match match in stepMatches)
        {
            var stepText = match.Groups[1].Value.Trim();
            if (!string.IsNullOrWhiteSpace(stepText))
            {
                result.ReasoningSteps.Add(stepText);
            }
        }

        // Try to extract final answer
        var answerMatch = Regex.Match(response, @"Final\s*Answer:\s*(.+?)$",
            RegexOptions.IgnoreCase | RegexOptions.Multiline | RegexOptions.Singleline, RegexTimeout);
        if (answerMatch.Success)
        {
            result.FinalAnswer = answerMatch.Groups[1].Value.Trim();
        }
        else if (result.ReasoningSteps.Count == 0)
        {
            // If we can't find structured steps, treat entire response as answer
            result.FinalAnswer = response.Trim();
        }

        return result;
    }

    /// <summary>
    /// Executes a tool with error handling.
    /// </summary>
    private string ExecuteTool(string toolName, string input)
    {
        var tool = FindTool(toolName);

        if (tool == null)
        {
            return $"Error: Tool '{toolName}' not found. Available tools: {string.Join(", ", Tools.Select(t => t.Name))}";
        }

        try
        {
            return tool.Execute(input);
        }
        catch (Exception ex)
        {
            // Rethrow critical exceptions
            if (ex is OutOfMemoryException || ex is StackOverflowException || ex is System.Threading.ThreadAbortException)
                throw;

            return $"Error executing tool '{toolName}': {ex.Message}";
        }
    }

    /// <summary>
    /// Represents a parsed Chain of Thought response.
    /// </summary>
    private class ChainOfThoughtResponse
    {
        public List<string> ReasoningSteps { get; set; } = new List<string>();
        public List<ToolCall> ToolCalls { get; set; } = new List<ToolCall>();
        public string? FinalAnswer { get; set; }
    }

    /// <summary>
    /// Represents a tool call request.
    /// </summary>
    private class ToolCall
    {
        public string ToolName { get; set; } = "";
        public string Input { get; set; } = "";
        public string? Result { get; set; }
    }
}
