using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Agents;

/// <summary>
/// Implements a Plan-and-Execute agent that decomposes complex tasks into a plan of subtasks,
/// then executes each subtask sequentially. This approach is particularly effective for
/// multi-step problems that require careful coordination of multiple operations.
/// </summary>
/// <typeparam name="T">The numeric type used for model parameters and operations (e.g., double, float).</typeparam>
/// <remarks>
/// For Beginners:
/// Plan-and-Execute is an agent pattern that works like a project manager:
/// 1. **Plan**: First, create a detailed plan of what needs to be done
/// 2. **Execute**: Then, execute each step of the plan in order
/// 3. **Revise**: Update the plan if new information changes what's needed
///
/// **When to use PlanAndExecuteAgent vs other agent types:**
///
/// Use PlanAndExecuteAgent when:
/// - Task has multiple clear steps that depend on each other
/// - You want to see the full plan before execution starts
/// - Task requires coordination across multiple tools
/// - Order of operations matters (step 2 depends on step 1, etc.)
///
/// Use Agent (ReAct) when:
/// - Task requires dynamic decision-making based on observations
/// - You don't know all the steps upfront
/// - Each action depends heavily on previous results
///
/// Use ChainOfThoughtAgent when:
/// - Problem is primarily logical/mathematical reasoning
/// - Minimal tool use required
/// - Focus is on showing reasoning steps
///
/// **Example workflow:**
/// <code>
/// User Query: "Find the current weather in the capital of France and convert the temperature to Fahrenheit"
///
/// === PLANNING PHASE ===
/// Plan:
/// 1. Search for "capital of France" to find the city name
/// 2. Use weather tool to get current temperature in that city
/// 3. Use calculator to convert Celsius to Fahrenheit
/// 4. Provide the final answer
///
/// === EXECUTION PHASE ===
/// Step 1: Search("capital of France")
/// Result: Paris
///
/// Step 2: Weather("Paris")
/// Result: 18°C, Partly Cloudy
///
/// Step 3: Calculator("(18 * 9/5) + 32")
/// Result: 64.4°F
///
/// Step 4: Formulate answer
/// Final Answer: The current weather in Paris (capital of France) is 18°C (64.4°F), Partly Cloudy
/// </code>
///
/// **Benefits:**
/// - Clear structure: You can see the entire plan upfront
/// - Better for complex tasks: Decomposes complexity into manageable steps
/// - Predictable: Each step builds on previous results
/// - Easier to debug: Can see exactly which step failed
///
/// **Research background:**
/// Based on techniques like "Least-to-Most Prompting" and "Plan-and-Solve" prompting,
/// which improve performance on complex multi-step reasoning tasks by explicit planning.
/// </remarks>
public class PlanAndExecuteAgent<T> : AgentBase<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly bool _allowPlanRevision;

    /// <summary>
    /// Initializes a new instance of the <see cref="PlanAndExecuteAgent{T}"/> class.
    /// </summary>
    /// <param name="chatModel">The chat model used for planning and execution.</param>
    /// <param name="tools">The collection of tools available to the agent.</param>
    /// <param name="allowPlanRevision">Whether to allow plan revision based on execution results (default: true).</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="chatModel"/> is null.</exception>
    /// <remarks>
    /// For Beginners:
    /// Creates a Plan-and-Execute agent that creates a plan first, then executes it.
    ///
    /// **Parameters:**
    /// - chatModel: The language model for planning and reasoning
    /// - tools: Tools the agent can use during execution
    /// - allowPlanRevision: Whether the agent can update its plan during execution
    ///
    /// **Plan Revision:**
    /// - When true (default): Agent can adapt plan if unexpected results occur
    ///   Example: If step 2 fails, agent can create a new plan to work around it
    ///
    /// - When false: Agent sticks to original plan no matter what
    ///   More predictable, but less flexible
    ///
    /// Example:
    /// <code>
    /// var tools = new ITool[] { new SearchTool(), new Calculator(), new WeatherTool() };
    ///
    /// // Flexible agent that can revise plan
    /// var flexibleAgent = new PlanAndExecuteAgent&lt;double&gt;(chatModel, tools);
    ///
    /// // Strict agent that follows original plan
    /// var strictAgent = new PlanAndExecuteAgent&lt;double&gt;(chatModel, tools, allowPlanRevision: false);
    /// </code>
    /// </remarks>
    public PlanAndExecuteAgent(IChatModel<T> chatModel, IEnumerable<ITool>? tools = null, bool allowPlanRevision = true)
        : base(chatModel, tools)
    {
        _allowPlanRevision = allowPlanRevision;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Note: For PlanAndExecuteAgent, the maxIterations parameter limits the number of plan
    /// revisions (not the number of steps in the plan). The agent will execute all steps in
    /// the plan, but if errors occur and plan revision is enabled, it can only revise the plan
    /// up to maxIterations times. This prevents infinite revision loops while allowing plans
    /// with legitimately many steps to complete.
    /// </remarks>
    public override async Task<string> RunAsync(string query, int maxIterations = 10)
    {
        ValidateMaxIterations(maxIterations);

        if (string.IsNullOrWhiteSpace(query))
        {
            throw new ArgumentException("Query cannot be null or whitespace.", nameof(query));
        }

        ClearScratchpad();
        AppendToScratchpad($"Query: {query}\n");

        // Phase 1: Generate the plan
        AppendToScratchpad("=== PLANNING PHASE ===");
        var plan = await GeneratePlanAsync(query);

        if (plan.Steps.Count == 0)
        {
            return "I was unable to create a plan to solve this query.";
        }

        AppendToScratchpad("Plan created:");
        for (int i = 0; i < plan.Steps.Count; i++)
        {
            AppendToScratchpad($"  {i + 1}. {plan.Steps[i].Description}");
        }
        AppendToScratchpad("");

        // Phase 2: Execute the plan
        AppendToScratchpad("=== EXECUTION PHASE ===");

        int revisionCount = 0; // Track number of plan revisions to enforce maxIterations limit

        for (int stepIndex = 0; stepIndex < plan.Steps.Count; stepIndex++)
        {
            var step = plan.Steps[stepIndex];
            AppendToScratchpad($"Step {stepIndex + 1}/{plan.Steps.Count}: {step.Description}");

            try
            {
                // Execute the step
                string result = await ExecuteStepAsync(step, query);
                step.Result = result;
                step.IsCompleted = true;

                AppendToScratchpad($"Result: {result}");
                AppendToScratchpad("");

                // Check if this was the final step
                if (step.IsFinalStep)
                {
                    AppendToScratchpad("=== PLAN COMPLETED ===");
                    return result;
                }
            }
            catch (Exception ex) when (ex is System.Net.Http.HttpRequestException || ex is System.IO.IOException || ex is TaskCanceledException)
            {
                step.Result = $"Error: {ex.Message}";
                AppendToScratchpad($"Error executing step: {ex.Message}");

                // If plan revision is allowed, try to create a new plan
                if (_allowPlanRevision && revisionCount < maxIterations)
                {
                    AppendToScratchpad($"Attempting to revise plan (revision {revisionCount + 1}/{maxIterations})...");
                    var revisedPlan = await RevisePlanAsync(query, plan, stepIndex);

                    if (revisedPlan.Steps.Count > 0)
                    {
                        plan = revisedPlan;
                        revisionCount++; // Increment revision counter
                        AppendToScratchpad("Plan revised:");
                        for (int i = 0; i < plan.Steps.Count; i++)
                        {
                            AppendToScratchpad($"  {i + 1}. {plan.Steps[i].Description}");
                        }
                        AppendToScratchpad("");

                        // Restart from the beginning of the new plan
                        stepIndex = -1; // Will be incremented to 0 at the start of next iteration
                        continue;
                    }
                    else
                    {
                        return $"I encountered an error at step {stepIndex + 1} and was unable to revise the plan: {ex.Message}";
                    }
                }
                else if (_allowPlanRevision && revisionCount >= maxIterations)
                {
                    AppendToScratchpad($"Maximum revisions ({maxIterations}) reached. Cannot revise plan further.");
                    return $"I encountered an error at step {stepIndex + 1} and reached the maximum number of plan revisions ({maxIterations}): {ex.Message}";
                }
                else
                {
                    return $"I encountered an error at step {stepIndex + 1}: {ex.Message}";
                }
            }
        }

        // If we completed all steps but didn't hit a final step, synthesize an answer
        AppendToScratchpad("=== SYNTHESIZING FINAL ANSWER ===");
        return await SynthesizeFinalAnswerAsync(query, plan);
    }

    /// <summary>
    /// Generates a plan to solve the query.
    /// </summary>
    /// <param name="query">The user's query.</param>
    /// <returns>A plan consisting of ordered steps.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method asks the language model to think about the task and create
    /// a step-by-step plan to solve it, like a project plan or recipe.
    ///
    /// The plan breaks down "What needs to be done?" into specific actions.
    /// </remarks>
    private async Task<Plan> GeneratePlanAsync(string query)
    {
        var prompt = $@"You are a helpful AI assistant that creates detailed plans to solve queries.

Query: {query}

Available tools:
{GetToolDescriptions()}

Create a step-by-step plan to solve this query. Each step should be a specific action.
If a step requires a tool, specify which tool to use.

Respond in this JSON format:
{{
  ""steps"": [
    {{
      ""description"": ""what this step does"",
      ""tool"": ""tool name to use (or empty if no tool needed)"",
      ""input"": ""input for the tool (or empty)"",
      ""is_final_step"": false
    }},
    ...
    {{
      ""description"": ""provide final answer based on all results"",
      ""tool"": """",
      ""input"": """",
      ""is_final_step"": true
    }}
  ]
}}

Important:
- Break down the task into clear, sequential steps
- Each step should build on previous results
- Last step should always be synthesizing/providing the final answer
- Mark the final answer step with is_final_step: true

Create the plan now:";

        try
        {
            var response = await ChatModel.GenerateResponseAsync(prompt);
            return ParsePlan(response);
        }
        catch (Exception ex) when (ex is System.Net.Http.HttpRequestException || ex is System.IO.IOException || ex is TaskCanceledException)
        {
            AppendToScratchpad($"Error generating plan: {ex.Message}");
            return new Plan();
        }
    }

    /// <summary>
    /// Revises the plan based on execution results so far.
    /// </summary>
    private async Task<Plan> RevisePlanAsync(string query, Plan currentPlan, int failedStepIndex)
    {
        var executedSteps = new System.Text.StringBuilder();
        for (int i = 0; i <= failedStepIndex && i < currentPlan.Steps.Count; i++)
        {
            var step = currentPlan.Steps[i];
            executedSteps.AppendLine($"Step {i + 1}: {step.Description}");
            executedSteps.AppendLine($"Result: {step.Result ?? "Not completed"}");
        }

        var prompt = $@"You are a helpful AI assistant that revises plans when problems occur.

Original Query: {query}

Available tools:
{GetToolDescriptions()}

Original plan execution so far:
{executedSteps}

The plan encountered a problem at step {failedStepIndex + 1}. Create a revised plan to complete the query,
taking into account what has been accomplished and what went wrong.

Respond in the same JSON format as before with a new list of steps:
{{
  ""steps"": [
    {{
      ""description"": ""what this step does"",
      ""tool"": ""tool name"",
      ""input"": ""input"",
      ""is_final_step"": false
    }},
    ...
  ]
}}

Create the revised plan now:";

        try
        {
            var response = await ChatModel.GenerateResponseAsync(prompt);
            return ParsePlan(response);
        }
        catch (Exception ex) when (ex is System.Net.Http.HttpRequestException || ex is System.IO.IOException || ex is TaskCanceledException)
        {
            AppendToScratchpad($"Error revising plan: {ex.Message}");
            return new Plan();
        }
    }

    /// <summary>
    /// Executes a single step of the plan.
    /// </summary>
    private async Task<string> ExecuteStepAsync(PlanStep step, string originalQuery)
    {
        // If this is a final synthesis step or no tool is specified, ask LLM to synthesize
        if (step.IsFinalStep || string.IsNullOrWhiteSpace(step.Tool))
        {
            // Get all previous results for context
            var context = new System.Text.StringBuilder();
            context.AppendLine($"Original query: {originalQuery}");
            context.AppendLine("\nPrevious step results:");
            context.AppendLine(Scratchpad);

            var prompt = $@"{context}

Based on the information above, {step.Description}

Provide a clear, concise answer:";

            return await ChatModel.GenerateResponseAsync(prompt);
        }

        // Otherwise, execute the tool
        var tool = FindTool(step.Tool);
        if (tool == null)
        {
            return $"Error: Tool '{step.Tool}' not found. Available: {string.Join(", ", Tools.Select(t => t.Name))}";
        }

        // If input is not specified, ask LLM to determine it based on context
        string toolInput = step.Input;
        if (string.IsNullOrWhiteSpace(toolInput))
        {
            var inputPrompt = $@"Based on this context:
{Scratchpad}

What input should be provided to the {step.Tool} tool to accomplish this step: {step.Description}

Provide only the input value, no explanation:";

            try
            {
                toolInput = await ChatModel.GenerateResponseAsync(inputPrompt);
                toolInput = toolInput.Trim();
            }
            catch (Exception ex) when (ex is System.Net.Http.HttpRequestException || ex is System.IO.IOException || ex is TaskCanceledException)
            {
                return $"Error determining tool input: {ex.Message}";
            }
        }

        // Execute the tool
        try
        {
            return tool.Execute(toolInput);
        }
        catch (Exception ex) when (ex is System.Net.Http.HttpRequestException || ex is System.IO.IOException || ex is TaskCanceledException)
        {
            // Transient errors - rethrow to trigger plan revision if enabled
            throw;
        }
        catch (Exception ex)
        {
            // Rethrow critical exceptions
            if (ex is OutOfMemoryException || ex is StackOverflowException || ex is System.Threading.ThreadAbortException)
                throw;

            return $"Error executing {step.Tool}: {ex.Message}";
        }
    }

    /// <summary>
    /// Synthesizes a final answer from all the plan execution results.
    /// </summary>
    private async Task<string> SynthesizeFinalAnswerAsync(string query, Plan plan)
    {
        var results = new System.Text.StringBuilder();
        results.AppendLine($"Query: {query}");
        results.AppendLine("\nExecution results:");

        for (int i = 0; i < plan.Steps.Count; i++)
        {
            var step = plan.Steps[i];
            results.AppendLine($"\nStep {i + 1}: {step.Description}");
            if (step.IsCompleted)
            {
                results.AppendLine($"Result: {step.Result}");
            }
            else
            {
                results.AppendLine("Result: Not completed");
            }
        }

        var prompt = $@"{results}

Based on all the step results above, provide a comprehensive final answer to the original query.

Final answer:";

        try
        {
            return await ChatModel.GenerateResponseAsync(prompt);
        }
        catch (Exception ex) when (ex is System.Net.Http.HttpRequestException || ex is System.IO.IOException || ex is TaskCanceledException)
        {
            return $"I completed the plan steps but encountered an error synthesizing the final answer: {ex.Message}";
        }
    }

    /// <summary>
    /// Parses a plan from the LLM response.
    /// </summary>
    private Plan ParsePlan(string response)
    {
        var plan = new Plan();

        if (string.IsNullOrWhiteSpace(response))
        {
            return plan;
        }

        // Extract JSON from response
        string jsonContent = ExtractJsonFromResponse(response);

        try
        {
            var root = JObject.Parse(jsonContent);

            if (root["steps"] is JArray stepsArray)
            {
                foreach (var stepElement in stepsArray)
                {
                    var step = new PlanStep
                    {
                        Description = stepElement["description"]?.Value<string>() ?? "",
                        Tool = stepElement["tool"]?.Value<string>() ?? "",
                        Input = stepElement["input"]?.Value<string>() ?? "",
                        IsFinalStep = stepElement["is_final_step"]?.Value<bool>() ?? false
                    };

                    if (!string.IsNullOrWhiteSpace(step.Description))
                    {
                        plan.Steps.Add(step);
                    }
                }
            }
        }
        catch (Newtonsoft.Json.JsonException)
        {
            // Fallback: try to parse with regex
            plan = ParsePlanWithRegex(response);
        }

        return plan;
    }

    /// <summary>
    /// Extracts JSON from markdown code blocks.
    /// </summary>
    private string ExtractJsonFromResponse(string response)
    {
        var jsonMatch = Regex.Match(response, @"```(?:json)?\s*(\{[\s\S]*?\})\s*```", RegexOptions.Multiline, RegexTimeout);
        if (jsonMatch.Success)
        {
            return jsonMatch.Groups[1].Value;
        }

        var jsonObjectMatch = Regex.Match(response, @"\{[\s\S]*?\}", RegexOptions.None, RegexTimeout);
        if (jsonObjectMatch.Success)
        {
            return jsonObjectMatch.Value;
        }

        return response;
    }

    /// <summary>
    /// Fallback regex parser for plans.
    /// </summary>
    private Plan ParsePlanWithRegex(string response)
    {
        var plan = new Plan();

        // Try to find numbered steps
        var stepMatches = Regex.Matches(response, @"(?:Step\s*)?(\d+)[.:\)]\s*(.+?)(?=(?:Step\s*)?\d+[.:\)]|$)",
            RegexOptions.IgnoreCase | RegexOptions.Multiline | RegexOptions.Singleline, RegexTimeout);

        foreach (Match match in stepMatches)
        {
            var stepText = match.Groups[2].Value.Trim();
            if (!string.IsNullOrWhiteSpace(stepText))
            {
                plan.Steps.Add(new PlanStep
                {
                    Description = stepText,
                    Tool = "",
                    Input = ""
                });
            }
        }

        // Mark last step as final
        if (plan.Steps.Count > 0)
        {
            plan.Steps[plan.Steps.Count - 1].IsFinalStep = true;
        }

        return plan;
    }

    /// <summary>
    /// Represents a plan consisting of ordered steps.
    /// </summary>
    private class Plan
    {
        public List<PlanStep> Steps { get; set; } = new List<PlanStep>();
    }

    /// <summary>
    /// Represents a single step in a plan.
    /// </summary>
    private class PlanStep
    {
        public string Description { get; set; } = "";
        public string Tool { get; set; } = "";
        public string Input { get; set; } = "";
        public bool IsFinalStep { get; set; }
        public bool IsCompleted { get; set; }
        public string? Result { get; set; }
    }
}
