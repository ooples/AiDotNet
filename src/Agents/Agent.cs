using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Agents;

/// <summary>
/// Implements a ReAct (Reasoning + Acting) agent that uses a language model and tools to solve problems.
/// The agent alternates between thinking about what to do and taking actions with tools.
/// </summary>
/// <typeparam name="T">The numeric type used for model parameters and operations (e.g., double, float).</typeparam>
/// <remarks>
/// For Beginners:
/// This is a concrete implementation of an AI agent that follows the ReAct pattern. ReAct stands for
/// "Reasoning and Acting" - the agent alternates between:
///
/// 1. **Reasoning (Thought)**: Thinking about what needs to be done next
/// 2. **Acting (Action)**: Using a tool to perform an operation
/// 3. **Observing (Observation)**: Seeing the result of the action
///
/// The agent repeats this cycle until it has enough information to answer the query or reaches
/// the maximum number of iterations.
///
/// Example of how the agent works:
/// <code>
/// User Query: "What is 25 * 4 + 10?"
///
/// Iteration 1:
/// Thought: I need to calculate 25 * 4 first
/// Action: Calculator
/// Action Input: 25 * 4
/// Observation: 100
///
/// Iteration 2:
/// Thought: Now I need to add 10 to 100
/// Action: Calculator
/// Action Input: 100 + 10
/// Observation: 110
///
/// Iteration 3:
/// Thought: I have the final answer
/// Final Answer: 110
/// </code>
///
/// This implementation uses JSON formatting for the LLM responses to make parsing easier and more reliable.
/// </remarks>
public class Agent<T> : AgentBase<T>
{

    /// <summary>
    /// Initializes a new instance of the <see cref="Agent{T}"/> class.
    /// </summary>
    /// <param name="chatModel">The chat model used for reasoning and decision-making.</param>
    /// <param name="tools">The collection of tools available to the agent.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="chatModel"/> is null.</exception>
    /// <remarks>
    /// For Beginners:
    /// This creates a new ReAct agent with the specified language model and tools.
    /// The agent will use the language model to think through problems and the tools
    /// to perform actions.
    /// </remarks>
    public Agent(IChatModel<T> chatModel, IEnumerable<ITool>? tools = null)
        : base(chatModel, tools)
    {
    }

    /// <inheritdoc/>
    public override async Task<string> RunAsync(string query, int maxIterations = 5)
    {
        ValidateMaxIterations(maxIterations);

        if (string.IsNullOrWhiteSpace(query))
        {
            throw new ArgumentException("Query cannot be null or whitespace.", nameof(query));
        }

        ClearScratchpad();
        AppendToScratchpad($"Query: {query}\n");

        for (int iteration = 1; iteration <= maxIterations; iteration++)
        {
            AppendToScratchpad($"=== Iteration {iteration} ===");

            // Build the prompt for the LLM
            string prompt = BuildPrompt(query, iteration, maxIterations);

            // Get response from the language model
            string llmResponse;
            try
            {
                llmResponse = await ChatModel.GenerateResponseAsync(prompt);
            }
            catch (System.Net.Http.HttpRequestException ex)
            {
                string errorMsg = $"Error communicating with chat model: {ex.Message}";
                AppendToScratchpad($"ERROR: {errorMsg}");
                return $"I encountered an error while thinking: {ex.Message}";
            }
            catch (System.IO.IOException ex)
            {
                string errorMsg = $"IO error communicating with chat model: {ex.Message}";
                AppendToScratchpad($"ERROR: {errorMsg}");
                return $"I encountered an IO error while thinking: {ex.Message}";
            }
            catch (TaskCanceledException ex)
            {
                string errorMsg = $"Timeout communicating with chat model: {ex.Message}";
                AppendToScratchpad($"ERROR: {errorMsg}");
                return $"I encountered a timeout while thinking: {ex.Message}";
            }

            AppendToScratchpad($"LLM Response: {llmResponse}\n");

            // Parse the response
            var parsedResponse = ParseLLMResponse(llmResponse);

            if (parsedResponse.HasFinalAnswer)
            {
                string finalAnswer = parsedResponse.FinalAnswer ?? "No answer provided.";
                AppendToScratchpad($"\n=== Final Answer ===");
                AppendToScratchpad(finalAnswer);
                return finalAnswer;
            }

            // Record the thought
            if (!string.IsNullOrWhiteSpace(parsedResponse.Thought))
            {
                AppendToScratchpad($"Thought: {parsedResponse.Thought}");
            }

            // Execute the action if specified
            // Explicit null check for net462 compatibility with nullable reference types
            if (parsedResponse.Action is not null && !string.IsNullOrWhiteSpace(parsedResponse.Action))
            {
                string action = parsedResponse.Action;
                AppendToScratchpad($"Action: {action}");
                AppendToScratchpad($"Action Input: {parsedResponse.ActionInput ?? ""}");

                string observation = ExecuteTool(action, parsedResponse.ActionInput ?? "");
                AppendToScratchpad($"Observation: {observation}\n");
            }
            else if (!parsedResponse.HasFinalAnswer)
            {
                AppendToScratchpad("Warning: No action specified and no final answer provided.\n");
            }
        }

        // Reached max iterations without final answer
        string fallbackMsg = $"I reached the maximum number of iterations ({maxIterations}) without finding a complete answer. " +
                           "Here's what I learned:\n" + Scratchpad;
        return fallbackMsg;
    }

    /// <summary>
    /// Builds the prompt to send to the language model.
    /// </summary>
    /// <param name="query">The user's original query.</param>
    /// <param name="currentIteration">The current iteration number.</param>
    /// <param name="maxIterations">The maximum number of iterations allowed.</param>
    /// <returns>A formatted prompt string for the language model.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method creates the instructions we send to the language model. The prompt tells the model:
    /// - What the user wants (the query)
    /// - What tools are available
    /// - How to format its response (using JSON)
    /// - What it has learned so far (from the scratchpad)
    ///
    /// Think of it like writing detailed instructions for someone: you explain the task, list the
    /// available resources, show previous work, and specify the expected format for their response.
    /// </remarks>
    private string BuildPrompt(string query, int currentIteration, int maxIterations)
    {
        var prompt = $@"You are a helpful AI agent that can use tools to solve problems. Your goal is to answer the following query:

Query: {query}

{GetToolDescriptions()}

You should follow this format in your response (use valid JSON):

{{
  ""thought"": ""your reasoning about what to do next"",
  ""action"": ""the name of the tool to use (or empty if you have the final answer)"",
  ""action_input"": ""the input to provide to the tool (or empty if you have the final answer)"",
  ""final_answer"": ""the final answer to the query (only fill this when you have the complete answer)""
}}

Important guidelines:
1. Think step by step about what you need to do
2. Use tools one at a time - you can only use one tool per iteration
3. When you use a tool, leave ""final_answer"" empty
4. When you have the complete answer, fill in ""final_answer"" and leave ""action"" and ""action_input"" empty
5. Make sure your response is valid JSON

This is iteration {currentIteration} of {maxIterations}.

Previous reasoning and observations:
{Scratchpad}

Respond now with your next step in JSON format:";

        return prompt;
    }

    /// <summary>
    /// Parses the language model's response to extract thought, action, and answer information.
    /// </summary>
    /// <param name="llmResponse">The raw response from the language model.</param>
    /// <returns>A parsed response object containing the extracted information.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method reads the language model's response and extracts the important parts:
    /// the thought (what it's thinking), the action (what tool to use), the action input
    /// (what to send to the tool), and the final answer (if it has one).
    ///
    /// The method tries to parse JSON first (the expected format), but also has fallback
    /// logic to handle cases where the LLM doesn't follow the exact format. This makes
    /// the agent more robust and able to work with different language models.
    /// </remarks>
    private ParsedResponse ParseLLMResponse(string llmResponse)
    {
        var response = new ParsedResponse();

        if (string.IsNullOrWhiteSpace(llmResponse))
        {
            return response;
        }

        // Try to extract JSON from the response (it might be wrapped in markdown code blocks)
        string jsonContent = ExtractJsonFromResponse(llmResponse);

        try
        {
            var root = JObject.Parse(jsonContent);

            if (root["thought"] != null)
            {
                response.Thought = root["thought"]?.Value<string>();
            }

            if (root["action"] != null)
            {
                response.Action = root["action"]?.Value<string>();
            }

            if (root["action_input"] != null)
            {
                response.ActionInput = root["action_input"]?.Value<string>();
            }

            if (root["final_answer"] != null)
            {
                var finalAnswerStr = root["final_answer"]?.Value<string>();
                if (!string.IsNullOrWhiteSpace(finalAnswerStr))
                {
                    response.FinalAnswer = finalAnswerStr;
                    response.HasFinalAnswer = true;
                }
            }
        }
        catch (Newtonsoft.Json.JsonException)
        {
            // Fallback: try to extract information using regex if JSON parsing fails
            response = ParseWithRegex(llmResponse);
        }

        return response;
    }

    /// <summary>
    /// Extracts JSON content from a response that might be wrapped in markdown code blocks.
    /// </summary>
    /// <param name="response">The raw response string.</param>
    /// <returns>The extracted JSON content.</returns>
    /// <remarks>
    /// For Beginners:
    /// Sometimes language models wrap JSON in markdown code blocks like:
    /// ```json
    /// {"key": "value"}
    /// ```
    ///
    /// This method removes those markdown wrappers to get just the JSON content,
    /// making it easier to parse.
    /// </remarks>
    private string ExtractJsonFromResponse(string response)
    {
        // Remove markdown code block markers if present
        var jsonMatch = RegexHelper.Match(response, @"```(?:json)?\s*(\{[\s\S]*?\})\s*```", RegexOptions.Multiline);
        if (jsonMatch.Success)
        {
            return jsonMatch.Groups[1].Value;
        }

        // Try to find JSON object without code blocks (non-greedy)
        var jsonObjectMatch = RegexHelper.Match(response, @"\{[\s\S]*?\}", RegexOptions.Multiline);
        if (jsonObjectMatch.Success)
        {
            return jsonObjectMatch.Value;
        }

        return response;
    }

    /// <summary>
    /// Fallback parser that uses regular expressions when JSON parsing fails.
    /// </summary>
    /// <param name="response">The raw response string.</param>
    /// <returns>A parsed response object.</returns>
    /// <remarks>
    /// For Beginners:
    /// This is a backup plan. If the language model doesn't provide valid JSON (maybe it forgot
    /// or used a different format), this method tries to extract the information using pattern
    /// matching (regular expressions).
    ///
    /// It looks for patterns like:
    /// - "Thought: ..." or "thought: ..."
    /// - "Action: ..." or "action: ..."
    /// - "Final Answer: ..." or "final_answer: ..."
    ///
    /// This makes the agent more forgiving and able to work even when the LLM doesn't
    /// follow instructions perfectly.
    /// </remarks>
    private ParsedResponse ParseWithRegex(string response)
    {
        var parsed = new ParsedResponse();

        // Try to extract thought
        var thoughtMatch = RegexHelper.Match(response, @"thought:\s*(.+?)(?=\n|$)", RegexOptions.IgnoreCase | RegexOptions.Multiline);
        if (thoughtMatch.Success)
        {
            parsed.Thought = thoughtMatch.Groups[1].Value.Trim();
        }

        // Try to extract action
        var actionMatch = RegexHelper.Match(response, @"action:\s*(.+?)(?=\n|$)", RegexOptions.IgnoreCase | RegexOptions.Multiline);
        if (actionMatch.Success)
        {
            parsed.Action = actionMatch.Groups[1].Value.Trim();
        }

        // Try to extract action input
        var actionInputMatch = RegexHelper.Match(response, @"action[_ ]input:\s*(.+?)(?=\n|$)", RegexOptions.IgnoreCase | RegexOptions.Multiline);
        if (actionInputMatch.Success)
        {
            parsed.ActionInput = actionInputMatch.Groups[1].Value.Trim();
        }

        // Try to extract final answer
        var finalAnswerMatch = RegexHelper.Match(response, @"final[_ ]answer:\s*(.+?)(?=\n|$)", RegexOptions.IgnoreCase | RegexOptions.Multiline);
        if (finalAnswerMatch.Success)
        {
            parsed.FinalAnswer = finalAnswerMatch.Groups[1].Value.Trim();
            parsed.HasFinalAnswer = true;
        }

        return parsed;
    }

    /// <summary>
    /// Executes a tool with the given input.
    /// </summary>
    /// <param name="toolName">The name of the tool to execute.</param>
    /// <param name="input">The input to provide to the tool.</param>
    /// <returns>The result of executing the tool, or an error message if the tool is not found or execution fails.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method is like reaching into a toolbox, grabbing the requested tool, and using it.
    /// If the tool doesn't exist or something goes wrong, it returns an error message instead
    /// of crashing, allowing the agent to potentially recover and try something else.
    /// </remarks>
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
    /// Represents a parsed response from the language model.
    /// </summary>
    private class ParsedResponse
    {
        public string? Thought { get; set; }
        public string? Action { get; set; }
        public string? ActionInput { get; set; }
        public string? FinalAnswer { get; set; }
        public bool HasFinalAnswer { get; set; }
    }
}



