using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.Reasoning.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Reasoning.Components;

/// <summary>
/// Generates alternative thoughts or reasoning steps from a current state.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The thought generator creates new ideas or next steps from where you are now.
/// Think of it like brainstorming - given your current position, what are different directions you could explore?
///
/// **Example:**
/// Current thought: "How can we reduce carbon emissions?"
/// Generated alternatives:
/// 1. "Increase adoption of renewable energy sources like solar and wind"
/// 2. "Improve energy efficiency in buildings and manufacturing"
/// 3. "Transition transportation to electric vehicles"
/// 4. "Implement carbon capture and storage technologies"
/// 5. "Reform agricultural practices to reduce methane"
///
/// The generator uses the language model to create these alternatives, with temperature controlling diversity.
/// </para>
/// </remarks>
internal class ThoughtGenerator<T> : IThoughtGenerator<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly IChatModel<T> _chatModel;

    /// <summary>
    /// Initializes a new instance of the <see cref="ThoughtGenerator{T}"/> class.
    /// </summary>
    /// <param name="chatModel">The chat model used to generate thoughts.</param>
    public ThoughtGenerator(IChatModel<T> chatModel)
    {
        _chatModel = chatModel ?? throw new ArgumentNullException(nameof(chatModel));
    }

    /// <inheritdoc/>
    public async Task<List<AiDotNet.Reasoning.Models.ThoughtNode<T>>> GenerateThoughtsAsync(
        AiDotNet.Reasoning.Models.ThoughtNode<T> currentNode,
        int numThoughts,
        ReasoningConfig config,
        CancellationToken cancellationToken = default)
    {
        if (currentNode == null)
            throw new ArgumentNullException(nameof(currentNode));

        if (numThoughts < 1)
            throw new ArgumentException("Must generate at least 1 thought", nameof(numThoughts));

        // Build the path to this node for context
        var pathFromRoot = currentNode.GetPathFromRoot();
        string context = string.Join(" → ", pathFromRoot);

        // Create prompt for generating alternative next steps
        string prompt = BuildGenerationPrompt(currentNode, numThoughts, context, config);

        // Generate thoughts from LLM
        string response = await _chatModel.GenerateResponseAsync(prompt);

        // Parse the response into thought nodes
        var thoughts = ParseThoughts(response, numThoughts);

        // Create nodes for each thought
        var nodes = new List<AiDotNet.Reasoning.Models.ThoughtNode<T>>();
        foreach (var thought in thoughts)
        {
            if (string.IsNullOrWhiteSpace(thought))
                continue;

            var node = new AiDotNet.Reasoning.Models.ThoughtNode<T>
            {
                Thought = thought.Trim(),
                Parent = currentNode,
                Depth = currentNode.Depth + 1,
                // EvaluationScore initialized by constructor, will be set by evaluator
                IsVisited = false
            };

            nodes.Add(node);
        }

        return nodes;
    }

    /// <summary>
    /// Builds the prompt for generating thoughts.
    /// </summary>
    private string BuildGenerationPrompt(
        AiDotNet.Reasoning.Models.ThoughtNode<T> currentNode,
        int numThoughts,
        string context,
        ReasoningConfig config)
    {
        string terminalNote = currentNode.IsTerminal
            ? ""
            : "\nNote: This thought does not yet represent a complete solution. Generate next steps toward solving the problem.";

        return $@"You are helping solve a problem by exploring different reasoning paths.

Current reasoning path:
{context}

Current position: ""{currentNode.Thought}""
{terminalNote}

Task: Generate {numThoughts} diverse alternative next steps or thoughts from this position.
Each thought should:
- Be a distinct approach or direction
- Build logically on the current path
- Move toward solving the original problem
- Be specific and actionable

Respond in JSON format:
{{
  ""thoughts"": [
    ""First alternative thought or next step"",
    ""Second alternative thought or next step"",
    ...
  ]
}}

Generate exactly {numThoughts} diverse thoughts:";
    }

    /// <summary>
    /// Parses thoughts from the LLM response.
    /// </summary>
    private List<string> ParseThoughts(string response, int expectedCount)
    {
        var thoughts = new List<string>();

        try
        {
            // Try JSON parsing
            string jsonContent = ExtractJsonFromResponse(response);
            var root = JObject.Parse(jsonContent);

            if (root["thoughts"] is JArray thoughtsArray)
            {
                foreach (var thought in thoughtsArray)
                {
                    string thoughtText = thought.Value<string>() ?? "";
                    if (!string.IsNullOrWhiteSpace(thoughtText))
                    {
                        thoughts.Add(thoughtText.Trim());
                    }
                }
            }
        }
        catch (JsonException)
        {
            // Fallback to line-based parsing
            thoughts = ParseThoughtsFromLines(response, expectedCount);
        }

        return thoughts;
    }

    /// <summary>
    /// Fallback parser for non-JSON responses.
    /// </summary>
    private List<string> ParseThoughtsFromLines(string response, int expectedCount)
    {
        var thoughts = new List<string>();

        // Split by numbered list patterns
        var lines = response.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (var line in lines)
        {
            // Match patterns like "1.", "1)", "•", "-", etc.
            var match = Regex.Match(line, @"^[\s]*(?:\d+[\.\)]\s*|[•\-\*]\s*)(.+)$", RegexOptions.None, RegexTimeout);
            if (match.Success)
            {
                string thought = match.Groups[1].Value.Trim();
                if (thought.Length > 10) // Minimum length filter
                {
                    thoughts.Add(thought);
                }
            }
        }

        return thoughts;
    }

    /// <summary>
    /// Extracts JSON content from markdown code blocks.
    /// </summary>
    private string ExtractJsonFromResponse(string response)
    {
        // Remove markdown code block markers
        var jsonMatch = Regex.Match(response, @"```(?:json)?\s*(\{[\s\S]*?\})\s*```", RegexOptions.Multiline, RegexTimeout);
        if (jsonMatch.Success)
        {
            return jsonMatch.Groups[1].Value;
        }

        // Try to find JSON object
        var jsonObjectMatch = Regex.Match(response, @"\{[\s\S]*?\}", RegexOptions.None, RegexTimeout);
        if (jsonObjectMatch.Success)
        {
            return jsonObjectMatch.Value;
        }

        return response;
    }
}
