# Issue #420: Junior Developer Implementation Guide

## Understanding Prompt Engineering and LLM Tools Framework

**Goal**: Build a framework for Large Language Model (LLM) integration with structured prompts, tool calling, and response parsing - enabling AI agents to use external tools and APIs.

---

## Key Concepts for Beginners

### What is Prompt Engineering?

**The Challenge**: Getting LLMs to produce consistent, structured, useful outputs.

**Example**:
```
Bad Prompt: "Tell me about cats"
Response: Random essay about cats (unpredictable format)

Good Prompt: "List 3 facts about cats in JSON format with 'fact' and 'category' fields"
Response: Structured JSON that you can parse programmatically
```

### What are LLM Tools/Function Calling?

**The Concept**: LLMs can "call" external functions/tools by outputting structured commands.

**Example**:
```
User: "What's the weather in Seattle?"

LLM thinks: "I need to call get_weather(location='Seattle')"

LLM outputs: {"tool": "get_weather", "args": {"location": "Seattle"}}

System calls actual weather API → gets result → feeds back to LLM

LLM: "The weather in Seattle is 65F and partly cloudy"
```

### Components of an LLM Framework

1. **Prompt Templates**: Reusable prompt structures
2. **Tool Definitions**: Describe available tools/functions
3. **Response Parsers**: Extract structured data from LLM output
4. **Tool Executor**: Actually run the tools LLM requests
5. **Conversation Management**: Track multi-turn dialogues

---

## Phase 1: Prompt Template System

### AC 1.1: Implement PromptTemplate

**What does this do?**
Allows creating reusable prompt templates with variable substitution.

**File**: `src/LLM/PromptTemplate.cs`

**Step 1**: Create basic template system

```csharp
// File: src/LLM/PromptTemplate.cs
namespace AiDotNet.LLM;

/// <summary>
/// Template for creating prompts with variable substitution.
/// Supports {{variable}} syntax for placeholders.
/// </summary>
public class PromptTemplate
{
    private readonly string _template;
    private readonly List<string> _requiredVariables;

    /// <summary>
    /// Creates a prompt template.
    /// </summary>
    /// <param name="template">Template string with {{variable}} placeholders</param>
    public PromptTemplate(string template)
    {
        _template = template ?? throw new ArgumentNullException(nameof(template));
        _requiredVariables = ExtractVariables(template);
    }

    /// <summary>
    /// Format the template by substituting variables.
    /// </summary>
    /// <param name="variables">Dictionary of variable name -> value</param>
    /// <returns>Formatted prompt</returns>
    public string Format(Dictionary<string, string> variables)
    {
        if (variables == null)
            throw new ArgumentNullException(nameof(variables));

        // Check all required variables are provided
        var missingVars = _requiredVariables.Where(v => !variables.ContainsKey(v)).ToList();
        if (missingVars.Any())
        {
            throw new ArgumentException(
                $"Missing required variables: {string.Join(", ", missingVars)}");
        }

        string result = _template;

        // Replace all variables
        foreach (var kvp in variables)
        {
            string placeholder = $"{{{{{kvp.Key}}}}}"; // {{variable}}
            result = result.Replace(placeholder, kvp.Value);
        }

        return result;
    }

    /// <summary>
    /// Get list of required variables in this template.
    /// </summary>
    public IReadOnlyList<string> RequiredVariables => _requiredVariables.AsReadOnly();

    private List<string> ExtractVariables(string template)
    {
        var variables = new List<string>();
        var regex = new System.Text.RegularExpressions.Regex(@"\{\{(\w+)\}\}");
        var matches = regex.Matches(template);

        foreach (System.Text.RegularExpressions.Match match in matches)
        {
            string varName = match.Groups[1].Value;
            if (!variables.Contains(varName))
            {
                variables.Add(varName);
            }
        }

        return variables;
    }
}
```

**Step 2**: Create common prompt templates

```csharp
// File: src/LLM/PromptTemplates.cs
namespace AiDotNet.LLM;

/// <summary>
/// Collection of common prompt templates for various tasks.
/// </summary>
public static class PromptTemplates
{
    /// <summary>
    /// Zero-shot classification template.
    /// Variables: text, categories (comma-separated)
    /// </summary>
    public static readonly PromptTemplate ZeroShotClassification = new PromptTemplate(
        @"Classify the following text into one of these categories: {{categories}}

Text: {{text}}

Category:");

    /// <summary>
    /// Few-shot classification template.
    /// Variables: examples, text
    /// </summary>
    public static readonly PromptTemplate FewShotClassification = new PromptTemplate(
        @"Here are some examples of text classification:

{{examples}}

Now classify this text:
Text: {{text}}

Category:");

    /// <summary>
    /// JSON extraction template.
    /// Variables: text, schema
    /// </summary>
    public static readonly PromptTemplate JsonExtraction = new PromptTemplate(
        @"Extract structured information from the following text as JSON.

Required schema:
{{schema}}

Text: {{text}}

JSON:");

    /// <summary>
    /// Question answering with context.
    /// Variables: context, question
    /// </summary>
    public static readonly PromptTemplate QuestionAnswering = new PromptTemplate(
        @"Answer the question based on the context below.

Context: {{context}}

Question: {{question}}

Answer:");

    /// <summary>
    /// Tool use template for function calling.
    /// Variables: tools, query
    /// </summary>
    public static readonly PromptTemplate ToolUse = new PromptTemplate(
        @"You have access to the following tools:

{{tools}}

User query: {{query}}

Which tool should you use and with what arguments? Respond in JSON format:
{""tool"": ""tool_name"", ""arguments"": {}}

Response:");
}
```

**Step 3**: Create unit tests

```csharp
// File: tests/UnitTests/LLM/PromptTemplateTests.cs
namespace AiDotNet.Tests.LLM;

public class PromptTemplateTests
{
    [Fact]
    public void Format_AllVariablesProvided_ReplacesCorrectly()
    {
        // Arrange
        var template = new PromptTemplate("Hello {{name}}, you are {{age}} years old.");
        var variables = new Dictionary<string, string>
        {
            { "name", "Alice" },
            { "age", "30" }
        };

        // Act
        var result = template.Format(variables);

        // Assert
        Assert.Equal("Hello Alice, you are 30 years old.", result);
    }

    [Fact]
    public void Format_MissingVariable_ThrowsException()
    {
        // Arrange
        var template = new PromptTemplate("Hello {{name}}, you are {{age}} years old.");
        var variables = new Dictionary<string, string>
        {
            { "name", "Alice" }
            // age is missing
        };

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => template.Format(variables));
        Assert.Contains("age", ex.Message);
    }

    [Fact]
    public void RequiredVariables_ExtractsAllVariables()
    {
        // Arrange
        var template = new PromptTemplate("{{var1}} and {{var2}} and {{var1}} again");

        // Act
        var variables = template.RequiredVariables;

        // Assert
        Assert.Equal(2, variables.Count); // var1 should only appear once
        Assert.Contains("var1", variables);
        Assert.Contains("var2", variables);
    }
}
```

---

## Phase 2: Tool Definition Framework

### AC 2.1: Implement ToolDefinition and ToolRegistry

**What is this?**
Describes tools/functions that an LLM can call, including parameters, types, and descriptions.

**File**: `src/LLM/Tools/ToolDefinition.cs`

```csharp
// File: src/LLM/Tools/ToolDefinition.cs
namespace AiDotNet.LLM.Tools;

/// <summary>
/// Defines a tool/function that can be called by an LLM.
/// </summary>
public class ToolDefinition
{
    /// <summary>Unique tool name (e.g., "get_weather", "search_web")</summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>Human-readable description of what the tool does</summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>Parameter definitions</summary>
    public List<ToolParameter> Parameters { get; set; } = new List<ToolParameter>();

    /// <summary>Return type description</summary>
    public string ReturnType { get; set; } = "string";

    /// <summary>
    /// Convert to JSON schema format (OpenAI function calling format).
    /// </summary>
    public string ToJsonSchema()
    {
        var parameters = new
        {
            type = "object",
            properties = Parameters.ToDictionary(
                p => p.Name,
                p => new
                {
                    type = p.Type,
                    description = p.Description,
                    @enum = p.AllowedValues.Count > 0 ? p.AllowedValues : null
                }
            ),
            required = Parameters.Where(p => p.Required).Select(p => p.Name).ToArray()
        };

        var schema = new
        {
            name = Name,
            description = Description,
            parameters = parameters
        };

        return System.Text.Json.JsonSerializer.Serialize(schema, new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true
        });
    }

    /// <summary>
    /// Convert to plain text description for prompt injection.
    /// </summary>
    public string ToTextDescription()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Tool: {Name}");
        sb.AppendLine($"Description: {Description}");
        sb.AppendLine("Parameters:");

        foreach (var param in Parameters)
        {
            string required = param.Required ? "(required)" : "(optional)";
            sb.AppendLine($"  - {param.Name} ({param.Type}) {required}: {param.Description}");

            if (param.AllowedValues.Count > 0)
            {
                sb.AppendLine($"    Allowed values: {string.Join(", ", param.AllowedValues)}");
            }
        }

        sb.AppendLine($"Returns: {ReturnType}");

        return sb.ToString();
    }
}

/// <summary>
/// Parameter definition for a tool.
/// </summary>
public class ToolParameter
{
    /// <summary>Parameter name</summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>Data type (string, number, boolean, array, object)</summary>
    public string Type { get; set; } = "string";

    /// <summary>Description of the parameter</summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>Whether this parameter is required</summary>
    public bool Required { get; set; } = true;

    /// <summary>Allowed values (for enum types)</summary>
    public List<string> AllowedValues { get; set; } = new List<string>();

    /// <summary>Default value if not provided</summary>
    public object? DefaultValue { get; set; }
}
```

**Step 2**: Create ToolRegistry

```csharp
// File: src/LLM/Tools/ToolRegistry.cs
namespace AiDotNet.LLM.Tools;

/// <summary>
/// Registry of available tools for LLM to use.
/// </summary>
public class ToolRegistry
{
    private readonly Dictionary<string, ToolDefinition> _tools;
    private readonly Dictionary<string, Func<Dictionary<string, object>, Task<object>>> _implementations;

    public ToolRegistry()
    {
        _tools = new Dictionary<string, ToolDefinition>();
        _implementations = new Dictionary<string, Func<Dictionary<string, object>, Task<object>>>();
    }

    /// <summary>
    /// Register a tool with its implementation.
    /// </summary>
    public void RegisterTool(
        ToolDefinition definition,
        Func<Dictionary<string, object>, Task<object>> implementation)
    {
        if (definition == null)
            throw new ArgumentNullException(nameof(definition));
        if (implementation == null)
            throw new ArgumentNullException(nameof(implementation));

        _tools[definition.Name] = definition;
        _implementations[definition.Name] = implementation;
    }

    /// <summary>
    /// Get tool definition by name.
    /// </summary>
    public ToolDefinition? GetTool(string name)
    {
        return _tools.TryGetValue(name, out var tool) ? tool : null;
    }

    /// <summary>
    /// Get all registered tools.
    /// </summary>
    public IReadOnlyList<ToolDefinition> GetAllTools()
    {
        return _tools.Values.ToList().AsReadOnly();
    }

    /// <summary>
    /// Execute a tool with given arguments.
    /// </summary>
    public async Task<object> ExecuteToolAsync(string toolName, Dictionary<string, object> arguments)
    {
        if (!_implementations.ContainsKey(toolName))
            throw new ArgumentException($"Tool '{toolName}' not found");

        // Validate arguments against definition
        var definition = _tools[toolName];
        ValidateArguments(definition, arguments);

        // Execute tool
        return await _implementations[toolName](arguments);
    }

    /// <summary>
    /// Get all tools formatted as text for prompt injection.
    /// </summary>
    public string GetToolsAsText()
    {
        var sb = new System.Text.StringBuilder();

        foreach (var tool in _tools.Values)
        {
            sb.AppendLine(tool.ToTextDescription());
            sb.AppendLine();
        }

        return sb.ToString();
    }

    /// <summary>
    /// Get all tools formatted as JSON schema.
    /// </summary>
    public string GetToolsAsJsonSchema()
    {
        var schemas = _tools.Values.Select(t => t.ToJsonSchema()).ToList();
        return $"[{string.Join(",", schemas)}]";
    }

    private void ValidateArguments(ToolDefinition definition, Dictionary<string, object> arguments)
    {
        // Check required parameters
        var requiredParams = definition.Parameters.Where(p => p.Required).Select(p => p.Name).ToList();
        var missingParams = requiredParams.Where(p => !arguments.ContainsKey(p)).ToList();

        if (missingParams.Any())
        {
            throw new ArgumentException(
                $"Missing required parameters for tool '{definition.Name}': {string.Join(", ", missingParams)}");
        }

        // Validate parameter types (basic validation)
        foreach (var param in definition.Parameters)
        {
            if (!arguments.ContainsKey(param.Name))
                continue;

            var value = arguments[param.Name];

            // Type checking
            bool validType = param.Type.ToLower() switch
            {
                "string" => value is string,
                "number" => value is int or long or float or double or decimal,
                "boolean" => value is bool,
                "array" => value is System.Collections.IEnumerable,
                "object" => value is Dictionary<string, object>,
                _ => true // Unknown type - skip validation
            };

            if (!validType)
            {
                throw new ArgumentException(
                    $"Parameter '{param.Name}' has invalid type. Expected {param.Type}, got {value?.GetType().Name}");
            }

            // Enum validation
            if (param.AllowedValues.Count > 0)
            {
                string strValue = value?.ToString() ?? "";
                if (!param.AllowedValues.Contains(strValue))
                {
                    throw new ArgumentException(
                        $"Parameter '{param.Name}' has invalid value. Allowed: {string.Join(", ", param.AllowedValues)}");
                }
            }
        }
    }
}
```

**Step 3**: Create example tools

```csharp
// File: src/LLM/Tools/BuiltInTools.cs
namespace AiDotNet.LLM.Tools;

/// <summary>
/// Collection of built-in tools for common tasks.
/// </summary>
public static class BuiltInTools
{
    /// <summary>
    /// Calculator tool for arithmetic operations.
    /// </summary>
    public static ToolDefinition Calculator = new ToolDefinition
    {
        Name = "calculator",
        Description = "Performs basic arithmetic operations",
        Parameters = new List<ToolParameter>
        {
            new ToolParameter
            {
                Name = "operation",
                Type = "string",
                Description = "The operation to perform",
                Required = true,
                AllowedValues = new List<string> { "add", "subtract", "multiply", "divide" }
            },
            new ToolParameter
            {
                Name = "a",
                Type = "number",
                Description = "First operand",
                Required = true
            },
            new ToolParameter
            {
                Name = "b",
                Type = "number",
                Description = "Second operand",
                Required = true
            }
        },
        ReturnType = "number"
    };

    /// <summary>
    /// Weather lookup tool (mock implementation).
    /// </summary>
    public static ToolDefinition GetWeather = new ToolDefinition
    {
        Name = "get_weather",
        Description = "Gets current weather for a location",
        Parameters = new List<ToolParameter>
        {
            new ToolParameter
            {
                Name = "location",
                Type = "string",
                Description = "City name or zip code",
                Required = true
            },
            new ToolParameter
            {
                Name = "units",
                Type = "string",
                Description = "Temperature units",
                Required = false,
                AllowedValues = new List<string> { "celsius", "fahrenheit" },
                DefaultValue = "fahrenheit"
            }
        },
        ReturnType = "object"
    };

    /// <summary>
    /// Web search tool.
    /// </summary>
    public static ToolDefinition WebSearch = new ToolDefinition
    {
        Name = "web_search",
        Description = "Searches the web for information",
        Parameters = new List<ToolParameter>
        {
            new ToolParameter
            {
                Name = "query",
                Type = "string",
                Description = "Search query",
                Required = true
            },
            new ToolParameter
            {
                Name = "max_results",
                Type = "number",
                Description = "Maximum number of results to return",
                Required = false,
                DefaultValue = 5
            }
        },
        ReturnType = "array"
    };

    /// <summary>
    /// Register all built-in tools to a registry with implementations.
    /// </summary>
    public static void RegisterAll(ToolRegistry registry)
    {
        // Calculator implementation
        registry.RegisterTool(Calculator, async args =>
        {
            string operation = args["operation"].ToString() ?? "";
            double a = Convert.ToDouble(args["a"]);
            double b = Convert.ToDouble(args["b"]);

            return operation switch
            {
                "add" => a + b,
                "subtract" => a - b,
                "multiply" => a * b,
                "divide" => a / b,
                _ => throw new ArgumentException($"Unknown operation: {operation}")
            };
        });

        // Weather implementation (mock)
        registry.RegisterTool(GetWeather, async args =>
        {
            string location = args["location"].ToString() ?? "";
            string units = args.ContainsKey("units") ? args["units"].ToString() ?? "fahrenheit" : "fahrenheit";

            // Mock weather data
            return new
            {
                location = location,
                temperature = units == "celsius" ? 20 : 68,
                units = units,
                condition = "Partly cloudy",
                humidity = 65
            };
        });

        // Web search implementation (mock)
        registry.RegisterTool(WebSearch, async args =>
        {
            string query = args["query"].ToString() ?? "";
            int maxResults = args.ContainsKey("max_results")
                ? Convert.ToInt32(args["max_results"])
                : 5;

            // Mock search results
            var results = new List<object>();
            for (int i = 0; i < Math.Min(maxResults, 3); i++)
            {
                results.Add(new
                {
                    title = $"Result {i + 1} for '{query}'",
                    url = $"https://example.com/result{i + 1}",
                    snippet = $"This is a mock search result for {query}"
                });
            }

            return results;
        });
    }
}
```

---

## Phase 3: Response Parsing

### AC 3.1: Implement JsonResponseParser

**What is this?**
Extracts structured data from LLM responses, even when LLM adds extra text.

**File**: `src/LLM/Parsing/JsonResponseParser.cs`

```csharp
// File: src/LLM/Parsing/JsonResponseParser.cs
namespace AiDotNet.LLM.Parsing;

/// <summary>
/// Parses JSON from LLM responses, handling various formats and errors.
/// </summary>
public class JsonResponseParser
{
    /// <summary>
    /// Extract and parse JSON from response text.
    /// Handles cases where JSON is embedded in natural language.
    /// </summary>
    public T? ParseJson<T>(string response) where T : class
    {
        if (string.IsNullOrWhiteSpace(response))
            return null;

        // Try parsing entire response as JSON
        try
        {
            return System.Text.Json.JsonSerializer.Deserialize<T>(response);
        }
        catch
        {
            // Not pure JSON - try extracting JSON from text
        }

        // Try to find JSON block in response
        string? jsonText = ExtractJsonBlock(response);

        if (jsonText != null)
        {
            try
            {
                return System.Text.Json.JsonSerializer.Deserialize<T>(jsonText);
            }
            catch (System.Text.Json.JsonException ex)
            {
                throw new InvalidOperationException(
                    $"Found JSON-like text but failed to parse: {ex.Message}", ex);
            }
        }

        throw new InvalidOperationException("No valid JSON found in response");
    }

    /// <summary>
    /// Extract JSON from markdown code blocks or embedded in text.
    /// </summary>
    private string? ExtractJsonBlock(string text)
    {
        // Pattern 1: Markdown code block with json tag
        var markdownMatch = System.Text.RegularExpressions.Regex.Match(
            text,
            @"```json\s*\n(.*?)\n```",
            System.Text.RegularExpressions.RegexOptions.Singleline
        );

        if (markdownMatch.Success)
        {
            return markdownMatch.Groups[1].Value.Trim();
        }

        // Pattern 2: Generic code block
        var codeBlockMatch = System.Text.RegularExpressions.Regex.Match(
            text,
            @"```\s*\n(.*?)\n```",
            System.Text.RegularExpressions.RegexOptions.Singleline
        );

        if (codeBlockMatch.Success)
        {
            string content = codeBlockMatch.Groups[1].Value.Trim();
            if (IsLikelyJson(content))
            {
                return content;
            }
        }

        // Pattern 3: Look for object or array directly in text
        var jsonObjectMatch = System.Text.RegularExpressions.Regex.Match(
            text,
            @"(\{.*\})",
            System.Text.RegularExpressions.RegexOptions.Singleline
        );

        if (jsonObjectMatch.Success)
        {
            return jsonObjectMatch.Groups[1].Value;
        }

        var jsonArrayMatch = System.Text.RegularExpressions.Regex.Match(
            text,
            @"(\[.*\])",
            System.Text.RegularExpressions.RegexOptions.Singleline
        );

        if (jsonArrayMatch.Success)
        {
            return jsonArrayMatch.Groups[1].Value;
        }

        return null;
    }

    private bool IsLikelyJson(string text)
    {
        text = text.Trim();
        return (text.StartsWith("{") && text.EndsWith("}")) ||
               (text.StartsWith("[") && text.EndsWith("]"));
    }
}
```

**Step 2**: Create tool call parser

```csharp
// File: src/LLM/Parsing/ToolCallParser.cs
namespace AiDotNet.LLM.Parsing;

/// <summary>
/// Parses tool call requests from LLM responses.
/// </summary>
public class ToolCallParser
{
    private readonly JsonResponseParser _jsonParser;

    public ToolCallParser()
    {
        _jsonParser = new JsonResponseParser();
    }

    /// <summary>
    /// Parse tool call from LLM response.
    /// </summary>
    public ToolCall? ParseToolCall(string response)
    {
        try
        {
            return _jsonParser.ParseJson<ToolCall>(response);
        }
        catch
        {
            // Try alternative formats
            return TryParseAlternativeFormats(response);
        }
    }

    private ToolCall? TryParseAlternativeFormats(string response)
    {
        // Format: "Use tool_name with argument1=value1, argument2=value2"
        var match = System.Text.RegularExpressions.Regex.Match(
            response,
            @"(?:use|call)\s+(\w+)\s+(?:with|using)\s+(.*)",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase
        );

        if (match.Success)
        {
            string toolName = match.Groups[1].Value;
            string argsText = match.Groups[2].Value;

            var arguments = ParseArgumentsList(argsText);

            return new ToolCall
            {
                Tool = toolName,
                Arguments = arguments
            };
        }

        return null;
    }

    private Dictionary<string, object> ParseArgumentsList(string argsText)
    {
        var arguments = new Dictionary<string, object>();

        // Parse "key=value, key2=value2" format
        var pairs = argsText.Split(',');

        foreach (var pair in pairs)
        {
            var parts = pair.Split('=');
            if (parts.Length == 2)
            {
                string key = parts[0].Trim();
                string value = parts[1].Trim().Trim('\'', '"');
                arguments[key] = value;
            }
        }

        return arguments;
    }
}

/// <summary>
/// Represents a tool call request from LLM.
/// </summary>
public class ToolCall
{
    public string Tool { get; set; } = string.Empty;
    public Dictionary<string, object> Arguments { get; set; } = new Dictionary<string, object>();
}
```

---

## Phase 4: Agent Orchestration

### AC 4.1: Implement LLMAgent

**What is this?**
Coordinates prompts, tool calls, and responses to create an autonomous agent.

**File**: `src/LLM/LLMAgent.cs`

```csharp
// File: src/LLM/LLMAgent.cs
namespace AiDotNet.LLM;

/// <summary>
/// LLM-powered agent that can use tools to accomplish tasks.
/// </summary>
public class LLMAgent
{
    private readonly ILanguageModel _llm;
    private readonly ToolRegistry _toolRegistry;
    private readonly ToolCallParser _toolParser;
    private readonly List<Message> _conversationHistory;
    private readonly int _maxIterations;

    /// <summary>
    /// Creates an LLM agent.
    /// </summary>
    /// <param name="llm">Language model to use</param>
    /// <param name="toolRegistry">Available tools</param>
    /// <param name="maxIterations">Max tool use iterations (prevents infinite loops)</param>
    public LLMAgent(ILanguageModel llm, ToolRegistry toolRegistry, int maxIterations = 10)
    {
        _llm = llm ?? throw new ArgumentNullException(nameof(llm));
        _toolRegistry = toolRegistry ?? throw new ArgumentNullException(nameof(toolRegistry));
        _toolParser = new ToolCallParser();
        _conversationHistory = new List<Message>();
        _maxIterations = maxIterations;
    }

    /// <summary>
    /// Execute a task using available tools.
    /// </summary>
    public async Task<AgentResponse> ExecuteAsync(string task)
    {
        _conversationHistory.Clear();

        // System message with tool descriptions
        string systemMessage = BuildSystemMessage();
        _conversationHistory.Add(new Message
        {
            Role = "system",
            Content = systemMessage
        });

        // User task
        _conversationHistory.Add(new Message
        {
            Role = "user",
            Content = task
        });

        var toolCalls = new List<(string Tool, Dictionary<string, object> Args, object Result)>();
        int iteration = 0;

        while (iteration < _maxIterations)
        {
            iteration++;

            // Get LLM response
            string response = await _llm.GenerateAsync(BuildPrompt());

            // Check if response contains tool call
            var toolCall = _toolParser.ParseToolCall(response);

            if (toolCall == null)
            {
                // No tool call - this is the final answer
                return new AgentResponse
                {
                    FinalAnswer = response,
                    ToolCalls = toolCalls,
                    ConversationHistory = new List<Message>(_conversationHistory),
                    Iterations = iteration
                };
            }

            // Execute tool
            try
            {
                var result = await _toolRegistry.ExecuteToolAsync(toolCall.Tool, toolCall.Arguments);

                toolCalls.Add((toolCall.Tool, toolCall.Arguments, result));

                // Add tool result to conversation
                _conversationHistory.Add(new Message
                {
                    Role = "assistant",
                    Content = $"Tool call: {toolCall.Tool} with arguments {System.Text.Json.JsonSerializer.Serialize(toolCall.Arguments)}"
                });

                _conversationHistory.Add(new Message
                {
                    Role = "system",
                    Content = $"Tool result: {System.Text.Json.JsonSerializer.Serialize(result)}"
                });
            }
            catch (Exception ex)
            {
                // Tool execution failed
                _conversationHistory.Add(new Message
                {
                    Role = "system",
                    Content = $"Tool execution failed: {ex.Message}"
                });
            }
        }

        // Max iterations reached
        return new AgentResponse
        {
            FinalAnswer = "Max iterations reached without final answer",
            ToolCalls = toolCalls,
            ConversationHistory = new List<Message>(_conversationHistory),
            Iterations = iteration,
            Error = "Maximum iterations exceeded"
        };
    }

    private string BuildSystemMessage()
    {
        return $@"You are a helpful AI agent with access to tools.

Available tools:
{_toolRegistry.GetToolsAsText()}

When you need to use a tool, respond with JSON in this format:
{{""tool"": ""tool_name"", ""arguments"": {{""arg1"": ""value1""}}}}

When you have the final answer, respond directly without using tool format.";
    }

    private string BuildPrompt()
    {
        var sb = new System.Text.StringBuilder();

        foreach (var message in _conversationHistory)
        {
            sb.AppendLine($"{message.Role}: {message.Content}");
            sb.AppendLine();
        }

        return sb.ToString();
    }
}

/// <summary>
/// Message in conversation history.
/// </summary>
public class Message
{
    public string Role { get; set; } = string.Empty; // system, user, assistant
    public string Content { get; set; } = string.Empty;
}

/// <summary>
/// Response from agent execution.
/// </summary>
public class AgentResponse
{
    public string FinalAnswer { get; set; } = string.Empty;
    public List<(string Tool, Dictionary<string, object> Args, object Result)> ToolCalls { get; set; }
        = new List<(string, Dictionary<string, object>, object)>();
    public List<Message> ConversationHistory { get; set; } = new List<Message>();
    public int Iterations { get; set; }
    public string? Error { get; set; }
}

/// <summary>
/// Interface for language model providers.
/// </summary>
public interface ILanguageModel
{
    Task<string> GenerateAsync(string prompt);
}
```

---

## Phase 5: Chain-of-Thought and ReAct

### AC 5.1: Implement ReActAgent

**What is ReAct?**
Reasoning + Acting: Agent alternates between reasoning (thinking) and acting (using tools).

**Pattern**:
```
Thought: I need to find the weather
Action: get_weather(location="Seattle")
Observation: Temperature is 65F
Thought: Now I can answer the question
Answer: The weather in Seattle is 65F
```

**File**: `src/LLM/ReActAgent.cs`

```csharp
// File: src/LLM/ReActAgent.cs
namespace AiDotNet.LLM;

/// <summary>
/// ReAct (Reasoning + Acting) agent implementation.
/// Alternates between reasoning steps and tool use.
/// </summary>
public class ReActAgent
{
    private readonly ILanguageModel _llm;
    private readonly ToolRegistry _toolRegistry;
    private readonly int _maxSteps;

    private static readonly PromptTemplate ReActTemplate = new PromptTemplate(
        @"Answer the following question using tools and step-by-step reasoning.

Available tools:
{{tools}}

Question: {{question}}

Use this format:
Thought: Your reasoning about what to do next
Action: tool_name(arg1=value1, arg2=value2)
Observation: [Result will be inserted here]
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Answer: The final answer

Begin!");

    public ReActAgent(ILanguageModel llm, ToolRegistry toolRegistry, int maxSteps = 10)
    {
        _llm = llm ?? throw new ArgumentNullException(nameof(llm));
        _toolRegistry = toolRegistry ?? throw new ArgumentNullException(nameof(toolRegistry));
        _maxSteps = maxSteps;
    }

    public async Task<ReActResponse> ExecuteAsync(string question)
    {
        var steps = new List<ReActStep>();

        string prompt = ReActTemplate.Format(new Dictionary<string, string>
        {
            { "tools", _toolRegistry.GetToolsAsText() },
            { "question", question }
        });

        string scratchpad = prompt;

        for (int step = 0; step < _maxSteps; step++)
        {
            // Generate next step
            string response = await _llm.GenerateAsync(scratchpad);

            // Parse response
            var parsedStep = ParseReActStep(response);

            if (parsedStep.IsAnswer)
            {
                // Found final answer
                return new ReActResponse
                {
                    Answer = parsedStep.Answer,
                    Steps = steps,
                    Success = true
                };
            }

            // Execute action
            try
            {
                var result = await _toolRegistry.ExecuteToolAsync(
                    parsedStep.Action,
                    parsedStep.ActionArguments
                );

                parsedStep.Observation = System.Text.Json.JsonSerializer.Serialize(result);
                steps.Add(parsedStep);

                // Update scratchpad
                scratchpad += $"\nThought: {parsedStep.Thought}";
                scratchpad += $"\nAction: {parsedStep.Action}({FormatArgs(parsedStep.ActionArguments)})";
                scratchpad += $"\nObservation: {parsedStep.Observation}\n";
            }
            catch (Exception ex)
            {
                parsedStep.Observation = $"Error: {ex.Message}";
                steps.Add(parsedStep);

                scratchpad += $"\nObservation: Error: {ex.Message}\n";
            }
        }

        return new ReActResponse
        {
            Answer = "Maximum steps reached without finding answer",
            Steps = steps,
            Success = false,
            Error = "Max steps exceeded"
        };
    }

    private ReActStep ParseReActStep(string response)
    {
        var step = new ReActStep();

        // Extract Thought
        var thoughtMatch = System.Text.RegularExpressions.Regex.Match(
            response,
            @"Thought:\s*(.+?)(?=\nAction:|\nAnswer:|$)",
            System.Text.RegularExpressions.RegexOptions.Singleline
        );

        if (thoughtMatch.Success)
        {
            step.Thought = thoughtMatch.Groups[1].Value.Trim();
        }

        // Check for Answer (final step)
        var answerMatch = System.Text.RegularExpressions.Regex.Match(
            response,
            @"Answer:\s*(.+)",
            System.Text.RegularExpressions.RegexOptions.Singleline
        );

        if (answerMatch.Success)
        {
            step.IsAnswer = true;
            step.Answer = answerMatch.Groups[1].Value.Trim();
            return step;
        }

        // Extract Action
        var actionMatch = System.Text.RegularExpressions.Regex.Match(
            response,
            @"Action:\s*(\w+)\((.*?)\)",
            System.Text.RegularExpressions.RegexOptions.Singleline
        );

        if (actionMatch.Success)
        {
            step.Action = actionMatch.Groups[1].Value.Trim();
            step.ActionArguments = ParseActionArguments(actionMatch.Groups[2].Value);
        }

        return step;
    }

    private Dictionary<string, object> ParseActionArguments(string argsText)
    {
        var arguments = new Dictionary<string, object>();

        if (string.IsNullOrWhiteSpace(argsText))
            return arguments;

        var pairs = argsText.Split(',');

        foreach (var pair in pairs)
        {
            var parts = pair.Split('=');
            if (parts.Length == 2)
            {
                string key = parts[0].Trim();
                string value = parts[1].Trim().Trim('\'', '"');
                arguments[key] = value;
            }
        }

        return arguments;
    }

    private string FormatArgs(Dictionary<string, object> args)
    {
        return string.Join(", ", args.Select(kvp => $"{kvp.Key}={kvp.Value}"));
    }
}

public class ReActStep
{
    public string Thought { get; set; } = string.Empty;
    public string Action { get; set; } = string.Empty;
    public Dictionary<string, object> ActionArguments { get; set; } = new Dictionary<string, object>();
    public string Observation { get; set; } = string.Empty;
    public bool IsAnswer { get; set; }
    public string Answer { get; set; } = string.Empty;
}

public class ReActResponse
{
    public string Answer { get; set; } = string.Empty;
    public List<ReActStep> Steps { get; set; } = new List<ReActStep>();
    public bool Success { get; set; }
    public string? Error { get; set; }
}
```

---

## Testing Strategy

### Unit Tests

```csharp
// File: tests/UnitTests/LLM/ToolRegistryTests.cs
namespace AiDotNet.Tests.LLM;

public class ToolRegistryTests
{
    [Fact]
    public async Task ExecuteToolAsync_ValidTool_ReturnsResult()
    {
        // Arrange
        var registry = new ToolRegistry();
        BuiltInTools.RegisterAll(registry);

        var args = new Dictionary<string, object>
        {
            { "operation", "add" },
            { "a", 5.0 },
            { "b", 3.0 }
        };

        // Act
        var result = await registry.ExecuteToolAsync("calculator", args);

        // Assert
        Assert.Equal(8.0, Convert.ToDouble(result));
    }

    [Fact]
    public async Task ExecuteToolAsync_MissingRequiredParameter_ThrowsException()
    {
        // Arrange
        var registry = new ToolRegistry();
        BuiltInTools.RegisterAll(registry);

        var args = new Dictionary<string, object>
        {
            { "operation", "add" },
            { "a", 5.0 }
            // Missing 'b'
        };

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(
            () => registry.ExecuteToolAsync("calculator", args));
    }
}
```

### Integration Tests

```csharp
// File: tests/IntegrationTests/LLM/AgentIntegrationTests.cs
namespace AiDotNet.Tests.LLM;

public class AgentIntegrationTests
{
    [Fact]
    public async Task Agent_MultiStepTask_UsesTools()
    {
        // Arrange
        var mockLLM = new MockLanguageModel();
        var registry = new ToolRegistry();
        BuiltInTools.RegisterAll(registry);

        var agent = new LLMAgent(mockLLM, registry);

        // Act
        var response = await agent.ExecuteAsync(
            "What is 15 + 27 multiplied by 3?");

        // Assert
        Assert.Contains("calculator",
            response.ToolCalls.Select(tc => tc.Tool));
        Assert.True(response.ToolCalls.Count >= 2); // Should use calculator twice
        Assert.NotEmpty(response.FinalAnswer);
    }
}
```

---

## Common Pitfalls

1. **LLM not following JSON format**: Use few-shot examples in system message
2. **Infinite tool calling loops**: Always set max iterations
3. **Missing error handling**: Tool execution can fail - handle gracefully
4. **Not validating tool arguments**: Always validate before execution
5. **Prompt too long**: Summarize conversation history after N turns

---

## Success Criteria Checklist

- [ ] PromptTemplate supports variable substitution
- [ ] ToolRegistry validates parameters before execution
- [ ] JsonResponseParser handles various formats (markdown, plain, embedded)
- [ ] LLMAgent successfully completes multi-step tasks
- [ ] ReActAgent shows clear reasoning steps
- [ ] All built-in tools execute correctly
- [ ] Error handling prevents crashes on malformed responses
- [ ] Unit tests cover all parsing edge cases
- [ ] Integration tests demonstrate end-to-end agent flow

---

## Example Usage After Implementation

```csharp
// Setup
var llm = new OpenAILanguageModel(apiKey: "...");
var registry = new ToolRegistry();
BuiltInTools.RegisterAll(registry);

// ReAct agent
var agent = new ReActAgent(llm, registry);
var response = await agent.ExecuteAsync(
    "What's the weather in Seattle? If it's above 60F, recommend outdoor activities.");

Console.WriteLine($"Answer: {response.Answer}");

foreach (var step in response.Steps)
{
    Console.WriteLine($"Thought: {step.Thought}");
    Console.WriteLine($"Action: {step.Action}");
    Console.WriteLine($"Observation: {step.Observation}");
}
```
