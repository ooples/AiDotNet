# Agent Framework

## Overview

The Agent Framework enables AI agents to intelligently use tools to solve complex problems. Agents follow the **ReAct** (Reasoning + Acting) pattern, which alternates between:

1. **Reasoning (Thought)**: Thinking about what to do next
2. **Acting (Action)**: Using a tool to perform an operation
3. **Observing (Observation)**: Seeing the result of the action

## Core Components

### Interfaces

- **`ITool`**: Defines a tool that agents can use (e.g., Calculator, Search)
- **`IChatModel<T>`**: Defines a language model interface for agent reasoning
- **`IAgent<T>`**: Defines an agent that can reason and use tools

### Classes

- **`AgentBase<T>`**: Abstract base class providing common agent functionality
- **`Agent<T>`**: Concrete ReAct agent implementation

### Built-in Tools

- **`CalculatorTool`**: Performs mathematical calculations
- **`SearchTool`**: Mock search tool for demonstration and testing

## Quick Start

### Example 1: Simple Agent with Calculator

```csharp
using AiDotNet.Agents;
using AiDotNet.Tools;
using AiDotNet.Interfaces;

// Create a chat model (you'll need to implement IChatModel<T> for your LLM provider)
IChatModel<double> chatModel = new YourChatModel<double>();

// Create tools
var tools = new List<ITool>
{
    new CalculatorTool()
};

// Create the agent
var agent = new Agent<double>(chatModel, tools);

// Run the agent
string result = await agent.RunAsync("What is (25 * 4) + 10?");
Console.WriteLine(result); // Output: 110

// View the reasoning process
Console.WriteLine("\nReasoning Process:");
Console.WriteLine(agent.Scratchpad);
```

### Example 2: Agent with Multiple Tools

```csharp
using AiDotNet.Agents;
using AiDotNet.Tools;
using AiDotNet.Interfaces;

// Create multiple tools
var tools = new List<ITool>
{
    new CalculatorTool(),
    new SearchTool()
};

// Create agent with multiple tools
var agent = new Agent<double>(chatModel, tools);

// The agent can now use both calculator and search
string result = await agent.RunAsync(
    "What is the capital of France and how many letters are in its name?");

Console.WriteLine(result);
Console.WriteLine("\n--- Scratchpad ---");
Console.WriteLine(agent.Scratchpad);
```

### Example 3: Custom Tool Implementation

```csharp
using AiDotNet.Interfaces;

public class WeatherTool : ITool
{
    public string Name => "Weather";

    public string Description =>
        "Gets current weather for a city. " +
        "Input should be a city name. " +
        "Returns temperature and conditions.";

    public string Execute(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            return "Error: City name cannot be empty.";
        }

        // In a real implementation, you would call a weather API
        // This is a mock for demonstration
        return $"Weather in {input}: Sunny, 72°F";
    }
}

// Use your custom tool
var tools = new List<ITool>
{
    new CalculatorTool(),
    new WeatherTool()
};

var agent = new Agent<double>(chatModel, tools);
string result = await agent.RunAsync("What's the weather in Paris?");
```

### Example 4: Implementing IChatModel<T>

Here's a skeleton for implementing your own chat model:

```csharp
using AiDotNet.Interfaces;

public class OpenAIChatModel<T> : IChatModel<T>
{
    private readonly string _apiKey;
    private readonly string _model;

    public OpenAIChatModel(string apiKey, string model = "gpt-4")
    {
        _apiKey = apiKey;
        _model = model;
    }

    public string ModelName => _model;

    public async Task<string> GenerateResponseAsync(string prompt)
    {
        // Implement your API call here
        // Example using HttpClient:

        using var client = new HttpClient();
        client.DefaultRequestHeaders.Add("Authorization", $"Bearer {_apiKey}");

        var requestBody = new
        {
            model = _model,
            messages = new[]
            {
                new { role = "user", content = prompt }
            }
        };

        var response = await client.PostAsJsonAsync(
            "https://api.openai.com/v1/chat/completions",
            requestBody);

        var result = await response.Content.ReadFromJsonAsync<OpenAIResponse>();
        return result?.Choices[0]?.Message?.Content ?? "No response";
    }
}
```

## How the ReAct Loop Works

When you call `agent.RunAsync("query")`, the agent:

1. **Iteration 1**:
   - Sends a prompt to the LLM with the query and available tools
   - LLM responds with a thought and action (or final answer)
   - If action specified, executes the tool and records the observation

2. **Iteration 2**:
   - Sends updated prompt including previous thoughts, actions, and observations
   - LLM decides next step based on accumulated context
   - Continues until final answer or max iterations reached

3. **Returns**: Final answer or partial results if max iterations reached

## Advanced Configuration

### Setting Max Iterations

```csharp
// Allow up to 10 reasoning steps
string result = await agent.RunAsync("Complex query", maxIterations: 10);
```

### Accessing the Scratchpad

The scratchpad contains the complete reasoning trace:

```csharp
string result = await agent.RunAsync("Calculate 5 + 3");
Console.WriteLine("Final Answer: " + result);
Console.WriteLine("\nReasoning Trace:");
Console.WriteLine(agent.Scratchpad);

/* Output:
Query: Calculate 5 + 3

=== Iteration 1 ===
Thought: I need to use the calculator
Action: Calculator
Action Input: 5 + 3
Observation: 8

=== Iteration 2 ===
Thought: I have the answer
Final Answer: 8
*/
```

## Testing

The framework includes comprehensive unit tests. See:
- `tests/UnitTests/Agents/AgentTests.cs`
- `tests/UnitTests/Tools/CalculatorToolTests.cs`
- `tests/UnitTests/Tools/SearchToolTests.cs`

### Mock Chat Model for Testing

```csharp
using AiDotNetTests.UnitTests.Agents;

var mockModel = new MockChatModel<double>(
    "{\"thought\": \"I'll calculate\", \"action\": \"Calculator\", \"action_input\": \"2 + 2\"}",
    "{\"thought\": \"Got it\", \"final_answer\": \"4\"}"
);

var agent = new Agent<double>(mockModel, tools);
var result = await agent.RunAsync("What is 2 + 2?");

Assert.Equal("4", result);
```

## LLM Response Format

The agent expects the LLM to respond in JSON format:

```json
{
  "thought": "I need to search for information about X",
  "action": "Search",
  "action_input": "query about X",
  "final_answer": ""
}
```

Or when done:

```json
{
  "thought": "I have all the information I need",
  "action": "",
  "action_input": "",
  "final_answer": "The answer is Y"
}
```

The agent also has fallback regex parsing if the LLM doesn't return valid JSON.

## Best Practices

1. **Tool Design**: Keep tools focused on a single responsibility
2. **Descriptions**: Write clear, detailed tool descriptions
3. **Error Handling**: Tools should return error messages, not throw exceptions
4. **Iterations**: Start with 5-10 max iterations; adjust based on task complexity
5. **Testing**: Use MockChatModel to test agent behavior without calling real APIs

## Integration with PredictionModelBuilder

While the agent framework is standalone, you can integrate it with the broader AiDotNet ecosystem:

```csharp
// Example: Use agent to help with feature selection
var featureAgent = new Agent<double>(chatModel, featureTools);
var recommendation = await featureAgent.RunAsync(
    "Which features should I select for predicting house prices?");
```

## Future Extensions

Potential areas for extension:

- **Memory**: Add long-term memory for agents
- **Multi-agent Systems**: Enable multiple agents to collaborate
- **Tool Learning**: Allow agents to learn new tools dynamically
- **Hierarchical Agents**: Parent agents that delegate to child agents
- **Evaluation Metrics**: Track success rates and reasoning quality
