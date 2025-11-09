# Junior Developer Implementation Guide: Issue #285
## Agents - Tool Use, Function Calling, and Memory Primitives

### Overview
This guide will walk you through implementing a complete agent framework for AiDotNet. Agents are AI systems that can reason, use tools, and maintain memory to solve complex tasks that go beyond simple LLM calls.

---

## Understanding Agents and Tool Use

### What Are AI Agents?

Think of an AI agent as a problem-solving assistant that can:

1. **Reason**: Think through problems step by step
2. **Act**: Use tools to gather information or perform actions
3. **Observe**: Learn from the results of actions
4. **Remember**: Maintain conversation history and long-term knowledge

**Real-World Analogy**:
Imagine a research assistant helping you plan a trip:
- **User**: "I need to visit Paris next month. What's the weather like and should I book hotels now?"
- **Agent Reasoning**: "I need current weather data and hotel pricing info"
- **Agent Action**: Uses WeatherTool and HotelSearchTool
- **Agent Observation**: Receives results from both tools
- **Agent Response**: "Paris in November averages 8°C with rain. Hotels are 20% cheaper if booked now. I recommend booking soon."

### Key Concepts

#### 1. ReAct Loop (Reason + Act)
The agent operates in a loop:
```
1. THOUGHT: Reason about what to do next
2. ACTION: Execute a tool with specific input
3. OBSERVATION: Receive and process the tool's output
4. Repeat until final answer is reached
```

#### 2. Tools (Function Calling)
Tools are functions the agent can call:
- Each tool has a name and description
- The LLM decides which tool to use based on the task
- Tools execute and return observations
- Examples: Calculator, Search, Database Query, API Call

#### 3. Memory Systems

**Short-Term Memory (Scratchpad)**:
- Stores the current conversation and reasoning steps
- Lives only for the current task
- Like working memory in humans

**Long-Term Memory (Vector Database)**:
- Stores facts and experiences across conversations
- Can be retrieved based on semantic similarity
- Like episodic/semantic memory in humans

#### 4. Function Calling Flow
```
User Query: "What is 25 * 4 + 10?"
    ↓
Agent Thought: "I need to calculate 25 * 4 first"
    ↓
Agent Action: {"tool": "Calculator", "input": "25 * 4"}
    ↓
Tool Execution: Calculator returns "100"
    ↓
Agent Observation: "25 * 4 = 100"
    ↓
Agent Thought: "Now I need to add 10"
    ↓
Agent Action: {"tool": "Calculator", "input": "100 + 10"}
    ↓
Tool Execution: Calculator returns "110"
    ↓
Agent Observation: "100 + 10 = 110"
    ↓
Agent Thought: "I have the final answer"
    ↓
Agent Action: {"tool": "FinalAnswer", "input": "110"}
    ↓
Return: "110"
```

---

## Architecture Overview

### File Structure
```
src/
├── Interfaces/
│   ├── ITool.cs                    # Tool interface
│   ├── IChatModel.cs               # Chat LLM interface
│   └── IMemoryStore.cs             # Memory storage interface
├── Agents/
│   ├── Agent.cs                    # Main agent orchestrator
│   ├── AgentAction.cs              # Action data structure
│   ├── AgentThought.cs             # Thought data structure
│   ├── Memory/
│   │   ├── ShortTermMemory.cs     # Scratchpad implementation
│   │   ├── LongTermMemory.cs      # Vector DB memory
│   │   └── MemoryEntry.cs         # Memory data structure
│   └── Tools/
│       ├── CalculatorTool.cs      # Math calculations
│       ├── SearchTool.cs          # Web search (mock)
│       └── MemoryTool.cs          # Memory retrieval tool

tests/
└── UnitTests/
    └── Agents/
        ├── AgentTests.cs           # Agent behavior tests
        ├── ToolTests.cs            # Tool execution tests
        └── MemoryTests.cs          # Memory system tests
```

### Inheritance Pattern
```
ITool (interface in src/Interfaces/)
    ↓
CalculatorTool / SearchTool / MemoryTool (concrete implementations)


IChatModel<T> (interface in src/Interfaces/)
    ↓
(To be implemented by LLM providers)


IMemoryStore (interface in src/Interfaces/)
    ↓
ShortTermMemory / LongTermMemory (concrete implementations)
```

---

## Step-by-Step Implementation

### Phase 1: Core Agent Abstractions

#### Step 1: Define ITool Interface

**File**: `src/Interfaces/ITool.cs`

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for tools that an agent can use to perform actions.
/// </summary>
/// <remarks>
/// <para>
/// Tools are the "hands" of an agent - they allow the agent to interact with the world.
/// Each tool has a specific purpose and can be invoked with string input to produce string output.
/// </para>
/// <para><b>For Beginners:</b> Tools are like apps on your phone.
///
/// Think of it this way:
/// - Calculator app: Takes "25 * 4" → Returns "100"
/// - Weather app: Takes "Paris" → Returns "15°C, Sunny"
/// - Maps app: Takes "Coffee near me" → Returns "Starbucks, 0.3 miles"
///
/// The agent decides which "app" (tool) to use based on what it needs to accomplish.
/// </para>
/// </remarks>
public interface ITool
{
    /// <summary>
    /// Gets the name of the tool.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The name is used by the agent to identify and select the tool.
    /// Should be concise and descriptive (e.g., "Calculator", "Search", "Database").
    /// </para>
    /// <para><b>For Beginners:</b> This is like the app name on your phone's home screen.
    /// It tells the agent what this tool does at a glance.
    /// </para>
    /// </remarks>
    string Name { get; }

    /// <summary>
    /// Gets a description of what the tool does and how to use it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The description helps the LLM understand when to use this tool and what input format it expects.
    /// Should be clear and include examples if the input format is complex.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the app description in the app store.
    ///
    /// Good description example:
    /// "Useful for solving mathematical expressions. Input should be a simple string like '25 * 4' or '100 + 10'."
    ///
    /// This tells the agent:
    /// - WHEN to use it: for math problems
    /// - HOW to use it: provide a math expression as a string
    /// - WHAT format: examples like '25 * 4'
    /// </para>
    /// </remarks>
    string Description { get; }

    /// <summary>
    /// Executes the tool with the given input and returns the result.
    /// </summary>
    /// <param name="input">The input string for the tool to process.</param>
    /// <returns>The result of the tool execution as a string.</returns>
    /// <remarks>
    /// <para>
    /// This method performs the actual work of the tool. It should:
    /// - Parse the input string
    /// - Perform the tool's specific operation
    /// - Return the result as a string
    /// - Handle errors gracefully and return error messages as strings
    /// </para>
    /// <para><b>For Beginners:</b> This is the "Run" button of the app.
    ///
    /// Execution flow:
    /// 1. Agent calls: tool.Execute("25 * 4")
    /// 2. Tool processes: Evaluates the math expression
    /// 3. Tool returns: "100"
    /// 4. Agent receives: "100" and continues reasoning
    ///
    /// If something goes wrong, return a helpful error message:
    /// "Error: Invalid expression '25 &amp; 4'. Expected format: '25 * 4'"
    /// </para>
    /// </remarks>
    string Execute(string input);
}
```

#### Step 2: Define IChatModel Interface

**File**: `src/Interfaces/IChatModel.cs`

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for chat-based language models that can hold conversations.
/// </summary>
/// <remarks>
/// <para>
/// Chat models are LLMs optimized for instruction-following and conversation.
/// They take a prompt (which may include conversation history) and return a response.
/// </para>
/// <para><b>For Beginners:</b> This is like ChatGPT's API interface.
///
/// You send:
/// - System instructions (how the model should behave)
/// - User message (the current question)
/// - Conversation history (previous messages)
///
/// You receive:
/// - Model's response (the answer or action)
///
/// For agents, we use this to make the LLM think and decide what to do next.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for any internal calculations.</typeparam>
public interface IChatModel<T>
{
    /// <summary>
    /// Sends a chat message and receives a response.
    /// </summary>
    /// <param name="prompt">The complete prompt including instructions, history, and current query.</param>
    /// <returns>The model's response as a string.</returns>
    /// <remarks>
    /// <para>
    /// This method sends a prompt to the LLM and receives its response.
    /// For agents, the prompt typically includes:
    /// - Instructions on how to format responses (e.g., as JSON)
    /// - Available tools and their descriptions
    /// - Conversation/reasoning history
    /// - The current task or question
    /// </para>
    /// <para><b>For Beginners:</b> This is the core "think" operation.
    ///
    /// Example agent prompt:
    /// ```
    /// You are a helpful agent with access to these tools:
    /// - Calculator: For math expressions like "25 * 4"
    /// - Search: For finding information
    ///
    /// History:
    /// User: What is 25 * 4?
    ///
    /// Respond with JSON:
    /// {{"thought": "I need to calculate 25 * 4", "action": {{"tool": "Calculator", "input": "25 * 4"}}}}
    /// ```
    ///
    /// The model responds with the JSON, which the agent parses and executes.
    /// </para>
    /// </remarks>
    string Chat(string prompt);
}
```

#### Step 3: Define IMemoryStore Interface

**File**: `src/Interfaces/IMemoryStore.cs`

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for storing and retrieving agent memories.
/// </summary>
/// <remarks>
/// <para>
/// Memory stores allow agents to remember information across conversations.
/// They support adding new memories and retrieving relevant memories based on queries.
/// </para>
/// <para><b>For Beginners:</b> Memory stores are like the agent's brain.
///
/// Two types of memory:
/// 1. Short-term (working memory): Current conversation and reasoning
/// 2. Long-term (episodic/semantic memory): Facts and experiences from past conversations
///
/// Think of it like studying for an exam:
/// - Short-term: You remember the specific problem you're solving right now
/// - Long-term: You remember concepts from lectures weeks ago
/// </para>
/// </remarks>
public interface IMemoryStore
{
    /// <summary>
    /// Stores a new memory entry.
    /// </summary>
    /// <param name="content">The content to remember.</param>
    /// <param name="metadata">Optional metadata about the memory (e.g., timestamp, source).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is like writing in a notebook.
    ///
    /// Examples of what to store:
    /// - User preferences: "User prefers metric units"
    /// - Facts learned: "Paris is the capital of France"
    /// - Completed actions: "Booked flight to Paris on 2024-11-15"
    /// </para>
    /// </remarks>
    void Store(string content, Dictionary<string, string>? metadata = null);

    /// <summary>
    /// Retrieves relevant memories based on a query.
    /// </summary>
    /// <param name="query">The query to search for relevant memories.</param>
    /// <param name="topK">The maximum number of memories to retrieve.</param>
    /// <returns>A list of relevant memory contents, ranked by relevance.</returns>
    /// <remarks>
    /// <para>
    /// Uses semantic similarity to find memories related to the query.
    /// More relevant memories are ranked higher.
    /// </para>
    /// <para><b>For Beginners:</b> This is like searching your notebook for related notes.
    ///
    /// Example:
    /// - Query: "What does the user prefer for measurements?"
    /// - Retrieved: ["User prefers metric units", "User is from Europe"]
    ///
    /// The agent can use these memories to provide personalized responses.
    /// </para>
    /// </remarks>
    List<string> Retrieve(string query, int topK = 5);

    /// <summary>
    /// Clears all stored memories.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This resets the agent's memory, like starting fresh.
    /// Useful for testing or when starting a completely new conversation context.
    /// </para>
    /// </remarks>
    void Clear();
}
```

### Phase 2: Agent Data Structures

#### Step 4: Create AgentAction Class

**File**: `src/Agents/AgentAction.cs`

```csharp
namespace AiDotNet.Agents;

/// <summary>
/// Represents an action that an agent plans to execute.
/// </summary>
/// <remarks>
/// <para>
/// An action specifies which tool to use and what input to provide.
/// The special tool name "FinalAnswer" indicates the agent has completed the task.
/// </para>
/// <para><b>For Beginners:</b> This is the agent's decision about what to do next.
///
/// Think of it like a to-do item:
/// - Tool: Calculator (what to use)
/// - Input: "25 * 4" (how to use it)
///
/// Or for finishing:
/// - Tool: FinalAnswer
/// - Input: "The answer is 100"
/// </para>
/// </remarks>
public class AgentAction
{
    /// <summary>
    /// Gets or sets the name of the tool to execute.
    /// </summary>
    /// <remarks>
    /// Use "FinalAnswer" to indicate task completion and return a final result.
    /// </remarks>
    public string ToolName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the input to provide to the tool.
    /// </summary>
    public string ToolInput { get; set; } = string.Empty;

    /// <summary>
    /// Determines if this action represents a final answer.
    /// </summary>
    public bool IsFinalAnswer => ToolName.Equals("FinalAnswer", StringComparison.OrdinalIgnoreCase);
}
```

#### Step 5: Create AgentThought Class

**File**: `src/Agents/AgentThought.cs`

```csharp
namespace AiDotNet.Agents;

/// <summary>
/// Represents a reasoning step in the agent's thought process.
/// </summary>
/// <remarks>
/// <para>
/// A thought captures the agent's internal reasoning before taking an action.
/// It includes both the reasoning text and the planned action.
/// </para>
/// <para><b>For Beginners:</b> This is the agent "thinking out loud."
///
/// Example thought process:
/// ```
/// Thought: "I need to calculate 25 * 4 first, then add 10 to the result"
/// Action: Use Calculator with "25 * 4"
/// ```
///
/// This helps us understand WHY the agent chose to do what it did.
/// </para>
/// </remarks>
public class AgentThought
{
    /// <summary>
    /// Gets or sets the agent's reasoning text.
    /// </summary>
    public string Reasoning { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the action the agent plans to take based on this thought.
    /// </summary>
    public AgentAction Action { get; set; } = new AgentAction();
}
```

### Phase 3: Memory Implementation

#### Step 6: Create MemoryEntry Class

**File**: `src/Agents/Memory/MemoryEntry.cs`

```csharp
namespace AiDotNet.Agents.Memory;

/// <summary>
/// Represents a single entry in the agent's memory.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A memory entry is like a note card with information.
///
/// It contains:
/// - The actual content (what to remember)
/// - Metadata (when it was stored, where it came from, etc.)
/// - An ID for retrieval
/// </para>
/// </remarks>
public class MemoryEntry
{
    /// <summary>
    /// Gets or sets the unique identifier for this memory.
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the content of the memory.
    /// </summary>
    public string Content { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets optional metadata about the memory.
    /// </summary>
    /// <remarks>
    /// Common metadata keys:
    /// - "timestamp": When the memory was created
    /// - "source": Where the information came from
    /// - "importance": How important this memory is (0-1)
    /// </remarks>
    public Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();

    /// <summary>
    /// Gets or sets the timestamp when this memory was created.
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}
```

#### Step 7: Implement ShortTermMemory

**File**: `src/Agents/Memory/ShortTermMemory.cs`

```csharp
namespace AiDotNet.Agents.Memory;

/// <summary>
/// Implements a simple in-memory store for short-term agent memory.
/// </summary>
/// <remarks>
/// <para>
/// Short-term memory is like a scratchpad or working memory.
/// It stores the current conversation and recent reasoning steps.
/// This memory is volatile and resets when the agent session ends.
/// </para>
/// <para><b>For Beginners:</b> This is the agent's "working memory."
///
/// Think of it like RAM in a computer:
/// - Fast access
/// - Stores current task information
/// - Cleared when you restart
///
/// Example usage in an agent:
/// - Stores each thought and observation
/// - Keeps track of what tools were used
/// - Maintains conversation context
/// - Resets at the start of each new task
/// </para>
/// </remarks>
public class ShortTermMemory : IMemoryStore
{
    private readonly List<MemoryEntry> _memories = new List<MemoryEntry>();
    private readonly int _maxSize;

    /// <summary>
    /// Initializes a new instance of the <see cref="ShortTermMemory"/> class.
    /// </summary>
    /// <param name="maxSize">
    /// The maximum number of memories to retain. Defaults to 100.
    /// When exceeded, oldest memories are removed (FIFO).
    /// </param>
    /// <remarks>
    /// <para><b>Default Value (100):</b> Based on typical conversation lengths.
    ///
    /// Most agent tasks complete within 5-10 reasoning steps, with each step
    /// generating 1-2 memory entries. A limit of 100 provides ample buffer
    /// while preventing unbounded memory growth.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when maxSize is less than 1.</exception>
    public ShortTermMemory(int maxSize = 100)
    {
        if (maxSize < 1)
        {
            throw new ArgumentException("Maximum size must be at least 1", nameof(maxSize));
        }

        _maxSize = maxSize;
    }

    /// <inheritdoc/>
    public void Store(string content, Dictionary<string, string>? metadata = null)
    {
        var entry = new MemoryEntry
        {
            Content = content,
            Metadata = metadata ?? new Dictionary<string, string>()
        };

        _memories.Add(entry);

        // Remove oldest memories if we exceed max size
        while (_memories.Count > _maxSize)
        {
            _memories.RemoveAt(0);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// For short-term memory, this performs simple string matching.
    /// Returns the most recent memories that contain the query string.
    /// </para>
    /// <para><b>For Beginners:</b> This is a basic search through recent memories.
    ///
    /// More sophisticated implementations might use:
    /// - Semantic similarity (embeddings)
    /// - Relevance scoring
    /// - Recency weighting
    ///
    /// For now, we keep it simple: find memories containing the search terms.
    /// </para>
    /// </remarks>
    public List<string> Retrieve(string query, int topK = 5)
    {
        // Simple keyword-based retrieval for short-term memory
        var matchingMemories = _memories
            .Where(m => m.Content.Contains(query, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(m => m.CreatedAt)
            .Take(topK)
            .Select(m => m.Content)
            .ToList();

        return matchingMemories;
    }

    /// <inheritdoc/>
    public void Clear()
    {
        _memories.Clear();
    }

    /// <summary>
    /// Gets all memories as a formatted string for inclusion in prompts.
    /// </summary>
    /// <returns>A string containing all memories, one per line.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This formats the memory for the agent to read.
    ///
    /// Example output:
    /// ```
    /// Thought: I need to calculate 25 * 4
    /// Action: Calculator("25 * 4")
    /// Observation: 100
    /// Thought: Now I need to add 10
    /// ```
    ///
    /// This formatted history is included in the prompt to the LLM,
    /// so the agent can see what it has already done.
    /// </para>
    /// </remarks>
    public string GetFormattedHistory()
    {
        return string.Join(Environment.NewLine, _memories.Select(m => m.Content));
    }
}
```

### Phase 4: Tool Implementations

#### Step 8: Implement CalculatorTool

**File**: `src/Agents/Tools/CalculatorTool.cs`

```csharp
using System.Data;
using AiDotNet.Interfaces;

namespace AiDotNet.Agents.Tools;

/// <summary>
/// A tool that evaluates mathematical expressions.
/// </summary>
/// <remarks>
/// <para>
/// Uses System.Data.DataTable.Compute() to safely evaluate math expressions.
/// Supports basic arithmetic operations: +, -, *, /, (), etc.
/// </para>
/// <para><b>For Beginners:</b> This is like a calculator app.
///
/// What it can do:
/// - Basic math: "25 * 4" → "100"
/// - Complex expressions: "(100 + 50) / 2" → "75"
/// - Decimal numbers: "3.14 * 2" → "6.28"
///
/// What it cannot do:
/// - Advanced functions: "sin(45)" → Error
/// - Variables: "x + 5" → Error
/// - Non-numeric input: "hello" → Error
///
/// For simple agent tasks, this is sufficient.
/// </para>
/// </remarks>
public class CalculatorTool : ITool
{
    /// <inheritdoc/>
    public string Name => "Calculator";

    /// <inheritdoc/>
    public string Description =>
        "Useful for solving mathematical expressions. Input should be a simple arithmetic expression like '25 * 4', '100 + 10', or '(50 - 20) / 3'. Supports +, -, *, /, and parentheses.";

    /// <inheritdoc/>
    public string Execute(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            return "Error: Calculator input cannot be empty.";
        }

        try
        {
            // Use DataTable.Compute for safe expression evaluation
            var table = new DataTable();
            var result = table.Compute(input, string.Empty);
            return result.ToString() ?? "Error: Unable to compute result.";
        }
        catch (Exception ex)
        {
            return $"Error: Invalid mathematical expression '{input}'. {ex.Message}";
        }
    }
}
```

#### Step 9: Implement SearchTool (Mock)

**File**: `src/Agents/Tools/SearchTool.cs`

```csharp
using AiDotNet.Interfaces;

namespace AiDotNet.Agents.Tools;

/// <summary>
/// A mock tool that simulates web search functionality.
/// </summary>
/// <remarks>
/// <para>
/// This is a placeholder implementation that returns hardcoded results.
/// In a production system, this would integrate with a real search API
/// (e.g., Google Search, Bing Search, or a custom search index).
/// </para>
/// <para><b>For Beginners:</b> This is a fake search engine for testing.
///
/// Why use a mock?
/// - No API keys or external dependencies needed
/// - Fast and reliable for testing
/// - Predictable results make tests easier
///
/// In a real agent system, you would replace this with:
/// - Google Custom Search API
/// - Bing Search API
/// - DuckDuckGo API
/// - Your own search index
///
/// For now, it returns canned responses based on keywords.
/// </para>
/// </remarks>
public class SearchTool : ITool
{
    private readonly Dictionary<string, string> _mockResponses;

    /// <inheritdoc/>
    public string Name => "Search";

    /// <inheritdoc/>
    public string Description =>
        "Useful for finding current information on the web. Input should be a search query like 'weather in Paris' or 'capital of France'.";

    /// <summary>
    /// Initializes a new instance of the <see cref="SearchTool"/> class.
    /// </summary>
    public SearchTool()
    {
        // Initialize mock responses for common queries
        _mockResponses = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            { "weather", "The current weather is partly cloudy with a temperature of 20°C (68°F)." },
            { "paris", "Paris is the capital and most populous city of France, with an estimated population of 2,165,423 residents." },
            { "moon landing", "The first man on the moon was Neil Armstrong on July 20, 1969, as part of the Apollo 11 mission." },
            { "python", "Python is a high-level, interpreted programming language known for its simplicity and readability." },
            { "ai", "Artificial Intelligence (AI) refers to computer systems designed to perform tasks that typically require human intelligence." }
        };
    }

    /// <inheritdoc/>
    public string Execute(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            return "Error: Search query cannot be empty.";
        }

        // Check if any keyword matches the input
        foreach (var kvp in _mockResponses)
        {
            if (input.Contains(kvp.Key, StringComparison.OrdinalIgnoreCase))
            {
                return kvp.Value;
            }
        }

        // Default response for unknown queries
        return $"Search results for '{input}': No specific information available. This is a mock search tool.";
    }
}
```

### Phase 5: Agent Implementation

#### Step 10: Implement the Agent Class

**File**: `src/Agents/Agent.cs`

```csharp
using System.Text.Json;
using AiDotNet.Interfaces;
using AiDotNet.Agents.Memory;

namespace AiDotNet.Agents;

/// <summary>
/// Implements a ReAct-style agent that can reason and use tools to solve problems.
/// </summary>
/// <remarks>
/// <para>
/// The agent operates in a thought-action-observation loop:
/// 1. Think about what to do next
/// 2. Execute an action (use a tool)
/// 3. Observe the result
/// 4. Repeat until the task is complete
/// </para>
/// <para><b>For Beginners:</b> The agent is the "brain" that orchestrates everything.
///
/// How it works:
/// 1. You ask a question: "What is 25 * 4 + 10?"
/// 2. Agent thinks: "I need to calculate 25 * 4 first"
/// 3. Agent acts: Uses Calculator tool with "25 * 4"
/// 4. Agent observes: Gets "100" as the result
/// 5. Agent thinks: "Now I need to add 10"
/// 6. Agent acts: Uses Calculator tool with "100 + 10"
/// 7. Agent observes: Gets "110" as the result
/// 8. Agent thinks: "I have the answer"
/// 9. Agent returns: "110"
///
/// This loop continues until the agent decides it has the final answer.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class Agent<T>
{
    private readonly IChatModel<T> _llm;
    private readonly Dictionary<string, ITool> _tools;
    private readonly ShortTermMemory _memory;
    private readonly int _maxIterations;

    /// <summary>
    /// Initializes a new instance of the <see cref="Agent{T}"/> class.
    /// </summary>
    /// <param name="llm">The language model to use for reasoning.</param>
    /// <param name="tools">The list of tools available to the agent.</param>
    /// <param name="maxIterations">
    /// The maximum number of reasoning iterations before giving up. Defaults to 5.
    /// </param>
    /// <remarks>
    /// <para><b>Default Value (maxIterations = 5):</b> Based on typical agent task complexity.
    ///
    /// Research on ReAct agents shows most tasks complete in 2-4 iterations.
    /// A limit of 5 provides buffer for complex tasks while preventing infinite loops.
    /// If your tasks consistently hit this limit, the agent may need:
    /// - Better tool descriptions
    /// - More specific prompts
    /// - Additional tools
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when llm or tools is null.</exception>
    /// <exception cref="ArgumentException">Thrown when maxIterations is less than 1.</exception>
    public Agent(IChatModel<T> llm, List<ITool> tools, int maxIterations = 5)
    {
        _llm = llm ?? throw new ArgumentNullException(nameof(llm));
        if (tools == null) throw new ArgumentNullException(nameof(tools));
        if (maxIterations < 1) throw new ArgumentException("Max iterations must be at least 1", nameof(maxIterations));

        _tools = tools.ToDictionary(t => t.Name, t => t, StringComparer.OrdinalIgnoreCase);
        _memory = new ShortTermMemory();
        _maxIterations = maxIterations;
    }

    /// <summary>
    /// Runs the agent to solve the given query.
    /// </summary>
    /// <param name="query">The user's question or task.</param>
    /// <returns>The agent's final answer.</returns>
    /// <remarks>
    /// <para>
    /// The agent will iterate through thought-action-observation cycles until:
    /// 1. It produces a final answer, or
    /// 2. It reaches the maximum iteration limit
    /// </para>
    /// <para><b>For Beginners:</b> This is the main entry point to use the agent.
    ///
    /// Example usage:
    /// ```csharp
    /// var agent = new Agent<double>(llm, tools);
    /// var answer = agent.Run("What is 25 * 4 + 10?");
    /// Console.WriteLine(answer); // Output: "110"
    /// ```
    ///
    /// What happens inside:
    /// - Agent maintains a scratchpad of all thoughts and observations
    /// - Each iteration, it sends the full history + query to the LLM
    /// - LLM responds with next thought and action
    /// - Agent executes the action and adds observation to scratchpad
    /// - Continues until LLM returns "FinalAnswer"
    /// </para>
    /// </remarks>
    public string Run(string query)
    {
        _memory.Clear(); // Start fresh for each query

        for (int i = 0; i < _maxIterations; i++)
        {
            // Build the prompt with current history
            string prompt = BuildPrompt(query);

            // Get the LLM's response
            string llmResponse = _llm.Chat(prompt);

            // Parse the response
            AgentThought? thought = ParseLLMResponse(llmResponse);
            if (thought == null)
            {
                return $"Error: Unable to parse LLM response: {llmResponse}";
            }

            // Store the thought in memory
            _memory.Store($"Thought: {thought.Reasoning}");

            // Check if this is the final answer
            if (thought.Action.IsFinalAnswer)
            {
                return thought.Action.ToolInput;
            }

            // Execute the action
            string observation = ExecuteAction(thought.Action);
            _memory.Store($"Action: {thought.Action.ToolName}(\"{thought.Action.ToolInput}\")");
            _memory.Store($"Observation: {observation}");
        }

        return $"Error: Agent reached maximum iterations ({_maxIterations}) without completing the task.";
    }

    /// <summary>
    /// Builds the prompt for the LLM including instructions, tools, and history.
    /// </summary>
    private string BuildPrompt(string query)
    {
        var toolDescriptions = string.Join("\n", _tools.Values.Select(t =>
            $"- {t.Name}: {t.Description}"));

        var history = _memory.GetFormattedHistory();

        return $@"You are a helpful agent with access to the following tools:

{toolDescriptions}

To use a tool, you must respond with a JSON object in this exact format:
{{
  ""thought"": ""your reasoning about what to do next"",
  ""action"": {{
    ""tool"": ""ToolName"",
    ""input"": ""input for the tool""
  }}
}}

When you have the final answer, use the special ""FinalAnswer"" tool:
{{
  ""thought"": ""I now have the final answer"",
  ""action"": {{
    ""tool"": ""FinalAnswer"",
    ""input"": ""your final answer here""
  }}
}}

History:
{history}

User Query: {query}

Respond with your next thought and action in JSON format:";
    }

    /// <summary>
    /// Parses the LLM response into an AgentThought object.
    /// </summary>
    private AgentThought? ParseLLMResponse(string response)
    {
        try
        {
            // Try to extract JSON from the response (may be wrapped in markdown)
            string json = ExtractJson(response);

            var parsed = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(json);
            if (parsed == null) return null;

            var thought = new AgentThought
            {
                Reasoning = parsed.ContainsKey("thought") ? parsed["thought"].GetString() ?? string.Empty : string.Empty
            };

            if (parsed.ContainsKey("action"))
            {
                var actionElement = parsed["action"];
                thought.Action = new AgentAction
                {
                    ToolName = actionElement.TryGetProperty("tool", out var toolProp) ? toolProp.GetString() ?? string.Empty : string.Empty,
                    ToolInput = actionElement.TryGetProperty("input", out var inputProp) ? inputProp.GetString() ?? string.Empty : string.Empty
                };
            }

            return thought;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Extracts JSON from response that may be wrapped in markdown code blocks.
    /// </summary>
    private string ExtractJson(string response)
    {
        // Remove markdown code blocks if present
        if (response.Contains("```json"))
        {
            int start = response.IndexOf("```json") + 7;
            int end = response.IndexOf("```", start);
            if (end > start)
            {
                return response.Substring(start, end - start).Trim();
            }
        }
        else if (response.Contains("```"))
        {
            int start = response.IndexOf("```") + 3;
            int end = response.IndexOf("```", start);
            if (end > start)
            {
                return response.Substring(start, end - start).Trim();
            }
        }

        return response.Trim();
    }

    /// <summary>
    /// Executes an action by calling the appropriate tool.
    /// </summary>
    private string ExecuteAction(AgentAction action)
    {
        if (!_tools.ContainsKey(action.ToolName))
        {
            return $"Error: Unknown tool '{action.ToolName}'. Available tools: {string.Join(", ", _tools.Keys)}";
        }

        var tool = _tools[action.ToolName];
        return tool.Execute(action.ToolInput);
    }
}
```

---

## Phase 6: Testing

### Step 11: Unit Tests for Tools

**File**: `tests/UnitTests/Agents/ToolTests.cs`

```csharp
using Xunit;
using AiDotNet.Agents.Tools;

namespace AiDotNet.Tests.UnitTests.Agents;

/// <summary>
/// Tests for agent tool implementations.
/// </summary>
public class ToolTests
{
    [Fact]
    public void CalculatorTool_BasicArithmetic_ReturnsCorrectResult()
    {
        // Arrange
        var calculator = new CalculatorTool();

        // Act
        var result = calculator.Execute("25 * 4");

        // Assert
        Assert.Equal("100", result);
    }

    [Fact]
    public void CalculatorTool_ComplexExpression_ReturnsCorrectResult()
    {
        // Arrange
        var calculator = new CalculatorTool();

        // Act
        var result = calculator.Execute("(100 + 50) / 2");

        // Assert
        Assert.Equal("75", result);
    }

    [Fact]
    public void CalculatorTool_InvalidExpression_ReturnsError()
    {
        // Arrange
        var calculator = new CalculatorTool();

        // Act
        var result = calculator.Execute("invalid expression");

        // Assert
        Assert.StartsWith("Error:", result);
    }

    [Fact]
    public void CalculatorTool_EmptyInput_ReturnsError()
    {
        // Arrange
        var calculator = new CalculatorTool();

        // Act
        var result = calculator.Execute(string.Empty);

        // Assert
        Assert.StartsWith("Error:", result);
    }

    [Fact]
    public void SearchTool_KnownKeyword_ReturnsExpectedResult()
    {
        // Arrange
        var search = new SearchTool();

        // Act
        var result = search.Execute("moon landing");

        // Assert
        Assert.Contains("Neil Armstrong", result);
    }

    [Fact]
    public void SearchTool_UnknownQuery_ReturnsDefaultMessage()
    {
        // Arrange
        var search = new SearchTool();

        // Act
        var result = search.Execute("unknown query xyz");

        // Assert
        Assert.Contains("mock search tool", result);
    }

    [Fact]
    public void SearchTool_EmptyInput_ReturnsError()
    {
        // Arrange
        var search = new SearchTool();

        // Act
        var result = search.Execute(string.Empty);

        // Assert
        Assert.StartsWith("Error:", result);
    }
}
```

### Step 12: Unit Tests for Memory

**File**: `tests/UnitTests/Agents/MemoryTests.cs`

```csharp
using Xunit;
using AiDotNet.Agents.Memory;

namespace AiDotNet.Tests.UnitTests.Agents;

/// <summary>
/// Tests for agent memory implementations.
/// </summary>
public class MemoryTests
{
    [Fact]
    public void ShortTermMemory_Store_AddsMemory()
    {
        // Arrange
        var memory = new ShortTermMemory();

        // Act
        memory.Store("Test memory content");

        // Assert
        var history = memory.GetFormattedHistory();
        Assert.Contains("Test memory content", history);
    }

    [Fact]
    public void ShortTermMemory_Retrieve_FindsMatchingMemories()
    {
        // Arrange
        var memory = new ShortTermMemory();
        memory.Store("The weather is sunny");
        memory.Store("The temperature is 25 degrees");
        memory.Store("Paris is beautiful");

        // Act
        var results = memory.Retrieve("weather");

        // Assert
        Assert.Single(results);
        Assert.Contains("sunny", results[0]);
    }

    [Fact]
    public void ShortTermMemory_Clear_RemovesAllMemories()
    {
        // Arrange
        var memory = new ShortTermMemory();
        memory.Store("Memory 1");
        memory.Store("Memory 2");

        // Act
        memory.Clear();

        // Assert
        var history = memory.GetFormattedHistory();
        Assert.Empty(history);
    }

    [Fact]
    public void ShortTermMemory_MaxSize_EnforcesLimit()
    {
        // Arrange
        var memory = new ShortTermMemory(maxSize: 3);

        // Act
        memory.Store("Memory 1");
        memory.Store("Memory 2");
        memory.Store("Memory 3");
        memory.Store("Memory 4");

        // Assert
        var history = memory.GetFormattedHistory();
        Assert.DoesNotContain("Memory 1", history); // First memory should be removed
        Assert.Contains("Memory 4", history); // Latest memory should exist
    }

    [Fact]
    public void ShortTermMemory_Constructor_InvalidMaxSize_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => new ShortTermMemory(maxSize: 0));
    }
}
```

### Step 13: Unit Tests for Agent

**File**: `tests/UnitTests/Agents/AgentTests.cs`

```csharp
using Xunit;
using Moq;
using AiDotNet.Agents;
using AiDotNet.Agents.Tools;
using AiDotNet.Interfaces;

namespace AiDotNet.Tests.UnitTests.Agents;

/// <summary>
/// Tests for the Agent class.
/// </summary>
public class AgentTests
{
    [Fact]
    public void Agent_SimpleCalculation_ReturnsCorrectAnswer()
    {
        // Arrange
        var mockLLM = new Mock<IChatModel<double>>();

        // First call: Agent decides to use calculator
        mockLLM.SetupSequence(m => m.Chat(It.IsAny<string>()))
            .Returns(@"{
                ""thought"": ""I need to calculate 25 * 4"",
                ""action"": {
                    ""tool"": ""Calculator"",
                    ""input"": ""25 * 4""
                }
            }")
            // Second call: Agent returns final answer
            .Returns(@"{
                ""thought"": ""I have the result"",
                ""action"": {
                    ""tool"": ""FinalAnswer"",
                    ""input"": ""100""
                }
            }");

        var tools = new List<ITool> { new CalculatorTool() };
        var agent = new Agent<double>(mockLLM.Object, tools);

        // Act
        var result = agent.Run("What is 25 * 4?");

        // Assert
        Assert.Equal("100", result);
        mockLLM.Verify(m => m.Chat(It.IsAny<string>()), Times.Exactly(2));
    }

    [Fact]
    public void Agent_MultiStepCalculation_ExecutesCorrectly()
    {
        // Arrange
        var mockLLM = new Mock<IChatModel<double>>();

        mockLLM.SetupSequence(m => m.Chat(It.IsAny<string>()))
            // Step 1: Calculate 25 * 4
            .Returns(@"{
                ""thought"": ""First I need to calculate 25 * 4"",
                ""action"": {
                    ""tool"": ""Calculator"",
                    ""input"": ""25 * 4""
                }
            }")
            // Step 2: Add 10 to result
            .Returns(@"{
                ""thought"": ""Now I need to add 10 to 100"",
                ""action"": {
                    ""tool"": ""Calculator"",
                    ""input"": ""100 + 10""
                }
            }")
            // Step 3: Return final answer
            .Returns(@"{
                ""thought"": ""I have the final answer"",
                ""action"": {
                    ""tool"": ""FinalAnswer"",
                    ""input"": ""110""
                }
            }");

        var tools = new List<ITool> { new CalculatorTool() };
        var agent = new Agent<double>(mockLLM.Object, tools);

        // Act
        var result = agent.Run("What is 25 * 4 + 10?");

        // Assert
        Assert.Equal("110", result);
        mockLLM.Verify(m => m.Chat(It.IsAny<string>()), Times.Exactly(3));
    }

    [Fact]
    public void Agent_UnknownTool_ReturnsError()
    {
        // Arrange
        var mockLLM = new Mock<IChatModel<double>>();

        mockLLM.Setup(m => m.Chat(It.IsAny<string>()))
            .Returns(@"{
                ""thought"": ""I'll use an unknown tool"",
                ""action"": {
                    ""tool"": ""UnknownTool"",
                    ""input"": ""test""
                }
            }");

        var tools = new List<ITool> { new CalculatorTool() };
        var agent = new Agent<double>(mockLLM.Object, tools);

        // Act
        var result = agent.Run("Test query");

        // Assert
        Assert.Contains("Error:", result);
        Assert.Contains("UnknownTool", result);
    }

    [Fact]
    public void Agent_MaxIterationsReached_ReturnsError()
    {
        // Arrange
        var mockLLM = new Mock<IChatModel<double>>();

        // Always return a non-final action
        mockLLM.Setup(m => m.Chat(It.IsAny<string>()))
            .Returns(@"{
                ""thought"": ""Still thinking..."",
                ""action"": {
                    ""tool"": ""Calculator"",
                    ""input"": ""1 + 1""
                }
            }");

        var tools = new List<ITool> { new CalculatorTool() };
        var agent = new Agent<double>(mockLLM.Object, tools, maxIterations: 3);

        // Act
        var result = agent.Run("Test query");

        // Assert
        Assert.Contains("Error:", result);
        Assert.Contains("maximum iterations", result);
        mockLLM.Verify(m => m.Chat(It.IsAny<string>()), Times.Exactly(3));
    }

    [Fact]
    public void Agent_Constructor_NullLLM_ThrowsException()
    {
        // Arrange
        var tools = new List<ITool> { new CalculatorTool() };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new Agent<double>(null!, tools));
    }

    [Fact]
    public void Agent_Constructor_NullTools_ThrowsException()
    {
        // Arrange
        var mockLLM = new Mock<IChatModel<double>>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new Agent<double>(mockLLM.Object, null!));
    }

    [Fact]
    public void Agent_Constructor_InvalidMaxIterations_ThrowsException()
    {
        // Arrange
        var mockLLM = new Mock<IChatModel<double>>();
        var tools = new List<ITool> { new CalculatorTool() };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new Agent<double>(mockLLM.Object, tools, maxIterations: 0));
    }
}
```

---

## Usage Examples

### Example 1: Simple Math Agent

```csharp
using AiDotNet.Agents;
using AiDotNet.Agents.Tools;

// Create tools
var tools = new List<ITool>
{
    new CalculatorTool(),
    new SearchTool()
};

// Create agent (assuming you have an LLM implementation)
var agent = new Agent<double>(llm, tools);

// Run queries
var answer1 = agent.Run("What is 25 * 4?");
Console.WriteLine(answer1); // Output: 100

var answer2 = agent.Run("Calculate (100 + 50) / 2");
Console.WriteLine(answer2); // Output: 75
```

### Example 2: Multi-Tool Agent

```csharp
// Agent can use multiple tools in sequence
var answer = agent.Run("Search for the first man on the moon, then calculate his birth year plus 50");

// Agent reasoning:
// 1. Uses SearchTool: "first man on the moon" → "Neil Armstrong, 1969"
// 2. Uses SearchTool: "Neil Armstrong birth year" → "1930"
// 3. Uses CalculatorTool: "1930 + 50" → "1980"
// 4. Returns: "1980"
```

---

## Key Concepts for Testing

### 1. Mocking the LLM
Always mock `IChatModel<T>` in tests to avoid:
- External API calls
- Non-deterministic behavior
- Slow tests

### 2. Testing Tool Execution
Test each tool independently:
- Valid inputs return expected results
- Invalid inputs return error messages
- Edge cases are handled gracefully

### 3. Testing Agent Loop
Test the agent's ability to:
- Execute multiple iterations
- Use tools correctly
- Return final answers
- Handle errors and unknown tools

---

## Common Pitfalls and Solutions

### Pitfall 1: Infinite Loops
**Problem**: Agent never returns FinalAnswer
**Solution**: Always set `maxIterations` limit (default: 5)

### Pitfall 2: Tool Not Found
**Problem**: Agent tries to use a tool that wasn't registered
**Solution**: Validate tool names in error messages and LLM prompt

### Pitfall 3: JSON Parsing Failures
**Problem**: LLM returns malformed JSON
**Solution**: Robust parsing with try-catch and markdown extraction

### Pitfall 4: Memory Overflow
**Problem**: Short-term memory grows unbounded
**Solution**: Set `maxSize` limit and remove oldest entries (FIFO)

---

## Next Steps and Extensions

### 1. Long-Term Memory with Vector DB
Implement `LongTermMemory` using embeddings:
```csharp
public class LongTermMemory : IMemoryStore
{
    private IEmbeddingModel<T> _embedder;
    private IDocumentStore _vectorDb;

    // Store memories with embeddings for semantic search
}
```

### 2. Custom Tools
Create domain-specific tools:
- DatabaseTool: Query SQL databases
- APITool: Call REST APIs
- FileTool: Read/write files
- EmailTool: Send emails

### 3. Multi-Agent Systems
Coordinate multiple agents:
- ManagerAgent: Delegates tasks to specialist agents
- ResearchAgent: Focuses on information gathering
- ActionAgent: Focuses on executing tasks

### 4. Streaming Responses
Update Agent to stream thoughts in real-time:
```csharp
public async IAsyncEnumerable<string> RunStreaming(string query)
{
    // Yield thoughts and observations as they happen
}
```

---

## Testing Checklist

- [ ] All tool classes implement ITool correctly
- [ ] CalculatorTool handles basic arithmetic
- [ ] CalculatorTool returns errors for invalid input
- [ ] SearchTool returns mock results for known queries
- [ ] ShortTermMemory stores and retrieves memories
- [ ] ShortTermMemory enforces max size limit
- [ ] Agent executes single-step tasks correctly
- [ ] Agent executes multi-step tasks correctly
- [ ] Agent handles unknown tools gracefully
- [ ] Agent respects maxIterations limit
- [ ] Agent parses JSON responses correctly
- [ ] All tests pass with at least 80% code coverage

---

## Summary

You have implemented a complete agent framework with:
- Tool abstraction (ITool interface)
- ReAct-style agent loop (thought-action-observation)
- Short-term memory (scratchpad)
- Two example tools (Calculator, Search)
- Comprehensive unit tests

This foundation supports building sophisticated AI agents that can:
- Reason through complex multi-step problems
- Use external tools to gather information
- Maintain conversation context
- Handle errors gracefully

The next phase would add long-term memory, custom tools, and production-ready LLM integration.
