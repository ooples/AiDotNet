namespace AiDotNet.Interfaces;

/// <summary>
/// Defines an agent that can reason and use tools to solve complex problems.
/// An agent combines a language model with a set of tools to autonomously work toward a goal.
/// </summary>
/// <typeparam name="T">The numeric type used for model parameters and operations (e.g., double, float).</typeparam>
/// <remarks>
/// For Beginners:
/// An agent is like an intelligent assistant that can think through problems and use tools to solve them.
/// Imagine asking someone to "find out the weather in Paris and calculate the temperature in Fahrenheit."
/// A human would:
/// 1. Think: "I need to search for Paris weather" (Thought)
/// 2. Act: Search for Paris weather (Action using Search tool)
/// 3. Observe: See the result is "15°C" (Observation)
/// 4. Think: "Now I need to convert Celsius to Fahrenheit" (Thought)
/// 5. Act: Calculate (15 * 9/5) + 32 (Action using Calculator tool)
/// 6. Observe: Get "59°F" (Observation)
/// 7. Think: "I have the answer" (Thought)
/// 8. Respond: "The weather in Paris is 59°F"
///
/// An AI agent works the same way, using a cycle of reasoning (via a language model) and tool use
/// to break down complex tasks into manageable steps.
///
/// This is called the ReAct pattern: Reasoning + Acting.
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("Agent")]
public interface IAgent<T>
{
    /// <summary>
    /// Gets the chat model used by the agent for reasoning and decision-making.
    /// </summary>
    /// <value>An instance of <see cref="IChatModel{T}"/> that generates responses and plans actions.</value>
    IChatModel<T> ChatModel { get; }

    /// <summary>
    /// Gets the collection of tools available to the agent.
    /// </summary>
    /// <value>A read-only list of <see cref="ITool"/> instances that the agent can use to perform actions.</value>
    IReadOnlyList<ITool> Tools { get; }

    /// <summary>
    /// Executes the agent's reasoning and action loop to answer a query or complete a task.
    /// </summary>
    /// <param name="query">The question or task for the agent to solve.</param>
    /// <param name="maxIterations">The maximum number of thought-action-observation cycles to prevent infinite loops.
    /// Default is 5.</param>
    /// <returns>A task that represents the asynchronous operation. The task result contains
    /// the agent's final answer or response.</returns>
    /// <remarks>
    /// For Beginners:
    /// The Run method is where the magic happens. When you give the agent a query like
    /// "What is the square root of 144 plus 5?", it will:
    ///
    /// 1. Think about what needs to be done
    /// 2. Decide which tool to use (Calculator)
    /// 3. Execute the tool with the right input ("sqrt(144)")
    /// 4. Observe the result ("12")
    /// 5. Think about the next step
    /// 6. Use the calculator again ("12 + 5")
    /// 7. Observe the result ("17")
    /// 8. Conclude and provide the final answer
    ///
    /// The maxIterations parameter is a safety feature. It prevents the agent from getting stuck
    /// in an endless loop if something goes wrong. If the agent can't solve the problem within
    /// the maximum iterations, it will stop and return what it has learned so far.
    ///
    /// Implementation notes for developers:
    /// - Maintain a "scratchpad" or conversation history throughout the iterations
    /// - Format prompts to instruct the LLM to respond in a structured format (e.g., JSON)
    /// - Parse the LLM's response to extract thoughts and actions
    /// - Execute the specified tool and capture the observation
    /// - Continue until the LLM indicates it has a final answer or max iterations is reached
    /// - Handle errors gracefully and provide meaningful error messages
    /// </remarks>
    Task<string> RunAsync(string query, int maxIterations = 5);

    /// <summary>
    /// Gets the reasoning history (scratchpad) from the agent's most recent execution.
    /// This includes all thoughts, actions, and observations from the reasoning process.
    /// </summary>
    /// <value>A string containing the complete reasoning trace, useful for debugging and understanding
    /// how the agent arrived at its answer.</value>
    /// <remarks>
    /// For Beginners:
    /// The scratchpad is like showing your work in math class. It's a record of everything
    /// the agent thought and did while solving the problem. This is incredibly useful for:
    /// - Understanding how the agent solved the problem
    /// - Debugging when the agent gives an unexpected answer
    /// - Learning how to improve prompts and tools
    /// - Transparency and explainability in AI systems
    ///
    /// Example scratchpad content:
    /// <code>
    /// Iteration 1:
    /// Thought: I need to calculate the square root of 144 first.
    /// Action: Calculator
    /// Action Input: sqrt(144)
    /// Observation: 12
    ///
    /// Iteration 2:
    /// Thought: Now I need to add 5 to the result.
    /// Action: Calculator
    /// Action Input: 12 + 5
    /// Observation: 17
    ///
    /// Iteration 3:
    /// Thought: I have the final answer.
    /// Final Answer: 17
    /// </code>
    /// </remarks>
    string Scratchpad { get; }
}
