namespace AiDotNet.Enums;

/// <summary>
/// Represents different types of prompt templates for language model interactions.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Prompt templates are pre-structured formats for communicating with language models.
///
/// Think of templates like email templates:
/// - Instead of writing each email from scratch, you use a template with placeholders
/// - You fill in the specific details (name, date, etc.) for each email
/// - The overall structure and tone remain consistent
///
/// Prompt templates work the same way:
/// - You create a template with placeholders for variable content
/// - You fill in specific values when you need to use the template
/// - The language model receives a well-structured, consistent prompt
///
/// Different template types serve different purposes, from simple variable substitution to
/// complex multi-turn conversations with examples and tool usage.
/// </para>
/// </remarks>
public enum PromptTemplateType
{
    /// <summary>
    /// Simple template with variable substitution using placeholders.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The simplest type of template where you replace placeholders with values.
    ///
    /// Example:
    /// Template: "Translate the following {source_language} text to {target_language}: {text}"
    /// Variables: source_language="English", target_language="Spanish", text="Hello world"
    /// Result: "Translate the following English text to Spanish: Hello world"
    ///
    /// Use this when:
    /// - You have a simple prompt structure
    /// - You need to insert specific values into predetermined positions
    /// - You don't need examples or complex formatting
    /// </para>
    /// </remarks>
    Simple,

    /// <summary>
    /// Template with few-shot learning examples to guide the model's output.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Few-shot templates include examples to show the model what you want.
    ///
    /// Think of it like teaching by example:
    /// - Instead of just telling the model what to do, you show it examples
    /// - The model learns the pattern from your examples
    /// - Then applies that pattern to new inputs
    ///
    /// Example:
    /// Template with examples:
    /// "Classify sentiment as positive or negative.
    ///
    /// Example: 'This movie was amazing!' → Positive
    /// Example: 'I hated the food.' → Negative
    /// Example: 'Best day ever!' → Positive
    ///
    /// Now classify: '{text}'"
    ///
    /// The examples help the model understand exactly what format and type of response you want.
    ///
    /// Use this when:
    /// - The task is complex or ambiguous
    /// - You want to control the output format
    /// - A few examples significantly improve results
    /// </para>
    /// </remarks>
    FewShot,

    /// <summary>
    /// Template for structured message-based conversations with roles (system, user, assistant).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Chat templates structure conversations with different roles.
    ///
    /// Modern language models understand different roles:
    /// - System: Instructions for how the assistant should behave
    /// - User: The person asking questions
    /// - Assistant: The AI's responses
    ///
    /// Example:
    /// System: "You are a helpful math tutor for elementary school students."
    /// User: "What is 25 + 17?"
    /// Assistant: "Great question! Let's solve this step by step..."
    ///
    /// This structure helps the model:
    /// - Understand its role and constraints (system message)
    /// - Distinguish between user questions and its own responses
    /// - Maintain consistent behavior across a conversation
    ///
    /// Use this when:
    /// - Building chatbots or conversational AI
    /// - You need to set specific behavior guidelines
    /// - Working with multi-turn conversations
    /// </para>
    /// </remarks>
    Chat,

    /// <summary>
    /// Template optimized through automated prompt engineering techniques.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Optimized templates are automatically refined to get better results.
    ///
    /// Think of it like A/B testing:
    /// - You provide a basic template
    /// - The system tries multiple variations
    /// - It measures which variation gets the best results
    /// - The winning version becomes your optimized template
    ///
    /// Optimization can include:
    /// - Testing different wordings or phrasings
    /// - Trying different instruction orderings
    /// - Adding or removing examples
    /// - Adjusting tone and specificity
    ///
    /// Example process:
    /// Original: "Summarize this text: {text}"
    /// Variation 1: "Provide a concise summary of the following text in 2-3 sentences: {text}"
    /// Variation 2: "What are the key points in this text? {text}"
    /// → System tests all variations and picks the one with best performance
    ///
    /// Use this when:
    /// - You want the best possible results
    /// - You have evaluation metrics to measure success
    /// - You're willing to invest time in optimization
    /// </para>
    /// </remarks>
    Optimized,

    /// <summary>
    /// Template that includes function/tool calling capabilities.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Tool templates allow the model to use external functions and APIs.
    ///
    /// Think of tools like giving the model superpowers:
    /// - Instead of just answering from its training, the model can:
    ///   * Search the web
    ///   * Query databases
    ///   * Do calculations
    ///   * Access current information
    ///   * Execute code
    ///
    /// How it works:
    /// 1. You define available tools (e.g., "get_weather", "search_database")
    /// 2. The model decides when to use these tools
    /// 3. The model generates structured tool calls
    /// 4. Your code executes the tools
    /// 5. The model uses the results to answer the user
    ///
    /// Example:
    /// User: "What's the weather in Paris?"
    /// Model: Calls tool: get_weather(city="Paris")
    /// Tool returns: {"temperature": 18, "condition": "Sunny"}
    /// Model: "The weather in Paris is sunny with a temperature of 18°C."
    ///
    /// Use this when:
    /// - You need real-time or external data
    /// - The task requires computation or database access
    /// - You want to extend the model's capabilities
    /// </para>
    /// </remarks>
    Tool,

    /// <summary>
    /// Template for chain-of-thought reasoning with step-by-step problem solving.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Chain-of-thought templates encourage the model to think step-by-step.
    ///
    /// Think of it like showing your work in math class:
    /// - Instead of jumping to the answer
    /// - The model explains its reasoning process
    /// - Each step builds on the previous one
    /// - The final answer is more accurate and explainable
    ///
    /// Example without chain-of-thought:
    /// Q: "If a store has 24 apples and sells 3/4 of them, how many are left?"
    /// A: "6 apples"
    ///
    /// Example with chain-of-thought:
    /// Q: "If a store has 24 apples and sells 3/4 of them, how many are left?"
    /// A: "Let me solve this step by step:
    ///     1. The store has 24 apples
    ///     2. They sell 3/4 of them
    ///     3. 3/4 of 24 = 24 × 3/4 = 18 apples sold
    ///     4. Apples left = 24 - 18 = 6 apples
    ///     Therefore, 6 apples are left."
    ///
    /// Benefits:
    /// - Better accuracy on complex problems
    /// - Explainable reasoning
    /// - Easier to debug errors
    /// - More reliable for multi-step tasks
    ///
    /// Use this when:
    /// - Tasks require multi-step reasoning
    /// - Accuracy is critical
    /// - You need to understand how the model reached its answer
    /// </para>
    /// </remarks>
    ChainOfThought,

    /// <summary>
    /// Template for ReAct (Reasoning + Acting) pattern combining thought and action.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ReAct templates combine reasoning with actions in an iterative loop.
    ///
    /// ReAct stands for Reasoning + Acting:
    /// - The model alternates between thinking and doing
    /// - Each thought informs the next action
    /// - Each action result informs the next thought
    /// - This continues until the task is complete
    ///
    /// The pattern:
    /// 1. Thought: "I need to find X to answer this question"
    /// 2. Action: Search for X
    /// 3. Observation: Here's what I found about X
    /// 4. Thought: "Now I need to verify Y"
    /// 5. Action: Calculate Y
    /// 6. Observation: Y equals Z
    /// 7. Thought: "I have enough information to answer"
    /// 8. Answer: Final response
    ///
    /// Example:
    /// User: "Who is the current leader of the country where the Eiffel Tower is located?"
    ///
    /// Thought: "I need to find out which country the Eiffel Tower is in."
    /// Action: search("Eiffel Tower location")
    /// Observation: "The Eiffel Tower is in Paris, France."
    ///
    /// Thought: "Now I need to find the current leader of France."
    /// Action: search("current president of France")
    /// Observation: "Emmanuel Macron is the current President of France."
    ///
    /// Thought: "I have all the information needed to answer."
    /// Answer: "Emmanuel Macron is the current leader of France, where the Eiffel Tower is located."
    ///
    /// Use this when:
    /// - Tasks require both reasoning and external actions
    /// - You need to combine multiple tools/data sources
    /// - The solution path isn't straightforward
    /// - You want transparent, step-by-step problem solving
    /// </para>
    /// </remarks>
    ReAct
}
