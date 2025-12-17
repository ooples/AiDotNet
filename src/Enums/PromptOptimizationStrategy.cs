namespace AiDotNet.Enums;

/// <summary>
/// Represents strategies for optimizing prompts to improve language model performance.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Prompt optimization strategies automatically improve prompts to get better results.
///
/// Think of it like tuning a recipe:
/// - You start with a basic recipe
/// - Try different variations (more salt? less sugar? different temperature?)
/// - Taste test each variation
/// - Keep the version that tastes best
///
/// Prompt optimization does the same thing:
/// - Start with a basic prompt
/// - Generate variations
/// - Test each variation's performance
/// - Keep the best-performing version
///
/// Different strategies use different approaches to search for better prompts.
/// </para>
/// </remarks>
public enum PromptOptimizationStrategy
{
    /// <summary>
    /// No optimization - use the prompt as provided.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This means using your prompt exactly as you wrote it, without any optimization.
    ///
    /// Use this when:
    /// - You've already manually crafted a great prompt
    /// - Speed is critical and you can't afford optimization time
    /// - You want complete control over the exact prompt text
    /// - You're in early development and want to test specific prompt ideas
    /// </para>
    /// </remarks>
    None,

    /// <summary>
    /// Discrete optimization that tests variations of prompt components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Discrete optimization tries different versions of prompt parts and picks what works best.
    ///
    /// Think of it like A/B testing:
    /// - Create multiple variations of different parts
    /// - Test each combination
    /// - Measure which performs best
    /// - Use the winning combination
    ///
    /// Example - Optimizing a classification prompt:
    ///
    /// Component 1 - Instruction phrase:
    /// A: "Classify the following"
    /// B: "Categorize this text as"
    /// C: "Determine if this is"
    ///
    /// Component 2 - Output format:
    /// A: "Respond with just the category"
    /// B: "Output format: [CATEGORY]"
    /// C: "Answer in one word"
    ///
    /// Component 3 - Examples:
    /// A: 2 examples
    /// B: 5 examples
    /// C: 10 examples
    ///
    /// The system tests combinations:
    /// - Instruction A + Format A + Examples A
    /// - Instruction A + Format B + Examples A
    /// - ...and so on
    ///
    /// After testing, it finds:
    /// Best combination: Instruction B + Format C + Examples B
    /// = "Categorize this text as... Answer in one word... [5 examples]"
    ///
    /// Advantages:
    /// - Systematic exploration of prompt variations
    /// - Finds good combinations of components
    /// - Interpretable results (you know what changed)
    /// - Works well for structured prompts
    ///
    /// Disadvantages:
    /// - Can be slow with many components
    /// - Only tests pre-defined variations
    /// - May miss optimal wordings not in the variation set
    ///
    /// Use this when:
    /// - You have ideas for prompt variations to test
    /// - Prompt has clear, separable components
    /// - You want to understand what makes prompts work
    /// - You have a good evaluation metric
    /// </para>
    /// </remarks>
    DiscreteSearch,

    /// <summary>
    /// Gradient-based optimization using automatic differentiation (APE - Automatic Prompt Engineering).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Gradient-based optimization automatically finds better prompts using math,
    /// similar to how neural networks learn.
    ///
    /// Think of it like a robot learning to navigate:
    /// - Instead of trying random directions
    /// - It calculates which direction leads "downhill" toward the goal
    /// - It takes steps in that direction
    /// - It repeats until it reaches the destination
    ///
    /// For prompts:
    /// - Start with an initial prompt
    /// - Calculate how to change it to improve performance
    /// - Make small adjustments
    /// - Repeat until performance is good enough
    ///
    /// This uses "soft prompts" or "continuous prompts" - mathematical representations that can
    /// be optimized with gradients, then converted back to text.
    ///
    /// Example process:
    /// Initial: "Classify this text"
    /// → Calculate gradient (math indicating how to improve)
    /// → Adjust toward "Categorize the sentiment of this text as"
    /// → Evaluate performance
    /// → Continue adjusting...
    /// Final: "Analyze and categorize the emotional tone of the following text as"
    ///
    /// Advantages:
    /// - Can find prompts you wouldn't think of manually
    /// - Efficient optimization process
    /// - Explores a continuous space of possibilities
    /// - State-of-the-art performance in research
    ///
    /// Disadvantages:
    /// - Complex to implement
    /// - Requires differentiable models
    /// - May produce unnatural-sounding prompts
    /// - Harder to interpret why it works
    ///
    /// Use this when:
    /// - You want cutting-edge performance
    /// - You have computational resources
    /// - You can use gradient-based optimization
    /// - Quality is more important than interpretability
    /// </para>
    /// </remarks>
    GradientBased,

    /// <summary>
    /// Ensemble multiple prompts and combine their outputs for better performance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Ensemble optimization uses multiple different prompts and combines their results.
    ///
    /// Think of it like asking multiple experts:
    /// - Expert 1 gives their opinion
    /// - Expert 2 gives their opinion
    /// - Expert 3 gives their opinion
    /// - You combine all opinions to reach a conclusion
    ///
    /// Often, the combined opinion is better than any single expert.
    ///
    /// How it works:
    /// 1. Create multiple diverse prompts (or use the best from optimization)
    /// 2. Send the query to the model with each prompt
    /// 3. Get multiple responses
    /// 4. Combine responses (voting, averaging, or consensus)
    ///
    /// Example - Sentiment classification:
    ///
    /// Prompt 1 (formal): "Analyze the sentiment of the following text. Classify as positive, negative, or neutral."
    /// → Output: "Positive"
    ///
    /// Prompt 2 (instructive): "You are a sentiment analysis expert. Categorize this review."
    /// → Output: "Positive"
    ///
    /// Prompt 3 (with examples): [Few-shot examples] "Now classify: [text]"
    /// → Output: "Positive"
    ///
    /// Prompt 4 (chain-of-thought): "Think step-by-step about the sentiment of this text."
    /// → Output: "The text mentions 'loved it' and 'amazing', which are positive. Negative words are absent. Therefore: Positive"
    ///
    /// Combined result: Positive (4/4 agree) → High confidence
    ///
    /// For more complex tasks:
    /// - If prompts disagree, use majority voting
    /// - Weight prompts by their historical accuracy
    /// - Use the most confident prediction
    ///
    /// Advantages:
    /// - More robust than single prompts
    /// - Reduces impact of prompt brittleness
    /// - Can provide confidence estimates
    /// - Often better accuracy
    ///
    /// Disadvantages:
    /// - More expensive (multiple API calls)
    /// - Slower response time
    /// - Complex to combine results for generative tasks
    ///
    /// Use this when:
    /// - Accuracy is critical
    /// - Cost and speed are acceptable trade-offs
    /// - Single prompts are too brittle
    /// - You need confidence estimates
    /// </para>
    /// </remarks>
    Ensemble,

    /// <summary>
    /// Monte Carlo optimization that randomly samples and tests prompt variations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Monte Carlo optimization tries many random variations and keeps what works.
    ///
    /// Think of it like:
    /// - Throwing darts at a dartboard
    /// - Some hit the bullseye, some miss
    /// - After many throws, you learn where to aim
    /// - You focus on the areas that work best
    ///
    /// For prompts:
    /// 1. Generate random variations of the prompt
    /// 2. Test each variation's performance
    /// 3. Keep track of what works
    /// 4. Generate more variations around successful ones
    /// 5. Repeat until performance is good enough
    ///
    /// Example:
    /// Base prompt: "Classify this text"
    ///
    /// Random variations generated:
    /// 1. "Categorize the following text"
    /// 2. "What is the category of this text?"
    /// 3. "Classify the sentiment of this text"
    /// 4. "Determine the classification of this text"
    /// 5. "Please classify this text into a category"
    /// ...50 more variations...
    ///
    /// Test all variations:
    /// Variation 3 ("Classify the sentiment of this text") → 92% accuracy
    /// Variation 5 ("Please classify this text into a category") → 89% accuracy
    /// Others: 75-85% accuracy
    ///
    /// Generate more variations similar to #3:
    /// - "Analyze and classify the sentiment of this text"
    /// - "Classify the emotional tone of this text"
    /// - etc.
    ///
    /// Continue until optimal performance is found.
    ///
    /// Advantages:
    /// - Simple to implement
    /// - No gradient computation needed
    /// - Can escape local optima (bad solutions)
    /// - Good for exploring large spaces
    ///
    /// Disadvantages:
    /// - Requires many evaluations
    /// - Can be slow and expensive
    /// - May miss optimal solutions
    /// - No guarantee of finding the best prompt
    ///
    /// Use this when:
    /// - You can't use gradient-based methods
    /// - You have computational budget for many tests
    /// - The optimization space is complex
    /// - You want a simple, robust approach
    /// </para>
    /// </remarks>
    MonteCarlo,

    /// <summary>
    /// Evolutionary optimization using genetic algorithms to evolve better prompts.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Evolutionary optimization mimics biological evolution to improve prompts.
    ///
    /// How evolution works in nature:
    /// 1. Start with a population of individuals (genetic variation)
    /// 2. Individuals with better traits survive (natural selection)
    /// 3. Survivors reproduce, creating new individuals (crossover)
    /// 4. Random mutations add new variations
    /// 5. Repeat for many generations
    /// 6. Eventually, you get highly adapted individuals
    ///
    /// For prompts:
    /// 1. Create a "population" of different prompts
    /// 2. Test each prompt's performance (fitness)
    /// 3. Keep the best-performing prompts
    /// 4. Create new prompts by combining parts of good prompts (crossover)
    /// 5. Add random changes to some prompts (mutation)
    /// 6. Repeat for many generations
    ///
    /// Example:
    /// Generation 1 - Random population:
    /// - Prompt A: "Classify this text" → 75% accuracy
    /// - Prompt B: "Categorize the sentiment" → 82% accuracy
    /// - Prompt C: "Determine the category" → 70% accuracy
    /// - Prompt D: "Analyze and classify this text" → 85% accuracy
    ///
    /// Selection: Keep B and D (best performers)
    ///
    /// Generation 2 - Crossover (combine parts of B and D):
    /// - New E: "Categorize and analyze this text" (from B + D)
    /// - New F: "Analyze the sentiment" (from D + B)
    ///
    /// Mutation (random changes):
    /// - New G: "Carefully analyze the sentiment" (added "carefully")
    /// - New H: "Categorize the emotional tone" (changed "sentiment" to "emotional tone")
    ///
    /// Test Generation 2:
    /// - Prompt E: 87% accuracy
    /// - Prompt F: 83% accuracy
    /// - Prompt G: 84% accuracy
    /// - Prompt H: 90% accuracy ← Best so far!
    ///
    /// Keep best from Gen 2, create Gen 3, and continue...
    ///
    /// After many generations:
    /// Final prompt: "Carefully categorize the specific emotional tone and sentiment of the following text"
    /// → 94% accuracy
    ///
    /// Advantages:
    /// - Can find creative solutions
    /// - Balances exploration and exploitation
    /// - Doesn't require gradients
    /// - Often finds good solutions
    ///
    /// Disadvantages:
    /// - Requires many evaluations
    /// - Can be slow
    /// - Complex to implement well
    /// - Results can be unpredictable
    ///
    /// Use this when:
    /// - You want to explore creative prompt variations
    /// - You have evaluation budget
    /// - Other methods haven't worked well
    /// - You enjoy bio-inspired algorithms
    /// </para>
    /// </remarks>
    Evolutionary
}
