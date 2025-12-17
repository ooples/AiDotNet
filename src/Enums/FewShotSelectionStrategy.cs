namespace AiDotNet.Enums;

/// <summary>
/// Represents strategies for selecting few-shot examples in prompt templates.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Few-shot selection strategies determine which examples to show the language model.
///
/// Think of it like choosing which practice problems to show a student:
/// - You could show random problems
/// - You could show problems similar to the current one
/// - You could show problems that cover diverse scenarios
/// - You could show the most helpful examples
///
/// The right strategy depends on what you're trying to teach and what works best.
/// Different strategies can significantly impact the model's performance.
/// </para>
/// </remarks>
public enum FewShotSelectionStrategy
{
    /// <summary>
    /// Select examples randomly from the available pool.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Random selection picks examples without any particular logic.
    ///
    /// Like drawing names from a hat:
    /// - Each example has an equal chance of being selected
    /// - No consideration of relevance or similarity
    /// - Simple and fast
    /// - Good baseline approach
    ///
    /// Example:
    /// Available examples: 100 sentiment classification examples
    /// Need: 3 examples
    /// Strategy: Randomly pick 3 from the 100
    ///
    /// Advantages:
    /// - Very fast
    /// - No special setup required
    /// - Unbiased
    /// - Good for diverse example sets
    ///
    /// Disadvantages:
    /// - May pick irrelevant examples
    /// - Performance varies between runs
    /// - Doesn't adapt to the specific query
    ///
    /// Use this when:
    /// - You have diverse, high-quality examples
    /// - All examples are roughly equally useful
    /// - Simplicity is important
    /// - You don't have semantic similarity infrastructure
    /// </para>
    /// </remarks>
    Random,

    /// <summary>
    /// Select examples most semantically similar to the current input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Semantic similarity picks examples that are most similar to the current query.
    ///
    /// Think of it like a tutor who:
    /// - Looks at the problem the student is struggling with
    /// - Finds previous problems that are similar
    /// - Shows those similar examples to help the student understand
    ///
    /// How it works:
    /// 1. Convert query and all examples into embeddings (mathematical representations)
    /// 2. Calculate similarity between query and each example
    /// 3. Select the most similar examples
    ///
    /// Example:
    /// Query: "This restaurant was absolutely terrible, worst experience ever."
    ///
    /// Example pool:
    /// - "The movie was amazing, I loved it!" (Low similarity)
    /// - "Bad service, would not recommend this place." (High similarity - both negative reviews)
    /// - "The weather is nice today." (Low similarity)
    /// - "Horrible food, never going back." (High similarity - both negative, about food)
    ///
    /// Selected examples:
    /// - "Bad service, would not recommend this place."
    /// - "Horrible food, never going back."
    ///
    /// Advantages:
    /// - Examples are relevant to the specific query
    /// - Often better performance than random
    /// - Adapts to different types of queries
    ///
    /// Disadvantages:
    /// - Requires embedding model
    /// - Slower than random (need to compute similarities)
    /// - May lack diversity if all examples are too similar
    ///
    /// Use this when:
    /// - Query types vary significantly
    /// - Relevant examples significantly improve performance
    /// - You have an embedding model available
    /// - Quality is more important than speed
    /// </para>
    /// </remarks>
    SemanticSimilarity,

    /// <summary>
    /// Select diverse examples to maximize coverage of different patterns.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Diversity selection picks examples that are different from each other.
    ///
    /// Think of it like choosing a diverse team:
    /// - You don't want everyone with the same skills
    /// - You want a variety of perspectives and strengths
    /// - Together, they cover more ground than similar individuals would
    ///
    /// The goal is to show the model different types of inputs and outputs so it understands
    /// the full range of possibilities.
    ///
    /// Example - Sentiment classification:
    /// Instead of all similar examples:
    /// - "Great product!" → Positive
    /// - "Loved it!" → Positive
    /// - "Excellent quality!" → Positive
    ///
    /// Diverse selection chooses:
    /// - "Great product!" → Positive (enthusiastic positive)
    /// - "It's okay, nothing special." → Neutral (mild, mixed)
    /// - "Complete waste of money!" → Negative (strong negative)
    /// - "Not bad, but could be better." → Mixed (constructive criticism)
    ///
    /// The diverse set teaches the model to handle different tones, lengths, and edge cases.
    ///
    /// Advantages:
    /// - Model learns broader patterns
    /// - Better generalization
    /// - Handles edge cases better
    /// - Less likely to overfit to specific patterns
    ///
    /// Disadvantages:
    /// - May not include the most relevant examples for specific queries
    /// - Requires clustering or diversity metrics
    /// - More complex to implement
    ///
    /// Use this when:
    /// - Input space is large and varied
    /// - You want robust, generalizable performance
    /// - Examples naturally cluster into groups
    /// - You're building a general-purpose system
    /// </para>
    /// </remarks>
    Diversity,

    /// <summary>
    /// Select examples based on maximum marginal relevance (balance between relevance and diversity).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MMR (Maximum Marginal Relevance) balances being relevant AND diverse.
    ///
    /// Think of it like building a playlist:
    /// - Pure relevance: All songs sound almost identical (boring!)
    /// - Pure diversity: Random songs that don't fit together (confusing!)
    /// - MMR: Similar style, but each song brings something different (perfect!)
    ///
    /// How it works:
    /// 1. Start with the most relevant example
    /// 2. For the next example, consider both:
    ///    - How relevant it is to the query (good)
    ///    - How different it is from already-selected examples (also good)
    /// 3. Pick the example with the best balance
    /// 4. Repeat until you have enough examples
    ///
    /// Example - Question answering about Python:
    /// Query: "How do I sort a list in Python?"
    ///
    /// Step 1: Most relevant example
    /// Selected: "How to reverse a list → list.reverse()" (very similar to sorting)
    ///
    /// Step 2: Balance relevance and diversity
    /// Candidates:
    /// - "How to sort a dictionary" (Relevant: 0.9, Diversity: 0.4) → Score: 0.65
    /// - "How to use list comprehensions" (Relevant: 0.5, Diversity: 0.9) → Score: 0.70
    /// - "How to remove duplicates" (Relevant: 0.7, Diversity: 0.7) → Score: 0.70
    ///
    /// Selected: "How to use list comprehensions" or "How to remove duplicates"
    ///
    /// The result: Examples that are all related to the query but show different aspects
    /// of working with lists.
    ///
    /// Advantages:
    /// - Best of both worlds (relevance + diversity)
    /// - Often outperforms pure relevance or pure diversity
    /// - Prevents redundant examples
    /// - Provides comprehensive coverage
    ///
    /// Disadvantages:
    /// - More complex to implement
    /// - Slower than simple strategies
    /// - Requires tuning the relevance/diversity balance parameter
    ///
    /// Use this when:
    /// - You want the best possible performance
    /// - You have the computational resources
    /// - Both relevance and diversity matter
    /// - Quality is the top priority
    /// </para>
    /// </remarks>
    MaximumMarginalRelevance,

    /// <summary>
    /// Select examples in a fixed, predetermined order.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Fixed ordering uses the same examples in the same order every time.
    ///
    /// Like a textbook that always shows the same examples:
    /// - Example 1 is always first
    /// - Example 2 is always second
    /// - And so on...
    ///
    /// Use cases:
    /// - You've carefully curated the best examples
    /// - Order matters (simple to complex, for instance)
    /// - Consistency across all queries is important
    /// - You want predictable, reproducible behavior
    ///
    /// Example - Teaching code style:
    /// Always show these examples in this order:
    /// 1. Simple variable naming
    /// 2. Function documentation
    /// 3. Error handling
    /// 4. Complex class structure
    ///
    /// This builds from basics to advanced, regardless of the query.
    ///
    /// Advantages:
    /// - Completely predictable
    /// - No computation needed
    /// - Fastest option
    /// - Can carefully control the learning progression
    ///
    /// Disadvantages:
    /// - Not adaptive to different queries
    /// - May include irrelevant examples
    /// - Less flexible
    ///
    /// Use this when:
    /// - You have a canonical set of examples
    /// - All queries are similar enough that the same examples work
    /// - Speed and consistency are critical
    /// - You want to maintain exact reproducibility
    /// </para>
    /// </remarks>
    Fixed,

    /// <summary>
    /// Select examples using a clustering approach to ensure broad coverage.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cluster-based selection groups similar examples and picks representatives.
    ///
    /// Think of organizing a photo collection:
    /// - Group 1: Beach photos
    /// - Group 2: Mountain photos
    /// - Group 3: City photos
    /// - Group 4: Portrait photos
    ///
    /// Instead of showing all beach photos, you show one representative from each group.
    ///
    /// How it works:
    /// 1. Cluster all examples into groups of similar items
    /// 2. Select examples from different clusters
    /// 3. Ensure each cluster is represented
    /// 4. Optionally, weight selection by cluster size or importance
    ///
    /// Example - Customer support queries:
    /// Clusters discovered:
    /// - Billing issues (100 examples)
    /// - Technical problems (150 examples)
    /// - Account questions (50 examples)
    /// - Shipping inquiries (75 examples)
    ///
    /// Selection for 4 examples:
    /// - Pick 1 from Billing issues
    /// - Pick 1 from Technical problems
    /// - Pick 1 from Account questions
    /// - Pick 1 from Shipping inquiries
    ///
    /// Result: Examples that cover all major types of customer queries.
    ///
    /// Advantages:
    /// - Systematic coverage of the example space
    /// - Prevents over-representation of any one type
    /// - Good for varied input distributions
    /// - Generalizes well
    ///
    /// Disadvantages:
    /// - Requires pre-computed clusters
    /// - May not adapt to individual queries
    /// - Setup overhead for clustering
    ///
    /// Use this when:
    /// - Examples naturally fall into categories
    /// - You want guaranteed coverage of all categories
    /// - Example pool has uneven distribution
    /// - Building a general-purpose system
    /// </para>
    /// </remarks>
    ClusterBased
}
