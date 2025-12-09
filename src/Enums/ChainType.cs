namespace AiDotNet.Enums;

/// <summary>
/// Represents different types of chains for composing language model operations.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Chains connect multiple language model operations together, like building blocks.
///
/// Think of chains like assembly lines:
/// - Each step does a specific task
/// - Output from one step becomes input for the next
/// - Complex workflows are built from simple components
/// - The final result is the combination of all steps
///
/// For example, a research assistant might:
/// 1. Search for relevant documents (Step 1)
/// 2. Summarize each document (Step 2)
/// 3. Combine summaries into a final report (Step 3)
///
/// Chains make this easy to build, test, and modify.
/// </para>
/// </remarks>
public enum ChainType
{
    /// <summary>
    /// Simple sequential chain where output of each step feeds into the next.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Sequential chains run steps one after another in order.
    ///
    /// Like a recipe:
    /// - Step 1: Gather ingredients
    /// - Step 2: Mix ingredients
    /// - Step 3: Bake mixture
    /// - Step 4: Serve result
    ///
    /// Each step must complete before the next begins.
    ///
    /// Example:
    /// Input: "Analyze customer feedback about our product"
    /// Step 1: Extract feedback → ["Great product!", "Too expensive", "Love it"]
    /// Step 2: Classify sentiment → [Positive, Negative, Positive]
    /// Step 3: Generate summary → "Overall positive (67%) with price concerns"
    /// Output: Summary report
    ///
    /// Use this when:
    /// - Steps must happen in a specific order
    /// - Each step depends on the previous step's output
    /// - The workflow is straightforward and linear
    /// </para>
    /// </remarks>
    Sequential,

    /// <summary>
    /// Conditional chain with branching logic based on intermediate results.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Conditional chains make decisions and take different paths based on results.
    ///
    /// Like a choose-your-own-adventure book:
    /// - Evaluate a condition
    /// - If true, go down path A
    /// - If false, go down path B
    /// - Different paths lead to different outcomes
    ///
    /// Example:
    /// Input: Customer email
    ///
    /// Step 1: Classify email type
    /// → If "complaint":
    ///    - Extract issue
    ///    - Check severity
    ///    - Route to appropriate team
    /// → If "question":
    ///    - Search knowledge base
    ///    - Generate answer
    /// → If "feedback":
    ///    - Categorize feedback
    ///    - Log to database
    ///
    /// The chain adapts based on what it encounters.
    ///
    /// Use this when:
    /// - Different inputs require different processing
    /// - You need to make decisions based on intermediate results
    /// - Workflows have multiple possible paths
    /// </para>
    /// </remarks>
    Conditional,

    /// <summary>
    /// Looping chain that repeats operations until a condition is met.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Looping chains repeat operations until they achieve a goal or meet a condition.
    ///
    /// Like a video game character searching for a key:
    /// - Check current room for key
    /// - If found: Stop searching
    /// - If not found: Move to next room and repeat
    /// - Continue until key is found or all rooms are checked
    ///
    /// Example - Research assistant:
    /// Goal: "Find 5 peer-reviewed articles about climate change"
    ///
    /// Loop iteration 1: Search → Found 2 articles → Continue
    /// Loop iteration 2: Refine search → Found 1 more (total 3) → Continue
    /// Loop iteration 3: Broaden search → Found 2 more (total 5) → Stop
    ///
    /// Result: 5 articles found
    ///
    /// Loop with refinement:
    /// Goal: "Generate a summary that's exactly 100 words"
    ///
    /// Attempt 1: Generated 85 words → Too short → Try again with "expand" instruction
    /// Attempt 2: Generated 112 words → Too long → Try again with "condense" instruction
    /// Attempt 3: Generated 98 words → Close enough → Accept
    ///
    /// Use this when:
    /// - You need to refine results iteratively
    /// - The exact number of steps isn't known in advance
    /// - Quality improves with iteration
    /// - You need to meet specific criteria
    /// </para>
    /// </remarks>
    Loop,

    /// <summary>
    /// Parallel chain that executes multiple independent operations simultaneously.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Parallel chains run multiple operations at the same time, then combine results.
    ///
    /// Like a team working on a project:
    /// - Different team members work on different tasks simultaneously
    /// - Each person completes their part independently
    /// - At the end, all parts are combined into the final product
    ///
    /// Example - Comprehensive product analysis:
    /// Input: Product URL
    ///
    /// Parallel operations (all at once):
    /// - Branch 1: Extract product specs
    /// - Branch 2: Analyze customer reviews
    /// - Branch 3: Find competitor prices
    /// - Branch 4: Check availability
    ///
    /// Combine results:
    /// "Product X costs $99, has 4.5-star rating, is in stock, and is cheaper than competitors"
    ///
    /// Benefits:
    /// - Much faster than sequential processing
    /// - Each branch is independent
    /// - Results are combined at the end
    ///
    /// Use this when:
    /// - Operations don't depend on each other
    /// - Speed is important
    /// - You need multiple perspectives on the same input
    /// - Tasks can be done independently
    /// </para>
    /// </remarks>
    Parallel,

    /// <summary>
    /// Map-reduce chain that processes collections by mapping operations and reducing results.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Map-reduce chains process large collections by working on items individually,
    /// then combining results.
    ///
    /// Two phases:
    /// 1. Map: Apply the same operation to each item in a collection
    /// 2. Reduce: Combine all the results into a single output
    ///
    /// Think of it like grading exams:
    /// - Map phase: Grade each student's exam individually (can be done in parallel)
    /// - Reduce phase: Calculate class average and statistics from all grades
    ///
    /// Example - Analyze multiple documents:
    /// Input: [Document1, Document2, Document3, Document4, Document5]
    ///
    /// Map phase (apply to each document):
    /// - Document1 → Extract key points → ["Point A", "Point B"]
    /// - Document2 → Extract key points → ["Point C", "Point D"]
    /// - Document3 → Extract key points → ["Point E"]
    /// - Document4 → Extract key points → ["Point F", "Point G"]
    /// - Document5 → Extract key points → ["Point H"]
    ///
    /// Reduce phase (combine all results):
    /// All points: ["Point A", "Point B", "Point C", ..., "Point H"]
    /// → Identify common themes
    /// → Rank by importance
    /// → Generate final summary
    ///
    /// Benefits:
    /// - Handles large collections efficiently
    /// - Map phase can be parallelized
    /// - Scales well with data size
    /// - Clean separation of processing and aggregation
    ///
    /// Use this when:
    /// - Processing multiple similar items
    /// - Each item needs the same analysis
    /// - Final result combines all individual analyses
    /// - Working with large document collections
    /// </para>
    /// </remarks>
    MapReduce,

    /// <summary>
    /// Router chain that directs inputs to specialized chains based on content.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Router chains analyze input and send it to the best specialized chain for that type.
    ///
    /// Like a receptionist at a hospital:
    /// - Patient arrives with symptoms
    /// - Receptionist determines the issue
    /// - Patient is directed to the right specialist
    /// - Specialist provides appropriate care
    ///
    /// Example - Customer support system:
    /// Input: Customer message
    ///
    /// Router analysis:
    /// - If about billing → Route to BillingChain
    /// - If about technical issue → Route to TechnicalSupportChain
    /// - If about product info → Route to ProductInfoChain
    /// - If about returns → Route to ReturnsChain
    ///
    /// Each specialized chain is optimized for its specific task:
    /// - BillingChain: Knows about invoices, payments, refunds
    /// - TechnicalSupportChain: Knows about troubleshooting, diagnostics
    /// - ProductInfoChain: Knows about features, specifications
    /// - ReturnsChain: Knows about return policies, shipping
    ///
    /// Benefits:
    /// - Each chain can be optimized for its specialty
    /// - Easier to maintain than one giant chain
    /// - Better results because of specialization
    /// - Can add new specialized chains easily
    ///
    /// Use this when:
    /// - Inputs fall into distinct categories
    /// - Different categories need different processing
    /// - You want specialized handling for each type
    /// - One-size-fits-all approach doesn't work well
    /// </para>
    /// </remarks>
    Router
}
