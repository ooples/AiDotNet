using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using System.Collections.Generic;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Represents a reasoning model capable of multi-step logical reasoning, chain-of-thought processing,
    /// and generating explanations for its decisions.
    /// </summary>
    /// <remarks>
    /// Reasoning models are designed to perform complex logical operations that require multiple steps
    /// of analysis, similar to how humans approach problem-solving. These models can:
    /// - Break down complex problems into manageable steps
    /// - Generate intermediate reasoning chains
    /// - Provide explanations for their conclusions
    /// - Handle uncertainty and ambiguity in reasoning
    /// 
    /// Common applications include:
    /// - Mathematical problem solving
    /// - Logical deduction and inference
    /// - Question answering with explanation
    /// - Code generation with step-by-step reasoning
    /// - Decision making with transparent rationale
    /// </remarks>
    /// <typeparam name="T">The numeric type used for model computations</typeparam>
    public interface IReasoningModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
    {
        /// <summary>
        /// Performs multi-step reasoning on the input, generating intermediate reasoning steps.
        /// </summary>
        /// <param name="input">The input tensor containing the problem or query</param>
        /// <param name="maxSteps">Maximum number of reasoning steps to perform</param>
        /// <returns>A list of tensors representing each reasoning step</returns>
        /// <remarks>
        /// This method allows the model to break down complex problems into smaller steps,
        /// similar to chain-of-thought reasoning. Each step builds upon previous ones.
        /// </remarks>
        List<Tensor<T>> ReasonStepByStep(Tensor<T> input, int maxSteps = 10);

        /// <summary>
        /// Generates an explanation for the model's prediction or reasoning process.
        /// </summary>
        /// <param name="input">The input tensor for which to generate an explanation</param>
        /// <param name="prediction">The prediction made by the model</param>
        /// <returns>A tensor representing the explanation in the model's internal representation</returns>
        /// <remarks>
        /// Explanations help users understand why the model arrived at a particular conclusion,
        /// which is crucial for trust and debugging in production scenarios.
        /// </remarks>
        Tensor<T> GenerateExplanation(Tensor<T> input, Tensor<T> prediction);

        /// <summary>
        /// Gets the confidence scores for each reasoning step in the last prediction.
        /// </summary>
        /// <returns>A vector of confidence scores, one for each reasoning step</returns>
        /// <remarks>
        /// Confidence scores indicate how certain the model is about each step in its reasoning
        /// process. Lower scores might indicate areas where the model is uncertain.
        /// </remarks>
        Vector<T> GetReasoningConfidence();

        /// <summary>
        /// Performs self-consistency checking by generating multiple reasoning paths and comparing results.
        /// </summary>
        /// <param name="input">The input tensor to reason about</param>
        /// <param name="numPaths">Number of independent reasoning paths to generate</param>
        /// <returns>A tensor representing the aggregated result from multiple reasoning paths</returns>
        /// <remarks>
        /// Self-consistency improves reliability by ensuring the model arrives at similar conclusions
        /// through different reasoning paths, reducing the impact of random variations.
        /// </remarks>
        Tensor<T> SelfConsistencyCheck(Tensor<T> input, int numPaths = 3);

        /// <summary>
        /// Gets whether the model supports iterative refinement of its reasoning.
        /// </summary>
        /// <remarks>
        /// Some reasoning models can refine their answers by reconsidering their initial reasoning
        /// in light of the conclusions they reached.
        /// </remarks>
        bool SupportsIterativeRefinement { get; }

        /// <summary>
        /// Refines the reasoning process by iteratively improving the solution.
        /// </summary>
        /// <param name="input">The original input tensor</param>
        /// <param name="initialReasoning">The initial reasoning result to refine</param>
        /// <param name="iterations">Number of refinement iterations</param>
        /// <returns>The refined reasoning result</returns>
        /// <remarks>
        /// Iterative refinement allows the model to improve its answers by repeatedly
        /// analyzing and correcting its own reasoning, similar to human reflection.
        /// </remarks>
        Tensor<T> RefineReasoning(Tensor<T> input, Tensor<T> initialReasoning, int iterations = 3);

        /// <summary>
        /// Gets the maximum reasoning depth the model can handle effectively.
        /// </summary>
        /// <remarks>
        /// Reasoning depth refers to how many logical steps the model can chain together
        /// while maintaining coherence and accuracy.
        /// </remarks>
        int MaxReasoningDepth { get; }

        /// <summary>
        /// Sets the reasoning strategy for the model.
        /// </summary>
        /// <param name="strategy">The reasoning strategy to use</param>
        /// <remarks>
        /// Different strategies might include:
        /// - Forward chaining (reasoning from premises to conclusion)
        /// - Backward chaining (reasoning from goal to prerequisites)
        /// - Bidirectional reasoning (combining both approaches)
        /// </remarks>
        void SetReasoningStrategy(ReasoningStrategy strategy);

        /// <summary>
        /// Gets the current reasoning strategy being used by the model.
        /// </summary>
        ReasoningStrategy CurrentStrategy { get; }

        /// <summary>
        /// Validates the logical consistency of a reasoning chain.
        /// </summary>
        /// <param name="reasoningSteps">The list of reasoning steps to validate</param>
        /// <returns>True if the reasoning chain is logically consistent, false otherwise</returns>
        /// <remarks>
        /// This method checks whether each step follows logically from the previous ones,
        /// helping to identify potential errors or contradictions in the reasoning process.
        /// </remarks>
        bool ValidateReasoningChain(List<Tensor<T>> reasoningSteps);

        /// <summary>
        /// Gets diagnostic information about the last reasoning process.
        /// </summary>
        /// <returns>A dictionary containing diagnostic metrics and information</returns>
        /// <remarks>
        /// Diagnostic information might include:
        /// - Time spent on each reasoning step
        /// - Memory usage during reasoning
        /// - Confidence variations
        /// - Detected inconsistencies
        /// </remarks>
        Dictionary<string, object> GetReasoningDiagnostics();
    }
}