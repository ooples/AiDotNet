namespace AiDotNet.Interfaces;

using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Defines the contract for AI alignment methods that ensure models behave according to human values and intentions.
/// </summary>
/// <remarks>
/// AI alignment focuses on making AI systems that reliably do what humans want them to do,
/// even in novel situations where their behavior wasn't explicitly programmed.
///
/// <b>For Beginners:</b> Think of AI alignment as "teaching good behavior" to AI systems.
/// Just like teaching children values and ethics so they make good decisions on their own,
/// alignment methods help AI systems understand and follow human intentions.
///
/// Common alignment approaches include:
/// - RLHF (Reinforcement Learning from Human Feedback): Train models using human preferences
/// - Constitutional AI: Teach models principles to guide their behavior
/// - Red Teaming: Systematically test for harmful or unintended behaviors
///
/// Why AI alignment matters:
/// - Prevents models from pursuing goals in harmful ways
/// - Ensures models are helpful, harmless, and honest
/// - Critical for deploying powerful AI systems safely
/// - Helps models generalize human values to new situations
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface IAlignmentMethod<T> : IModelSerializer
{
    /// <summary>
    /// Aligns a model using feedback from human evaluators or preferences.
    /// </summary>
    /// <remarks>
    /// This method takes a base model and improves it using human feedback to better
    /// align with human values and intentions.
    ///
    /// <b>For Beginners:</b> This is like having a teacher grade your AI's homework
    /// and help it learn what responses are good vs. bad. The AI learns from examples
    /// of what humans prefer.
    ///
    /// The process typically involves:
    /// 1. Generate multiple outputs from the model for various inputs
    /// 2. Collect human feedback ranking which outputs are better
    /// 3. Train a reward model that predicts human preferences
    /// 4. Use reinforcement learning to optimize the model according to the reward
    /// </remarks>
    /// <param name="baseModel">The initial model to align.</param>
    /// <param name="feedbackData">Human feedback or preference data.</param>
    /// <returns>An aligned model that better matches human preferences.</returns>
    IPredictiveModel<T, Vector<T>, Vector<T>> AlignModel(IPredictiveModel<T, Vector<T>, Vector<T>> baseModel, AlignmentFeedbackData<T> feedbackData);

    /// <summary>
    /// Evaluates how well a model is aligned with human values.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This tests whether the AI is behaving the way humans want it to.
    /// It's like giving the AI a test to see if it learned the right lessons.
    /// </remarks>
    /// <param name="model">The model to evaluate.</param>
    /// <param name="evaluationData">Test cases for alignment evaluation.</param>
    /// <returns>Alignment metrics including helpfulness, harmlessness, and honesty scores.</returns>
    AlignmentMetrics<T> EvaluateAlignment(IPredictiveModel<T, Vector<T>, Vector<T>> model, AlignmentEvaluationData<T> evaluationData);

    /// <summary>
    /// Applies constitutional principles to guide model behavior.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This gives the AI a set of rules or principles to follow,
    /// like a constitution. The AI learns to critique and improve its own outputs based
    /// on these principles.
    ///
    /// For example, principles might include:
    /// - "Choose responses that are helpful and informative"
    /// - "Avoid responses that could cause harm"
    /// - "Be honest and don't make up information"
    /// </remarks>
    /// <param name="model">The model to apply constitutional principles to.</param>
    /// <param name="principles">The constitutional principles to follow.</param>
    /// <returns>A model that follows the specified principles.</returns>
    IPredictiveModel<T, Vector<T>, Vector<T>> ApplyConstitutionalPrinciples(IPredictiveModel<T, Vector<T>, Vector<T>> model, string[] principles);

    /// <summary>
    /// Performs red teaming to identify potential misalignment or harmful behaviors.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Red teaming is like hiring someone to try to break your system.
    /// You deliberately try to make the AI misbehave so you can find and fix problems
    /// before deploying it to real users.
    ///
    /// Red teamers might try to:
    /// - Get the AI to give harmful advice
    /// - Trick it into revealing private information
    /// - Make it behave inconsistently with its values
    /// - Find edge cases where alignment breaks down
    /// </remarks>
    /// <param name="model">The model to red team.</param>
    /// <param name="adversarialPrompts">Test prompts designed to elicit misaligned behavior.</param>
    /// <returns>Red teaming results identifying vulnerabilities and failure modes.</returns>
    RedTeamingResults<T> PerformRedTeaming(IPredictiveModel<T, Vector<T>, Vector<T>> model, Matrix<T> adversarialPrompts);

    /// <summary>
    /// Gets the configuration options for the alignment method.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These settings control how the alignment process works,
    /// like how much to weight human feedback or which principles to prioritize.
    /// </remarks>
    /// <returns>The configuration options for alignment.</returns>
    AlignmentMethodOptions<T> GetOptions();

    /// <summary>
    /// Resets the alignment method state.
    /// </summary>
    void Reset();
}
