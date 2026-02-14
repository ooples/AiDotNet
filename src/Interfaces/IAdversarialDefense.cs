namespace AiDotNet.Interfaces;

using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Defines the contract for adversarial defense mechanisms that protect models against attacks.
/// </summary>
/// <remarks>
/// An adversarial defense is a technique to make machine learning models more resistant
/// to adversarial attacks and improve their robustness.
///
/// <b>For Beginners:</b> Think of adversarial defenses as "armor" for your AI model.
/// Just like armor protects a knight from attacks, these defenses protect your model
/// from adversarial examples that try to fool it.
///
/// Common examples of adversarial defenses include:
/// - Adversarial Training: Training the model on adversarial examples to make it robust
/// - Input Transformations: Preprocessing inputs to remove adversarial perturbations
/// - Ensemble Methods: Using multiple models to make predictions more reliable
///
/// Why adversarial defenses matter:
/// - They make models safer for real-world deployment
/// - They improve model reliability under attack
/// - They're critical for security-sensitive applications
/// - They help models generalize better to unusual inputs
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type for the model (e.g., Vector&lt;T&gt;, string).</typeparam>
/// <typeparam name="TOutput">The output data type for the model (e.g., Vector&lt;T&gt;, int).</typeparam>
[AiDotNet.Configuration.YamlConfigurable("AdversarialDefense")]
public interface IAdversarialDefense<T, TInput, TOutput> : IModelSerializer
{
    /// <summary>
    /// Trains or hardens a model to be more resistant to adversarial attacks.
    /// </summary>
    /// <remarks>
    /// This method applies defensive techniques to improve model robustness.
    ///
    /// <b>For Beginners:</b> This is like training a model to recognize and resist tricks.
    /// The defense mechanism teaches the model to handle adversarial examples correctly,
    /// making it harder for attackers to fool it.
    ///
    /// The process typically involves:
    /// 1. Generating adversarial examples during training
    /// 2. Training the model on both clean and adversarial data
    /// 3. Optimizing the model to be robust against perturbations
    /// 4. Validating improved robustness on test adversarial examples
    /// </remarks>
    /// <param name="trainingData">The training data to use for defensive training.</param>
    /// <param name="labels">The labels for the training data.</param>
    /// <param name="model">The model to harden against attacks.</param>
    /// <returns>The defended/hardened model.</returns>
    IFullModel<T, TInput, TOutput> ApplyDefense(TInput[] trainingData, TOutput[] labels, IFullModel<T, TInput, TOutput> model);

    /// <summary>
    /// Preprocesses input data to remove or reduce adversarial perturbations.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is like a "filter" that cleans up suspicious inputs
    /// before they reach your model. It tries to detect and remove malicious changes
    /// that an attacker might have added.
    /// </remarks>
    /// <param name="input">The potentially adversarial input.</param>
    /// <returns>The cleaned/defended input.</returns>
    TInput PreprocessInput(TInput input);

    /// <summary>
    /// Evaluates the robustness of a defended model against attacks.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This tests how well your defense works by trying attacks
    /// on the defended model and measuring how often it resists them successfully.
    /// </remarks>
    /// <param name="model">The defended model to evaluate.</param>
    /// <param name="testData">Test data to use for evaluation.</param>
    /// <param name="labels">The true labels for test data.</param>
    /// <param name="attack">The attack to test against.</param>
    /// <returns>Robustness metrics including clean accuracy and adversarial accuracy.</returns>
    RobustnessMetrics<T> EvaluateRobustness(
        IFullModel<T, TInput, TOutput> model,
        TInput[] testData,
        TOutput[] labels,
        IAdversarialAttack<T, TInput, TOutput> attack);

    /// <summary>
    /// Gets the configuration options for the adversarial defense.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These are the "settings" for the defense mechanism,
    /// controlling how aggressively it protects the model and what techniques it uses.
    /// </remarks>
    /// <returns>The configuration options for the defense.</returns>
    AdversarialDefenseOptions<T> GetOptions();

    /// <summary>
    /// Resets the defense state to prepare for a fresh defense application.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This clears any saved state from previous defense operations.
    /// </remarks>
    void Reset();
}
