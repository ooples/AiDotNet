namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for adversarial defense mechanisms.
/// </summary>
/// <remarks>
/// <para>
/// These options control how models are defended against adversarial attacks through
/// training procedures, preprocessing, and ensemble methods.
/// </para>
/// <para><b>For Beginners:</b> These settings control how your "armor" protects the AI model.
/// You can adjust how the defense is applied, how strong it should be, and what techniques to use.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public class AdversarialDefenseOptions<T>
{
    /// <summary>
    /// Gets or sets the ratio of adversarial examples to include in training.
    /// </summary>
    /// <value>The adversarial ratio, defaulting to 0.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When training, this controls what percentage of examples
    /// should be adversarial. 0.5 means 50% clean and 50% adversarial, providing a balance
    /// between robustness and normal accuracy.</para>
    /// </remarks>
    public double AdversarialRatio { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the perturbation budget for adversarial training.
    /// </summary>
    /// <value>The epsilon value, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how strong the adversarial examples during training are.
    /// Training on stronger attacks makes the model more robust but might reduce clean accuracy.</para>
    /// </remarks>
    public double Epsilon { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>The number of epochs, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> An epoch is one complete pass through all training data.
    /// More epochs allow the model to learn better defenses but take longer.</para>
    /// </remarks>
    public int TrainingEpochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to use input preprocessing for defense.
    /// </summary>
    /// <value>True to use preprocessing, false otherwise (default: true).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Preprocessing cleans up inputs before they reach the model,
    /// potentially removing adversarial perturbations.</para>
    /// </remarks>
    public bool UsePreprocessing { get; set; } = true;

    /// <summary>
    /// Gets or sets the preprocessing method to use.
    /// </summary>
    /// <value>The preprocessing method name, defaulting to "JPEG".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Different preprocessing methods work better for different types
    /// of data. JPEG compression, for example, can remove small adversarial changes from images.</para>
    /// </remarks>
    public string PreprocessingMethod { get; set; } = "JPEG";

    /// <summary>
    /// Gets or sets whether to use ensemble defenses.
    /// </summary>
    /// <value>True to use ensembles, false otherwise (default: false).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Ensemble defenses use multiple models to make predictions.
    /// It's harder to fool all models at once, making the system more robust.</para>
    /// </remarks>
    public bool UseEnsemble { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of models in the ensemble.
    /// </summary>
    /// <value>The ensemble size, defaulting to 3.</value>
    public int EnsembleSize { get; set; } = 3;

    /// <summary>
    /// Gets or sets the attack method to use during adversarial training.
    /// </summary>
    /// <value>The attack method name, defaulting to "PGD".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This determines what type of attacks to train against.
    /// PGD is a strong iterative attack that provides good robustness when used for training.</para>
    /// </remarks>
    public string AttackMethod { get; set; } = "PGD";
}
