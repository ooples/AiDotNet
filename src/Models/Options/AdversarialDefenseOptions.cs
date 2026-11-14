using AiDotNet.Enums;
using AiDotNet.Interfaces;
using Newtonsoft.Json;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for adversarial defense mechanisms.
/// </summary>
/// <remarks>
/// <para>
/// These options control adversarial training (Madry et al. 2018, "Towards Deep Learning Models Resistant to
/// Adversarial Attacks"), the clean/adversarial mixing ratio of Goodfellow et al. 2015 ("Explaining and Harnessing
/// Adversarial Examples"), input preprocessing defenses, and ensembles of independently trained defended models.
/// </para>
/// <para><b>For Beginners:</b> These settings control how your "armor" protects the AI model.
/// You can adjust how the defense is applied, how strong it should be, and what techniques to use.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public class AdversarialDefenseOptions<T> : ModelOptions
{
    /// <summary>
    /// Gets or sets the share of every training mini-batch that is replaced by adversarial examples.
    /// </summary>
    /// <value>The adversarial ratio in [0, 1], defaulting to 0.5.</value>
    /// <remarks>
    /// <para>
    /// Goodfellow et al. 2015 train on alpha J(x, y) + (1 - alpha) J(x_adv, y) with alpha = 0.5. With per-sample
    /// updates this is a mini-batch in which round(AdversarialRatio * batch) samples are adversarial and the rest
    /// clean. 1.0 is the pure min-max objective of Madry et al. 2018; 0.0 is standard training.
    /// </para>
    /// <para><b>For Beginners:</b> 0.5 means 50% clean and 50% adversarial examples, balancing
    /// robustness and normal accuracy.</para>
    /// </remarks>
    public double AdversarialRatio { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the perturbation budget of the training attack.
    /// </summary>
    /// <value>The epsilon value, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how strong the adversarial examples during training are.
    /// Training on stronger attacks makes the model more robust but might reduce clean accuracy.</para>
    /// </remarks>
    public double Epsilon { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the number of adversarial training epochs.
    /// </summary>
    /// <value>The number of epochs, defaulting to 100.</value>
    /// <remarks>
    /// <para>Each epoch is one shuffled pass over the training data in mini-batches of <see cref="BatchSize"/>.
    /// 0 skips training, leaving only the preprocessing defense.</para>
    /// <para><b>For Beginners:</b> An epoch is one complete pass through all training data.
    /// More epochs allow the model to learn better defenses but take longer.</para>
    /// </remarks>
    public int TrainingEpochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the mini-batch size of adversarial training.
    /// </summary>
    /// <value>The batch size, defaulting to 128 (Madry et al. 2018).</value>
    public int BatchSize { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of iterations of the PGD training attack.
    /// </summary>
    /// <value>Defaults to 7, with a step of <see cref="Epsilon"/> / 4: the CIFAR-10 setting of Madry et al. 2018
    /// (7 steps of size 2 at epsilon 8).</value>
    public int AttackIterations { get; set; } = 7;

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
    /// Gets or sets the built-in preprocessing defense.
    /// </summary>
    /// <value>Defaults to <see cref="AdversarialPreprocessingMethod.JpegCompression"/>.</value>
    /// <remarks>
    /// <para>Ignored when <see cref="CustomPreprocessor"/> is set.</para>
    /// <para><b>For Beginners:</b> Different preprocessing methods work better for different types
    /// of data. JPEG compression, for example, can remove small adversarial changes from images.</para>
    /// </remarks>
    public AdversarialPreprocessingMethod PreprocessingMethod { get; set; } = AdversarialPreprocessingMethod.JpegCompression;

    /// <summary>
    /// Gets or sets a custom preprocessing defense that replaces <see cref="PreprocessingMethod"/>.
    /// </summary>
    /// <value>Null (the default) uses the built-in method.</value>
    [JsonIgnore]
    public IAdversarialInputPreprocessor<T>? CustomPreprocessor { get; set; }

    /// <summary>
    /// Gets or sets the bit depth of <see cref="AdversarialPreprocessingMethod.BitDepthReduction"/>.
    /// </summary>
    /// <value>Defaults to 4 bits, the ImageNet setting of Xu et al. 2018 (they use 1 bit for MNIST, 5 for CIFAR-10).</value>
    public int BitDepth { get; set; } = 4;

    /// <summary>
    /// Gets or sets the window side of <see cref="AdversarialPreprocessingMethod.MedianSmoothing"/>.
    /// </summary>
    /// <value>Defaults to 2, the 2 x 2 window Xu et al. 2018 use for MNIST and CIFAR-10 (3 x 3 for ImageNet).</value>
    public int MedianFilterSize { get; set; } = 2;

    /// <summary>
    /// Gets or sets whether to train an ensemble of defended models and average their outputs.
    /// </summary>
    /// <value>True to use ensembles, false otherwise (default: false).</value>
    /// <remarks>
    /// <para>Each member is a deep copy of the model trained adversarially with its own shuffling and attack random
    /// starts. Predictions are the mean of the members' outputs, so an attack must fool the average.</para>
    /// <para><b>For Beginners:</b> Ensemble defenses use multiple models to make predictions.
    /// It's harder to fool all models at once, making the system more robust.</para>
    /// </remarks>
    public bool UseEnsemble { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of models in the ensemble.
    /// </summary>
    /// <value>The ensemble size, defaulting to 3. Read only when <see cref="UseEnsemble"/> is true.</value>
    public int EnsembleSize { get; set; } = 3;

    /// <summary>
    /// Gets or sets the attack that generates the adversarial training examples.
    /// </summary>
    /// <value>Defaults to <see cref="AdversarialAttackMethod.Pgd"/>, the attack of Madry et al. 2018.</value>
    /// <remarks>
    /// <para>To train against a custom attack, pass it to the
    /// <see cref="AiDotNet.AdversarialRobustness.Defenses.AdversarialTraining{T, TInput, TOutput}"/> constructor.</para>
    /// <para><b>For Beginners:</b> This determines what type of attacks to train against.
    /// PGD is a strong iterative attack that provides good robustness when used for training.</para>
    /// </remarks>
    public AdversarialAttackMethod AttackMethod { get; set; } = AdversarialAttackMethod.Pgd;
}
