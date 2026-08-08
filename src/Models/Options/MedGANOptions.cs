using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for medGAN, the medical Generative Adversarial Network of
/// Choi et al. (arXiv:1703.06490) for generating synthetic patient records.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// medGAN generates high-dimensional discrete patient records by combining an AUTOENCODER with a
/// GAN. The autoencoder is pre-trained on the real records; the GAN's generator then learns to
/// produce points in the autoencoder's LATENT space, and the pre-trained decoder converts those to
/// records. The paper's three efficiency devices are minibatch averaging (to avoid mode collapse),
/// batch normalization, and shortcut connections in the generator.
/// </para>
/// <para>
/// Defaults reproduce the paper's stated hyperparameters: 128-dimensional embedding, a generator of
/// two 128-wide shortcut layers, a discriminator of 256 then 128, Adam at 1e-3, minibatches of
/// 1,000 records, and k = 2 discriminator updates per generator update.
/// </para>
/// <para>
/// <b>For Beginners:</b> Hospitals cannot share real patient records, but they can share fake ones
/// that have the same statistical shape. medGAN learns that shape in two stages: first it learns to
/// compress and rebuild real records (the autoencoder), then it learns to invent new compressed
/// codes that rebuild into records a critic cannot tell from real ones.
/// </para>
/// <example>
/// <code>
/// var options = new MedGANOptions&lt;double&gt;
/// {
///     EmbeddingDimension = 128,
///     DataType = MedGANDataType.Binary,
/// };
/// var medgan = new MedGANGenerator&lt;double&gt;(architecture, options);
/// medgan.Fit(data, columns, epochs: 1000);
/// var synthetic = medgan.Generate(1000);
/// </code>
/// </example>
/// </remarks>
public class MedGANOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the autoencoder's embedding dimension — the width of the latent space the
    /// generator produces into and the decoder consumes.
    /// </summary>
    /// <remarks>
    /// The paper compresses records to a 128-dimensional representation. The generator's shortcut
    /// connections add a layer's output to its input, which requires equal widths, so this value
    /// also fixes the generator's hidden width and the prior dimension unless
    /// <see cref="NoiseDimension"/> is set.
    /// </remarks>
    /// <value>Embedding dimension, defaulting to 128 (the paper's value).</value>
    public int EmbeddingDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the dimension r of the random prior z ~ N(0, 1).
    /// </summary>
    /// <remarks>
    /// The reference implementation uses the same value as <see cref="EmbeddingDimension"/> so that
    /// every generator layer, including the first, can carry a shortcut connection. When this is
    /// null the embedding dimension is used.
    /// </remarks>
    /// <value>Prior dimension, defaulting to null (match <see cref="EmbeddingDimension"/>).</value>
    public int? NoiseDimension { get; set; }

    /// <summary>
    /// Gets or sets the autoencoder's hidden layer widths BEFORE the embedding layer.
    /// </summary>
    /// <remarks>
    /// Empty by default, matching the paper's single-layer feedforward autoencoder: the encoder is
    /// one projection from the record width straight to <see cref="EmbeddingDimension"/>, and the
    /// decoder is its mirror. Supplying widths here deepens both halves symmetrically.
    /// </remarks>
    /// <value>Hidden widths, defaulting to empty.</value>
    public int[] AutoencoderDimensions { get; set; } = [];

    /// <summary>
    /// Gets or sets the generator's hidden layer widths, each carrying a shortcut connection.
    /// </summary>
    /// <remarks>
    /// The paper: the generator has two hidden layers of 128 dimensions each. Because the shortcut
    /// is an addition, every width here must equal <see cref="EmbeddingDimension"/>; any other value
    /// is rejected at construction rather than silently dropping the shortcut.
    /// </remarks>
    /// <value>Generator hidden widths, defaulting to [128, 128] (the paper's value).</value>
    public int[] GeneratorDimensions { get; set; } = [128, 128];

    /// <summary>
    /// Gets or sets the discriminator's hidden layer widths.
    /// </summary>
    /// <value>Discriminator hidden widths, defaulting to [256, 128] (the paper's value).</value>
    public int[] DiscriminatorDimensions { get; set; } = [256, 128];

    /// <summary>
    /// Gets or sets the kind of variables the records hold, which selects the autoencoder's
    /// activations and pre-training loss.
    /// </summary>
    /// <value>Defaults to <see cref="MedGANDataType.MixedTabular"/>, the shape produced by this
    /// library's tabular transformer. Set <see cref="MedGANDataType.Binary"/> for the paper's
    /// binary-code EHR matrices.</value>
    public MedGANDataType DataType { get; set; } = MedGANDataType.MixedTabular;

    /// <summary>
    /// Gets or sets the number of epochs spent pre-training the autoencoder before the GAN begins.
    /// </summary>
    /// <remarks>
    /// The paper pre-trains the autoencoder until convergence and only then starts the GAN, after
    /// which the decoder continues to be updated adversarially. A bounded budget is used here so
    /// <c>Fit</c> terminates; when this is null it is derived from
    /// <see cref="AutoencoderPretrainFraction"/>.
    /// </remarks>
    /// <value>Explicit pre-training epochs, defaulting to null (derive from the fraction).</value>
    public int? AutoencoderPretrainEpochs { get; set; }

    /// <summary>
    /// Gets or sets the share of the total epoch budget spent pre-training the autoencoder when
    /// <see cref="AutoencoderPretrainEpochs"/> is null.
    /// </summary>
    /// <value>Fraction in (0, 1), defaulting to 0.1.</value>
    public double AutoencoderPretrainFraction { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the decay used by the generator's batch normalization moving averages.
    /// </summary>
    /// <value>Decay, defaulting to 0.99 (the paper's value).</value>
    public double BatchNormDecay { get; set; } = 0.99;

    /// <summary>
    /// Gets or sets whether the discriminator sees each sample concatenated with the minibatch
    /// average, medGAN's remedy for mode collapse.
    /// </summary>
    /// <remarks>
    /// On by default because it is one of the paper's three named contributions. Turning it off
    /// halves the discriminator's input width and reproduces a plain GAN discriminator; it exists
    /// only so the mechanism's effect can be measured against its absence.
    /// </remarks>
    /// <value>True by default.</value>
    public bool UseMinibatchAveraging { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of discriminator updates per generator update (k).
    /// </summary>
    /// <value>Defaults to 2 (the paper's value).</value>
    public int DiscriminatorSteps { get; set; } = 2;

    /// <summary>
    /// Gets or sets the training minibatch size.
    /// </summary>
    /// <value>Defaults to 1000 (the paper's value).</value>
    public int BatchSize { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>Defaults to 1000 (the paper's value).</value>
    public int Epochs { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the Adam learning rate.
    /// </summary>
    /// <value>Defaults to 1e-3 (the paper's value).</value>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the number of VGM modes used to transform continuous columns.
    /// </summary>
    /// <remarks>
    /// Applies only to <see cref="MedGANDataType.MixedTabular"/>; the paper's binary and count
    /// matrices need no such transform.
    /// </remarks>
    /// <value>Number of modes, defaulting to 10.</value>
    public int VGMModes { get; set; } = 10;

    /// <summary>
    /// Gets or sets whether to train the discriminator with differentially private SGD.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Beyond the paper.</b> medGAN offers no formal privacy guarantee — it measures the residual
    /// risk of identity and attribute disclosure empirically and reports it as limited, but that is
    /// an observation, not a bound. Enabling this trains the discriminator under the DP-SGD
    /// mechanism of Abadi et al. (2016), which does give a bound; the generator needs no separate
    /// treatment because it never touches real data, so its privacy follows by post-processing.
    /// </para>
    /// <para>
    /// Off by default so the default configuration is exactly the paper's.
    /// </para>
    /// </remarks>
    /// <value>False by default.</value>
    public bool EnablePrivacy { get; set; } = false;

    /// <summary>
    /// Gets or sets the privacy budget epsilon used when <see cref="EnablePrivacy"/> is set.
    /// </summary>
    /// <value>Epsilon, defaulting to 3.0.</value>
    public double Epsilon { get; set; } = 3.0;

    /// <summary>
    /// Gets or sets the per-example gradient clipping norm C used when <see cref="EnablePrivacy"/>
    /// is set.
    /// </summary>
    /// <value>Clip norm, defaulting to 1.0.</value>
    public double ClipNorm { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the weight on a penalty for generated values outside the per-column range
    /// observed in the training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Beyond the paper.</b> medGAN has no validity term; its outputs are constrained only by the
    /// decoder's output activation. This penalty is available for domains where an out-of-range
    /// value is not merely unlikely but clinically impossible.
    /// </para>
    /// <para>
    /// Zero by default so the default objective is exactly the paper's.
    /// </para>
    /// </remarks>
    /// <value>Weight, defaulting to 0.0 (disabled).</value>
    public double ConstraintWeight { get; set; } = 0.0;
}
