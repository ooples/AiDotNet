namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the FactorVAE model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FactorVAE is a variational autoencoder tailored to learning disentangled financial factors.
/// These options control the latent space size, factor count, and regularization strengths.
/// </para>
/// <para>
/// <b>For Beginners:</b> FactorVAE learns a compressed representation of market data.
/// You can decide how many hidden factors to discover and how strongly to enforce disentanglement.
/// </para>
/// </remarks>
public class FactorVAEOptions<T> : ModelOptions
{
    /// <summary>
    /// Gets or sets the number of latent factors to learn.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the number of independent drivers the model tries to uncover.
    /// </para>
    /// </remarks>
    public int NumFactors { get; set; } = 10;

    /// <summary>
    /// Gets or sets the number of assets covered by the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Controls output size for asset-level reconstructions.
    /// </para>
    /// </remarks>
    public int NumAssets { get; set; } = 500;

    /// <summary>
    /// Gets or sets the number of input features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Features can include prices, returns, and technical indicators.
    /// </para>
    /// </remarks>
    public int NumFeatures { get; set; } = 50;

    /// <summary>
    /// Gets or sets the width of hidden layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Larger hidden layers can capture more complex patterns.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the dimension of the latent space.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the size of the compressed representation.
    /// A larger value preserves more detail but is harder to disentangle.
    /// </para>
    /// </remarks>
    public int LatentDimension { get; set; } = 32;

    /// <summary>
    /// Gets or sets the input sequence length.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many time steps the model looks at for each example.
    /// </para>
    /// </remarks>
    public int SequenceLength { get; set; } = 60;

    /// <summary>
    /// Gets or sets the prediction horizon.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many steps into the future the model is trained to predict.
    /// </para>
    /// </remarks>
    public int PredictionHorizon { get; set; } = 20;

    /// <summary>
    /// Gets or sets the beta coefficient for the VAE KL term.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Beta controls how strongly the latent space is regularized.
    /// Higher beta encourages more structured, disentangled factors.
    /// </para>
    /// </remarks>
    public double Beta { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the gamma coefficient for the factor disentanglement penalty.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Gamma increases the penalty for entangled factors,
    /// encouraging each latent dimension to capture a distinct driver.
    /// </para>
    /// </remarks>
    public double Gamma { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the weight on the prior-posterior KL divergence.
    /// </summary>
    /// <value>Defaults to 1.0, keeping the reconstruction term dominant.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> during training the model may look at the realized returns to work out
    /// what the hidden factors were. At prediction time it cannot. This weight controls how strongly it
    /// is pushed to reach the same conclusion WITHOUT peeking, which is what makes the model usable for
    /// prediction at all.
    /// </para>
    /// <para>
    /// Deliberately separate from <see cref="Gamma"/>. Gamma is the disentanglement / total-correlation
    /// coefficient belonging to the OTHER paper also called FactorVAE (Kim &amp; Mnih, <i>Disentangling
    /// by Factorising</i>), where 10.0 is a normal value. Reusing it for this KL would apply a 10x
    /// weight that swamps the reconstruction term and redefine a public property's meaning, so the
    /// prior-posterior term gets its own knob.
    /// </para>
    /// </remarks>
    public double KlWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the dropout rate used for regularization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dropout reduces overfitting by randomly disabling neurons during training.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether the model's default Adam optimizer uses the AMSGrad second-moment maximum.
    /// </summary>
    /// <value>Defaults to <see langword="true"/>.</value>
    /// <remarks>
    /// <para>
    /// FactorVAE samples latent factors during training. On this stochastic objective, plain Adam can
    /// drift slightly after reaching a good solution because its effective denominator may decrease.
    /// AMSGrad keeps that denominator non-decreasing and stabilizes longer training runs.
    /// </para>
    /// <para>
    /// This setting applies only when no optimizer is supplied to the model constructor. A
    /// caller-provided optimizer remains authoritative.
    /// </para>
    /// </remarks>
    public bool UseAMSGrad { get; set; } = true;

    /// <summary>
    /// Validates the options and throws if any value is invalid.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This ensures your settings are reasonable before training starts.
    /// </para>
    /// </remarks>
    public void Validate()
    {
        if (NumFactors <= 0)
            throw new ArgumentOutOfRangeException(nameof(NumFactors), "NumFactors must be positive.");
        if (NumAssets <= 0)
            throw new ArgumentOutOfRangeException(nameof(NumAssets), "NumAssets must be positive.");
        if (NumFeatures <= 0)
            throw new ArgumentOutOfRangeException(nameof(NumFeatures), "NumFeatures must be positive.");
        if (HiddenDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(HiddenDimension), "HiddenDimension must be positive.");
        if (LatentDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(LatentDimension), "LatentDimension must be positive.");
        if (SequenceLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(SequenceLength), "SequenceLength must be positive.");
        if (PredictionHorizon <= 0)
            throw new ArgumentOutOfRangeException(nameof(PredictionHorizon), "PredictionHorizon must be positive.");
        if (Beta <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(Beta), "Beta must be positive.");
        if (Gamma <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(Gamma), "Gamma must be positive.");
        if (DropoutRate < 0.0 || DropoutRate >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(DropoutRate), "DropoutRate must be in [0, 1).");
    }
}
