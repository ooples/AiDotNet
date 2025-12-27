namespace AiDotNet.SelfSupervisedLearning.Core;

/// <summary>
/// BYOL-specific configuration settings.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> BYOL (Bootstrap Your Own Latent) learns without negative samples
/// by using an asymmetric architecture with a predictor network.</para>
/// </remarks>
public class BYOLConfig
{
    /// <summary>
    /// Gets or sets the base momentum for the target encoder.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.996</c></para>
    /// <para>BYOL typically schedules momentum from 0.996 to 1.0 during training.</para>
    /// </remarks>
    public double? BaseMomentum { get; set; }

    /// <summary>
    /// Gets or sets the final momentum (for momentum scheduling).
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>1.0</c></para>
    /// </remarks>
    public double? FinalMomentum { get; set; }

    /// <summary>
    /// Gets or sets the hidden dimension of the predictor MLP.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>4096</c></para>
    /// </remarks>
    public int? PredictorHiddenDimension { get; set; }

    /// <summary>
    /// Gets or sets the output dimension of the predictor.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>256</c></para>
    /// </remarks>
    public int? PredictorOutputDimension { get; set; }

    /// <summary>
    /// Gets the configuration as a dictionary.
    /// </summary>
    public IDictionary<string, object> GetConfiguration()
    {
        var config = new Dictionary<string, object>();
        if (BaseMomentum.HasValue) config["baseMomentum"] = BaseMomentum.Value;
        if (FinalMomentum.HasValue) config["finalMomentum"] = FinalMomentum.Value;
        if (PredictorHiddenDimension.HasValue) config["predictorHiddenDimension"] = PredictorHiddenDimension.Value;
        if (PredictorOutputDimension.HasValue) config["predictorOutputDimension"] = PredictorOutputDimension.Value;
        return config;
    }
}
