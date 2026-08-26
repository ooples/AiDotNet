using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TOTO (Datadog's Time Series Foundation Model for Observability).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TOTO is Datadog's domain-specific time series foundation model optimized for IT operations,
/// infrastructure monitoring, and observability. Pre-trained on 1 trillion data points from
/// the Datadog observability platform, it excels at SRE metrics and anomaly detection.
/// </para>
/// <para>
/// <b>Reference:</b> Datadog, "Introducing Toto: A state-of-the-art time series foundation model", 2025.
/// </para>
/// </remarks>
public class TOTOOptions<T> : TimeSeriesRegressionOptions<T>
{
    public TOTOOptions() { }

    public TOTOOptions(TOTOOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        // Seed is declared on ModelOptions rather than in this file, so a copy constructor
        // written from the local declarations alone misses it. Losing it on a clone silently
        // changes deterministic initialization.
        Seed = other.Seed;
        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        IntermediateSize = other.IntermediateSize;
        DropoutRate = other.DropoutRate;
        ModelSize = other.ModelSize;
        LearningRate = other.LearningRate;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        WeightDecay = other.WeightDecay;
        WarmupSteps = other.WarmupSteps;
        TotalTrainingSteps = other.TotalTrainingSteps;
    }

    /// <summary>Gets or sets the maximum historical context used for forecasting.</summary>
    /// <value>Defaults to 2048 samples.</value>
    /// <remarks><para><b>For Beginners:</b> This is how much recent history Toto sees at once.</para></remarks>
    public int ContextLength { get; set; } = 2048;

    /// <summary>Gets or sets the number of future samples produced by the native forecast head.</summary>
    /// <value>Defaults to 96 samples.</value>
    /// <remarks><para><b>For Beginners:</b> This is how far into the future one call predicts.</para></remarks>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>Gets or sets the causal patch size.</summary>
    /// <value>Defaults to 32, matching Appendix A, Table A.1 of the Toto paper.</value>
    /// <remarks><para><b>For Beginners:</b> Each token summarizes this many adjacent time points.</para></remarks>
    public int PatchLength { get; set; } = 32;

    /// <summary>Gets or sets the transformer embedding dimension.</summary>
    /// <value>Defaults to 512, matching Appendix A, Table A.1 of the Toto paper.</value>
    /// <remarks><para><b>For Beginners:</b> This is the width of every transformer token.</para></remarks>
    public int HiddenDimension { get; set; } = 512;

    /// <summary>Gets or sets the number of transformer layers.</summary>
    /// <value>Defaults to 24, matching Appendix A, Table A.1 of the Toto paper.</value>
    /// <remarks><para><b>For Beginners:</b> More layers let the model learn deeper temporal patterns.</para></remarks>
    public int NumLayers { get; set; } = 24;

    /// <summary>Gets or sets the number of attention heads.</summary>
    /// <value>Defaults to 8, matching Appendix A, Table A.1 of the Toto paper.</value>
    /// <remarks><para><b>For Beginners:</b> Heads let attention examine several relationships in parallel.</para></remarks>
    public int NumHeads { get; set; } = 8;

    /// <summary>Gets or sets the transformer MLP dimension.</summary>
    /// <value>Defaults to 2048, matching Appendix A, Table A.1 of the Toto paper.</value>
    /// <remarks><para><b>For Beginners:</b> This is the width of each layer's feed-forward block.</para></remarks>
    public int IntermediateSize { get; set; } = 2048;

    /// <summary>Gets or sets the dropout probability.</summary>
    /// <value>Defaults to 0.1.</value>
    /// <remarks><para><b>For Beginners:</b> Dropout randomly hides activations during training to reduce overfitting.</para></remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>Gets or sets the public size label for this configuration.</summary>
    /// <value>Defaults to <see cref="FoundationModelSize.Base"/>.</value>
    /// <remarks><para><b>For Beginners:</b> This label describes the default paper-scale configuration.</para></remarks>
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Base;

    /// <summary>Gets or sets AdamW's peak learning rate.</summary>
    /// <value>Defaults to 0.001, matching Appendix A, Table A.1 of the Toto paper.</value>
    /// <remarks><para><b>For Beginners:</b> This controls the largest optimizer update after warmup.</para></remarks>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>Gets or sets AdamW's first moment coefficient.</summary>
    /// <value>Defaults to 0.9, matching Appendix A, Table A.1 of the Toto paper.</value>
    /// <remarks><para><b>For Beginners:</b> Beta1 smooths the recent gradient direction.</para></remarks>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>Gets or sets AdamW's second moment coefficient.</summary>
    /// <value>Defaults to 0.95, matching Appendix A, Table A.1 of the Toto paper.</value>
    /// <remarks><para><b>For Beginners:</b> Beta2 smooths recent squared gradients for stable scaling.</para></remarks>
    public double Beta2 { get; set; } = 0.95;

    /// <summary>Gets or sets AdamW's decoupled weight decay.</summary>
    /// <value>Defaults to 0.01, matching Appendix A, Table A.1 of the Toto paper.</value>
    /// <remarks><para><b>For Beginners:</b> Weight decay discourages unnecessarily large weights.</para></remarks>
    public double WeightDecay { get; set; } = 0.01;

    /// <summary>Gets or sets the number of linear learning-rate warmup steps.</summary>
    /// <value>Defaults to 5000, matching Appendix A, Table A.1 of the Toto paper.</value>
    /// <remarks><para><b>For Beginners:</b> Warmup gradually raises the update size at the start of training.</para></remarks>
    public int WarmupSteps { get; set; } = 5000;

    /// <summary>Gets or sets the total number of pretraining scheduler steps.</summary>
    /// <value>Defaults to 193000, matching Appendix A, Table A.1 of the Toto paper.</value>
    /// <remarks><para><b>For Beginners:</b> This tells cosine decay when the full paper training run ends.</para></remarks>
    public int TotalTrainingSteps { get; set; } = 193000;
}
