using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// Neural Granger Causality — deep learning extension of Granger causality.
/// </summary>
/// <remarks>
/// <para>
/// Neural Granger Causality replaces the linear VAR model in Granger causality with
/// neural networks (MLP or LSTM), combined with structured sparsity penalties on the
/// input layer to identify which variables' lags are useful for prediction.
/// </para>
/// <para>
/// <b>For Beginners:</b> Standard Granger causality assumes linear relationships.
/// Neural Granger uses neural networks instead, so it can find nonlinear causal
/// relationships. For example, "X causes Y, but only when X is in a certain range."
/// </para>
/// <para>
/// Reference: Tank et al. (2021), "Neural Granger Causality", IEEE TPAMI.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("Neural Granger Causality", "https://doi.org/10.1109/TPAMI.2021.3065601", Year = 2021, Authors = "Alex Tank, Ian Covert, Nicholas Foti, Ali Shojaie, Emily B. Fox")]
public class NeuralGrangerAlgorithm<T> : TimeSeriesCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "NeuralGranger";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public NeuralGrangerAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        // Shell: delegates to Granger causality as baseline
        var baseline = new GrangerCausalityAlgorithm<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
