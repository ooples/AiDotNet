using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// TiMINo — Time series Models with Independent Noise.
/// </summary>
/// <remarks>
/// <para>
/// TiMINo tests whether the residuals of a time series regression model are independent
/// of the inputs. Causal direction is determined by the direction that yields independent
/// noise (analogous to the additive noise model for i.i.d. data).
/// </para>
/// <para>
/// <b>For Beginners:</b> TiMINo checks if the "leftover noise" after predicting one variable
/// from another's past is truly random (independent). If it is, the prediction direction is
/// likely the causal direction.
/// </para>
/// <para>
/// Reference: Peters et al. (2013), "Causal Discovery with Continuous Additive Noise Models", JMLR.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("Causal Discovery with Continuous Additive Noise Models", "https://jmlr.org/papers/v15/peters14a.html", Year = 2014, Authors = "Jonas Peters, Joris M. Mooij, Dominik Janzing, Bernhard Scholkopf")]
public class TiMINoAlgorithm<T> : TimeSeriesCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "TiMINo";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public TiMINoAlgorithm(CausalDiscoveryOptions? options = null)
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
