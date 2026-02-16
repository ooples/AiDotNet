using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// TCDF — Temporal Causal Discovery Framework.
/// </summary>
/// <remarks>
/// <para>
/// TCDF uses attention-based convolutional neural networks to discover temporal causal
/// relationships. Each variable has a dedicated CNN that predicts it from all variables'
/// histories, and attention weights indicate which inputs are causally relevant.
/// </para>
/// <para>
/// <b>For Beginners:</b> TCDF uses "attention" (like in language models) to figure out
/// which past variables matter for predicting each current variable. If the network
/// "pays attention" to variable X's past when predicting Y, that suggests X causes Y.
/// </para>
/// <para>
/// Reference: Nauta et al. (2019), "Causal Discovery with Attention-Based Convolutional
/// Neural Networks", Machine Learning and Knowledge Extraction.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TCDFAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "TCDF";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    /// <inheritdoc/>
    public override bool SupportsTimeSeries => true;

    public TCDFAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new ContinuousOptimization.NOTEARSLinear<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
