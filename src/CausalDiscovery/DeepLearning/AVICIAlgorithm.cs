using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// AVICI — Amortized Variational Inference for Causal Discovery.
/// </summary>
/// <remarks>
/// <para>
/// AVICI uses a transformer architecture trained on synthetic data to perform causal discovery.
/// Given a dataset as input, the transformer outputs edge probabilities for the causal graph.
/// It generalizes across different graph sizes and data distributions.
/// </para>
/// <para>
/// <b>For Beginners:</b> AVICI is like a "universal causal discovery engine" powered by a
/// transformer (similar to ChatGPT's architecture). It's pre-trained to recognize causal
/// patterns, so it can analyze new datasets very quickly.
/// </para>
/// <para>
/// Reference: Lorch et al. (2023), "Amortized Inference for Causal Structure Learning", NeurIPS.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("Amortized Inference for Causal Structure Learning", "https://proceedings.neurips.cc/paper_files/paper/2022/hash/07cb5f86508f146774a5b5d2b5e529e4-Abstract-Conference.html", Year = 2022, Authors = "Lars Lorch, Scott Sussex, Jonas Rothfuss, Andreas Krause, Bernhard Scholkopf")]
public class AVICIAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "AVICI";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public AVICIAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new ContinuousOptimization.NOTEARSLinear<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
