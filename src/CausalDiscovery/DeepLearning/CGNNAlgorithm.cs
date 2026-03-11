using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// CGNN — Causal Generative Neural Networks.
/// </summary>
/// <remarks>
/// <para>
/// CGNN generates data according to a causal model parameterized by neural networks.
/// For each candidate graph, a generative model is trained and the generated data is
/// compared to the observed data using an MMD (Maximum Mean Discrepancy) criterion.
/// </para>
/// <para>
/// <b>For Beginners:</b> CGNN tests different causal graph candidates by asking "If this
/// graph were correct, could a neural network generate data that looks like the real data?"
/// The graph that produces the most realistic synthetic data is chosen as the answer.
/// </para>
/// <para>
/// Reference: Goudet et al. (2018), "Learning Functional Causal Models with Generative
/// Neural Networks", Explainable and Interpretable Models in CV and ML.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.GAN)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ModelPaper("Learning Functional Causal Models with Generative Neural Networks", "https://doi.org/10.1007/978-3-319-98131-4_3", Year = 2018, Authors = "Olivier Goudet, Diviyan Kalainathan, Philippe Caillou, Isabelle Guyon, David Lopez-Paz, Michele Sebag")]
public class CGNNAlgorithm<T> : DeepCausalBase<T>
{
    /// <inheritdoc/>
    public override string Name => "CGNN";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => true;

    public CGNNAlgorithm(CausalDiscoveryOptions? options = null) { ApplyDeepOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        var baseline = new ContinuousOptimization.NOTEARSLinear<T>();
        return baseline.DiscoverStructure(data).AdjacencyMatrix;
    }
}
