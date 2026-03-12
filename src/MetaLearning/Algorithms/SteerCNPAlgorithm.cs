using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>Implementation of Steerable Conditional Neural Process (Holderrieth et al., 2021).</summary>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Equivariant Learning of Stochastic Fields",
    "https://arxiv.org/abs/2106.10972",
    Year = 2021,
    Authors = "Peter Holderrieth, Michael J. Hutchinson, Yee Whye Teh")]
public class SteerCNPAlgorithm<T, TInput, TOutput> : NeuralProcessBase<T, TInput, TOutput>
{
    private readonly SteerCNPOptions<T, TInput, TOutput> _algoOptions;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.SteerCNP;

    public SteerCNPAlgorithm(SteerCNPOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               options.RepresentationDim)
    {
        if (options.RepresentationDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "RepresentationDim must be positive.");

        _algoOptions = options;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
        => StandardNPMetaTrain(taskBatch, _algoOptions.OuterLearningRate);

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
        => StandardNPAdapt(task);
}
