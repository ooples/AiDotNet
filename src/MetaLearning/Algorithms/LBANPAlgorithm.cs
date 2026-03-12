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

/// <summary>Implementation of Latent Bottleneck Attentive Neural Process (Feng et al., ICML 2023).</summary>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Latent Bottleneck Attentive Neural Processes",
    "https://arxiv.org/abs/2211.08458",
    Year = 2023,
    Authors = "Leo Feng, Hossein Hajimirsadeghi, Yoshua Bengio, Mohamed Osama Ahmed")]
public class LBANPAlgorithm<T, TInput, TOutput> : NeuralProcessBase<T, TInput, TOutput>
{
    private readonly LBANPOptions<T, TInput, TOutput> _algoOptions;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.LBANP;

    public LBANPAlgorithm(LBANPOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               options.RepresentationDim)
    {
        _algoOptions = options;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
        => StandardNPMetaTrain(taskBatch, _algoOptions.OuterLearningRate);

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
        => StandardNPAdapt(task);
}
