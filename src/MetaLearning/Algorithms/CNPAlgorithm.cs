using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Conditional Neural Process (CNP) (Garnelo et al., ICML 2018).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// CNP encodes each context pair independently, aggregates via mean pooling, and decodes
/// to predict target values. It provides amortized inference without gradient-based adaptation.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// For each task:
///   1. Encode each (x_c, y_c) pair: r_i = encoder(x_c_i, y_c_i)
///   2. Aggregate: r = mean(r_1, ..., r_n)
///   3. For each target x_t: y_t = decoder(r, x_t)
///   4. Loss = MSE(y_t, y_true) or NLL
/// </code>
/// </para>
/// </remarks>
public class CNPAlgorithm<T, TInput, TOutput> : NeuralProcessBase<T, TInput, TOutput>
{
    private readonly CNPOptions<T, TInput, TOutput> _cnpOptions;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.CNP;

    public CNPAlgorithm(CNPOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               options.RepresentationDim)
    {
        _cnpOptions = options;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
        => StandardNPMetaTrain(taskBatch, _cnpOptions.OuterLearningRate);

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
        => StandardNPAdapt(task);
}
