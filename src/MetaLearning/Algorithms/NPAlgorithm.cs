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

/// <summary>
/// Implementation of Neural Process (NP) (Garnelo et al., 2018).
/// Extends CNP with a latent variable z for modeling uncertainty and function correlations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>Algorithm:</b>
/// <code>
/// For each task:
///   1. Encode context: r_c = aggregate(encoder(x_c_i, y_c_i))
///   2. Compute latent: mu, sigma = latent_encoder(r_c)
///   3. Sample: z ~ N(mu, sigma) via reparameterization
///   4. Decode: y_t = decoder(z, x_t) for each target
///   5. Loss = -log_likelihood + KL(q(z|context,target) || q(z|context))
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Neural Processes",
    "https://arxiv.org/abs/1807.01622",
    Year = 2018,
    Authors = "Marta Garnelo, Jonathan Schwarz, Dan Rosenbaum, Fabio Viola, et al.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class NPAlgorithm<T, TInput, TOutput> : NeuralProcessBase<T, TInput, TOutput>
{
    private readonly NPOptions<T, TInput, TOutput> _npOptions;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _latentEncoderParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.NP;

    public NPAlgorithm(NPOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               options.RepresentationDim)
    {
        if (options.LatentDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "LatentDim must be positive.");

        _npOptions = options;
        _latentEncoderParams = InitializeParams(options.LatentDim * 4);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Garnelo et al. (2018), Eq. 4: each task minimises the negative ELBO
    /// -E_{q(z|C,T)}[log p(y_T | z, x_T)] + KL(q(z|C,T) || q(z|C)). The expectation is a Monte Carlo average over
    /// <see cref="NPOptions{T, TInput, TOutput}.NumLatentSamples"/> reparameterised posterior draws; the decoder
    /// (backbone) gradient is the gradient of that reconstruction average, because the KL term does not depend on
    /// the decoder. The latent encoder is trained on the full negative ELBO.
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = InterfaceGuard.Parameterizable(MetaModel).GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            InterfaceGuard.Parameterizable(MetaModel).SetParameters(initParams);
            var (loss, gradient) = NegativeElbo(task, initParams, withGradient: true);
            losses.Add(loss);
            if (gradient is not null) metaGradients.Add(gradient);
        }

        ApplyOuterUpdate(initParams, metaGradients, _npOptions.OuterLearningRate);

        // Update auxiliary params
        UpdateAuxiliaryParamsSPSA(taskBatch, ref EncoderParams, _npOptions.OuterLearningRate, ComputeAuxLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _latentEncoderParams, _npOptions.OuterLearningRate, ComputeAuxLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = InterfaceGuard.Parameterizable(MetaModel).GetParameters();

        var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
        var supportLabels = ConvertToVector(task.SupportOutput);
        var contextRep = AggregateRepresentations(BuildContextRepresentations(supportFeatures, supportLabels));

        var (mean, logvar) = ComputeLatentDistribution(contextRep);
        // Use mean for deterministic prediction at adaptation time
        var z = mean;

        return new NeuralProcessModel<T, TInput, TOutput>(MetaModel, ScaleVector(currentParams, ComputeModScale(z)), contextRep);
    }

    /// <summary>
    /// The task's negative ELBO and, when requested, the decoder gradient of its reconstruction term, each averaged
    /// over <see cref="NPOptions{T, TInput, TOutput}.NumLatentSamples"/> posterior draws.
    /// </summary>
    private (T Loss, Vector<T>? Gradient) NegativeElbo(
        IMetaLearningTask<T, TInput, TOutput> task, Vector<T> initParams, bool withGradient)
    {
        // Encode context (support)
        var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
        var supportLabels = ConvertToVector(task.SupportOutput);
        var contextRep = AggregateRepresentations(BuildContextRepresentations(supportFeatures, supportLabels));

        // Prior q(z | C) from the context alone
        var (priorMean, priorLogVar) = ComputeLatentDistribution(contextRep);

        // Posterior q(z | C, T) from context and targets
        var queryFeatures = ConvertToVector(MetaModel.Predict(task.QueryInput));
        var queryLabels = ConvertToVector(task.QueryOutput);
        var fullRep = AggregateRepresentations(BuildContextRepresentations(
            ConcatVectors(supportFeatures, queryFeatures),
            ConcatVectors(supportLabels, queryLabels)));
        var (postMean, postLogVar) = ComputeLatentDistribution(fullRep);

        int samples = Math.Max(1, _npOptions.NumLatentSamples);
        var reconstruction = new List<T>(samples);
        var gradients = new List<Vector<T>>(samples);
        for (int s = 0; s < samples; s++)
        {
            // z ~ q(z | C, T) by reparameterisation, then decode the targets with it
            var z = ReparameterizeSample(postMean, postLogVar);
            ModulateParameters(initParams, ComputeModScale(z));
            reconstruction.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            if (withGradient)
                gradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        var kl = GaussianKL(postMean, postLogVar, priorMean, priorLogVar);
        var loss = NumOps.Add(ComputeMean(reconstruction), NumOps.Multiply(NumOps.FromDouble(_npOptions.KLWeight), kl));
        return (loss, withGradient ? AverageVectors(gradients) : null);
    }

    /// <summary>
    /// KL(N(mean1, exp(logvar1)) || N(mean2, exp(logvar2))) for diagonal Gaussians:
    /// 0.5 * sum(logvar2 - logvar1 + (exp(logvar1) + (mean1 - mean2)^2) / exp(logvar2) - 1).
    /// </summary>
    internal static T GaussianKL(Vector<T> mean1, Vector<T> logvar1, Vector<T> mean2, Vector<T> logvar2)
    {
        var engine = AiDotNetEngine.Current;
        var difference = engine.Subtract(mean1, mean2);
        var ratio = engine.Divide(
            engine.Add(engine.Exp(logvar1), engine.Multiply(difference, difference)), engine.Exp(logvar2));
        var terms = engine.Add(engine.Subtract(logvar2, logvar1), ratio);
        return NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Subtract(engine.Sum(terms), NumOps.FromDouble(mean1.Length)));
    }

    /// <summary>
    /// The latent encoder: mean = tanh(W_mu r) and log-variance = clamp(W_sigma r, -4, 2), with both weight blocks
    /// read cyclically from the latent encoder parameters.
    /// </summary>
    private (Vector<T> mean, Vector<T> logvar) ComputeLatentDistribution(Vector<T> representation)
    {
        int latentDim = _npOptions.LatentDim;
        int width = Math.Min(representation.Length, RepresentationDim);
        if (width == 0)
        {
            // No representation: both linear maps are zero, so mean = tanh(0) = 0 and log-variance = 0.
            return (new Vector<T>(latentDim), new Vector<T>(latentDim));
        }

        var input = representation.GetSubVector(0, width);
        var mean = Engine.MatrixVectorMultiply(EncoderBlock(0, latentDim, width), input);
        var logvar = Engine.Clamp(
            Engine.MatrixVectorMultiply(EncoderBlock(latentDim * RepresentationDim, latentDim, width), input),
            NumOps.FromDouble(-4), NumOps.FromDouble(2));

        return (VectorTanh(mean), logvar);
    }

    /// <summary>
    /// The rows x cols weight block whose (i, j) entry is the latent encoder parameter at
    /// (offset + i * RepresentationDim + j) modulo the parameter count.
    /// </summary>
    private Matrix<T> EncoderBlock(int offset, int rows, int cols)
    {
        int length = _latentEncoderParams.Length;
        int stride = RepresentationDim;
        var indices = Enumerable.Range(0, rows * cols)
            .Select(k => (int)((offset + (long)(k / cols) * stride + k % cols) % length))
            .ToArray();
        return Engine.TensorIndexSelect(
                Tensor<T>.FromVector(_latentEncoderParams), new Tensor<int>(indices, new[] { indices.Length }), 0)
            .Reshape(rows, cols).ToMatrix();
    }

    private double ComputeAuxLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var initParams = InterfaceGuard.Parameterizable(MetaModel).GetParameters();
        var losses = new List<T>();

        foreach (var task in taskBatch.Tasks)
        {
            InterfaceGuard.Parameterizable(MetaModel).SetParameters(initParams);
            losses.Add(NegativeElbo(task, initParams, withGradient: false).Loss);
        }

        InterfaceGuard.Parameterizable(MetaModel).SetParameters(initParams);
        return losses.Count == 0 ? 0.0 : NumOps.ToDouble(ComputeMean(losses));
    }
}
