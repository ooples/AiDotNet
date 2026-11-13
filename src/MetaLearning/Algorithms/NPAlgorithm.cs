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

    // Relative step of the feature-path perturbation: probe = FeaturePathProbeScale (1 + |theta|) / sqrt(n).
    private const double FeaturePathProbeScale = 1e-3;

    // Seed of the posterior draws shared by every loss evaluation of one SPSA step, so the paired losses differ by
    // the perturbation alone and not by Monte Carlo noise; null outside an SPSA step.
    private int? _spsaDrawSeed;

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
    /// (backbone) gradient covers both of its roles: the decoder path exactly and the feature path (through both
    /// latent distributions and the KL term) by a paired perturbation over the same draws. The encoders are trained
    /// on the full negative ELBO, with every evaluation of one SPSA step replaying the same draws.
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = InterfaceGuard.Parameterizable(MetaModel).GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            InterfaceGuard.Parameterizable(MetaModel).SetParameters(initParams);
            var (loss, gradient) = NegativeElbo(task, initParams, withGradient: true, RandomGenerator.Next());
            losses.Add(loss);
            if (gradient is not null) metaGradients.Add(gradient);
        }

        ApplyOuterUpdate(initParams, metaGradients, _npOptions.OuterLearningRate);

        // Update auxiliary params
        _spsaDrawSeed = RandomGenerator.Next();
        try
        {
            UpdateAuxiliaryParamsSPSA(taskBatch, ref EncoderParams, _npOptions.OuterLearningRate, ComputeAuxLoss);
            _spsaDrawSeed = RandomGenerator.Next();
            UpdateAuxiliaryParamsSPSA(taskBatch, ref _latentEncoderParams, _npOptions.OuterLearningRate, ComputeAuxLoss);
        }
        finally
        {
            _spsaDrawSeed = null;
        }

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
    /// The task's negative ELBO and, when requested, its gradient with respect to the backbone parameters, each
    /// averaged over <see cref="NPOptions{T, TInput, TOutput}.NumLatentSamples"/> posterior draws taken from
    /// <paramref name="drawSeed"/>.
    /// </summary>
    /// <remarks>
    /// The backbone is both the decoder and the feature extractor of both latent distributions. The decoder path is
    /// differentiated exactly (the installed parameters are scale * initParams, so each draw's gradient carries its
    /// scale). The feature path, through the posterior draw, the modulation scale and the KL term, has no tape here;
    /// a two-sided simultaneous-perturbation estimate over the same draws (Spall 1992) supplies it, with the decoder
    /// held at <paramref name="initParams"/>.
    /// </remarks>
    private (T Loss, Vector<T>? Gradient) NegativeElbo(
        IMetaLearningTask<T, TInput, TOutput> task, Vector<T> initParams, bool withGradient, int drawSeed)
    {
        var (loss, decoderGradient) = SampledObjective(task, initParams, initParams, drawSeed, withGradient);
        if (!withGradient || decoderGradient is null) return (loss, null);

        int n = initParams.Length;
        var signs = Enumerable.Range(0, n).Select(_ => RandomGenerator.NextDouble() > 0.5 ? 1.0 : -1.0).ToArray();
        var direction = TensorCopyHelper.CastLogical<double, T>(new Tensor<double>(signs, new[] { n })).ToVector();
        double probe = FeaturePathProbeScale * (1.0 + NumOps.ToDouble(Engine.Norm(initParams))) / Math.Sqrt(Math.Max(n, 1));
        var step = Engine.Multiply(direction, NumOps.FromDouble(probe));
        T plus = SampledObjective(task, Engine.Add(initParams, step), initParams, drawSeed, false).Loss;
        T minus = SampledObjective(task, Engine.Subtract(initParams, step), initParams, drawSeed, false).Loss;
        double slope = (NumOps.ToDouble(plus) - NumOps.ToDouble(minus)) / (2.0 * probe);
        var featureGradient = Engine.Multiply(direction, NumOps.FromDouble(slope));

        ParamModel.SetParameters(initParams);
        return (loss, ClipGradients(Engine.Add(decoderGradient, featureGradient)));
    }

    /// <summary>
    /// The negative ELBO with the latent distributions encoded by <paramref name="featureParams"/> and the targets
    /// decoded by <paramref name="decoderParams"/> modulated by each draw, and (when requested) the gradient of its
    /// reconstruction term with respect to <paramref name="decoderParams"/>.
    /// </summary>
    private (T Loss, Vector<T>? Gradient) SampledObjective(
        IMetaLearningTask<T, TInput, TOutput> task, Vector<T> featureParams, Vector<T> decoderParams, int drawSeed,
        bool withGradient)
    {
        ParamModel.SetParameters(featureParams);

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

        var draws = RandomHelper.CreateSeededRandom(drawSeed);
        var location = Tensor<T>.FromVector(postMean);
        var spread = Tensor<T>.FromVector(Engine.Exp(Engine.Multiply(postLogVar, NumOps.FromDouble(0.5))));
        int samples = Math.Max(1, _npOptions.NumLatentSamples);
        var reconstruction = new List<T>(samples);
        var gradients = new List<Vector<T>>(samples);
        for (int s = 0; s < samples; s++)
        {
            // z ~ q(z | C, T) by reparameterisation, then decode the targets with it
            var z = Engine.TensorNormal(location, spread, draws.Next()).ToVector();
            double scale = ComputeModScale(z);
            ModulateParameters(decoderParams, scale);
            reconstruction.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            if (withGradient)
                gradients.Add(ScaleVector(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput), scale));
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
        using var indexTensor = new Tensor<int>(indices, new[] { indices.Length });
        return Engine.TensorIndexSelect(Tensor<T>.FromVector(_latentEncoderParams), indexTensor, 0)
            .Reshape(rows, cols).ToMatrix();
    }

    private double ComputeAuxLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var initParams = InterfaceGuard.Parameterizable(MetaModel).GetParameters();
        var losses = new List<T>();
        int drawSeed = _spsaDrawSeed ?? RandomGenerator.Next();

        foreach (var task in taskBatch.Tasks)
        {
            InterfaceGuard.Parameterizable(MetaModel).SetParameters(initParams);
            losses.Add(NegativeElbo(task, initParams, withGradient: false, drawSeed).Loss);
        }

        InterfaceGuard.Parameterizable(MetaModel).SetParameters(initParams);
        return losses.Count == 0 ? 0.0 : NumOps.ToDouble(ComputeMean(losses));
    }
}
