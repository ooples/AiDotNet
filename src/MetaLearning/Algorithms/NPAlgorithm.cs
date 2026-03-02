using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

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
public class NPAlgorithm<T, TInput, TOutput> : NeuralProcessBase<T, TInput, TOutput>
{
    private readonly NPOptions<T, TInput, TOutput> _npOptions;
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
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);

            // Encode context (support)
            var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
            var supportLabels = ConvertToVector(task.SupportOutput);
            var contextRep = AggregateRepresentations(BuildContextRepresentations(supportFeatures, supportLabels));

            // Compute latent distribution from context
            var (priorMean, priorLogVar) = ComputeLatentDistribution(contextRep);

            // Encode full set (context + target) for posterior
            var queryFeatures = ConvertToVector(MetaModel.Predict(task.QueryInput));
            var queryLabels = ConvertToVector(task.QueryOutput);
            var fullRep = AggregateRepresentations(BuildContextRepresentations(
                ConcatVectors(supportFeatures, queryFeatures),
                ConcatVectors(supportLabels, queryLabels)));
            var (postMean, postLogVar) = ComputeLatentDistribution(fullRep);

            // Sample z from posterior and modulate backbone
            var z = ReparameterizeSample(postMean, postLogVar);
            ModulateParameters(initParams, ComputeModScale(z));

            // Compute reconstruction loss + KL
            double reconLoss = NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            double klLoss = KLDivergenceGaussian(postMean, postLogVar, priorMean, priorLogVar);
            double totalLoss = reconLoss + _npOptions.KLWeight * klLoss;

            losses.Add(NumOps.FromDouble(totalLoss));

            // Compute task gradient from reconstruction loss
            var reconGrad = ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput));

            // Approximate KL gradient contribution via finite differences on the latent modulation
            // The KL term regularizes the posterior toward the prior; its gradient w.r.t. parameters
            // is propagated by scaling the reconstruction gradient proportionally
            double klScale = _npOptions.KLWeight * klLoss / (reconLoss + 1e-10);
            var metaGrad = new Vector<T>(reconGrad.Length);
            for (int d = 0; d < reconGrad.Length; d++)
            {
                double rg = NumOps.ToDouble(reconGrad[d]);
                metaGrad[d] = NumOps.FromDouble(rg * (1.0 + klScale));
            }
            metaGradients.Add(metaGrad);
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
        var currentParams = MetaModel.GetParameters();

        var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
        var supportLabels = ConvertToVector(task.SupportOutput);
        var contextRep = AggregateRepresentations(BuildContextRepresentations(supportFeatures, supportLabels));

        var (mean, logvar) = ComputeLatentDistribution(contextRep);
        // Use mean for deterministic prediction at adaptation time
        var z = mean;

        return new NeuralProcessModel<T, TInput, TOutput>(MetaModel, ScaleVector(currentParams, ComputeModScale(z)), contextRep);
    }

    private (Vector<T> mean, Vector<T> logvar) ComputeLatentDistribution(Vector<T> representation)
    {
        int latentDim = _npOptions.LatentDim;
        var mean = new Vector<T>(latentDim);
        var logvar = new Vector<T>(latentDim);

        for (int i = 0; i < latentDim; i++)
        {
            double sum = 0;
            for (int j = 0; j < Math.Min(representation.Length, RepresentationDim); j++)
            {
                int paramIdx = (i * RepresentationDim + j) % _latentEncoderParams.Length;
                sum += NumOps.ToDouble(representation[j]) * NumOps.ToDouble(_latentEncoderParams[paramIdx]);
            }
            mean[i] = NumOps.FromDouble(Math.Tanh(sum));

            double sumVar = 0;
            for (int j = 0; j < Math.Min(representation.Length, RepresentationDim); j++)
            {
                int paramIdx = (latentDim * RepresentationDim + i * RepresentationDim + j) % _latentEncoderParams.Length;
                sumVar += NumOps.ToDouble(representation[j]) * NumOps.ToDouble(_latentEncoderParams[paramIdx]);
            }
            logvar[i] = NumOps.FromDouble(Math.Max(-4, Math.Min(2, sumVar)));
        }

        return (mean, logvar);
    }

    private double ComputeAuxLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var initParams = MetaModel.GetParameters();
        double totalLoss = 0;

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);
            var sf = ConvertToVector(MetaModel.Predict(task.SupportInput));
            var sl = ConvertToVector(task.SupportOutput);
            var cr = AggregateRepresentations(BuildContextRepresentations(sf, sl));
            var (mean, logvar) = ComputeLatentDistribution(cr);
            var z = ReparameterizeSample(mean, logvar);
            ModulateParameters(initParams, ComputeModScale(z));
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
