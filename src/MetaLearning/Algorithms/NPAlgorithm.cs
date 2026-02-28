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
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               options.RepresentationDim)
    {
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
            var contextRep = EncodeAndAggregateNP(supportFeatures, supportLabels);

            // Compute latent distribution from context
            var (priorMean, priorLogVar) = ComputeLatentDistribution(contextRep);

            // Encode full set (context + target) for posterior
            var queryFeatures = ConvertToVector(MetaModel.Predict(task.QueryInput));
            var queryLabels = ConvertToVector(task.QueryOutput);
            var fullRep = EncodeAndAggregateNP(
                ConcatVectors(supportFeatures, queryFeatures),
                ConcatVectors(supportLabels, queryLabels));
            var (postMean, postLogVar) = ComputeLatentDistribution(fullRep);

            // Sample z from posterior and modulate backbone
            var z = ReparameterizeSample(postMean, postLogVar);
            ModulateWithLatent(z, initParams);

            // Compute reconstruction loss + KL
            double reconLoss = NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            double klLoss = KLDivergenceGaussian(postMean, postLogVar, priorMean, priorLogVar);
            double totalLoss = reconLoss + _npOptions.KLWeight * klLoss;

            losses.Add(NumOps.FromDouble(totalLoss));
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _npOptions.OuterLearningRate));
        }

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
        var contextRep = EncodeAndAggregateNP(supportFeatures, supportLabels);

        var (mean, logvar) = ComputeLatentDistribution(contextRep);
        // Use mean for deterministic prediction at adaptation time
        var z = mean;

        var modParams = new Vector<T>(currentParams.Length);
        double scale = ComputeLatentScale(z);
        for (int i = 0; i < currentParams.Length; i++)
            modParams[i] = NumOps.Multiply(currentParams[i], NumOps.FromDouble(scale));

        return new NeuralProcessModel<T, TInput, TOutput>(MetaModel, modParams, contextRep);
    }

    private Vector<T> EncodeAndAggregateNP(Vector<T>? features, Vector<T>? labels)
    {
        if (features == null || labels == null || features.Length == 0)
            return new Vector<T>(RepresentationDim);

        var representations = new List<Vector<T>>();
        int numExamples = Math.Max(1, labels.Length);
        int featureDim = Math.Max(1, features.Length / numExamples);

        for (int i = 0; i < numExamples; i++)
        {
            int fStart = i * featureDim;
            int fLen = Math.Min(featureDim, features.Length - fStart);
            if (fLen <= 0) break;

            var f = new Vector<T>(fLen);
            for (int j = 0; j < fLen; j++) f[j] = features[fStart + j];

            var l = new Vector<T>(1);
            l[0] = labels[Math.Min(i, labels.Length - 1)];

            representations.Add(EncodeContextPair(f, l));
        }

        return AggregateRepresentations(representations);
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

    private void ModulateWithLatent(Vector<T> z, Vector<T> initParams)
    {
        double scale = ComputeLatentScale(z);
        var modulated = new Vector<T>(initParams.Length);
        for (int i = 0; i < initParams.Length; i++)
            modulated[i] = NumOps.Multiply(initParams[i], NumOps.FromDouble(scale));
        MetaModel.SetParameters(modulated);
    }

    private double ComputeLatentScale(Vector<T> z)
    {
        double norm = 0;
        for (int i = 0; i < z.Length; i++)
            norm += NumOps.ToDouble(z[i]) * NumOps.ToDouble(z[i]);
        norm = Math.Sqrt(norm / Math.Max(z.Length, 1));
        return 0.5 + 0.5 / (1.0 + Math.Exp(-norm + 1.0));
    }

    private Vector<T>? ConcatVectors(Vector<T>? a, Vector<T>? b)
    {
        if (a == null) return b;
        if (b == null) return a;
        var result = new Vector<T>(a.Length + b.Length);
        for (int i = 0; i < a.Length; i++) result[i] = a[i];
        for (int i = 0; i < b.Length; i++) result[a.Length + i] = b[i];
        return result;
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
            var cr = EncodeAndAggregateNP(sf, sl);
            var (mean, logvar) = ComputeLatentDistribution(cr);
            var z = ReparameterizeSample(mean, logvar);
            ModulateWithLatent(z, initParams);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
