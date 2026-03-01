using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Attentive Neural Process (ANP) (Kim et al., ICLR 2019).
/// Adds cross-attention from targets to context for better predictions on top of NP.
/// </summary>
public class ANPAlgorithm<T, TInput, TOutput> : NeuralProcessBase<T, TInput, TOutput>
{
    private readonly ANPOptions<T, TInput, TOutput> _anpOptions;
    private Vector<T> _latentEncoderParams;
    private Vector<T> _attentionParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ANP;

    public ANPAlgorithm(ANPOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               options.RepresentationDim)
    {
        _anpOptions = options;
        _latentEncoderParams = InitializeParams(options.LatentDim * 4);
        _attentionParams = InitializeParams(options.RepresentationDim * 2);
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

            var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
            var supportLabels = ConvertToVector(task.SupportOutput);
            var queryFeatures = ConvertToVector(MetaModel.Predict(task.QueryInput));

            // Deterministic path: cross-attention from query to context
            var contextReps = EncodeContextReps(supportFeatures, supportLabels);
            var attendedRep = CrossAttend(queryFeatures, contextReps);

            // Latent path: aggregate and sample
            var aggRep = AggregateRepresentations(contextReps);
            var (priorMean, priorLogVar) = ComputeLatent(aggRep);

            // Posterior from full set
            var queryLabels = ConvertToVector(task.QueryOutput);
            var fullReps = EncodeContextReps(
                ConcatVecs(supportFeatures, queryFeatures),
                ConcatVecs(supportLabels, queryLabels));
            var fullAgg = AggregateRepresentations(fullReps);
            var (postMean, postLogVar) = ComputeLatent(fullAgg);

            var z = ReparameterizeSample(postMean, postLogVar);

            // Modulate backbone with attended rep + latent
            ModulateWithAttentionAndLatent(attendedRep, z, initParams);

            double reconLoss = NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            double klLoss = KLDivergenceGaussian(postMean, postLogVar, priorMean, priorLogVar);
            losses.Add(NumOps.FromDouble(reconLoss + _anpOptions.KLWeight * klLoss));
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
            MetaModel.SetParameters(ApplyGradients(initParams, AverageVectors(metaGradients), _anpOptions.OuterLearningRate));

        UpdateAuxiliaryParamsSPSA(taskBatch, ref EncoderParams, _anpOptions.OuterLearningRate, ComputeAuxLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _attentionParams, _anpOptions.OuterLearningRate, ComputeAuxLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();
        var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
        var supportLabels = ConvertToVector(task.SupportOutput);

        var contextReps = EncodeContextReps(supportFeatures, supportLabels);
        var aggRep = AggregateRepresentations(contextReps);
        var (mean, _) = ComputeLatent(aggRep);

        double scale = ComputeScale(mean, aggRep);
        var modParams = new Vector<T>(currentParams.Length);
        for (int i = 0; i < currentParams.Length; i++)
            modParams[i] = NumOps.Multiply(currentParams[i], NumOps.FromDouble(scale));

        return new NeuralProcessModel<T, TInput, TOutput>(MetaModel, modParams, aggRep);
    }

    private List<Vector<T>> EncodeContextReps(Vector<T>? features, Vector<T>? labels)
    {
        var reps = new List<Vector<T>>();
        if (features == null || labels == null || features.Length == 0) return reps;

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
            reps.Add(EncodeContextPair(f, l));
        }
        return reps;
    }

    private Vector<T> CrossAttend(Vector<T>? queryFeatures, List<Vector<T>> contextReps)
    {
        if (queryFeatures == null || contextReps.Count == 0)
            return new Vector<T>(RepresentationDim);

        // Simplified dot-product attention: query attends to context representations
        var result = new Vector<T>(RepresentationDim);
        var weights = new double[contextReps.Count];
        double sumW = 0;

        for (int c = 0; c < contextReps.Count; c++)
        {
            double dot = 0;
            for (int i = 0; i < Math.Min(queryFeatures.Length, contextReps[c].Length); i++)
            {
                int pIdx = i % _attentionParams.Length;
                dot += NumOps.ToDouble(queryFeatures[i % queryFeatures.Length]) *
                       NumOps.ToDouble(contextReps[c][i]) *
                       NumOps.ToDouble(_attentionParams[pIdx]);
            }
            weights[c] = Math.Exp(dot / Math.Sqrt(RepresentationDim));
            sumW += weights[c];
        }

        if (sumW > 0)
        {
            for (int c = 0; c < contextReps.Count; c++)
            {
                double w = weights[c] / sumW;
                for (int i = 0; i < RepresentationDim && i < contextReps[c].Length; i++)
                    result[i] = NumOps.Add(result[i], NumOps.Multiply(contextReps[c][i], NumOps.FromDouble(w)));
            }
        }

        return result;
    }

    private (Vector<T> mean, Vector<T> logvar) ComputeLatent(Vector<T> representation)
    {
        int latentDim = _anpOptions.LatentDim;
        var mean = new Vector<T>(latentDim);
        var logvar = new Vector<T>(latentDim);

        for (int i = 0; i < latentDim; i++)
        {
            double sum = 0;
            for (int j = 0; j < Math.Min(representation.Length, RepresentationDim); j++)
            {
                int pIdx = (i * RepresentationDim + j) % _latentEncoderParams.Length;
                sum += NumOps.ToDouble(representation[j]) * NumOps.ToDouble(_latentEncoderParams[pIdx]);
            }
            mean[i] = NumOps.FromDouble(Math.Tanh(sum));

            double sumVar = 0;
            for (int j = 0; j < Math.Min(representation.Length, RepresentationDim); j++)
            {
                int pIdx = (latentDim * RepresentationDim + i * RepresentationDim + j) % _latentEncoderParams.Length;
                sumVar += NumOps.ToDouble(representation[j]) * NumOps.ToDouble(_latentEncoderParams[pIdx]);
            }
            logvar[i] = NumOps.FromDouble(Math.Max(-4, Math.Min(2, sumVar)));
        }
        return (mean, logvar);
    }

    private void ModulateWithAttentionAndLatent(Vector<T> attended, Vector<T> z, Vector<T> initParams)
    {
        double scale = ComputeScale(z, attended);
        var modulated = new Vector<T>(initParams.Length);
        for (int i = 0; i < initParams.Length; i++)
            modulated[i] = NumOps.Multiply(initParams[i], NumOps.FromDouble(scale));
        MetaModel.SetParameters(modulated);
    }

    private double ComputeScale(Vector<T> z, Vector<T> attended)
    {
        double norm = 0;
        for (int i = 0; i < z.Length; i++) norm += NumOps.ToDouble(z[i]) * NumOps.ToDouble(z[i]);
        for (int i = 0; i < attended.Length; i++) norm += NumOps.ToDouble(attended[i]) * NumOps.ToDouble(attended[i]);
        norm = Math.Sqrt(norm / Math.Max(z.Length + attended.Length, 1));
        return 0.5 + 0.5 / (1.0 + Math.Exp(-norm + 1.0));
    }

    private Vector<T>? ConcatVecs(Vector<T>? a, Vector<T>? b)
    {
        if (a == null) return b;
        if (b == null) return a;
        var r = new Vector<T>(a.Length + b.Length);
        for (int i = 0; i < a.Length; i++) r[i] = a[i];
        for (int i = 0; i < b.Length; i++) r[a.Length + i] = b[i];
        return r;
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
            var cr = EncodeContextReps(sf, sl);
            var att = CrossAttend(ConvertToVector(MetaModel.Predict(task.QueryInput)), cr);
            var agg = AggregateRepresentations(cr);
            var (m, lv) = ComputeLatent(agg);
            var z = ReparameterizeSample(m, lv);
            ModulateWithAttentionAndLatent(att, z, initParams);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
