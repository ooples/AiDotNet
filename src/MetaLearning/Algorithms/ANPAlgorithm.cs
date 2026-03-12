using AiDotNet.Attributes;
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
/// Implementation of Attentive Neural Process (ANP) (Kim et al., ICLR 2019).
/// Adds cross-attention from targets to context for better predictions on top of NP.
/// </summary>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Attentive Neural Processes", "https://arxiv.org/abs/1901.05761", Year = 2019, Authors = "Kim et al.")]
public class ANPAlgorithm<T, TInput, TOutput> : NeuralProcessBase<T, TInput, TOutput>
{
    private readonly ANPOptions<T, TInput, TOutput> _anpOptions;
    private Vector<T> _latentEncoderParams;
    private Vector<T> _attentionParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ANP;

    public ANPAlgorithm(ANPOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               options.RepresentationDim)
    {
        if (options.RepresentationDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "RepresentationDim must be positive.");
        if (options.LatentDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "LatentDim must be positive.");

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
            var contextReps = BuildContextRepresentations(supportFeatures, supportLabels);
            var attendedRep = CrossAttend(queryFeatures, contextReps);

            // Latent path: aggregate and sample
            var aggRep = AggregateRepresentations(contextReps);
            var (priorMean, priorLogVar) = ComputeLatent(aggRep);

            // Posterior from full set
            var queryLabels = ConvertToVector(task.QueryOutput);
            var fullReps = BuildContextRepresentations(
                ConcatVectors(supportFeatures, queryFeatures),
                ConcatVectors(supportLabels, queryLabels));
            var fullAgg = AggregateRepresentations(fullReps);
            var (postMean, postLogVar) = ComputeLatent(fullAgg);

            var z = ReparameterizeSample(postMean, postLogVar);

            // Modulate backbone with attended rep + latent
            ModulateParameters(initParams, ComputeScale(z, attendedRep));

            double reconLoss = NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            double klLoss = KLDivergenceGaussian(postMean, postLogVar, priorMean, priorLogVar);
            losses.Add(NumOps.FromDouble(reconLoss + _anpOptions.KLWeight * klLoss));
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        ApplyOuterUpdate(initParams, metaGradients, _anpOptions.OuterLearningRate);

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

        var contextReps = BuildContextRepresentations(supportFeatures, supportLabels);
        var aggRep = AggregateRepresentations(contextReps);
        var (mean, _) = ComputeLatent(aggRep);

        return new NeuralProcessModel<T, TInput, TOutput>(MetaModel, ScaleVector(currentParams, ComputeScale(mean, aggRep)), aggRep);
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

    private double ComputeScale(Vector<T> z, Vector<T> attended)
    {
        double norm = 0;
        for (int i = 0; i < z.Length; i++) norm += NumOps.ToDouble(z[i]) * NumOps.ToDouble(z[i]);
        for (int i = 0; i < attended.Length; i++) norm += NumOps.ToDouble(attended[i]) * NumOps.ToDouble(attended[i]);
        norm = Math.Sqrt(norm / Math.Max(z.Length + attended.Length, 1));
        return 0.5 + 0.5 / (1.0 + Math.Exp(-norm + 1.0));
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
            var cr = BuildContextRepresentations(sf, sl);
            var att = CrossAttend(ConvertToVector(MetaModel.Predict(task.QueryInput)), cr);
            var agg = AggregateRepresentations(cr);
            var (m, lv) = ComputeLatent(agg);
            var z = ReparameterizeSample(m, lv);
            ModulateParameters(initParams, ComputeScale(z, att));
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
