using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>Implementation of Swin Transformer Neural Process (2024).</summary>
public class SwinTNPAlgorithm<T, TInput, TOutput> : NeuralProcessBase<T, TInput, TOutput>
{
    private readonly SwinTNPOptions<T, TInput, TOutput> _algoOptions;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.SwinTNP;

    public SwinTNPAlgorithm(SwinTNPOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               options.RepresentationDim)
    {
        _algoOptions = options;
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

            var contextReps = new List<Vector<T>>();
            if (supportFeatures != null && supportLabels != null && supportFeatures.Length > 0)
            {
                int numEx = Math.Max(1, supportLabels.Length);
                int fDim = Math.Max(1, supportFeatures.Length / numEx);
                for (int i = 0; i < numEx; i++)
                {
                    int fStart = i * fDim;
                    int fLen = Math.Min(fDim, supportFeatures.Length - fStart);
                    if (fLen <= 0) break;
                    var f = new Vector<T>(fLen);
                    for (int j = 0; j < fLen; j++) f[j] = supportFeatures[fStart + j];
                    var l = new Vector<T>(1);
                    l[0] = supportLabels[Math.Min(i, supportLabels.Length - 1)];
                    contextReps.Add(EncodeContextPair(f, l));
                }
            }

            var aggRep = AggregateRepresentations(contextReps);
            double scale = ComputeModScale(aggRep);
            var modParams = new Vector<T>(initParams.Length);
            for (int i = 0; i < initParams.Length; i++)
                modParams[i] = NumOps.Multiply(initParams[i], NumOps.FromDouble(scale));
            MetaModel.SetParameters(modParams);

            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
            MetaModel.SetParameters(ApplyGradients(initParams, AverageVectors(metaGradients), _algoOptions.OuterLearningRate));

        UpdateAuxiliaryParamsSPSA(taskBatch, ref EncoderParams, _algoOptions.OuterLearningRate, ComputeAuxLoss);
        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();
        var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
        var supportLabels = ConvertToVector(task.SupportOutput);

        var contextReps = new List<Vector<T>>();
        if (supportFeatures != null && supportLabels != null && supportFeatures.Length > 0)
        {
            int numEx = Math.Max(1, supportLabels.Length);
            int fDim = Math.Max(1, supportFeatures.Length / numEx);
            for (int i = 0; i < numEx; i++)
            {
                int fStart = i * fDim;
                int fLen = Math.Min(fDim, supportFeatures.Length - fStart);
                if (fLen <= 0) break;
                var f = new Vector<T>(fLen);
                for (int j = 0; j < fLen; j++) f[j] = supportFeatures[fStart + j];
                var l = new Vector<T>(1);
                l[0] = supportLabels[Math.Min(i, supportLabels.Length - 1)];
                contextReps.Add(EncodeContextPair(f, l));
            }
        }

        var aggRep = AggregateRepresentations(contextReps);
        double sc = ComputeModScale(aggRep);
        var modParams = new Vector<T>(currentParams.Length);
        for (int i = 0; i < currentParams.Length; i++)
            modParams[i] = NumOps.Multiply(currentParams[i], NumOps.FromDouble(sc));

        return new NeuralProcessModel<T, TInput, TOutput>(MetaModel, modParams, aggRep);
    }

    private double ComputeModScale(Vector<T> rep)
    {
        double norm = 0;
        for (int i = 0; i < rep.Length; i++) norm += NumOps.ToDouble(rep[i]) * NumOps.ToDouble(rep[i]);
        norm = Math.Sqrt(norm / Math.Max(rep.Length, 1));
        return 0.5 + 0.5 / (1.0 + Math.Exp(-norm + 1.0));
    }

    private double ComputeAuxLoss(TaskBatch<T, TInput, TOutput> tb)
    {
        var ip = MetaModel.GetParameters();
        double total = 0;
        foreach (var t in tb.Tasks)
        {
            MetaModel.SetParameters(ip);
            total += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(t.QueryInput), t.QueryOutput));
        }
        MetaModel.SetParameters(ip);
        return total / Math.Max(tb.Tasks.Length, 1);
    }
}
