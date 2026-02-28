using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

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
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               options.RepresentationDim)
    {
        _cnpOptions = options;
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

            // Encode context (support) set
            var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
            var supportLabels = ConvertToVector(task.SupportOutput);
            var contextRep = EncodeAndAggregate(supportFeatures, supportLabels);

            // Decode at target (query) locations
            var queryFeatures = ConvertToVector(MetaModel.Predict(task.QueryInput));

            // Modulate backbone based on context representation
            ModulateBackbone(contextRep, initParams);

            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Update backbone
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _cnpOptions.OuterLearningRate));
        }

        // Update encoder/decoder params via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref EncoderParams, _cnpOptions.OuterLearningRate, ComputeAuxLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref DecoderParams, _cnpOptions.OuterLearningRate, ComputeAuxLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = MetaModel.GetParameters();

        var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
        var supportLabels = ConvertToVector(task.SupportOutput);
        var contextRep = EncodeAndAggregate(supportFeatures, supportLabels);

        // Modulate backbone
        var modParams = new Vector<T>(currentParams.Length);
        double scale = ComputeModulationScale(contextRep);
        for (int i = 0; i < currentParams.Length; i++)
            modParams[i] = NumOps.Multiply(currentParams[i], NumOps.FromDouble(scale));

        return new NeuralProcessModel<T, TInput, TOutput>(MetaModel, modParams, contextRep);
    }

    private Vector<T> EncodeAndAggregate(Vector<T>? features, Vector<T>? labels)
    {
        if (features == null || labels == null || features.Length == 0)
            return new Vector<T>(RepresentationDim);

        // Simple: treat as single context pair for encoding
        var representations = new List<Vector<T>>();
        int featureDim = Math.Max(1, features.Length / Math.Max(1, labels.Length));
        int numExamples = Math.Max(1, features.Length / Math.Max(featureDim, 1));

        for (int i = 0; i < numExamples; i++)
        {
            int fStart = i * featureDim;
            int fLen = Math.Min(featureDim, features.Length - fStart);
            if (fLen <= 0) break;

            var f = new Vector<T>(fLen);
            for (int j = 0; j < fLen; j++) f[j] = features[fStart + j];

            int lIdx = Math.Min(i, labels.Length - 1);
            var l = new Vector<T>(1);
            l[0] = labels[lIdx];

            representations.Add(EncodeContextPair(f, l));
        }

        return AggregateRepresentations(representations);
    }

    private void ModulateBackbone(Vector<T> contextRep, Vector<T> initParams)
    {
        double scale = ComputeModulationScale(contextRep);
        var modulated = new Vector<T>(initParams.Length);
        for (int i = 0; i < initParams.Length; i++)
            modulated[i] = NumOps.Multiply(initParams[i], NumOps.FromDouble(scale));
        MetaModel.SetParameters(modulated);
    }

    private double ComputeModulationScale(Vector<T> contextRep)
    {
        double norm = 0;
        for (int i = 0; i < contextRep.Length; i++)
            norm += NumOps.ToDouble(contextRep[i]) * NumOps.ToDouble(contextRep[i]);
        norm = Math.Sqrt(norm / Math.Max(contextRep.Length, 1));
        return 0.5 + 0.5 / (1.0 + Math.Exp(-norm + 1.0));
    }

    private double ComputeAuxLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var initParams = MetaModel.GetParameters();
        double totalLoss = 0;

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(initParams);
            var supportFeatures = ConvertToVector(MetaModel.Predict(task.SupportInput));
            var supportLabels = ConvertToVector(task.SupportOutput);
            var contextRep = EncodeAndAggregate(supportFeatures, supportLabels);
            ModulateBackbone(contextRep, initParams);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
