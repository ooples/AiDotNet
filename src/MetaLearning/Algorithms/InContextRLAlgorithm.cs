using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of In-Context RL: meta-RL via in-context learning without
/// explicit gradient updates at test time.
/// </summary>
/// <remarks>
/// <para>
/// In-Context RL trains a model to adapt through its forward pass by conditioning on a
/// growing context buffer. The context buffer stores compressed representations of past
/// (input, prediction, loss) triplets. A context aggregator (learned attention-like
/// mechanism) combines the buffer entries into a context vector that modulates the
/// model's parameters. At test time, no gradients are needed — the model improves
/// purely by observing more support examples in its context.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Context buffer: B = [(h_1, l_1), ..., (h_t, l_t)]
///   h_i = hash(input_i, prediction_i), l_i = loss_i
///
/// Context aggregation: c = Σ_i softmax(-l_i / τ) * h_i  (loss-weighted average)
/// Parameter modulation: θ' = θ + W_c * c
///
/// Meta-training: build context sequentially
///   For step k:
///     Predict on support, record (h, loss) in buffer
///     Aggregate context → modulate params
///     Compute loss with modulated params
///
/// Outer loop: update θ, W_c, context encoder
/// </code>
/// </para>
/// </remarks>
public class InContextRLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly InContextRLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _contextDim;
    private readonly int _compressedDim;

    /// <summary>Context encoder parameters: gradient → context entry.</summary>
    private Vector<T> _contextEncoderParams;

    /// <summary>Context-to-parameter modulation projection.</summary>
    private Vector<T> _modulationParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.InContextRL;

    public InContextRLAlgorithm(InContextRLOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _contextDim = options.ContextEmbeddingDim;
        _compressedDim = Math.Min(_paramDim, 64);

        // Context encoder: compressedDim → contextDim
        _contextEncoderParams = new Vector<T>(_compressedDim * _contextDim);
        for (int i = 0; i < _contextEncoderParams.Length; i++)
            _contextEncoderParams[i] = NumOps.FromDouble(0.01 * (RandomGenerator.NextDouble() - 0.5));

        // Modulation: contextDim → compressedDim
        _modulationParams = new Vector<T>(_contextDim * _compressedDim);
        for (int i = 0; i < _modulationParams.Length; i++)
            _modulationParams[i] = NumOps.FromDouble(0.01 * (RandomGenerator.NextDouble() - 0.5));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Build context buffer sequentially
            var contextEntries = new List<double[]>();
            var contextLosses = new List<double>();
            var currentParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) currentParams[d] = initParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                // Predict and record context
                MetaModel.SetParameters(currentParams);
                var pred = MetaModel.Predict(task.SupportInput);
                double stepLoss = NumOps.ToDouble(ComputeLossFromOutput(pred, task.SupportOutput));

                // Encode context entry from gradient
                var grad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                var entry = EncodeContextEntry(grad);
                contextEntries.Add(entry);
                contextLosses.Add(stepLoss);

                // Trim buffer to max size
                while (contextEntries.Count > _algoOptions.ContextBufferSize)
                {
                    contextEntries.RemoveAt(0);
                    contextLosses.RemoveAt(0);
                }

                // Aggregate context and modulate parameters
                var contextVec = AggregateContext(contextEntries, contextLosses);
                var modulation = ComputeModulation(contextVec);
                for (int d = 0; d < _paramDim; d++)
                {
                    int cd = d % _compressedDim;
                    currentParams[d] = NumOps.Add(initParams[d], NumOps.FromDouble(modulation[cd]));
                }
            }

            MetaModel.SetParameters(currentParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Context prediction auxiliary loss
            double ctxPredLoss = 0;
            if (contextLosses.Count > 1)
            {
                double meanLoss = 0;
                foreach (var l in contextLosses) meanLoss += l;
                meanLoss /= contextLosses.Count;
                foreach (var l in contextLosses) ctxPredLoss += (l - meanLoss) * (l - meanLoss);
                ctxPredLoss /= contextLosses.Count;
            }

            var totalLoss = NumOps.Add(queryLoss, NumOps.FromDouble(_algoOptions.ContextPredictionWeight * ctxPredLoss));
            losses.Add(totalLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _contextEncoderParams, _algoOptions.OuterLearningRate * 0.1, ComputeInContextLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _modulationParams, _algoOptions.OuterLearningRate * 0.1, ComputeInContextLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();
        var contextEntries = new List<double[]>();
        var contextLosses = new List<double>();
        var currentParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) currentParams[d] = initParams[d];

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(currentParams);
            double stepLoss = NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.SupportInput), task.SupportOutput));
            var grad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            contextEntries.Add(EncodeContextEntry(grad));
            contextLosses.Add(stepLoss);

            while (contextEntries.Count > _algoOptions.ContextBufferSize)
            { contextEntries.RemoveAt(0); contextLosses.RemoveAt(0); }

            var contextVec = AggregateContext(contextEntries, contextLosses);
            var modulation = ComputeModulation(contextVec);
            for (int d = 0; d < _paramDim; d++)
                currentParams[d] = NumOps.Add(initParams[d], NumOps.FromDouble(modulation[d % _compressedDim]));
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, currentParams);
    }

    private double[] EncodeContextEntry(Vector<T> grad)
    {
        var entry = new double[_contextDim];
        for (int c = 0; c < _contextDim; c++)
        {
            double sum = 0;
            for (int d = 0; d < _compressedDim && d < grad.Length; d++)
                sum += NumOps.ToDouble(grad[d]) * NumOps.ToDouble(_contextEncoderParams[c * _compressedDim + d]);
            entry[c] = Math.Tanh(sum);
        }
        return entry;
    }

    private double[] AggregateContext(List<double[]> entries, List<double> entryLosses)
    {
        var result = new double[_contextDim];
        if (entries.Count == 0) return result;

        // Loss-weighted attention: lower loss → higher weight
        var weights = new double[entries.Count];
        double maxNegLoss = double.NegativeInfinity;
        for (int i = 0; i < entries.Count; i++)
        {
            weights[i] = -entryLosses[i];
            if (weights[i] > maxNegLoss) maxNegLoss = weights[i];
        }
        double sumW = 0;
        for (int i = 0; i < entries.Count; i++) { weights[i] = Math.Exp(weights[i] - maxNegLoss); sumW += weights[i]; }
        for (int i = 0; i < entries.Count; i++) weights[i] /= (sumW + 1e-10);

        for (int i = 0; i < entries.Count; i++)
            for (int c = 0; c < _contextDim; c++)
                result[c] += weights[i] * entries[i][c];
        return result;
    }

    private double[] ComputeModulation(double[] contextVec)
    {
        var mod = new double[_compressedDim];
        for (int d = 0; d < _compressedDim; d++)
        {
            double sum = 0;
            for (int c = 0; c < _contextDim; c++)
                sum += contextVec[c] * NumOps.ToDouble(_modulationParams[d * _contextDim + c]);
            mod[d] = sum;
        }
        return mod;
    }

    private double ComputeInContextLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var initParams = MetaModel.GetParameters();
        foreach (var task in taskBatch.Tasks)
        {
            var entries = new List<double[]>();
            var eLosses = new List<double>();
            var cp = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) cp[d] = initParams[d];
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(cp);
                double sl = NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.SupportInput), task.SupportOutput));
                var g = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                entries.Add(EncodeContextEntry(g));
                eLosses.Add(sl);
                var cv = AggregateContext(entries, eLosses);
                var m = ComputeModulation(cv);
                for (int d = 0; d < _paramDim; d++) cp[d] = NumOps.Add(initParams[d], NumOps.FromDouble(m[d % _compressedDim]));
            }
            MetaModel.SetParameters(cp);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
