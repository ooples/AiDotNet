using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of ATAML: Attention-based Task-Adaptive Meta-Learning.
/// </summary>
/// <remarks>
/// <para>
/// ATAML meta-learns a task-adaptive attention mechanism that produces per-parameter
/// learning rate scaling factors based on the task's gradient profile. A learned projection
/// maps compressed gradient features to attention weights over parameter dimensions,
/// enabling the inner loop to focus adaptation on the most relevant parameters for each task.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Compressed gradient: g_c = compress(grad) ∈ R^AttentionDim
///
/// Attention logits: a_d = W_attn[d, :] · g_c   (per compressed dimension)
/// Attention weights: w = softmax(a / τ) * D   (scale to mean=1)
///
/// Inner loop: θ_d ← θ_d - η * w[d % compressedDim] * grad_d
///
/// Entropy regularization: H(w) = -Σ w_d * log(w_d)
/// L_meta = L_query + AttentionEntropyWeight * (-H(w))
///
/// Outer loop: update θ, W_attn (via SPSA)
/// </code>
/// </para>
/// </remarks>
public class ATAMLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly ATAMLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _compressedDim;

    /// <summary>Attention projection: compressedDim × attentionDim.</summary>
    private Vector<T> _attentionParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ATAML;

    public ATAMLAlgorithm(ATAMLOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _compressedDim = Math.Min(_paramDim, 64);

        // Attention projection: compressedDim × attentionDim
        _attentionParams = new Vector<T>(_compressedDim * options.AttentionDim);
        double scale = 1.0 / Math.Sqrt(options.AttentionDim);
        for (int i = 0; i < _attentionParams.Length; i++)
            _attentionParams[i] = NumOps.FromDouble(scale * (RandomGenerator.NextDouble() - 0.5));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            double lastEntropy = 0;
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Compute compressed gradient features
                var compressed = CompressGradient(grad);

                // Compute attention weights
                var (weights, entropy) = ComputeAttention(compressed);
                lastEntropy = entropy;

                // Apply attention-weighted gradient
                for (int d = 0; d < _paramDim; d++)
                {
                    int cd = d % _compressedDim;
                    adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                        NumOps.FromDouble(_algoOptions.InnerLearningRate * weights[cd] * NumOps.ToDouble(grad[d])));
                }
            }

            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Entropy regularization (penalize low entropy to prevent attention collapse)
            var totalLoss = NumOps.Subtract(queryLoss,
                NumOps.FromDouble(_algoOptions.AttentionEntropyWeight * lastEntropy));

            losses.Add(totalLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _attentionParams, _algoOptions.OuterLearningRate * 0.1, ComputeATAMLLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();
        var adaptedParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            var compressed = CompressGradient(grad);
            var (weights, _) = ComputeAttention(compressed);

            for (int d = 0; d < _paramDim; d++)
            {
                int cd = d % _compressedDim;
                adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                    NumOps.FromDouble(_algoOptions.InnerLearningRate * weights[cd] * NumOps.ToDouble(grad[d])));
            }
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    private double[] CompressGradient(Vector<T> grad)
    {
        var result = new double[_algoOptions.AttentionDim];
        for (int a = 0; a < _algoOptions.AttentionDim; a++)
        {
            double sum = 0;
            for (int d = 0; d < _compressedDim && d < grad.Length; d++)
                sum += NumOps.ToDouble(grad[d]) * NumOps.ToDouble(grad[d]); // squared gradient features
            result[a] = Math.Tanh(sum / _compressedDim);
        }
        return result;
    }

    private (double[] weights, double entropy) ComputeAttention(double[] compressed)
    {
        var logits = new double[_compressedDim];
        for (int d = 0; d < _compressedDim; d++)
        {
            double sum = 0;
            for (int a = 0; a < _algoOptions.AttentionDim; a++)
                sum += compressed[a] * NumOps.ToDouble(_attentionParams[d * _algoOptions.AttentionDim + a]);
            logits[d] = sum / _algoOptions.AttentionTemperature;
        }

        // Softmax
        double maxLogit = logits[0];
        for (int d = 1; d < _compressedDim; d++)
            if (logits[d] > maxLogit) maxLogit = logits[d];
        double sumExp = 0;
        for (int d = 0; d < _compressedDim; d++) { logits[d] = Math.Exp(logits[d] - maxLogit); sumExp += logits[d]; }
        for (int d = 0; d < _compressedDim; d++) logits[d] /= (sumExp + 1e-10);

        // Scale to mean=1 (so average LR stays the same)
        for (int d = 0; d < _compressedDim; d++) logits[d] *= _compressedDim;

        // Compute entropy
        double entropy = 0;
        for (int d = 0; d < _compressedDim; d++)
        {
            double p = logits[d] / _compressedDim; // back to probability for entropy
            if (p > 1e-10) entropy -= p * Math.Log(p);
        }

        return (logits, entropy);
    }

    private double ComputeATAMLLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var initParams = MetaModel.GetParameters();
        foreach (var task in taskBatch.Tasks)
        {
            var ap = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) ap[d] = initParams[d];
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(ap);
                var g = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                var c = CompressGradient(g);
                var (w, _) = ComputeAttention(c);
                for (int d = 0; d < _paramDim; d++)
                    ap[d] = NumOps.Subtract(ap[d],
                        NumOps.FromDouble(_algoOptions.InnerLearningRate * w[d % _compressedDim] * NumOps.ToDouble(g[d])));
            }
            MetaModel.SetParameters(ap);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
