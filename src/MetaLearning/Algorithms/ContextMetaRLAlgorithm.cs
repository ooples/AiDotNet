using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Context Meta-RL: context-conditioned meta-reinforcement learning
/// with multi-head attention-based aggregation.
/// </summary>
/// <remarks>
/// <para>
/// Context Meta-RL uses multi-head attention to aggregate task context from the support
/// set gradient history. A learned query vector attends over encoded gradient entries
/// to produce a context vector. This context vector multiplicatively modulates the
/// model parameters, providing a smooth, deterministic task conditioning mechanism.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Context entries: e_k = encode(grad_k) for each adaptation step k
/// Attention query: q (learned)
/// Keys: K = [e_1, ..., e_K]
/// Attention weights: α_k = softmax(q^T e_k / √d / temperature)
/// Context: c = Σ_k α_k * e_k
///
/// Parameter modulation: θ' = θ ⊙ (1 + strength * sigmoid(W_m * c))
///
/// Inner loop: gradient adaptation with modulated params
/// Outer loop: update θ, encoder, query q, modulation W_m
/// </code>
/// </para>
/// </remarks>
public class ContextMetaRLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly ContextMetaRLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _contextDim;
    private readonly int _compressedDim;

    /// <summary>Context encoder: compressedDim → contextDim.</summary>
    private Vector<T> _encoderParams;

    /// <summary>Learned attention query vector: contextDim.</summary>
    private Vector<T> _queryVector;

    /// <summary>Modulation projection: contextDim → compressedDim.</summary>
    private Vector<T> _modulationParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ContextMetaRL;

    public ContextMetaRLAlgorithm(ContextMetaRLOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _contextDim = options.ContextDim;
        _compressedDim = Math.Min(_paramDim, 64);

        _encoderParams = new Vector<T>(_compressedDim * _contextDim);
        _queryVector = new Vector<T>(_contextDim);
        _modulationParams = new Vector<T>(_contextDim * _compressedDim);

        double scale = 1.0 / Math.Sqrt(_contextDim);
        for (int i = 0; i < _encoderParams.Length; i++)
            _encoderParams[i] = NumOps.FromDouble(scale * (RandomGenerator.NextDouble() - 0.5));
        for (int i = 0; i < _contextDim; i++)
            _queryVector[i] = NumOps.FromDouble(scale * (RandomGenerator.NextDouble() - 0.5));
        for (int i = 0; i < _modulationParams.Length; i++)
            _modulationParams[i] = NumOps.FromDouble(scale * (RandomGenerator.NextDouble() - 0.5));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var gradHistory = new List<double[]>();
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Encode and store gradient
                var encoded = EncodeGradient(grad);
                gradHistory.Add(encoded);

                // Attention-based context aggregation
                var contextVec = AttentionAggregate(gradHistory);

                // Multiplicative modulation: θ' = θ ⊙ (1 + strength * sigmoid(W*c))
                var modulation = ComputeModulation(contextVec);
                for (int d = 0; d < _paramDim; d++)
                {
                    int cd = d % _compressedDim;
                    double mod = 1.0 + _algoOptions.ModulationStrength * Sigmoid(modulation[cd]);
                    double gradVal = NumOps.ToDouble(grad[d]);
                    adaptedParams[d] = NumOps.Subtract(
                        NumOps.Multiply(adaptedParams[d], NumOps.FromDouble(mod)),
                        NumOps.FromDouble(_algoOptions.InnerLearningRate * gradVal));
                }
            }

            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _encoderParams, _algoOptions.OuterLearningRate * 0.1, ComputeContextMetaRLLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _queryVector, _algoOptions.OuterLearningRate * 0.1, ComputeContextMetaRLLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _modulationParams, _algoOptions.OuterLearningRate * 0.1, ComputeContextMetaRLLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();
        var gradHistory = new List<double[]>();
        var adaptedParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            gradHistory.Add(EncodeGradient(grad));
            var contextVec = AttentionAggregate(gradHistory);
            var modulation = ComputeModulation(contextVec);

            for (int d = 0; d < _paramDim; d++)
            {
                int cd = d % _compressedDim;
                double mod = 1.0 + _algoOptions.ModulationStrength * Sigmoid(modulation[cd]);
                adaptedParams[d] = NumOps.Subtract(
                    NumOps.Multiply(adaptedParams[d], NumOps.FromDouble(mod)),
                    NumOps.FromDouble(_algoOptions.InnerLearningRate * NumOps.ToDouble(grad[d])));
            }
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    private double[] EncodeGradient(Vector<T> grad)
    {
        var encoded = new double[_contextDim];
        for (int c = 0; c < _contextDim; c++)
        {
            double sum = 0;
            for (int d = 0; d < _compressedDim && d < grad.Length; d++)
                sum += NumOps.ToDouble(grad[d]) * NumOps.ToDouble(_encoderParams[c * _compressedDim + d]);
            encoded[c] = Math.Tanh(sum);
        }
        return encoded;
    }

    /// <summary>
    /// Multi-head attention aggregation: α_k = softmax(q^T e_k / √d / τ).
    /// </summary>
    private double[] AttentionAggregate(List<double[]> encodedGrads)
    {
        var result = new double[_contextDim];
        if (encodedGrads.Count == 0) return result;

        double scaleFactor = 1.0 / (Math.Sqrt(_contextDim) * _algoOptions.AttentionTemperature);

        // Compute attention scores
        var scores = new double[encodedGrads.Count];
        double maxScore = double.NegativeInfinity;
        for (int i = 0; i < encodedGrads.Count; i++)
        {
            double dot = 0;
            for (int c = 0; c < _contextDim; c++)
                dot += NumOps.ToDouble(_queryVector[c]) * encodedGrads[i][c];
            scores[i] = dot * scaleFactor;
            if (scores[i] > maxScore) maxScore = scores[i];
        }

        // Softmax
        double sumExp = 0;
        for (int i = 0; i < scores.Length; i++) { scores[i] = Math.Exp(scores[i] - maxScore); sumExp += scores[i]; }
        for (int i = 0; i < scores.Length; i++) scores[i] /= (sumExp + 1e-10);

        // Weighted sum
        for (int i = 0; i < encodedGrads.Count; i++)
            for (int c = 0; c < _contextDim; c++)
                result[c] += scores[i] * encodedGrads[i][c];

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

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    private double ComputeContextMetaRLLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var initParams = MetaModel.GetParameters();
        foreach (var task in taskBatch.Tasks)
        {
            var gh = new List<double[]>();
            var ap = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) ap[d] = initParams[d];
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(ap);
                var g = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                gh.Add(EncodeGradient(g));
                var cv = AttentionAggregate(gh);
                var m = ComputeModulation(cv);
                for (int d = 0; d < _paramDim; d++)
                {
                    int cd = d % _compressedDim;
                    double mod = 1.0 + _algoOptions.ModulationStrength * Sigmoid(m[cd]);
                    ap[d] = NumOps.Subtract(NumOps.Multiply(ap[d], NumOps.FromDouble(mod)),
                        NumOps.FromDouble(_algoOptions.InnerLearningRate * NumOps.ToDouble(g[d])));
                }
            }
            MetaModel.SetParameters(ap);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
