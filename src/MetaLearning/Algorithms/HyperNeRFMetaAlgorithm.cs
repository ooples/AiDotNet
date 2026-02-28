using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of HyperNeRF Meta: Positional-Encoding-Conditioned Meta-Learning.
/// </summary>
/// <remarks>
/// <para>
/// Combines hypernetwork conditioning with NeRF-style sinusoidal positional encoding.
/// Each parameter index is encoded using multi-frequency sin/cos functions, providing
/// structural awareness of where parameters are located in the network. These positional
/// features are combined with a task-specific latent code (from gradient statistics) to
/// produce per-parameter learning rate modulation through a learned conditioning MLP.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Positional encoding for parameter index d (normalized to [0,1]):
///   PE(d) = [d̂, sin(2⁰πd̂), cos(2⁰πd̂), ..., sin(2^{L-1}πd̂), cos(2^{L-1}πd̂)]
///   dim_PE = 2 * NumFrequencyBands + 1
///
/// Task latent code: z = compress(grad_support)  →  [LatentDim]
///
/// Conditioning MLP (per parameter group):
///   input = [PE(d), z]  →  dim_PE + LatentDim
///   γ_d = sigmoid(W_cond · input)
///   modulation_d = 1 + ConditioningStrength * (γ_d - 0.5)
///
/// Inner loop: θ_d ← θ_d - η * modulation_d * grad_d
///
/// L_meta = L_query + ConditioningRegWeight * ||W_cond||²
/// Outer: update θ, update W_cond via SPSA
/// </code>
/// </para>
/// </remarks>
public class HyperNeRFMetaAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly HyperNeRFMetaOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _peDim;
    private readonly int _numGroups;
    private readonly int _groupSize;

    /// <summary>Conditioning MLP weights: (peDim + latentDim) × numGroups.</summary>
    private Vector<T> _conditioningWeights;

    /// <summary>Pre-computed positional encodings per group.</summary>
    private readonly double[][] _positionalEncodings;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.HyperNeRFMeta;

    public HyperNeRFMetaAlgorithm(HyperNeRFMetaOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _peDim = 2 * options.NumFrequencyBands + 1;

        // Group parameters to reduce conditioning MLP calls
        _numGroups = Math.Min(_paramDim, 64);
        _groupSize = (_paramDim + _numGroups - 1) / _numGroups;

        int inputDim = _peDim + options.LatentDim;
        // One weight vector per group: inputDim weights → 1 output (sigmoid)
        int totalWeights = _numGroups * inputDim;
        _conditioningWeights = new Vector<T>(totalWeights);
        double scale = 1.0 / Math.Sqrt(inputDim);
        for (int i = 0; i < totalWeights; i++)
            _conditioningWeights[i] = NumOps.FromDouble(scale * (RandomGenerator.NextDouble() - 0.5));

        // Pre-compute positional encodings for each group
        _positionalEncodings = new double[_numGroups][];
        for (int g = 0; g < _numGroups; g++)
        {
            _positionalEncodings[g] = new double[_peDim];
            double normalizedPos = (double)g / Math.Max(_numGroups - 1, 1);
            _positionalEncodings[g][0] = normalizedPos;
            for (int f = 0; f < options.NumFrequencyBands; f++)
            {
                double freq = Math.Pow(2.0, f) * Math.PI;
                _positionalEncodings[g][1 + 2 * f] = Math.Sin(freq * normalizedPos);
                _positionalEncodings[g][2 + 2 * f] = Math.Cos(freq * normalizedPos);
            }
        }
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

            // Compute task latent code from support gradient
            MetaModel.SetParameters(adaptedParams);
            var supportGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            var latentCode = CompressGradient(supportGrad);

            // Compute per-group modulation factors
            var modulation = ComputeModulation(latentCode);

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Apply position-aware, task-conditioned modulation
                for (int d = 0; d < _paramDim; d++)
                {
                    int g = Math.Min(d / _groupSize, _numGroups - 1);
                    adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                        NumOps.FromDouble(_algoOptions.InnerLearningRate * modulation[g] * NumOps.ToDouble(grad[d])));
                }

                // Recompute latent code for dynamic conditioning (optional: every other step)
                if (step < _algoOptions.AdaptationSteps - 1 && step % 2 == 0)
                {
                    MetaModel.SetParameters(adaptedParams);
                    var newGrad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                    latentCode = CompressGradient(newGrad);
                    modulation = ComputeModulation(latentCode);
                }
            }

            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Conditioning weight regularization
            double condReg = 0;
            for (int i = 0; i < _conditioningWeights.Length; i++)
                condReg += NumOps.ToDouble(_conditioningWeights[i]) * NumOps.ToDouble(_conditioningWeights[i]);

            var totalLoss = NumOps.Add(queryLoss,
                NumOps.FromDouble(_algoOptions.ConditioningRegWeight * condReg / _conditioningWeights.Length));

            losses.Add(totalLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _conditioningWeights, _algoOptions.OuterLearningRate * 0.1, ComputeNeRFLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();
        var adaptedParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) adaptedParams[d] = initParams[d];

        MetaModel.SetParameters(adaptedParams);
        var supportGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
        var latentCode = CompressGradient(supportGrad);
        var modulation = ComputeModulation(latentCode);

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

            for (int d = 0; d < _paramDim; d++)
            {
                int g = Math.Min(d / _groupSize, _numGroups - 1);
                adaptedParams[d] = NumOps.Subtract(adaptedParams[d],
                    NumOps.FromDouble(_algoOptions.InnerLearningRate * modulation[g] * NumOps.ToDouble(grad[d])));
            }
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    private double[] CompressGradient(Vector<T> grad)
    {
        int latentDim = _algoOptions.LatentDim;
        var result = new double[latentDim];
        int bucketSize = Math.Max(1, _paramDim / latentDim);
        for (int e = 0; e < latentDim; e++)
        {
            double sum = 0;
            int start = e * bucketSize;
            for (int d = start; d < start + bucketSize && d < grad.Length; d++)
                sum += NumOps.ToDouble(grad[d]);
            result[e] = Math.Tanh(sum / bucketSize);
        }
        return result;
    }

    private double[] ComputeModulation(double[] latentCode)
    {
        int latentDim = _algoOptions.LatentDim;
        int inputDim = _peDim + latentDim;
        var modulation = new double[_numGroups];

        for (int g = 0; g < _numGroups; g++)
        {
            // Concatenate [PE(g), latentCode]
            double sum = 0;
            int wOffset = g * inputDim;

            // PE features
            for (int p = 0; p < _peDim; p++)
                sum += NumOps.ToDouble(_conditioningWeights[wOffset + p]) * _positionalEncodings[g][p];

            // Latent code features
            for (int l = 0; l < latentDim; l++)
                sum += NumOps.ToDouble(_conditioningWeights[wOffset + _peDim + l]) * latentCode[l];

            // Sigmoid → modulation
            double gamma = 1.0 / (1.0 + Math.Exp(-sum));
            modulation[g] = 1.0 + _algoOptions.ConditioningStrength * (gamma - 0.5);
        }

        return modulation;
    }

    private double ComputeNeRFLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var initParams = MetaModel.GetParameters();
        foreach (var task in taskBatch.Tasks)
        {
            var ap = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) ap[d] = initParams[d];
            MetaModel.SetParameters(ap);
            var sg = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            var lc = CompressGradient(sg);
            var mod = ComputeModulation(lc);
            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(ap);
                var g = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                for (int d = 0; d < _paramDim; d++)
                {
                    int grp = Math.Min(d / _groupSize, _numGroups - 1);
                    ap[d] = NumOps.Subtract(ap[d],
                        NumOps.FromDouble(_algoOptions.InnerLearningRate * mod[grp] * NumOps.ToDouble(g[d])));
                }
            }
            MetaModel.SetParameters(ap);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
