using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of ICM-Fusion: In-Context Meta-Optimized LoRA Fusion (2025).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// ICM-Fusion addresses the problem of fusing multiple task-specific parameter deltas (task vectors)
/// by encoding them into a shared latent space via a Fusion-VAE. The VAE learns a manifold where
/// task vector arithmetic resolves inter-weight conflicts that arise when naively averaging or
/// summing adapters from different tasks.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Meta-learned state: base params θ, VAE encoder/decoder params, fusion component history
///
/// For each task:
///   1. Compute task vector: Δθ = θ_adapted - θ_base (via inner-loop gradient steps)
///   2. Encode task vector: (μ, log_σ²) = Encoder(Δθ)
///   3. Sample latent: z = μ + σ * ε  (reparameterization trick)
///   4. Average latent z with stored component latents (exponential decay weighting)
///   5. Decode fused adapter: Δθ_fused = Decoder(z_avg)
///   6. Adapted params: θ' = θ + Δθ_fused
///   7. Loss = L_task(θ') + KLWeight * KL(q(z|Δθ) || N(0,I))
///
/// Outer loop: update VAE encoder/decoder and base params via meta-gradients
/// </code>
/// </para>
/// </remarks>
public class ICMFusionAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly ICMFusionOptions<T, TInput, TOutput> _algoOptions;

    /// <summary>VAE encoder params: maps task vectors to (μ, log_σ²) in latent space.</summary>
    private Vector<T> _encoderParams;

    /// <summary>VAE decoder params: maps latent vectors back to parameter deltas.</summary>
    private Vector<T> _decoderParams;

    /// <summary>Stored latent codes from recent tasks for fusion (circular buffer).</summary>
    private readonly double[][] _componentLatents;
    private int _componentIdx;
    private int _componentCount;

    private readonly int _paramDim;
    private readonly int _latentDim;
    private readonly int _compressedDim;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ICMFusion;

    public ICMFusionAlgorithm(ICMFusionOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _latentDim = Math.Max(1, options.LatentDim);

        // Compress param space for the VAE to keep encoder/decoder tractable
        _compressedDim = Math.Min(_paramDim, 128);

        // Encoder: compressedDim → 2 * latentDim (mu + log_var)
        int encoderSize = _compressedDim * (2 * _latentDim) + 2 * _latentDim;
        _encoderParams = new Vector<T>(encoderSize);
        InitializeGaussian(_encoderParams, 1.0 / Math.Sqrt(_compressedDim));

        // Decoder: latentDim → compressedDim
        int decoderSize = _latentDim * _compressedDim + _compressedDim;
        _decoderParams = new Vector<T>(decoderSize);
        InitializeGaussian(_decoderParams, 1.0 / Math.Sqrt(_latentDim));

        // Component history
        _componentLatents = new double[options.NumFusionComponents][];
        for (int i = 0; i < options.NumFusionComponents; i++)
            _componentLatents[i] = new double[_latentDim];
        _componentIdx = 0;
        _componentCount = 0;
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var baseParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Inner loop: compute task-specific adapted params
            MetaModel.SetParameters(baseParams);
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = baseParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
            }

            // Compute task vector Δθ = adapted - base (compressed)
            var taskVector = CompressTaskVector(baseParams, adaptedParams);

            // Encode: (μ, log_σ²) = Encoder(Δθ)
            EncodeTaskVector(taskVector, out var mu, out var logVar);

            // Reparameterize: z = μ + σ * ε
            var z = Reparameterize(mu, logVar);

            // Store and fuse with historical components
            StoreComponent(z);
            var zFused = FuseLatents(z);

            // Decode: Δθ_fused = Decoder(z_fused)
            var fusedDelta = DecodeLatent(zFused);

            // Apply fused delta to base params
            var fusedParams = ApplyCompressedDelta(baseParams, fusedDelta);
            MetaModel.SetParameters(fusedParams);

            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Add KL divergence penalty
            double klDiv = ComputeKLDivergence(mu, logVar);
            var totalLoss = NumOps.Add(queryLoss, NumOps.FromDouble(_algoOptions.KLWeight * klDiv));
            losses.Add(totalLoss);

            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Outer loop: update base params
        MetaModel.SetParameters(baseParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(baseParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        // Update VAE params via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _encoderParams, _algoOptions.OuterLearningRate, ComputeVAELoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _decoderParams, _algoOptions.OuterLearningRate, ComputeVAELoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var baseParams = MetaModel.GetParameters();

        // Inner loop adaptation
        var adaptedParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) adaptedParams[d] = baseParams[d];

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
        }

        // Encode and decode through VAE for fusion
        var taskVector = CompressTaskVector(baseParams, adaptedParams);
        EncodeTaskVector(taskVector, out var mu, out _);
        var zFused = FuseLatents(mu); // Use mean (no sampling) for deterministic adaptation
        var fusedDelta = DecodeLatent(zFused);
        var finalParams = ApplyCompressedDelta(baseParams, fusedDelta);

        MetaModel.SetParameters(baseParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, finalParams);
    }

    private double[] CompressTaskVector(Vector<T> baseParams, Vector<T> adaptedParams)
    {
        var compressed = new double[_compressedDim];
        // Strided sampling + averaging for compression
        int stride = Math.Max(1, _paramDim / _compressedDim);
        for (int i = 0; i < _compressedDim; i++)
        {
            int idx = Math.Min(i * stride, _paramDim - 1);
            compressed[i] = NumOps.ToDouble(adaptedParams[idx]) - NumOps.ToDouble(baseParams[idx]);
        }
        return compressed;
    }

    private void EncodeTaskVector(double[] taskVector, out double[] mu, out double[] logVar)
    {
        mu = new double[_latentDim];
        logVar = new double[_latentDim];
        int biasOffset = _compressedDim * (2 * _latentDim);

        for (int o = 0; o < _latentDim; o++)
        {
            double sumMu = 0, sumLogVar = 0;
            for (int i = 0; i < _compressedDim; i++)
            {
                int muIdx = o * _compressedDim + i;
                int lvIdx = (_latentDim + o) * _compressedDim + i;
                sumMu += taskVector[i] * NumOps.ToDouble(_encoderParams[muIdx]);
                sumLogVar += taskVector[i] * NumOps.ToDouble(_encoderParams[lvIdx]);
            }
            mu[o] = sumMu + NumOps.ToDouble(_encoderParams[biasOffset + o]);
            logVar[o] = sumLogVar + NumOps.ToDouble(_encoderParams[biasOffset + _latentDim + o]);
            // Clamp log_var for numerical stability
            logVar[o] = Math.Max(-10.0, Math.Min(10.0, logVar[o]));
        }
    }

    private double[] Reparameterize(double[] mu, double[] logVar)
    {
        var z = new double[_latentDim];
        for (int i = 0; i < _latentDim; i++)
        {
            double u1 = 1.0 - RandomGenerator.NextDouble();
            double u2 = RandomGenerator.NextDouble();
            double eps = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            z[i] = mu[i] + Math.Exp(0.5 * logVar[i]) * eps;
        }
        return z;
    }

    private double[] DecodeLatent(double[] z)
    {
        var output = new double[_compressedDim];
        int biasOffset = _latentDim * _compressedDim;

        for (int o = 0; o < _compressedDim; o++)
        {
            double sum = 0;
            for (int i = 0; i < _latentDim; i++)
                sum += z[i] * NumOps.ToDouble(_decoderParams[o * _latentDim + i]);

            if (biasOffset + o < _decoderParams.Length)
                sum += NumOps.ToDouble(_decoderParams[biasOffset + o]);

            output[o] = Math.Tanh(sum); // Bounded output
        }
        return output;
    }

    private Vector<T> ApplyCompressedDelta(Vector<T> baseParams, double[] compressedDelta)
    {
        var result = new Vector<T>(_paramDim);
        int stride = Math.Max(1, _paramDim / _compressedDim);

        for (int d = 0; d < _paramDim; d++)
            result[d] = baseParams[d];

        // Apply decoded delta at strided positions with interpolation
        for (int i = 0; i < _compressedDim; i++)
        {
            int start = i * stride;
            int end = Math.Min((i + 1) * stride, _paramDim);
            T delta = NumOps.FromDouble(compressedDelta[i]);
            for (int d = start; d < end; d++)
                result[d] = NumOps.Add(result[d], delta);
        }
        return result;
    }

    private void StoreComponent(double[] z)
    {
        Array.Copy(z, _componentLatents[_componentIdx], _latentDim);
        _componentIdx = (_componentIdx + 1) % _algoOptions.NumFusionComponents;
        if (_componentCount < _algoOptions.NumFusionComponents) _componentCount++;
    }

    private double[] FuseLatents(double[] currentZ)
    {
        if (_componentCount == 0) return currentZ;

        var fused = new double[_latentDim];
        double totalWeight = 1.0;

        // Current component gets weight 1.0
        for (int d = 0; d < _latentDim; d++)
            fused[d] = currentZ[d];

        // Historical components with exponential decay
        double decay = _algoOptions.FusionDecay;
        double weight = decay;
        for (int k = 0; k < _componentCount; k++)
        {
            int idx = ((_componentIdx - 1 - k) % _algoOptions.NumFusionComponents + _algoOptions.NumFusionComponents) % _algoOptions.NumFusionComponents;
            for (int d = 0; d < _latentDim; d++)
                fused[d] += weight * _componentLatents[idx][d];
            totalWeight += weight;
            weight *= decay;
        }

        // Normalize
        for (int d = 0; d < _latentDim; d++)
            fused[d] /= totalWeight;

        return fused;
    }

    private static double ComputeKLDivergence(double[] mu, double[] logVar)
    {
        double kl = 0;
        for (int i = 0; i < mu.Length; i++)
            kl += -0.5 * (1.0 + logVar[i] - mu[i] * mu[i] - Math.Exp(logVar[i]));
        return kl;
    }

    private double ComputeVAELoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var baseParams = MetaModel.GetParameters();
        double totalLoss = 0;

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(baseParams);
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = baseParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
            }

            var tv = CompressTaskVector(baseParams, adaptedParams);
            EncodeTaskVector(tv, out var mu, out var logVar);
            var z = Reparameterize(mu, logVar);
            var decoded = DecodeLatent(FuseLatents(z));
            var fusedParams = ApplyCompressedDelta(baseParams, decoded);

            MetaModel.SetParameters(fusedParams);
            double taskLoss = NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            double klLoss = ComputeKLDivergence(mu, logVar);
            totalLoss += taskLoss + _algoOptions.KLWeight * klLoss;
        }

        MetaModel.SetParameters(baseParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }

    private void InitializeGaussian(Vector<T> v, double stdDev)
    {
        for (int i = 0; i < v.Length; i++)
        {
            double u1 = 1.0 - RandomGenerator.NextDouble();
            double u2 = RandomGenerator.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            v[i] = NumOps.FromDouble(z * stdDev);
        }
    }
}
