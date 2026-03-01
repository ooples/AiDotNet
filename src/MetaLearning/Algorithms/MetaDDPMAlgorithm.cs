using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Meta-DDPM: Meta-Learning with Denoising Diffusion Probabilistic Models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Meta-DDPM meta-learns a denoising diffusion model that generates task-specific model
/// weights conditioned on support set embeddings. It uses the full DDPM framework with
/// a linear noise schedule and EMA parameter averaging for stable generation.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// DDPM Schedule: β_t linear from β_start to β_end, α_t = 1-β_t, ᾱ_t = Πα_s
///
/// Training (noise prediction objective):
///   1. For task τ: get target weights w_0 = base_params + gradient_adapted_delta
///   2. Compute task condition: c = TaskEncoder(avg(model(x_support)))
///   3. Sample t ~ U(1,T), ε ~ N(0,I)
///   4. Noised: w_t = √ᾱ_t · w_0 + √(1-ᾱ_t) · ε
///   5. Predicted noise: ε̂ = UNet(w_t, c, t)
///   6. L = ||ε - ε̂||²
///   7. EMA update denoiser params: φ_ema = decay * φ_ema + (1-decay) * φ
///
/// Generation (DDPM sampling):
///   1. w_T ~ N(0,I)
///   2. For t = T,...,1:
///      μ_θ = (1/√α_t)(w_t - β_t/√(1-ᾱ_t) · ε_θ(w_t,c,t))
///      w_{t-1} = μ_θ + σ_t · z,  z ~ N(0,I), σ_t = √β_t
///   3. θ_adapted = θ_base + w_0
/// </code>
/// </para>
/// <para><b>Key difference from MetaDiff:</b> Meta-DDPM uses the standard DDPM sampling
/// algorithm with full variance (σ_t = √β_t), while MetaDiff uses the deterministic
/// DDIM-style denoising. Meta-DDPM also includes EMA for denoiser parameter stability.
/// </para>
/// </remarks>
public class MetaDDPMAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MetaDDPMOptions<T, TInput, TOutput> _algoOptions;

    private readonly double[] _betas;
    private readonly double[] _alphas;
    private readonly double[] _alphasCumprod;
    private readonly double[] _sqrtAlphasCumprod;
    private readonly double[] _sqrtOneMinusAlphasCumprod;
    private readonly double[] _posteriorVariance;

    /// <summary>Denoiser (noise predictor) parameters.</summary>
    private Vector<T> _denoiserParams;

    /// <summary>EMA copy of denoiser parameters for stable generation.</summary>
    private Vector<T> _denoiserEma;

    /// <summary>Task encoder parameters.</summary>
    private Vector<T> _taskEncoderParams;

    private readonly int _paramDim;
    private readonly int _condDim;
    private readonly int _compressedDim;
    private readonly int _numTimesteps;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaDDPM;

    public MetaDDPMAlgorithm(MetaDDPMOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _condDim = Math.Max(1, options.TaskConditionDim);
        _compressedDim = Math.Min(_paramDim, 128);
        _numTimesteps = Math.Max(1, options.NumTimesteps);

        // Precompute DDPM noise schedule quantities
        _betas = new double[_numTimesteps];
        _alphas = new double[_numTimesteps];
        _alphasCumprod = new double[_numTimesteps];
        _sqrtAlphasCumprod = new double[_numTimesteps];
        _sqrtOneMinusAlphasCumprod = new double[_numTimesteps];
        _posteriorVariance = new double[_numTimesteps];

        for (int t = 0; t < _numTimesteps; t++)
        {
            _betas[t] = options.BetaStart + (options.BetaEnd - options.BetaStart) * t / Math.Max(_numTimesteps - 1, 1);
            _alphas[t] = 1.0 - _betas[t];
            _alphasCumprod[t] = t == 0 ? _alphas[0] : _alphasCumprod[t - 1] * _alphas[t];
            _sqrtAlphasCumprod[t] = Math.Sqrt(_alphasCumprod[t]);
            _sqrtOneMinusAlphasCumprod[t] = Math.Sqrt(1.0 - _alphasCumprod[t]);

            // Posterior variance: β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
            if (t > 0)
            {
                _posteriorVariance[t] = _betas[t] * (1.0 - _alphasCumprod[t - 1]) / (1.0 - _alphasCumprod[t] + 1e-10);
            }
        }

        // Task encoder: featureDim → condDim
        int featureDim = Math.Min(_paramDim, 64);
        int encoderSize = featureDim * _condDim + _condDim;
        _taskEncoderParams = new Vector<T>(encoderSize);
        InitGaussian(_taskEncoderParams, 1.0 / Math.Sqrt(featureDim));

        // Denoiser MLP: (compressedDim + condDim + 1) → hidden → compressedDim
        int denoiserInput = _compressedDim + _condDim + 1;
        int hidden = Math.Max(64, _compressedDim);
        int denoiserSize = denoiserInput * hidden + hidden + hidden * _compressedDim + _compressedDim;
        _denoiserParams = new Vector<T>(denoiserSize);
        InitGaussian(_denoiserParams, 1.0 / Math.Sqrt(denoiserInput));

        // Initialize EMA as copy of denoiser
        _denoiserEma = new Vector<T>(denoiserSize);
        for (int i = 0; i < denoiserSize; i++)
            _denoiserEma[i] = _denoiserParams[i];
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var baseParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(baseParams);

            // Get target: gradient-adapted weights (compressed delta)
            var targetDelta = ComputeTargetDelta(baseParams, task);

            // Task conditioning
            var condition = ComputeTaskCondition(task.SupportInput);

            // Sample timestep and noise
            int t = RandomGenerator.Next(_numTimesteps);
            var epsilon = SampleGaussian(_compressedDim);

            // Forward diffusion: w_t = √ᾱ_t * w_0 + √(1-ᾱ_t) * ε
            var wt = new double[_compressedDim];
            for (int i = 0; i < _compressedDim; i++)
                wt[i] = _sqrtAlphasCumprod[t] * targetDelta[i] + _sqrtOneMinusAlphasCumprod[t] * epsilon[i];

            // Predict noise using denoiser
            var predictedNoise = PredictNoise(wt, condition, t, useEma: false);

            // Noise prediction loss
            double noiseLoss = 0;
            for (int i = 0; i < _compressedDim; i++)
            {
                double diff = epsilon[i] - predictedNoise[i];
                noiseLoss += diff * diff;
            }
            noiseLoss /= _compressedDim;

            // Also evaluate actual task performance
            var generated = SampleDDPM(condition, Math.Min(_algoOptions.SamplingSteps, 5));
            var adaptedParams = ApplyCompressedDelta(baseParams, generated);
            MetaModel.SetParameters(adaptedParams);

            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(NumOps.Add(queryLoss, NumOps.FromDouble(noiseLoss)));

            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Update base params
        MetaModel.SetParameters(baseParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(baseParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        // Update denoiser and task encoder via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _denoiserParams, _algoOptions.OuterLearningRate, ComputeDDPMLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _taskEncoderParams, _algoOptions.OuterLearningRate, ComputeDDPMLoss);

        // EMA update of denoiser
        double decay = _algoOptions.EmaDecay;
        for (int i = 0; i < _denoiserEma.Length; i++)
        {
            double ema = decay * NumOps.ToDouble(_denoiserEma[i]) + (1.0 - decay) * NumOps.ToDouble(_denoiserParams[i]);
            _denoiserEma[i] = NumOps.FromDouble(ema);
        }

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var baseParams = MetaModel.GetParameters();
        MetaModel.SetParameters(baseParams);

        var condition = ComputeTaskCondition(task.SupportInput);
        var generated = SampleDDPM(condition, _algoOptions.SamplingSteps);
        var adaptedParams = ApplyCompressedDelta(baseParams, generated);

        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// Full DDPM sampling (reverse process) with standard variance.
    /// </summary>
    private double[] SampleDDPM(double[] condition, int numSteps)
    {
        var wt = SampleGaussian(_compressedDim);
        int stepSize = Math.Max(1, _numTimesteps / Math.Max(numSteps, 1));

        for (int step = _numTimesteps - 1; step >= 0; step -= stepSize)
        {
            int t = step;
            var predictedNoise = PredictNoise(wt, condition, t, useEma: true);

            // Compute posterior mean: μ_θ = (1/√α_t)(w_t - β_t/√(1-ᾱ_t) * ε_θ)
            double sqrtAlpha = Math.Sqrt(_alphas[t]);
            double betaOverSqrt = _betas[t] / (_sqrtOneMinusAlphasCumprod[t] + 1e-10);

            for (int i = 0; i < _compressedDim; i++)
                wt[i] = (wt[i] - betaOverSqrt * predictedNoise[i]) / sqrtAlpha;

            // Add noise with posterior variance (except at t=0)
            if (t > 0)
            {
                double sigma = Math.Sqrt(_posteriorVariance[t]);
                var z = SampleGaussian(_compressedDim);
                for (int i = 0; i < _compressedDim; i++)
                    wt[i] += sigma * z[i];
            }
        }

        return wt;
    }

    private double[] ComputeTargetDelta(Vector<T> baseParams, IMetaLearningTask<T, TInput, TOutput> task)
    {
        var current = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) current[d] = baseParams[d];

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(current);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            current = ApplyGradients(current, grad, _algoOptions.InnerLearningRate);
        }

        var delta = new double[_compressedDim];
        int stride = Math.Max(1, _paramDim / _compressedDim);
        for (int i = 0; i < _compressedDim; i++)
        {
            int idx = Math.Min(i * stride, _paramDim - 1);
            delta[i] = NumOps.ToDouble(current[idx]) - NumOps.ToDouble(baseParams[idx]);
        }
        return delta;
    }

    private double[] ComputeTaskCondition(TInput supportInput)
    {
        var features = ConvertToVector(MetaModel.Predict(supportInput));
        int featureDim = Math.Min(_paramDim, 64);
        var condition = new double[_condDim];
        if (features == null) return condition;

        int biasOffset = featureDim * _condDim;
        for (int o = 0; o < _condDim; o++)
        {
            double sum = 0;
            int fLen = Math.Min(features.Length, featureDim);
            for (int i = 0; i < fLen; i++)
                sum += NumOps.ToDouble(features[i]) * NumOps.ToDouble(_taskEncoderParams[o * featureDim + i]);
            if (biasOffset + o < _taskEncoderParams.Length)
                sum += NumOps.ToDouble(_taskEncoderParams[biasOffset + o]);
            condition[o] = Math.Tanh(sum);
        }
        return condition;
    }

    private double[] PredictNoise(double[] wt, double[] condition, int timestep, bool useEma)
    {
        var denoiser = useEma ? _denoiserEma : _denoiserParams;
        int inputDim = _compressedDim + _condDim + 1;
        int hidden = Math.Max(64, _compressedDim);

        var input = new double[inputDim];
        Array.Copy(wt, 0, input, 0, _compressedDim);
        Array.Copy(condition, 0, input, _compressedDim, _condDim);
        input[_compressedDim + _condDim] = (double)timestep / _numTimesteps;

        // Hidden layer (ReLU)
        int b1Off = inputDim * hidden;
        var h = new double[hidden];
        for (int o = 0; o < hidden; o++)
        {
            double sum = 0;
            for (int i = 0; i < inputDim; i++)
            {
                int idx = o * inputDim + i;
                if (idx < denoiser.Length) sum += input[i] * NumOps.ToDouble(denoiser[idx]);
            }
            if (b1Off + o < denoiser.Length) sum += NumOps.ToDouble(denoiser[b1Off + o]);
            h[o] = Math.Max(0, sum); // ReLU
        }

        // Output layer
        int w2Off = b1Off + hidden;
        int b2Off = w2Off + hidden * _compressedDim;
        var output = new double[_compressedDim];
        for (int o = 0; o < _compressedDim; o++)
        {
            double sum = 0;
            for (int i = 0; i < hidden; i++)
            {
                int idx = w2Off + o * hidden + i;
                if (idx < denoiser.Length) sum += h[i] * NumOps.ToDouble(denoiser[idx]);
            }
            if (b2Off + o < denoiser.Length) sum += NumOps.ToDouble(denoiser[b2Off + o]);
            output[o] = sum;
        }
        return output;
    }

    private Vector<T> ApplyCompressedDelta(Vector<T> baseParams, double[] delta)
    {
        var result = new Vector<T>(_paramDim);
        int stride = Math.Max(1, _paramDim / _compressedDim);
        for (int d = 0; d < _paramDim; d++) result[d] = baseParams[d];

        for (int i = 0; i < _compressedDim; i++)
        {
            int start = i * stride;
            int end = Math.Min((i + 1) * stride, _paramDim);
            T dt = NumOps.FromDouble(delta[i]);
            for (int d = start; d < end; d++)
                result[d] = NumOps.Add(result[d], dt);
        }
        return result;
    }

    private double ComputeDDPMLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var baseParams = MetaModel.GetParameters();
        double totalLoss = 0;

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(baseParams);
            var target = ComputeTargetDelta(baseParams, task);
            int t = RandomGenerator.Next(_numTimesteps);
            var eps = SampleGaussian(_compressedDim);

            var noised = new double[_compressedDim];
            for (int i = 0; i < _compressedDim; i++)
                noised[i] = _sqrtAlphasCumprod[t] * target[i] + _sqrtOneMinusAlphasCumprod[t] * eps[i];

            var cond = ComputeTaskCondition(task.SupportInput);
            var pred = PredictNoise(noised, cond, t, useEma: false);

            double mse = 0;
            for (int i = 0; i < _compressedDim; i++) { double d = eps[i] - pred[i]; mse += d * d; }
            totalLoss += mse / _compressedDim;
        }

        MetaModel.SetParameters(baseParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }

    private double[] SampleGaussian(int length)
    {
        var arr = new double[length];
        for (int i = 0; i < length; i++)
        {
            double u1 = 1.0 - RandomGenerator.NextDouble();
            double u2 = RandomGenerator.NextDouble();
            arr[i] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }
        return arr;
    }

    private void InitGaussian(Vector<T> v, double stdDev)
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
