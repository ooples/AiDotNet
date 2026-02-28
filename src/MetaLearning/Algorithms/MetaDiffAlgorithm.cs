using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of MetaDiff: Meta-Learning with Conditional Diffusion for Few-Shot Learning
/// (Zhang et al., AAAI 2024).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// MetaDiff replaces the gradient-based inner loop of MAML with a learned diffusion-based
/// denoising process. Starting from Gaussian noise, a task-conditional denoising network
/// (TCUNet) iteratively produces task-specific weight parameters. The key insight is that
/// gradient descent (random init → optimal weights) is analogous to the diffusion reverse
/// process (noise → clean signal).
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Noise schedule: β_t linearly from β_start to β_end, α_t = 1 - β_t, ᾱ_t = Π α_s
///
/// Training (outer loop):
///   1. For task τ: compute "target weights" w_0 via support-set gradient adaptation
///   2. Sample t ~ Uniform(1, T), ε ~ N(0, I)
///   3. Noised weights: w_t = √ᾱ_t * w_0 + √(1-ᾱ_t) * ε
///   4. Task condition: c = TaskEncoder(support_features)
///   5. Predicted noise: ε̂ = Denoiser(w_t, c, t)
///   6. Loss = ||ε - ε̂||²
///
/// Inference (adaptation):
///   1. Start from w_T ~ N(0, I)
///   2. For t = T, T-1, ..., 1:
///      w_{t-1} = (1/√α_t)(w_t - β_t/√(1-ᾱ_t) * Denoiser(w_t, c, t)) + σ_t * z
///   3. Adapted params: θ' = θ_base + w_0
/// </code>
/// </para>
/// <para><b>Key advantage:</b> GPU memory is constant regardless of adaptation steps
/// (unlike MAML which scales linearly). The denoising process can be run for as many
/// steps as desired without backpropagating through the full chain.
/// </para>
/// </remarks>
public class MetaDiffAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MetaDiffOptions<T, TInput, TOutput> _algoOptions;

    /// <summary>Noise schedule: β_t values.</summary>
    private readonly double[] _betas;

    /// <summary>α_t = 1 - β_t.</summary>
    private readonly double[] _alphas;

    /// <summary>ᾱ_t = cumulative product of α.</summary>
    private readonly double[] _alphasCumprod;

    /// <summary>Denoiser network parameters (task-conditional noise predictor).</summary>
    private Vector<T> _denoiserParams;

    /// <summary>Task encoder parameters: maps support features to conditioning vector.</summary>
    private Vector<T> _taskEncoderParams;

    private readonly int _paramDim;
    private readonly int _condDim;
    private readonly int _compressedDim;
    private readonly int _diffusionSteps;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaDiff;

    public MetaDiffAlgorithm(MetaDiffOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _condDim = Math.Max(1, options.TaskConditionDim);
        _compressedDim = Math.Min(_paramDim, 128);
        _diffusionSteps = Math.Max(1, options.DiffusionSteps);

        // Compute linear noise schedule
        _betas = new double[_diffusionSteps];
        _alphas = new double[_diffusionSteps];
        _alphasCumprod = new double[_diffusionSteps];

        for (int t = 0; t < _diffusionSteps; t++)
        {
            _betas[t] = options.BetaStart + (options.BetaEnd - options.BetaStart) * t / Math.Max(_diffusionSteps - 1, 1);
            _alphas[t] = 1.0 - _betas[t];
            _alphasCumprod[t] = t == 0 ? _alphas[0] : _alphasCumprod[t - 1] * _alphas[t];
        }

        // Task encoder: feature_dim → condDim (linear + tanh)
        int featureDim = Math.Min(_paramDim, 64);
        int encoderSize = featureDim * _condDim + _condDim;
        _taskEncoderParams = new Vector<T>(encoderSize);
        InitGaussian(_taskEncoderParams, 1.0 / Math.Sqrt(featureDim));

        // Denoiser: (compressedDim + condDim + 1_timestep) → compressedDim
        // Simple MLP: input → hidden → output
        int denoiserInput = _compressedDim + _condDim + 1;
        int hidden = Math.Max(32, _compressedDim);
        int denoiserSize = denoiserInput * hidden + hidden + hidden * _compressedDim + _compressedDim;
        _denoiserParams = new Vector<T>(denoiserSize);
        InitGaussian(_denoiserParams, 1.0 / Math.Sqrt(denoiserInput));
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

            // Step 1: Compute "target weights" w_0 via gradient adaptation on support set
            var targetDelta = ComputeTargetDelta(baseParams, task);

            // Step 2: Sample random timestep and noise
            int t = RandomGenerator.Next(_diffusionSteps);
            var noise = SampleGaussianArray(_compressedDim);

            // Step 3: Add noise to target delta: w_t = √ᾱ_t * w_0 + √(1-ᾱ_t) * ε
            double sqrtAlphaCumprod = Math.Sqrt(_alphasCumprod[t]);
            double sqrtOneMinusAlphaCumprod = Math.Sqrt(1.0 - _alphasCumprod[t]);
            var noisedDelta = new double[_compressedDim];
            for (int i = 0; i < _compressedDim; i++)
                noisedDelta[i] = sqrtAlphaCumprod * targetDelta[i] + sqrtOneMinusAlphaCumprod * noise[i];

            // Step 4: Compute task conditioning
            var taskCondition = ComputeTaskCondition(task.SupportInput);

            // Step 5: Predict noise
            var predictedNoise = PredictNoise(noisedDelta, taskCondition, t);

            // Step 6: Noise prediction loss (MSE)
            double noiseLoss = 0;
            for (int i = 0; i < _compressedDim; i++)
            {
                double diff = noise[i] - predictedNoise[i];
                noiseLoss += diff * diff;
            }
            noiseLoss /= _compressedDim;

            // Also evaluate task performance for meta-gradient on base params
            var denoised = RunDenoisingInference(taskCondition, Math.Min(_algoOptions.SamplingSteps, 5));
            var adaptedParams = ApplyCompressedDelta(baseParams, denoised);
            MetaModel.SetParameters(adaptedParams);

            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            var totalLoss = NumOps.Add(queryLoss, NumOps.FromDouble(noiseLoss));
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

        // Update denoiser and task encoder via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _denoiserParams, _algoOptions.OuterLearningRate, ComputeDiffusionLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _taskEncoderParams, _algoOptions.OuterLearningRate, ComputeDiffusionLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var baseParams = MetaModel.GetParameters();
        MetaModel.SetParameters(baseParams);

        // Compute task conditioning from support set
        var taskCondition = ComputeTaskCondition(task.SupportInput);

        // Run full denoising inference
        var denoisedDelta = RunDenoisingInference(taskCondition, _algoOptions.SamplingSteps);
        var adaptedParams = ApplyCompressedDelta(baseParams, denoisedDelta);

        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// Computes target parameter delta by gradient adaptation on support set (compressed).
    /// </summary>
    private double[] ComputeTargetDelta(Vector<T> baseParams, IMetaLearningTask<T, TInput, TOutput> task)
    {
        var currentParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) currentParams[d] = baseParams[d];

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(currentParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            currentParams = ApplyGradients(currentParams, grad, _algoOptions.InnerLearningRate);
        }

        // Compress delta
        var delta = new double[_compressedDim];
        int stride = Math.Max(1, _paramDim / _compressedDim);
        for (int i = 0; i < _compressedDim; i++)
        {
            int idx = Math.Min(i * stride, _paramDim - 1);
            delta[i] = NumOps.ToDouble(currentParams[idx]) - NumOps.ToDouble(baseParams[idx]);
        }
        return delta;
    }

    /// <summary>
    /// Computes task conditioning vector from support features.
    /// </summary>
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

    /// <summary>
    /// Denoiser MLP: predicts noise given noised weights, task condition, and timestep.
    /// Architecture: concat(w_t, c, t/T) → hidden(ReLU) → output.
    /// </summary>
    private double[] PredictNoise(double[] noisedDelta, double[] taskCondition, int timestep)
    {
        int inputDim = _compressedDim + _condDim + 1;
        int hidden = Math.Max(32, _compressedDim);

        // Build input vector
        var input = new double[inputDim];
        Array.Copy(noisedDelta, 0, input, 0, _compressedDim);
        Array.Copy(taskCondition, 0, input, _compressedDim, _condDim);
        input[_compressedDim + _condDim] = (double)timestep / _diffusionSteps; // Normalized timestep

        // Layer 1: input → hidden (ReLU)
        int w1Offset = 0;
        int b1Offset = inputDim * hidden;
        var h = new double[hidden];
        for (int o = 0; o < hidden; o++)
        {
            double sum = 0;
            for (int i = 0; i < inputDim; i++)
            {
                int idx = w1Offset + o * inputDim + i;
                if (idx < _denoiserParams.Length)
                    sum += input[i] * NumOps.ToDouble(_denoiserParams[idx]);
            }
            if (b1Offset + o < _denoiserParams.Length)
                sum += NumOps.ToDouble(_denoiserParams[b1Offset + o]);
            h[o] = Math.Max(0, sum); // ReLU
        }

        // Layer 2: hidden → output
        int w2Offset = b1Offset + hidden;
        int b2Offset = w2Offset + hidden * _compressedDim;
        var output = new double[_compressedDim];
        for (int o = 0; o < _compressedDim; o++)
        {
            double sum = 0;
            for (int i = 0; i < hidden; i++)
            {
                int idx = w2Offset + o * hidden + i;
                if (idx < _denoiserParams.Length)
                    sum += h[i] * NumOps.ToDouble(_denoiserParams[idx]);
            }
            if (b2Offset + o < _denoiserParams.Length)
                sum += NumOps.ToDouble(_denoiserParams[b2Offset + o]);
            output[o] = sum;
        }

        return output;
    }

    /// <summary>
    /// Runs the reverse diffusion process (denoising) to generate task-specific weight deltas.
    /// </summary>
    private double[] RunDenoisingInference(double[] taskCondition, int numSteps)
    {
        // Start from pure noise
        var wt = SampleGaussianArray(_compressedDim);
        int stepSize = Math.Max(1, _diffusionSteps / Math.Max(numSteps, 1));

        for (int step = _diffusionSteps - 1; step >= 0; step -= stepSize)
        {
            int t = step;
            var predictedNoise = PredictNoise(wt, taskCondition, t);

            double sqrtAlpha = Math.Sqrt(_alphas[t]);
            double betaOverSqrt = _betas[t] / Math.Sqrt(1.0 - _alphasCumprod[t] + 1e-10);

            for (int i = 0; i < _compressedDim; i++)
            {
                wt[i] = (wt[i] - betaOverSqrt * predictedNoise[i]) / sqrtAlpha;
            }

            // Add noise (except for t=0)
            if (t > 0)
            {
                double sigma = Math.Sqrt(_betas[t]);
                var z = SampleGaussianArray(_compressedDim);
                for (int i = 0; i < _compressedDim; i++)
                    wt[i] += sigma * z[i];
            }
        }

        return wt;
    }

    private Vector<T> ApplyCompressedDelta(Vector<T> baseParams, double[] compressedDelta)
    {
        var result = new Vector<T>(_paramDim);
        int stride = Math.Max(1, _paramDim / _compressedDim);

        for (int d = 0; d < _paramDim; d++)
            result[d] = baseParams[d];

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

    private double[] SampleGaussianArray(int length)
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

    private double ComputeDiffusionLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var baseParams = MetaModel.GetParameters();
        double totalLoss = 0;

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(baseParams);
            var targetDelta = ComputeTargetDelta(baseParams, task);
            int t = RandomGenerator.Next(_diffusionSteps);
            var noise = SampleGaussianArray(_compressedDim);

            double sqrtAC = Math.Sqrt(_alphasCumprod[t]);
            double sqrtOMAC = Math.Sqrt(1.0 - _alphasCumprod[t]);
            var noised = new double[_compressedDim];
            for (int i = 0; i < _compressedDim; i++)
                noised[i] = sqrtAC * targetDelta[i] + sqrtOMAC * noise[i];

            var cond = ComputeTaskCondition(task.SupportInput);
            var pred = PredictNoise(noised, cond, t);

            double mse = 0;
            for (int i = 0; i < _compressedDim; i++)
            {
                double diff = noise[i] - pred[i];
                mse += diff * diff;
            }
            totalLoss += mse / _compressedDim;
        }

        MetaModel.SetParameters(baseParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
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
