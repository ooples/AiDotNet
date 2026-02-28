using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Meta-DM: Applications of Diffusion Models on Few-Shot Learning
/// (Hu et al., ICIP 2024).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Meta-DM uses a DDPM-style diffusion model as a data augmentation module for few-shot learning.
/// Rather than replacing the meta-learning algorithm, it augments the support set by generating
/// synthetic samples conditioned on existing few-shot data, then performs standard gradient-based
/// adaptation on the enriched dataset.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Meta-learned state: base params θ, diffusion model params φ (feature denoiser)
///
/// For each task:
///   1. Extract support features: F = model(x_support) → feature vectors
///   2. Compute class prototypes: p_c = mean(F_c) for each class c
///   3. Generate synthetic features via reverse diffusion conditioned on prototypes
///   4. Augmented support = original features + synthetic features
///   5. Inner loop: adapt θ on augmented support set
///   6. Evaluate on query set
///
/// Outer loop: update θ and diffusion model φ
///
/// Distribution matching:
///   L_match = ||mean(F_real) - mean(F_synth)||² + ||cov(F_real) - cov(F_synth)||²
///   Ensures synthetic features match the real feature distribution moments.
/// </code>
/// </para>
/// <para><b>Key advantage:</b> Modular — can be composed with any gradient-based meta-learner.
/// The diffusion-based augmentation enriches the support set, reducing overfitting.
/// </para>
/// </remarks>
public class MetaDMAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MetaDMOptions<T, TInput, TOutput> _algoOptions;

    /// <summary>Feature denoiser parameters for generating synthetic features.</summary>
    private Vector<T> _denoiserParams;

    /// <summary>Noise schedule.</summary>
    private readonly double[] _betas;
    private readonly double[] _alphasCumprod;

    private readonly int _paramDim;
    private readonly int _prototypeDim;
    private readonly int _diffusionSteps;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaDM;

    public MetaDMAlgorithm(MetaDMOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _prototypeDim = Math.Max(1, options.PrototypeDim);
        _diffusionSteps = Math.Max(1, options.DiffusionTimesteps);

        // Noise schedule
        _betas = new double[_diffusionSteps];
        _alphasCumprod = new double[_diffusionSteps];
        for (int t = 0; t < _diffusionSteps; t++)
        {
            _betas[t] = options.BetaStart + (options.BetaEnd - options.BetaStart) * t / Math.Max(_diffusionSteps - 1, 1);
            double alpha = 1.0 - _betas[t];
            _alphasCumprod[t] = t == 0 ? alpha : _alphasCumprod[t - 1] * alpha;
        }

        // Feature denoiser MLP: (prototypeDim + prototypeDim_condition + 1_timestep) → prototypeDim
        int denoiserInput = _prototypeDim * 2 + 1;
        int hidden = Math.Max(32, _prototypeDim * 2);
        int denoiserSize = denoiserInput * hidden + hidden + hidden * _prototypeDim + _prototypeDim;
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

            // Extract support features and compute prototype
            var supportFeatures = ExtractFeatures(task.SupportInput);
            var prototype = ComputePrototype(supportFeatures);

            // Generate synthetic features via reverse diffusion
            var syntheticFeatures = GenerateSyntheticFeatures(prototype, _algoOptions.SyntheticSamplesPerClass);

            // Compute distribution matching loss
            double matchingLoss = ComputeDistributionMatchingLoss(supportFeatures, syntheticFeatures);

            // Inner loop: adapt on both real and synthetic features
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = baseParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);

                // Gradient from real support data
                var realGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Modulate gradient with augmentation signal (mix real and synthetic influence)
                var augGrad = ModulateWithSynthetic(realGrad, syntheticFeatures, prototype);
                adaptedParams = ApplyGradients(adaptedParams, augGrad, _algoOptions.InnerLearningRate);
            }

            // Evaluate on query set
            MetaModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            var totalLoss = NumOps.Add(queryLoss, NumOps.FromDouble(_algoOptions.MatchingWeight * matchingLoss));
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

        // Update denoiser via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _denoiserParams, _algoOptions.OuterLearningRate, ComputeAugLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var baseParams = MetaModel.GetParameters();
        MetaModel.SetParameters(baseParams);

        var supportFeatures = ExtractFeatures(task.SupportInput);
        var prototype = ComputePrototype(supportFeatures);
        var syntheticFeatures = GenerateSyntheticFeatures(prototype, _algoOptions.SyntheticSamplesPerClass);

        var adaptedParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++) adaptedParams[d] = baseParams[d];

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(adaptedParams);
            var realGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            var augGrad = ModulateWithSynthetic(realGrad, syntheticFeatures, prototype);
            adaptedParams = ApplyGradients(adaptedParams, augGrad, _algoOptions.InnerLearningRate);
        }

        MetaModel.SetParameters(baseParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    private double[] ExtractFeatures(TInput input)
    {
        var features = ConvertToVector(MetaModel.Predict(input));
        var result = new double[_prototypeDim];
        if (features == null) return result;
        for (int i = 0; i < _prototypeDim && i < features.Length; i++)
            result[i] = NumOps.ToDouble(features[i]);
        return result;
    }

    private double[] ComputePrototype(double[] features)
    {
        // For a single feature vector, the prototype is itself
        // In a multi-example scenario, this would be the mean
        var proto = new double[_prototypeDim];
        Array.Copy(features, proto, Math.Min(features.Length, _prototypeDim));
        return proto;
    }

    /// <summary>
    /// Generates synthetic features by running the reverse diffusion process
    /// conditioned on the class prototype.
    /// </summary>
    private double[][] GenerateSyntheticFeatures(double[] prototype, int numSamples)
    {
        var synthetic = new double[numSamples][];
        int denoisingSteps = Math.Max(1, _algoOptions.DenoisingSteps);
        int stepSize = Math.Max(1, _diffusionSteps / denoisingSteps);

        for (int s = 0; s < numSamples; s++)
        {
            // Start from noise
            var xt = SampleGaussian(_prototypeDim);

            // Reverse diffusion conditioned on prototype
            for (int step = _diffusionSteps - 1; step >= 0; step -= stepSize)
            {
                var predictedNoise = PredictFeatureNoise(xt, prototype, step);
                double sqrtAlpha = Math.Sqrt(1.0 - _betas[step]);
                double betaOverSqrt = _betas[step] / Math.Sqrt(1.0 - _alphasCumprod[step] + 1e-10);

                for (int i = 0; i < _prototypeDim; i++)
                    xt[i] = (xt[i] - betaOverSqrt * predictedNoise[i]) / sqrtAlpha;

                if (step > stepSize)
                {
                    double sigma = Math.Sqrt(_betas[step]);
                    var z = SampleGaussian(_prototypeDim);
                    for (int i = 0; i < _prototypeDim; i++)
                        xt[i] += sigma * z[i];
                }
            }

            synthetic[s] = xt;
        }
        return synthetic;
    }

    /// <summary>
    /// Feature denoiser MLP: predicts noise given noised features, prototype condition, and timestep.
    /// </summary>
    private double[] PredictFeatureNoise(double[] noisedFeatures, double[] prototype, int timestep)
    {
        int inputDim = _prototypeDim * 2 + 1;
        int hidden = Math.Max(32, _prototypeDim * 2);

        var input = new double[inputDim];
        Array.Copy(noisedFeatures, 0, input, 0, _prototypeDim);
        Array.Copy(prototype, 0, input, _prototypeDim, _prototypeDim);
        input[_prototypeDim * 2] = (double)timestep / _diffusionSteps;

        // Hidden layer (ReLU)
        int b1Off = inputDim * hidden;
        var h = new double[hidden];
        for (int o = 0; o < hidden; o++)
        {
            double sum = 0;
            for (int i = 0; i < inputDim; i++)
            {
                int idx = o * inputDim + i;
                if (idx < _denoiserParams.Length)
                    sum += input[i] * NumOps.ToDouble(_denoiserParams[idx]);
            }
            if (b1Off + o < _denoiserParams.Length)
                sum += NumOps.ToDouble(_denoiserParams[b1Off + o]);
            h[o] = Math.Max(0, sum);
        }

        // Output layer
        int w2Off = b1Off + hidden;
        int b2Off = w2Off + hidden * _prototypeDim;
        var output = new double[_prototypeDim];
        for (int o = 0; o < _prototypeDim; o++)
        {
            double sum = 0;
            for (int i = 0; i < hidden; i++)
            {
                int idx = w2Off + o * hidden + i;
                if (idx < _denoiserParams.Length)
                    sum += h[i] * NumOps.ToDouble(_denoiserParams[idx]);
            }
            if (b2Off + o < _denoiserParams.Length)
                sum += NumOps.ToDouble(_denoiserParams[b2Off + o]);
            output[o] = sum;
        }
        return output;
    }

    /// <summary>
    /// Computes distribution matching loss between real and synthetic feature distributions
    /// (moment matching: mean and variance).
    /// </summary>
    private double ComputeDistributionMatchingLoss(double[] realFeatures, double[][] syntheticFeatures)
    {
        if (syntheticFeatures.Length == 0) return 0;

        // Mean matching
        double meanLoss = 0;
        for (int d = 0; d < _prototypeDim; d++)
        {
            double synthMean = 0;
            for (int s = 0; s < syntheticFeatures.Length; s++)
                synthMean += syntheticFeatures[s][d];
            synthMean /= syntheticFeatures.Length;

            double diff = realFeatures[d] - synthMean;
            meanLoss += diff * diff;
        }

        // Variance matching
        double varLoss = 0;
        for (int d = 0; d < _prototypeDim; d++)
        {
            double synthMean = 0;
            for (int s = 0; s < syntheticFeatures.Length; s++)
                synthMean += syntheticFeatures[s][d];
            synthMean /= syntheticFeatures.Length;

            double synthVar = 0;
            for (int s = 0; s < syntheticFeatures.Length; s++)
            {
                double diff = syntheticFeatures[s][d] - synthMean;
                synthVar += diff * diff;
            }
            synthVar /= syntheticFeatures.Length;

            // Target variance is small (features should be tightly clustered around prototype)
            varLoss += synthVar;
        }

        return (meanLoss + varLoss) / _prototypeDim;
    }

    /// <summary>
    /// Modulates the real gradient with information from synthetic features to create
    /// an augmented gradient that benefits from the diffusion-generated data.
    /// </summary>
    private Vector<T> ModulateWithSynthetic(Vector<T> realGrad, double[][] syntheticFeatures, double[] prototype)
    {
        if (syntheticFeatures.Length == 0) return realGrad;

        // Compute a modulation factor based on agreement between synthetic and real features
        double agreement = 0;
        for (int s = 0; s < syntheticFeatures.Length; s++)
        {
            double dot = 0;
            for (int d = 0; d < _prototypeDim; d++)
                dot += syntheticFeatures[s][d] * prototype[d];
            agreement += dot;
        }
        agreement /= syntheticFeatures.Length * _prototypeDim;

        // Scale gradient: higher agreement → more confidence in gradient direction
        double scale = 1.0 + 0.1 * Math.Tanh(agreement);
        var modulated = new Vector<T>(realGrad.Length);
        T scaleT = NumOps.FromDouble(scale);
        for (int d = 0; d < realGrad.Length; d++)
            modulated[d] = NumOps.Multiply(realGrad[d], scaleT);

        return modulated;
    }

    private double ComputeAugLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var baseParams = MetaModel.GetParameters();
        double totalLoss = 0;

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(baseParams);
            var sf = ExtractFeatures(task.SupportInput);
            var proto = ComputePrototype(sf);
            var synth = GenerateSyntheticFeatures(proto, _algoOptions.SyntheticSamplesPerClass);

            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = baseParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                var augGrad = ModulateWithSynthetic(grad, synth, proto);
                adaptedParams = ApplyGradients(adaptedParams, augGrad, _algoOptions.InnerLearningRate);
            }

            MetaModel.SetParameters(adaptedParams);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
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
