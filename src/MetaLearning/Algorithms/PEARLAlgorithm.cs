using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of PEARL: Probabilistic Embeddings for Actor-critic RL
/// (Rakelly et al., ICML 2019).
/// </summary>
/// <remarks>
/// <para>
/// PEARL infers a latent task variable z from context (support data) using a probabilistic
/// encoder q(z|c). The encoder produces a Gaussian posterior, and z is sampled via
/// reparameterization. The policy/model is conditioned on z through parameter modulation.
/// At test time, z is inferred from support data without gradient updates.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Context encoder: q(z|c) = N(μ(c), σ²(c))
///   c = aggregate of support (input, output) pairs
///   μ, log(σ²) = encoder_params applied to aggregated context
///
/// Inner loop:
///   1. Encode context: z ~ q(z|support)
///   2. Modulate params: θ_z = θ_0 + W_z * z  (linear projection of z)
///   3. Gradient adapt θ_z on support with modulated params
///
/// Outer loop:
///   L = E_z[L_query(θ_z)] + β * KL(q(z|c) || N(0,I))
///   Update θ_0, encoder params, W_z
/// </code>
/// </para>
/// </remarks>
public class PEARLAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly PEARLOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _latentDim;
    private readonly int _compressedDim;

    /// <summary>Encoder parameters: maps compressed gradient → (μ, log_σ²) of size 2*latentDim.</summary>
    private Vector<T> _encoderParams;

    /// <summary>Projection matrix W_z: maps z (latentDim) → parameter delta (compressedDim).</summary>
    private Vector<T> _projectionParams;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.PEARL;

    public PEARLAlgorithm(PEARLOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _latentDim = options.LatentDim;
        _compressedDim = Math.Min(_paramDim, options.EncoderHiddenDim);

        // Encoder: compressed context → hidden → (μ, log_σ²)
        int encoderSize = _compressedDim * _algoOptions.EncoderHiddenDim + _algoOptions.EncoderHiddenDim * 2 * _latentDim;
        _encoderParams = new Vector<T>(encoderSize);
        for (int i = 0; i < encoderSize; i++)
        {
            double u1 = Math.Max(1e-10, RandomGenerator.NextDouble());
            double u2 = RandomGenerator.NextDouble();
            _encoderParams[i] = NumOps.FromDouble(0.01 * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }

        // Projection: z → parameter delta
        _projectionParams = new Vector<T>(_latentDim * _compressedDim);
        for (int i = 0; i < _projectionParams.Length; i++)
        {
            double u1 = Math.Max(1e-10, RandomGenerator.NextDouble());
            double u2 = RandomGenerator.NextDouble();
            _projectionParams[i] = NumOps.FromDouble(0.01 * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
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
            // Encode context from support set
            MetaModel.SetParameters(initParams);
            var contextVec = ComputeContext(task.SupportInput, task.SupportOutput, initParams);
            var (mu, logVar) = Encode(contextVec);

            // Sample z and compute KL
            double totalKL = 0;
            var sampleLosses = new List<T>();
            var sampleGrads = new List<Vector<T>>();

            int numSamples = _algoOptions.NumPosteriorSamples;
            for (int s = 0; s < numSamples; s++)
            {
                var z = Reparameterize(mu, logVar);
                var paramDelta = ProjectZ(z);

                // Modulate parameters
                var modulatedParams = new Vector<T>(_paramDim);
                for (int d = 0; d < _paramDim; d++)
                {
                    int cd = d % _compressedDim;
                    modulatedParams[d] = NumOps.Add(initParams[d], NumOps.FromDouble(paramDelta[cd]));
                }

                // Gradient adaptation
                for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
                {
                    MetaModel.SetParameters(modulatedParams);
                    var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                    modulatedParams = ApplyGradients(modulatedParams, grad, _algoOptions.InnerLearningRate);
                }

                MetaModel.SetParameters(modulatedParams);
                sampleLosses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
                sampleGrads.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
            }

            // KL(q(z|c) || N(0,I))
            for (int i = 0; i < _latentDim; i++)
                totalKL += 0.5 * (Math.Exp(logVar[i]) + mu[i] * mu[i] - 1.0 - logVar[i]);

            var taskLoss = NumOps.Add(ComputeMean(sampleLosses), NumOps.FromDouble(_algoOptions.KLWeight * totalKL));
            losses.Add(taskLoss);
            metaGradients.Add(AverageVectors(sampleGrads));
        }

        // Outer loop
        MetaModel.SetParameters(initParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(initParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        // Update encoder and projection via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _encoderParams, _algoOptions.OuterLearningRate * 0.1, ComputePEARLLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _projectionParams, _algoOptions.OuterLearningRate * 0.1, ComputePEARLLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = MetaModel.GetParameters();
        var contextVec = ComputeContext(task.SupportInput, task.SupportOutput, initParams);
        var (mu, _) = Encode(contextVec);

        // Use posterior mean (no sampling for deterministic adaptation)
        var paramDelta = ProjectZ(mu);
        var modulatedParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
        {
            int cd = d % _compressedDim;
            modulatedParams[d] = NumOps.Add(initParams[d], NumOps.FromDouble(paramDelta[cd]));
        }

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            MetaModel.SetParameters(modulatedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            modulatedParams = ApplyGradients(modulatedParams, grad, _algoOptions.InnerLearningRate);
        }

        MetaModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, modulatedParams);
    }

    private double[] ComputeContext(TInput supportInput, TOutput supportOutput, Vector<T> currentParams)
    {
        MetaModel.SetParameters(currentParams);
        var grad = ComputeGradients(MetaModel, supportInput, supportOutput);
        var ctx = new double[_compressedDim];
        for (int d = 0; d < _compressedDim && d < grad.Length; d++)
            ctx[d] = NumOps.ToDouble(grad[d]);
        return ctx;
    }

    private (double[] mu, double[] logVar) Encode(double[] context)
    {
        int hiddenDim = _algoOptions.EncoderHiddenDim;
        var hidden = new double[hiddenDim];
        int offset = 0;

        // Layer 1: context → hidden (ReLU)
        for (int h = 0; h < hiddenDim; h++)
        {
            double sum = 0;
            for (int c = 0; c < _compressedDim; c++)
                sum += context[c] * NumOps.ToDouble(_encoderParams[offset + h * _compressedDim + c]);
            hidden[h] = Math.Max(0, sum); // ReLU
        }
        offset += _compressedDim * hiddenDim;

        // Layer 2: hidden → (μ, log_σ²)
        var mu = new double[_latentDim];
        var logVar = new double[_latentDim];
        for (int l = 0; l < _latentDim; l++)
        {
            double sumMu = 0, sumLogVar = 0;
            for (int h = 0; h < hiddenDim; h++)
            {
                sumMu += hidden[h] * NumOps.ToDouble(_encoderParams[offset + l * hiddenDim + h]);
                sumLogVar += hidden[h] * NumOps.ToDouble(_encoderParams[offset + _latentDim * hiddenDim + l * hiddenDim + h]);
            }
            mu[l] = sumMu;
            logVar[l] = Math.Max(-10, Math.Min(2, sumLogVar)); // Clamp
        }

        return (mu, logVar);
    }

    private double[] Reparameterize(double[] mu, double[] logVar)
    {
        var z = new double[_latentDim];
        for (int i = 0; i < _latentDim; i++)
        {
            double u1 = Math.Max(1e-10, RandomGenerator.NextDouble());
            double u2 = RandomGenerator.NextDouble();
            double eps = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            z[i] = mu[i] + Math.Exp(0.5 * logVar[i]) * eps;
        }
        return z;
    }

    private double[] ProjectZ(double[] z)
    {
        var delta = new double[_compressedDim];
        for (int d = 0; d < _compressedDim; d++)
        {
            double sum = 0;
            for (int l = 0; l < _latentDim; l++)
                sum += z[l] * NumOps.ToDouble(_projectionParams[d * _latentDim + l]);
            delta[d] = sum;
        }
        return delta;
    }

    private double ComputePEARLLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        double totalLoss = 0;
        var initParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var contextVec = ComputeContext(task.SupportInput, task.SupportOutput, initParams);
            var (mu, logVar) = Encode(contextVec);
            var paramDelta = ProjectZ(mu);

            var modulatedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++)
                modulatedParams[d] = NumOps.Add(initParams[d], NumOps.FromDouble(paramDelta[d % _compressedDim]));

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(modulatedParams);
                var grad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                modulatedParams = ApplyGradients(modulatedParams, grad, _algoOptions.InnerLearningRate);
            }

            MetaModel.SetParameters(modulatedParams);
            double loss = NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
            double kl = 0;
            for (int i = 0; i < _latentDim; i++)
                kl += 0.5 * (Math.Exp(logVar[i]) + mu[i] * mu[i] - 1.0 - logVar[i]);
            totalLoss += loss + _algoOptions.KLWeight * kl;
        }

        MetaModel.SetParameters(initParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
