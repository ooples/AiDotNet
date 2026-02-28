using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of LoRA-Recycle (Hu et al., CVPR 2025).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// LoRA-Recycle enables tuning-free few-shot adaptation by recycling pre-tuned LoRA adapters.
/// It maintains a bank of task-specific LoRA adapters and a prototype-based selection mechanism
/// that computes task embeddings from the support set to weight and combine stored adapters
/// without any gradient-based inner-loop optimization.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Meta-learned state: base params θ, adapter bank {A_1,...,A_K}, prototype encoder E
///
/// Adaptation (gradient-free, single forward pass):
///   1. Compute support set embedding: z = E(support_features)
///   2. Compute similarity to each adapter's prototype: s_k = -||z - p_k||^2
///   3. Compute adapter weights: w_k = softmax(s_k / temperature)
///   4. Fuse adapters: Δθ = Σ w_k * A_k
///   5. Adapted params: θ' = θ + Δθ
///
/// Meta-training:
///   1. For each task: gradient-adapt a fresh LoRA, add to adapter bank
///   2. Train prototype encoder using KL distillation loss
///   3. Update base params via meta-gradients
/// </code>
/// </para>
/// <para><b>Key difference from MAML:</b> No gradient-based inner-loop optimization at
/// inference time. Adaptation is a single forward pass through the prototype encoder
/// followed by adapter selection and fusion.
/// </para>
/// </remarks>
public class LoRARecycleAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly LoRARecycleOptions<T, TInput, TOutput> _algoOptions;

    /// <summary>
    /// Bank of LoRA adapters. Each adapter is a flat vector of parameter deltas (length = paramDim).
    /// </summary>
    private readonly List<Vector<T>> _adapterBank;

    /// <summary>
    /// Prototype embedding for each adapter in the bank (length = prototypeDim per adapter).
    /// </summary>
    private readonly List<Vector<T>> _adapterPrototypes;

    /// <summary>
    /// Prototype encoder parameters: maps feature vectors to prototype space.
    /// Stored as a flat vector of length (paramDim * prototypeDim + prototypeDim).
    /// </summary>
    private Vector<T> _encoderParams;

    private readonly int _paramDim;
    private readonly int _rank;
    private readonly int _prototypeDim;
    private int _bankInsertIdx;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.LoRARecycle;

    public LoRARecycleAlgorithm(LoRARecycleOptions<T, TInput, TOutput> options)
        : base(options.MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer)
    {
        _algoOptions = options;
        _paramDim = options.MetaModel.GetParameters().Length;
        _rank = Math.Max(1, options.Rank);
        _prototypeDim = Math.Max(1, options.PrototypeDim);

        _adapterBank = new List<Vector<T>>();
        _adapterPrototypes = new List<Vector<T>>();
        _bankInsertIdx = 0;

        // Initialize the prototype encoder: linear projection from feature space to prototype space
        // Size: min(paramDim, 128) input features → prototypeDim output (keep manageable)
        int encoderInputDim = Math.Min(_paramDim, 128);
        int encoderSize = encoderInputDim * _prototypeDim + _prototypeDim; // weights + bias
        _encoderParams = new Vector<T>(encoderSize);
        double initScale = 1.0 / Math.Sqrt(encoderInputDim);
        for (int i = 0; i < encoderSize; i++)
        {
            double u1 = 1.0 - RandomGenerator.NextDouble();
            double u2 = RandomGenerator.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            _encoderParams[i] = NumOps.FromDouble(z * initScale);
        }
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var baseParams = MetaModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            // Phase 1: Gradient-adapt a fresh LoRA adapter for this task
            MetaModel.SetParameters(baseParams);
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++)
                adaptedParams[d] = baseParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
            }

            // Compute adapter delta: Δθ = adaptedParams - baseParams
            var adapterDelta = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++)
                adapterDelta[d] = NumOps.Subtract(adaptedParams[d], baseParams[d]);

            // Compute support set embedding for this task
            MetaModel.SetParameters(baseParams);
            var supportEmbed = ComputeTaskEmbedding(task.SupportInput);

            // Add adapter and prototype to bank (circular buffer)
            AddToBank(adapterDelta, supportEmbed);

            // Phase 2: Evaluate via recycled adapter fusion (gradient-free)
            if (_adapterBank.Count > 0)
            {
                var fusedDelta = FuseAdapters(supportEmbed);
                var fusedParams = new Vector<T>(_paramDim);
                for (int d = 0; d < _paramDim; d++)
                    fusedParams[d] = NumOps.Add(baseParams[d], fusedDelta[d]);

                MetaModel.SetParameters(fusedParams);
            }
            else
            {
                MetaModel.SetParameters(adaptedParams);
            }

            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(queryLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Outer loop: update base parameters
        MetaModel.SetParameters(baseParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            MetaModel.SetParameters(ApplyGradients(baseParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        // Update encoder params via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _encoderParams, _algoOptions.OuterLearningRate, ComputeRecycleLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var baseParams = MetaModel.GetParameters();

        if (_adapterBank.Count == 0)
        {
            // Fall back to gradient-based adaptation if no adapters are available yet
            var adaptedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++) adaptedParams[d] = baseParams[d];

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                MetaModel.SetParameters(adaptedParams);
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
                adaptedParams = ApplyGradients(adaptedParams, grad, _algoOptions.InnerLearningRate);
            }
            MetaModel.SetParameters(baseParams);
            return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
        }

        // Gradient-free adaptation: compute task embedding and fuse adapters
        var taskEmbed = ComputeTaskEmbedding(task.SupportInput);
        var fusedDelta = FuseAdapters(taskEmbed);
        var finalParams = new Vector<T>(_paramDim);
        for (int d = 0; d < _paramDim; d++)
            finalParams[d] = NumOps.Add(baseParams[d], fusedDelta[d]);

        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, finalParams);
    }

    /// <summary>
    /// Computes a task embedding by running support data through the model and projecting
    /// the averaged feature representation through the prototype encoder.
    /// </summary>
    private Vector<T> ComputeTaskEmbedding(TInput supportInput)
    {
        var features = ConvertToVector(MetaModel.Predict(supportInput));
        if (features == null || features.Length == 0)
            return new Vector<T>(_prototypeDim);

        // Take first min(paramDim, 128) features as encoder input
        int inputDim = Math.Min(_paramDim, 128);
        int featureLen = Math.Min(features.Length, inputDim);

        // Linear projection: output = W * input + bias
        var embedding = new Vector<T>(_prototypeDim);
        int biasOffset = inputDim * _prototypeDim;

        for (int o = 0; o < _prototypeDim; o++)
        {
            double sum = 0;
            for (int i = 0; i < featureLen; i++)
                sum += NumOps.ToDouble(features[i]) * NumOps.ToDouble(_encoderParams[o * inputDim + i]);

            // Add bias
            if (biasOffset + o < _encoderParams.Length)
                sum += NumOps.ToDouble(_encoderParams[biasOffset + o]);

            // Tanh activation for bounded embeddings
            embedding[o] = NumOps.FromDouble(Math.Tanh(sum));
        }

        return embedding;
    }

    /// <summary>
    /// Fuses adapters from the bank using softmax-weighted combination based on
    /// similarity between the task embedding and adapter prototypes.
    /// </summary>
    private Vector<T> FuseAdapters(Vector<T> taskEmbedding)
    {
        int numAdapters = _adapterBank.Count;
        var similarities = new double[numAdapters];
        double maxSim = double.NegativeInfinity;

        // Compute negative squared Euclidean distance to each prototype
        for (int k = 0; k < numAdapters; k++)
        {
            double dist = 0;
            for (int d = 0; d < _prototypeDim; d++)
            {
                double diff = NumOps.ToDouble(taskEmbedding[d]) - NumOps.ToDouble(_adapterPrototypes[k][d]);
                dist += diff * diff;
            }
            similarities[k] = -dist / _algoOptions.SelectionTemperature;
            if (similarities[k] > maxSim) maxSim = similarities[k];
        }

        // Softmax to get adapter weights
        var weights = new double[numAdapters];
        double sumExp = 0;
        for (int k = 0; k < numAdapters; k++)
        {
            weights[k] = Math.Exp(similarities[k] - maxSim);
            sumExp += weights[k];
        }
        for (int k = 0; k < numAdapters; k++)
            weights[k] /= sumExp;

        // Weighted combination of adapter deltas
        var fused = new Vector<T>(_paramDim);
        for (int k = 0; k < numAdapters; k++)
        {
            if (weights[k] < 1e-10) continue;
            T w = NumOps.FromDouble(weights[k]);
            for (int d = 0; d < _paramDim; d++)
                fused[d] = NumOps.Add(fused[d], NumOps.Multiply(_adapterBank[k][d], w));
        }

        return fused;
    }

    /// <summary>
    /// Adds an adapter and its prototype to the circular bank.
    /// </summary>
    private void AddToBank(Vector<T> adapterDelta, Vector<T> prototype)
    {
        if (_adapterBank.Count < _algoOptions.NumRecycledAdapters)
        {
            _adapterBank.Add(adapterDelta);
            _adapterPrototypes.Add(prototype);
        }
        else
        {
            _adapterBank[_bankInsertIdx] = adapterDelta;
            _adapterPrototypes[_bankInsertIdx] = prototype;
            _bankInsertIdx = (_bankInsertIdx + 1) % _algoOptions.NumRecycledAdapters;
        }
    }

    private double ComputeRecycleLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var baseParams = MetaModel.GetParameters();
        double totalLoss = 0;

        foreach (var task in taskBatch.Tasks)
        {
            MetaModel.SetParameters(baseParams);
            if (_adapterBank.Count == 0) continue;

            var taskEmbed = ComputeTaskEmbedding(task.SupportInput);
            var fusedDelta = FuseAdapters(taskEmbed);
            var fusedParams = new Vector<T>(_paramDim);
            for (int d = 0; d < _paramDim; d++)
                fusedParams[d] = NumOps.Add(baseParams[d], fusedDelta[d]);

            MetaModel.SetParameters(fusedParams);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        MetaModel.SetParameters(baseParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
