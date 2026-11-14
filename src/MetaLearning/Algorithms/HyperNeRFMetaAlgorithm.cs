using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("HyperNeRF: A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields",
    "https://arxiv.org/abs/2106.13228",
    Year = 2021,
    Authors = "Keunhong Park, Utkarsh Sinha, Peter Hedman, et al.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class HyperNeRFMetaAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly HyperNeRFMetaOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;
    private readonly int _peDim;
    private readonly int _numGroups;
    private readonly int _groupSize;

    /// <summary>Conditioning MLP weights: (peDim + latentDim) × numGroups.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _conditioningWeights;

    /// <summary>Pre-computed positional encodings per group, [numGroups, peDim].</summary>
    private readonly Tensor<T> _positionalEncodings;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.HyperNeRFMeta;

    public HyperNeRFMetaAlgorithm(HyperNeRFMetaOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Regression),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        if (_paramDim == 0)
            throw new ArgumentException("MetaModel has zero parameters. HyperNeRF requires a model with at least one parameter.");
        if (options.NumFrequencyBands <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "NumFrequencyBands must be positive.");
        if (options.LatentDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "LatentDim must be positive.");
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

        // Pre-compute positional encodings for each group:
        // [pos, sin(2^0 pi pos), cos(2^0 pi pos), ..., sin(2^(F-1) pi pos), cos(2^(F-1) pi pos)], pos = g / max(G - 1, 1)
        int numBands = options.NumFrequencyBands;
        var positions = Engine.TensorDivideScalar(
            Engine.TensorArange<T>(0, _numGroups, 1), NumOps.FromDouble(Math.Max(_numGroups - 1, 1)));
        var frequencies = Engine.TensorMultiplyScalar(
            Engine.TensorPower(Tensor<T>.FromVector(Engine.Fill(numBands, NumOps.FromDouble(2.0))), Engine.TensorArange<T>(0, numBands, 1)),
            NumOps.FromDouble(Math.PI));
        var angles = Engine.TensorOuter(positions, frequencies);
        var sinCos = Engine.Reshape(
            Engine.TensorStack(new[] { Engine.TensorSin(angles), Engine.TensorCos(angles) }, 2), new[] { _numGroups, 2 * numBands });
        _positionalEncodings = Engine.TensorConcatenate(
            new[] { Engine.Reshape(positions, new[] { _numGroups, 1 }), sinCos }, 1);
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var adaptedParams = AdaptConditioned(task, initParams, clip: true, recondition: true);

            ParamModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Conditioning weight regularization
            T condReg = Engine.DotProduct(_conditioningWeights, _conditioningWeights);

            var totalLoss = NumOps.Add(queryLoss, NumOps.Multiply(
                NumOps.FromDouble(_algoOptions.ConditioningRegWeight / _conditioningWeights.Length), condReg));

            losses.Add(totalLoss);
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _conditioningWeights, _algoOptions.OuterLearningRate * 0.1, ComputeNeRFLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = ParamModel.GetParameters();
        var adaptedParams = AdaptConditioned(task, initParams, clip: true, recondition: false);

        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// The conditioned inner loop: a latent code from the support gradient conditions per-group step sizes,
    /// theta_d -= lr * modulation[group(d)] * g_d. With <paramref name="recondition"/> the code is recomputed
    /// from a fresh unclipped gradient after every even step except the last.
    /// </summary>
    private Vector<T> AdaptConditioned(IMetaLearningTask<T, TInput, TOutput> task, Vector<T> initParams, bool clip, bool recondition)
    {
        var adaptedParams = initParams.Clone();

        // Compute task latent code from support gradient
        ParamModel.SetParameters(adaptedParams);
        var supportGrad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
        var latentCode = CompressGradient(clip ? ClipGradients(supportGrad) : supportGrad);

        // Compute per-group modulation factors
        var modulation = ComputeModulation(latentCode);
        T learningRate = NumOps.FromDouble(_algoOptions.InnerLearningRate);

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adaptedParams);
            var raw = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
            var grad = (clip ? ClipGradients(raw) : raw).GetSubVector(0, _paramDim);

            // Apply position-aware, task-conditioned modulation
            var scaled = Engine.Multiply(PerParameter(modulation), grad);
            adaptedParams = Engine.Subtract(adaptedParams, Engine.Multiply(scaled, learningRate));

            // Recompute latent code for dynamic conditioning (optional: every other step)
            if (recondition && step < _algoOptions.AdaptationSteps - 1 && step % 2 == 0)
            {
                ParamModel.SetParameters(adaptedParams);
                var newGrad = ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput);
                modulation = ComputeModulation(CompressGradient(newGrad));
            }
        }

        return adaptedParams;
    }

    /// <summary>Each parameter's group value: group g covers [g * groupSize, (g + 1) * groupSize).</summary>
    private Vector<T> PerParameter(Vector<T> groupValues)
    {
        var repeated = Engine.TensorBroadcastTo(
            Tensor<T>.FromVector(groupValues, new[] { _numGroups, 1 }), new[] { _numGroups, _groupSize });
        return Engine.Reshape(repeated, new[] { _numGroups * _groupSize }).ToVector().GetSubVector(0, _paramDim);
    }

    /// <summary>
    /// tanh of the bucket means of the gradient: bucket e covers [e * b, e * b + b) clipped to the gradient,
    /// with b = ceil(paramDim / latentDim); empty buckets give 0.
    /// </summary>
    private Vector<T> CompressGradient(Vector<T> grad)
    {
        int latentDim = _algoOptions.LatentDim;
        // Use ceiling division so all parameters are covered (no remainder dropped)
        int bucketSize = Math.Max(1, (_paramDim + latentDim - 1) / latentDim);
        int covered = latentDim * bucketSize;
        int used = Math.Min(grad.Length, covered);

        var values = used == grad.Length ? grad : grad.GetSubVector(0, used);
        var present = Engine.Fill(used, NumOps.One);
        if (used < covered)
        {
            values = Vector<T>.Concatenate(values, new Vector<T>(covered - used));
            present = Vector<T>.Concatenate(present, new Vector<T>(covered - used));
        }

        var shape = new[] { latentDim, bucketSize };
        var sums = Engine.ReduceSum(Tensor<T>.FromVector(values, shape), new[] { 1 }, keepDims: false);
        var counts = Engine.TensorMax(Engine.ReduceSum(Tensor<T>.FromVector(present, shape), new[] { 1 }, keepDims: false), NumOps.One);
        return Engine.TensorTanh(Engine.TensorDivide(sums, counts)).ToVector();
    }

    /// <summary>
    /// 1 + strength * (sigmoid(w_g . [PE(g), z]) - 0.5) per group, with w_g row g of the [numGroups, peDim + latentDim]
    /// conditioning weights.
    /// </summary>
    private Vector<T> ComputeModulation(Vector<T> latentCode)
    {
        int latentDim = _algoOptions.LatentDim;
        var weights = Tensor<T>.FromVector(_conditioningWeights, new[] { _numGroups, _peDim + latentDim });

        // PE features
        var peWeights = Engine.TensorSlice(weights, new[] { 0, 0 }, new[] { _numGroups, _peDim });
        var peTerm = Engine.ReduceSum(Engine.TensorMultiply(peWeights, _positionalEncodings), new[] { 1 }, keepDims: false);

        // Latent code features
        var latentWeights = Engine.TensorSlice(weights, new[] { 0, _peDim }, new[] { _numGroups, latentDim });
        var latentTerm = Engine.TensorMatMul(latentWeights, Tensor<T>.FromVector(latentCode, new[] { latentDim, 1 }));

        // Sigmoid -> modulation
        var gamma = Engine.TensorSigmoid(Engine.TensorAdd(peTerm, Engine.Reshape(latentTerm, new[] { _numGroups })));
        return Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(Engine.TensorSubtractScalar(gamma, NumOps.FromDouble(0.5)), NumOps.FromDouble(_algoOptions.ConditioningStrength)),
            NumOps.One).ToVector();
    }

    private double ComputeNeRFLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>(taskBatch.Tasks.Length);
        var initParams = ParamModel.GetParameters();
        foreach (var task in taskBatch.Tasks)
        {
            var ap = AdaptConditioned(task, initParams, clip: false, recondition: false);
            ParamModel.SetParameters(ap);
            losses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }
        ParamModel.SetParameters(initParams);

        // The SPSA callback contract is a double; the batch mean is reduced in T and converted once.
        return NumOps.ToDouble(ComputeMean(losses));
    }
}
