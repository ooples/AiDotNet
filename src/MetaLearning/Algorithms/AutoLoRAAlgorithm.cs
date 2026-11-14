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
/// Implementation of AutoLoRA: Automatically Tuning Matrix Ranks in Low-Rank Adaptation
/// Based on Meta Learning (Zhang et al., NAACL 2024).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// AutoLoRA addresses LoRA's limitation of uniform rank assignment across all layers by
/// using a bi-level optimization to automatically discover the optimal rank for each
/// parameter group.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// State: base params θ, rank-1 component bank {Δ_g,j}, selection logits {β_g,j}
///
/// Inner loop (weight update on support/train data):
///   For each group g, compute selection weights: α_g,j = softmax(β_g / temperature)
///   Δθ_g = Σ_j α_g,j * Δ_g,j  (weighted sum of rank-1 components)
///   θ' = θ + Δθ
///   Update {Δ_g,j} via gradient descent on support loss L(θ')
///
/// Outer loop (rank selection on query/val data):
///   Update {β_g,j} via gradient descent on query loss L(θ')
///   Add rank regularization: L_reg = Σ_g Σ_j α_g,j  (encourages sparsity)
///
/// Rank determination (after meta-training):
///   For group g: effective_rank = |{j : α_g,j ≥ threshold}|
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("AutoLoRA: Automatically Tuning Matrix Ranks in Low-Rank Adaptation Based on Meta Learning",
    "https://arxiv.org/abs/2403.09113",
    Year = 2024,
    Authors = "Ruiyi Zhang, Rushi Shah, Haiwen Hong, et al.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class AutoLoRAAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly AutoLoRAOptions<T, TInput, TOutput> _algoOptions;

    /// <summary>
    /// Rank-1 components for all groups. Stored as flat vector:
    /// [group_0_comp_0, group_0_comp_1, ..., group_G_comp_R]
    /// Each component has length paramsPerGroup.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _rankComponents;

    /// <summary>
    /// Selection logits β for each (group, component) pair.
    /// Length = numGroups * maxRank.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _selectionLogits;

    private readonly int _paramDim;
    private readonly int _numGroups;
    private readonly int _maxRank;
    private readonly int _paramsPerGroup;

    /// <summary>SPSA learning rate multiplier for auxiliary parameter updates.</summary>
    private const double SpsaLearningRateMultiplier = 0.1;

    /// <summary>
    /// A selection weight (or a group's thresholded weight sum) at or below this value counts as inactive.
    /// </summary>
    private const double ActiveWeightEpsilon = 1e-10;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.AutoLoRA;

    public AutoLoRAAlgorithm(AutoLoRAOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        if (options.NumRankGroups <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "NumRankGroups must be positive.");
        if (options.MaxRank <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "MaxRank must be positive.");

        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        _numGroups = Math.Max(1, options.NumRankGroups);
        _maxRank = Math.Max(1, options.MaxRank);
        _paramsPerGroup = (_paramDim + _numGroups - 1) / _numGroups;

        // Initialize rank-1 components (small random values)
        int totalComponents = _numGroups * _maxRank * _paramsPerGroup;
        _rankComponents = new Vector<T>(totalComponents);
        double initScale = 0.01;
        for (int i = 0; i < totalComponents; i++)
            _rankComponents[i] = NumOps.FromDouble(SampleNormal() * initScale);

        // Initialize selection logits to uniform (all components equally likely)
        _selectionLogits = new Vector<T>(_numGroups * _maxRank);
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null) throw new ArgumentNullException(nameof(taskBatch));
        if (taskBatch.Tasks.Length == 0) return NumOps.Zero;

        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var baseParams = ParamModel.GetParameters();

        // Compute current selection weights via softmax
        var selectionWeights = ComputeSelectionWeights();
        T rankRegularization = NumOps.FromDouble(_algoOptions.RankRegularization);

        foreach (var task in taskBatch.Tasks)
        {
            // Inner loop: update rank components on support data
            // Compose adapted params using current selection weights
            var adaptedParams = ComposeAdaptedParams(baseParams, selectionWeights);
            ParamModel.SetParameters(adaptedParams);

            for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
            {
                var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

                // Update rank components (inner loop) by projecting gradient onto each component
                UpdateRankComponents(grad, selectionWeights);

                // Recompose with updated components
                adaptedParams = ComposeAdaptedParams(baseParams, selectionWeights);
                ParamModel.SetParameters(adaptedParams);
            }

            // Evaluate on query set
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Add rank regularization: penalize active rank (encourages sparsity)
            T rankReg = ComputeRankRegularization(selectionWeights);
            var totalLoss = NumOps.Add(queryLoss, NumOps.Multiply(rankRegularization, rankReg));
            losses.Add(totalLoss);

            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        // Outer loop: update base params
        ParamModel.SetParameters(baseParams);
        if (metaGradients.Count > 0)
        {
            var avgGrad = AverageVectors(metaGradients);
            ParamModel.SetParameters(ApplyOuterGradients(baseParams, avgGrad, _algoOptions.OuterLearningRate));
        }

        // Update selection logits and rank components via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _selectionLogits, _algoOptions.OuterLearningRate, ComputeAutoLoRALoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _rankComponents, _algoOptions.OuterLearningRate * SpsaLearningRateMultiplier, ComputeAutoLoRALoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var baseParams = ParamModel.GetParameters();
        var selectionWeights = ComputeSelectionWeights();

        // Apply thresholding: only use components with α ≥ threshold (rank determination)
        var thresholdedWeights = ApplyThreshold(selectionWeights);

        var adaptedParams = ComposeAdaptedParams(baseParams, thresholdedWeights);
        ParamModel.SetParameters(adaptedParams);

        // Fine-tune rank components on support set
        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            UpdateRankComponents(grad, thresholdedWeights);
            adaptedParams = ComposeAdaptedParams(baseParams, thresholdedWeights);
            ParamModel.SetParameters(adaptedParams);
        }

        ParamModel.SetParameters(baseParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// Computes softmax selection weights for each (group, rank) pair.
    /// Returns a [group, rank] tensor.
    /// </summary>
    private Tensor<T> ComputeSelectionWeights()
        => Engine.Softmax(Engine.Reshape(Tensor<T>.FromVector(_selectionLogits), new[] { _numGroups, _maxRank }), 1);

    /// <summary>
    /// The selection weights with every weight below <see cref="ActiveWeightEpsilon"/> set to zero: those
    /// components are skipped.
    /// </summary>
    private Tensor<T> ActiveWeights(Tensor<T> selectionWeights)
        => Engine.TensorMultiply(selectionWeights, Engine.TensorSubtract(
            Engine.TensorOnesLike(selectionWeights),
            Engine.TensorLessThan(selectionWeights, NumOps.FromDouble(ActiveWeightEpsilon))));

    /// <summary>The rank components as a [group, rank, paramsPerGroup] tensor.</summary>
    private Tensor<T> ComponentTensor()
        => Engine.Reshape(Tensor<T>.FromVector(_rankComponents), new[] { _numGroups, _maxRank, _paramsPerGroup });

    /// <summary>A parameter-space vector zero-padded to numGroups * paramsPerGroup and shaped [group, 1, paramsPerGroup].</summary>
    private Tensor<T> GroupedParameters(Vector<T> values)
    {
        int padded = _numGroups * _paramsPerGroup;
        var full = values.Length == padded ? values : Vector<T>.Concatenate(values, new Vector<T>(padded - values.Length));
        return Engine.Reshape(Tensor<T>.FromVector(full), new[] { _numGroups, 1, _paramsPerGroup });
    }

    /// <summary>
    /// Composes adapted parameters by adding weighted rank-1 components to base params.
    /// </summary>
    private Vector<T> ComposeAdaptedParams(Vector<T> baseParams, Tensor<T> selectionWeights)
    {
        // delta[g, p] = sum_j w[g, j] C[g, j, p]; group g covers parameters [g P, g P + P), so the flattened
        // delta lines up with the parameter vector and its padded tail is dropped.
        var shape = new[] { _numGroups, _maxRank, _paramsPerGroup };
        var weights = Engine.TensorBroadcastTo(
            Engine.Reshape(ActiveWeights(selectionWeights), new[] { _numGroups, _maxRank, 1 }), shape);
        var delta = Engine.ReduceSum(Engine.TensorMultiply(weights, ComponentTensor()), new[] { 1 }, false).ToVector();
        return Engine.Add(baseParams.GetSubVector(0, _paramDim), delta.GetSubVector(0, _paramDim));
    }

    /// <summary>
    /// Updates rank-1 components using projected gradients from the full parameter gradient.
    /// </summary>
    private void UpdateRankComponents(Vector<T> fullGrad, Tensor<T> selectionWeights)
    {
        // C[g, j, p] -= lr * w[g, j] * grad[g P + p]; the zero-padded tail of the gradient leaves its slots unchanged.
        var shape = new[] { _numGroups, _maxRank, _paramsPerGroup };
        var weights = Engine.TensorBroadcastTo(
            Engine.Reshape(ActiveWeights(selectionWeights), new[] { _numGroups, _maxRank, 1 }), shape);
        var gradient = Engine.TensorBroadcastTo(GroupedParameters(fullGrad.GetSubVector(0, _paramDim)), shape);
        var step = Engine.TensorMultiplyScalar(
            Engine.TensorMultiply(weights, gradient), NumOps.FromDouble(_algoOptions.InnerLearningRate));
        _rankComponents = Engine.TensorSubtract(ComponentTensor(), step).ToVector();
    }

    /// <summary>
    /// Applies threshold to selection weights, zeroing out components below threshold.
    /// </summary>
    private Tensor<T> ApplyThreshold(Tensor<T> selectionWeights)
    {
        var shape = new[] { _numGroups, _maxRank };
        var kept = Engine.TensorSubtract(Engine.TensorOnesLike(selectionWeights),
            Engine.TensorLessThan(selectionWeights, NumOps.FromDouble(_algoOptions.RankThreshold)));
        var thresholded = Engine.TensorMultiply(selectionWeights, kept);

        // Re-normalize each group; a group with nothing left falls back to uniform weights.
        var sums = Engine.TensorBroadcastTo(Engine.ReduceSum(thresholded, new[] { 1 }, true), shape);
        var valid = Engine.TensorGreaterThan(sums, NumOps.FromDouble(ActiveWeightEpsilon));
        var normalized = Engine.TensorDivide(thresholded, Engine.TensorWhere(valid, sums, Engine.TensorOnesLike(sums)));
        var uniform = Engine.TensorFullLike(selectionWeights, NumOps.FromDouble(1.0 / _maxRank));
        return Engine.TensorWhere(valid, normalized, uniform);
    }

    // L1 on selection weights
    private T ComputeRankRegularization(Tensor<T> selectionWeights) => Engine.TensorSum(selectionWeights);

    private double ComputeAutoLoRALoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var baseParams = ParamModel.GetParameters();
        var selectionWeights = ComputeSelectionWeights();
        double totalLoss = 0;

        foreach (var task in taskBatch.Tasks)
        {
            var adaptedParams = ComposeAdaptedParams(baseParams, selectionWeights);
            ParamModel.SetParameters(adaptedParams);
            totalLoss += NumOps.ToDouble(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        totalLoss += _algoOptions.RankRegularization * NumOps.ToDouble(ComputeRankRegularization(selectionWeights));
        ParamModel.SetParameters(baseParams);
        return totalLoss / Math.Max(taskBatch.Tasks.Length, 1);
    }
}
