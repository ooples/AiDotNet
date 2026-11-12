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
/// Implementation of Meta-LoRA Bank (2024).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// Meta-LoRA Bank maintains a bank of K diverse LoRA modules, each representing a
/// learned adaptation pattern. For a new task, a gating network computes task-conditioned
/// scores and selects the top-K modules via sparse gating. The selected modules are
/// combined with learned gating weights to produce the task-specific adaptation.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// Meta-learned state: base params θ, bank of LoRA modules {M_1,...,M_K}, gating network G
///
/// For each task:
///   1. Compute task embedding from support features: e = avg(model(x_support))
///   2. Compute gating scores: s_k = G(e) for each module k
///   3. Select top-K modules by score
///   4. Compute gating weights: w = softmax(top-K scores / temperature)
///   5. Combine modules: Δθ = Σ_{k ∈ top-K} w_k * M_k
///   6. Adapted params: θ' = θ + Δθ
///   7. Fine-tune coefficients on support set (inner loop on gating weights only)
///
/// Outer loop: update base params, all LoRA modules, and gating network
/// </code>
/// </para>
/// <para><b>Advantages:</b> Each module specializes in a different type of task adaptation.
/// The gating mechanism enables compositional generalization — novel tasks can be handled
/// by new combinations of existing modules.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("LoRAHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition",
    "https://arxiv.org/abs/2307.13269",
    Year = 2023,
    Authors = "Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, Min Lin")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class MetaLoRABankAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly MetaLoRABankOptions<T, TInput, TOutput> _algoOptions;

    /// <summary>
    /// Bank of LoRA modules. Each module is a low-rank basis vector in parameter space
    /// (length = rank * paramDim). Module k's basis vectors start at k * rank * paramDim.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _moduleBank;

    /// <summary>
    /// Gating network parameters: maps task embeddings to module scores.
    /// Linear: embeddingDim → bankSize (weights + bias).
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _gatingParams;

    private readonly int _paramDim;
    private readonly int _bankSize;
    private readonly int _rank;
    private readonly int _topK;
    private readonly int _embeddingDim;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaLoRABank;

    public MetaLoRABankAlgorithm(MetaLoRABankOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               optimizerUse: MetaLearnerOptimizerUse.InnerAndOuter)
    {
        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;
        _bankSize = Math.Max(1, options.BankSize);
        _rank = Math.Max(1, options.Rank);
        _topK = Math.Max(1, Math.Min(options.TopK, _bankSize));
        _embeddingDim = Math.Min(_paramDim, 64);

        // Initialize module bank: each module has rank basis vectors of length paramDim
        int moduleSize = _rank * _paramDim;
        _moduleBank = new Vector<T>(_bankSize * moduleSize);
        double initScale = 0.01;
        for (int i = 0; i < _moduleBank.Length; i++)
            _moduleBank[i] = NumOps.FromDouble(SampleNormal() * initScale);

        // Initialize gating network: linear(embeddingDim → bankSize)
        int gatingSize = _embeddingDim * _bankSize + _bankSize;
        _gatingParams = new Vector<T>(gatingSize);
        double gateScale = 1.0 / Math.Sqrt(_embeddingDim);
        for (int i = 0; i < gatingSize; i++)
            _gatingParams[i] = NumOps.FromDouble(SampleNormal() * gateScale);
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var baseParams = ParamModel.GetParameters();
        var moduleUsageCounts = new int[_bankSize];

        foreach (var task in taskBatch.Tasks)
        {
            ParamModel.SetParameters(baseParams);

            // Compute task embedding from support features
            var taskEmbed = ComputeTaskEmbedding(task.SupportInput);

            // Compute gating scores and select top-K modules
            var scores = ComputeGatingScores(taskEmbed);
            var (selectedIndices, gatingWeights) = SelectTopK(scores);

            // Track module usage for load balancing
            foreach (int idx in selectedIndices)
                moduleUsageCounts[idx]++;

            // Combine selected modules into task-specific parameter delta
            var paramDelta = CombineModules(selectedIndices, gatingWeights);

            // Inner loop: fine-tune gating weights on support set
            var currentWeights = FineTuneGatingWeights(task, baseParams, selectedIndices, gatingWeights);

            // Evaluate on query set with final adapted params
            var finalDelta = CombineModulesWithWeights(selectedIndices, currentWeights);
            var finalParams = AddVectors(baseParams, finalDelta);
            ParamModel.SetParameters(finalParams);

            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);

            // Add load balancing regularization
            T loadBalanceLoss = ComputeLoadBalanceLoss(moduleUsageCounts, taskBatch.Tasks.Length);
            var totalLoss = NumOps.Add(queryLoss, NumOps.Multiply(NumOps.FromDouble(_algoOptions.LoadBalanceRegularization), loadBalanceLoss));
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

        // Update module bank and gating network via SPSA
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _moduleBank, _algoOptions.OuterLearningRate * 0.1, ComputeBankLoss);
        UpdateAuxiliaryParamsSPSA(taskBatch, ref _gatingParams, _algoOptions.OuterLearningRate, ComputeBankLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var baseParams = ParamModel.GetParameters();
        ParamModel.SetParameters(baseParams);

        var taskEmbed = ComputeTaskEmbedding(task.SupportInput);
        var scores = ComputeGatingScores(taskEmbed);
        var (selectedIndices, gatingWeights) = SelectTopK(scores);

        // Fine-tune gating weights on support set
        var currentWeights = FineTuneGatingWeights(task, baseParams, selectedIndices, gatingWeights);

        var finalDelta = CombineModulesWithWeights(selectedIndices, currentWeights);
        var finalParams = AddVectors(baseParams, finalDelta);
        ParamModel.SetParameters(baseParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, finalParams);
    }

    /// <summary>
    /// Inner loop: steps the selected modules' gating weights along dL/dw_k (the gradient's dot product with
    /// each module's mean basis vector) and re-normalizes them to sum to 1 after every step.
    /// </summary>
    private Vector<T> FineTuneGatingWeights(
        IMetaLearningTask<T, TInput, TOutput> task, Vector<T> baseParams, int[] selectedIndices, Vector<T> gatingWeights)
    {
        var currentWeights = gatingWeights.Clone();
        var moduleMeans = SelectedModuleMeans(selectedIndices);
        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            var delta = CombineModulesWithWeights(selectedIndices, currentWeights);
            var adaptedParams = AddVectors(baseParams, delta);
            ParamModel.SetParameters(adaptedParams);

            var fullGrad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));

            // Compute gradient w.r.t. each gating weight
            var stepGradient = ComputeModuleGradient(fullGrad, moduleMeans);
            currentWeights = ApplyInnerGradients(currentWeights, stepGradient, _algoOptions.InnerLearningRate, step == 0);

            // Re-normalize weights to sum to 1
            currentWeights = NormalizeWeights(currentWeights);
        }

        return currentWeights;
    }

    private Vector<T> ComputeTaskEmbedding(TInput supportInput)
    {
        var features = ConvertToVector(MetaModel.Predict(supportInput));
        if (features == null || features.Length == 0) return new Vector<T>(_embeddingDim);

        // The first embeddingDim features, zero-padded when there are fewer
        return features.Length >= _embeddingDim
            ? features.GetSubVector(0, _embeddingDim)
            : Vector<T>.Concatenate(features.Clone(), new Vector<T>(_embeddingDim - features.Length));
    }

    /// <summary>Gating scores W e + b for the linear gate embeddingDim -> bankSize.</summary>
    private Vector<T> ComputeGatingScores(Vector<T> taskEmbed)
    {
        int weightCount = _embeddingDim * _bankSize;
        var weights = Engine.Reshape(Tensor<T>.FromVector(_gatingParams.GetSubVector(0, weightCount)), new[] { _bankSize, _embeddingDim });
        var scores = Engine.TensorMatMul(weights, Tensor<T>.FromVector(taskEmbed, new[] { _embeddingDim, 1 })).ToVector();
        int biasCount = Math.Min(_bankSize, _gatingParams.Length - weightCount);
        if (biasCount <= 0) return scores;
        var bias = _gatingParams.GetSubVector(weightCount, biasCount);
        return Engine.Add(scores, biasCount == _bankSize ? bias : Vector<T>.Concatenate(bias, new Vector<T>(_bankSize - biasCount)));
    }

    /// <summary>
    /// The top-K modules by score (ties broken by the lower module index) and the temperature softmax of
    /// their scores, shifted by the best score.
    /// </summary>
    private (int[] indices, Vector<T> weights) SelectTopK(Vector<T> scores)
    {
        var engine = Engine;
        var square = new[] { _bankSize, _bankSize };
        var s = Tensor<T>.FromVector(scores);

        // rank_k = #{j : s_j > s_k} + #{j < k : s_j == s_k}: a unique position in descending order
        var others = engine.TensorBroadcastTo(engine.Reshape(s, new[] { 1, _bankSize }), square);
        var own = engine.TensorBroadcastTo(engine.Reshape(s, new[] { _bankSize, 1 }), square);
        var index = engine.TensorArange<T>(0, _bankSize, 1);
        var earlier = engine.TensorLessThan(
            engine.TensorBroadcastTo(engine.Reshape(index, new[] { 1, _bankSize }), square),
            engine.TensorBroadcastTo(engine.Reshape(index, new[] { _bankSize, 1 }), square));
        var ahead = engine.TensorAdd(engine.TensorGreaterThan(others, own),
            engine.TensorMultiply(engine.TensorEquals(others, own), earlier));
        var rank = engine.ReduceSum(ahead, new[] { 1 }, keepDims: false);

        // Row p of the selection picks the module ranked p
        var positions = engine.TensorArange<T>(0, _topK, 1);
        var selection = engine.TensorEquals(
            engine.TensorBroadcastTo(engine.Reshape(rank, new[] { 1, _bankSize }), new[] { _topK, _bankSize }),
            engine.TensorBroadcastTo(engine.Reshape(positions, new[] { _topK, 1 }), new[] { _topK, _bankSize }));
        var indices = engine.TensorRound(engine.TensorMatMul(selection, engine.Reshape(index, new[] { _bankSize, 1 })))
            .Cast<int>().ToArray();
        var topScores = engine.TensorMatMul(selection, engine.Reshape(s, new[] { _bankSize, 1 })).ToVector();

        // Softmax over top-K scores
        T maxScore = Engine.TensorMaxValue(Tensor<T>.FromVector(topScores));
        var exps = Engine.Exp(Engine.Divide(Engine.Subtract(topScores, Engine.Fill(_topK, maxScore)),
            NumOps.FromDouble(_algoOptions.GatingTemperature)));
        return (indices, Engine.Divide(exps, Engine.Sum(exps)));
    }

    private Vector<T> CombineModules(int[] selectedIndices, Vector<T> weights)
    {
        return CombineModulesWithWeights(selectedIndices, weights);
    }

    /// <summary>The mean of each selected module's rank basis vectors, [K, paramDim].</summary>
    private Tensor<T> SelectedModuleMeans(int[] selectedIndices)
    {
        var bank = Engine.Reshape(Tensor<T>.FromVector(_moduleBank), new[] { _bankSize, _rank, _paramDim });
        var means = Engine.ReduceMean(bank, new[] { 1 }, keepDims: false);
        return Engine.TensorIndexSelect(means, new Tensor<int>(selectedIndices, new[] { selectedIndices.Length }), 0);
    }

    /// <summary>
    /// delta = sum_k w_k * mean_r(B[idx_k, r]), skipping weights below 1e-10 in magnitude.
    /// </summary>
    private Vector<T> CombineModulesWithWeights(int[] selectedIndices, Vector<T> weights)
    {
        var w = Tensor<T>.FromVector(weights, new[] { 1, selectedIndices.Length });
        var negligible = Engine.TensorLessThan(Engine.TensorAbs(w), NumOps.FromDouble(1e-10));
        var kept = Engine.TensorMultiply(w, Engine.ScalarMinusTensor(NumOps.One, negligible));
        return Engine.TensorMatMul(kept, SelectedModuleMeans(selectedIndices)).ToVector();
    }

    /// <summary>
    /// dL/dw_k: the dot product of the full gradient with each selected module's mean basis vector.
    /// </summary>
    private Vector<T> ComputeModuleGradient(Vector<T> fullGrad, Tensor<T> moduleMeans)
    {
        var g = fullGrad.Length == _paramDim ? fullGrad : fullGrad.GetSubVector(0, _paramDim);
        return Engine.TensorMatMul(moduleMeans, Tensor<T>.FromVector(g, new[] { _paramDim, 1 })).ToVector();
    }

    private Vector<T> AddVectors(Vector<T> a, Vector<T> b)
    {
        return Engine.Add(a.Length == _paramDim ? a : a.GetSubVector(0, _paramDim),
            b.Length == _paramDim ? b : b.GetSubVector(0, _paramDim));
    }

    /// <summary>Clamps the weights at zero and rescales them to sum to 1 (uniform when they sum to ~0).</summary>
    private Vector<T> NormalizeWeights(Vector<T> weights)
    {
        var nonNegative = Engine.Max(weights, Engine.Fill(weights.Length, NumOps.Zero)); // Ensure non-negative
        T sum = Engine.Sum(nonNegative);
        return NumOps.GreaterThan(sum, NumOps.FromDouble(1e-10))
            ? Engine.Divide(nonNegative, sum)
            : Engine.Fill(weights.Length, NumOps.FromDouble(1.0 / weights.Length));
    }

    /// <summary>Population variance of the per-module usage fractions count_m / numTasks.</summary>
    private T ComputeLoadBalanceLoss(int[] usageCounts, int numTasks)
    {
        if (numTasks == 0) return NumOps.Zero;
        var fractions = Engine.Divide(new Tensor<int>(usageCounts, new[] { usageCounts.Length }).Cast<T>().ToVector(),
            NumOps.FromDouble(numTasks));
        var centered = Engine.Subtract(fractions, Engine.Fill(fractions.Length, Engine.Mean(fractions)));
        return NumOps.Divide(Engine.DotProduct(centered, centered), NumOps.FromDouble(usageCounts.Length));
    }

    private double ComputeBankLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        var baseParams = ParamModel.GetParameters();
        var losses = new List<T>(taskBatch.Tasks.Length);

        foreach (var task in taskBatch.Tasks)
        {
            ParamModel.SetParameters(baseParams);
            var taskEmbed = ComputeTaskEmbedding(task.SupportInput);
            var scores = ComputeGatingScores(taskEmbed);
            var (selectedIndices, gatingWeights) = SelectTopK(scores);
            var delta = CombineModulesWithWeights(selectedIndices, gatingWeights);
            var adaptedParams = AddVectors(baseParams, delta);
            ParamModel.SetParameters(adaptedParams);
            losses.Add(ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput));
        }

        ParamModel.SetParameters(baseParams);

        // The SPSA callback contract is a double; the batch mean is reduced in T and converted once.
        return NumOps.ToDouble(ComputeMean(losses));
    }
}
