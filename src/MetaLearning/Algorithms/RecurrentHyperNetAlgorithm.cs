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
/// Implementation of Recurrent HyperNetwork for meta-learning.
/// </summary>
/// <remarks>
/// <para>
/// A GRU-like recurrent cell processes compressed gradient information at each adaptation
/// step, maintaining hidden state that captures the optimization trajectory. The recurrent
/// output is used to compute per-parameter learning rate modulation factors, enabling
/// adaptive step sizes that evolve through the inner loop based on gradient history.
/// </para>
/// <para><b>Algorithm:</b>
/// <code>
/// h_0 = 0  (zero-initialized hidden state)
///
/// For each inner step t:
///   x_t = compress(grad_t)                               →  [InputDim]
///   z_t = sigmoid(W_z · [h_{t-1}, x_t] + ForgetBias)    (update gate)
///   r_t = sigmoid(W_r · [h_{t-1}, x_t])                  (reset gate)
///   h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t])             (candidate)
///   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t             (new state)
///
///   Modulation: m_d = 1 + tanh(h_t[d % HiddenDim])
///   θ_d ← θ_d - η * m_d * grad_d
///
/// L_meta = L_query + CellRegWeight * Σ h_T²
/// Outer: update θ via meta-gradient, update GRU weights via SPSA
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("HyperNetworks",
    "https://arxiv.org/abs/1609.09106",
    Year = 2016,
    Authors = "David Ha, Andrew Dai, Quoc V. Le")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class RecurrentHyperNetAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly RecurrentHyperNetOptions<T, TInput, TOutput> _algoOptions;
    private readonly int _paramDim;

    private const double SpsaLearningRateMultiplier = 0.1;

    /// <summary>GRU weights: W_z, W_r, W_h — each (hidDim + inputDim) × hidDim.</summary>
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _gruWeights;

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.RecurrentHyperNet;

    public RecurrentHyperNetAlgorithm(RecurrentHyperNetOptions<T, TInput, TOutput> options)
        : base((options ?? throw new ArgumentNullException(nameof(options))).MetaModel,
               options.LossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.MultiClassClassification),
               options, options.DataLoader, options.MetaOptimizer, options.InnerOptimizer,
               optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        if (options.HiddenStateDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "HiddenStateDim must be positive.");
        if (options.InputDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "InputDim must be positive.");

        _algoOptions = options;
        _paramDim = InterfaceGuard.Parameterizable(options.MetaModel).GetParameters().Length;

        int hidDim = options.HiddenStateDim;
        int inDim = options.InputDim;
        int concatDim = hidDim + inDim;
        // 3 gates (z, r, h̃), each concatDim × hidDim
        int totalWeights = 3 * concatDim * hidDim;
        // U(-0.5, 0.5) / sqrt(concatDim): one NextDouble per weight, in order; the loop only stores the draws.
        var draws = new Vector<T>(totalWeights);
        for (int i = 0; i < totalWeights; i++)
            draws[i] = NumOps.FromDouble(RandomGenerator.NextDouble());
        _gruWeights = Engine.Multiply(
            Engine.Subtract(draws, Engine.Fill(totalWeights, NumOps.FromDouble(0.5))),
            NumOps.FromDouble(1.0 / Math.Sqrt(concatDim)));
    }

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null) throw new ArgumentNullException(nameof(taskBatch));
        if (taskBatch.Tasks.Length == 0) return NumOps.Zero;

        var losses = new List<T>();
        var metaGradients = new List<Vector<T>>();
        var initParams = ParamModel.GetParameters();

        foreach (var task in taskBatch.Tasks)
        {
            var (adaptedParams, hidden) = AdaptRecurrent(initParams, task);

            ParamModel.SetParameters(adaptedParams);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(WithCellRegularization(queryLoss, hidden));
            metaGradients.Add(ClipGradients(ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput)));
        }

        ApplyOuterUpdate(initParams, metaGradients, _algoOptions.OuterLearningRate);

        UpdateAuxiliaryParamsSPSA(taskBatch, ref _gruWeights, _algoOptions.OuterLearningRate * SpsaLearningRateMultiplier, ComputeRecurrentLoss);

        return ComputeMean(losses);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        var initParams = ParamModel.GetParameters();
        var (adaptedParams, _) = AdaptRecurrent(initParams, task);
        ParamModel.SetParameters(initParams);
        return new AdaptedMetaModel<T, TInput, TOutput>(MetaModel, adaptedParams);
    }

    /// <summary>
    /// The modulated inner loop: h_t = GRU(h_{t-1}, compress(grad_t)), then theta_d -= eta (1 + tanh(h_t[d % H])) grad_d.
    /// </summary>
    private (Vector<T> Adapted, Vector<T> Hidden) AdaptRecurrent(
        Vector<T> initParams, IMetaLearningTask<T, TInput, TOutput> task)
    {
        int hidDim = _algoOptions.HiddenStateDim;
        var adaptedParams = initParams.Clone();
        var hidden = new Vector<T>(hidDim);
        var lr = NumOps.FromDouble(_algoOptions.InnerLearningRate);

        for (int step = 0; step < _algoOptions.AdaptationSteps; step++)
        {
            ParamModel.SetParameters(adaptedParams);
            var grad = ClipGradients(ComputeGradients(MetaModel, task.SupportInput, task.SupportOutput));
            hidden = GRUStep(hidden, CompressVector(grad, _algoOptions.InputDim));

            // m_d = 1 + tanh(h[d % H]), in [0, 2]: the hidden modulation tiled across the parameters.
            var perUnit = Engine.Add(Engine.Fill(hidDim, NumOps.One), Engine.Tanh(hidden));
            int copies = (_paramDim + hidDim - 1) / hidDim;
            var modulation = Vector<T>.Concatenate(Enumerable.Repeat(perUnit, copies).ToArray())
                .GetSubVector(0, _paramDim);
            adaptedParams = Engine.Subtract(adaptedParams, Engine.Multiply(Engine.Multiply(modulation, grad), lr));
        }

        return (adaptedParams, hidden);
    }

    /// <summary>queryLoss + CellRegWeight * ||h||^2 / H.</summary>
    private T WithCellRegularization(T queryLoss, Vector<T> hidden)
        => NumOps.Add(queryLoss, NumOps.Multiply(
            NumOps.FromDouble(_algoOptions.CellRegWeight / _algoOptions.HiddenStateDim),
            Engine.DotProduct(hidden, hidden)));

    /// <summary>
    /// One GRU step: z = sigmoid(W_z [h, x] + ForgetBias), r = sigmoid(W_r [h, x]),
    /// h~ = tanh(W_h [r * h, x]), h' = (1 - z) h + z h~.
    /// </summary>
    private Vector<T> GRUStep(Vector<T> prevHidden, Vector<T> input)
    {
        int hidDim = _algoOptions.HiddenStateDim;
        int inDim = _algoOptions.InputDim;
        int concatDim = hidDim + inDim;
        int gateSize = concatDim * hidDim;

        // CompressVector returns exactly InputDim values.
        var x = input.Length == inDim ? input : input.GetSubVector(0, Math.Min(inDim, input.Length));
        if (x.Length < inDim) x = Vector<T>.Concatenate(new[] { x, new Vector<T>(inDim - x.Length) });

        var concat = Vector<T>.Concatenate(new[] { prevHidden, x });
        var z = Engine.Sigmoid(Engine.Add(Gate(0, gateSize, concat), Engine.Fill(hidDim, NumOps.FromDouble(_algoOptions.ForgetBias))));
        var r = Engine.Sigmoid(Gate(gateSize, gateSize, concat));
        var resetConcat = Vector<T>.Concatenate(new[] { Engine.Multiply(r, prevHidden), x });
        var candidate = Engine.Tanh(Gate(2 * gateSize, gateSize, resetConcat));

        // h' = h + z (h~ - h).
        return Engine.Add(prevHidden, Engine.Multiply(z, Engine.Subtract(candidate, prevHidden)));
    }

    /// <summary>W · v for the [H, concatDim] row-major gate matrix stored at <paramref name="offset"/>.</summary>
    private Vector<T> Gate(int offset, int size, Vector<T> v)
    {
        int hidDim = _algoOptions.HiddenStateDim;
        var w = Tensor<T>.FromVector(_gruWeights.GetSubVector(offset, size), new[] { hidDim, v.Length });
        return Engine.TensorMatMul(w, Tensor<T>.FromVector(v, new[] { v.Length, 1 })).ToVector();
    }

    private double ComputeRecurrentLoss(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        // The SPSA objective is the training loss, cell regularization included.
        var losses = new List<T>();
        var initParams = ParamModel.GetParameters();
        foreach (var task in taskBatch.Tasks)
        {
            var (ap, hid) = AdaptRecurrent(initParams, task);
            ParamModel.SetParameters(ap);
            var queryLoss = ComputeLossFromOutput(MetaModel.Predict(task.QueryInput), task.QueryOutput);
            losses.Add(WithCellRegularization(queryLoss, hid));
        }
        ParamModel.SetParameters(initParams);
        return NumOps.ToDouble(ComputeMean(losses));
    }
}
