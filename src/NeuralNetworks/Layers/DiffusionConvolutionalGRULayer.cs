using System;
using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements the diffusion-convolutional GRU cell introduced by DCRNN.
/// </summary>
/// <remarks>
/// Each reset, update, and candidate transform is learned over the identity plus powers of
/// the graph random-walk supports. The transition matrices are constants; the two projection
/// layers own all trainable parameters and are shared across graph nodes and time steps.
/// </remarks>
[LayerCategory(LayerCategory.Graph)]
[LayerCategory(LayerCategory.Recurrent)]
[LayerTask(LayerTask.GraphProcessing)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, HasTrainingMode = true, ChangesShape = true,
    Cost = ComputeCost.High, TestInputShape = "4, 3, 2", TestConstructorArgs = "2, 4, 4, 2")]
[TensorLayout(TensorAxis.Other, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input,
    Note = "Graph nodes, sequence steps, and features; weights are shared across nodes.")]
[TensorLayout(TensorAxis.Other, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output,
    Note = "Graph nodes, sequence steps, and hidden features.")]
internal sealed class DiffusionConvolutionalGRULayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inputSize;
    private readonly int _hiddenSize;
    private readonly int _numNodes;
    private readonly int _maxDiffusionStep;
    private readonly bool _useBackwardSupport;
    private readonly DenseLayer<T> _gateProjection;
    private readonly DenseLayer<T> _candidateProjection;
    private readonly List<Tensor<T>> _supports = new();
    private Tensor<T>? _ones;

    /// <summary>The final hidden state produced by the most recent sequence forward.</summary>
    internal Tensor<T>? LastState { get; private set; }

    internal int HiddenSize => _hiddenSize;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 3) return null;

        return
        [
            new OutputAxisContract(TensorAxis.Other, AxisRelation.Same(TensorAxis.Other)),
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_hiddenSize)),
        ];
    }

    /// <summary>
    /// Creates a DCGRU cell. Null transition matrices select identity random walks, which is the
    /// graph-neutral fallback used when a caller has not supplied an adjacency matrix.
    /// </summary>
    internal DiffusionConvolutionalGRULayer(
        int inputSize,
        int hiddenSize,
        int numNodes,
        int maxDiffusionStep = 2,
        double[,]? forwardTransition = null,
        double[,]? backwardTransition = null,
        bool useBackwardSupport = true)
        : base(
            [-1, -1, inputSize],
            [-1, -1, hiddenSize],
            (IActivationFunction<T>)new TanhActivation<T>())
    {
        if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (numNodes <= 0) throw new ArgumentOutOfRangeException(nameof(numNodes));
        if (maxDiffusionStep < 0) throw new ArgumentOutOfRangeException(nameof(maxDiffusionStep));

        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _numNodes = numNodes;
        _maxDiffusionStep = maxDiffusionStep;
        _useBackwardSupport = useBackwardSupport;

        _gateProjection = new DenseLayer<T>(hiddenSize * 2);
        _candidateProjection = new DenseLayer<T>(hiddenSize);
        RegisterSubLayer(_gateProjection);
        RegisterSubLayer(_candidateProjection);

        SetDiffusionMatrices(
            forwardTransition ?? CreateIdentity(numNodes),
            backwardTransition ?? CreateIdentity(numNodes));
    }

    /// <summary>Rebuilds the constant random-walk support powers after graph restoration.</summary>
    internal void SetDiffusionMatrices(double[,] forwardTransition, double[,] backwardTransition)
    {
        ValidateTransition(forwardTransition, nameof(forwardTransition));
        ValidateTransition(backwardTransition, nameof(backwardTransition));

        _supports.Clear();
        AddSupportPowers(forwardTransition);
        if (_useBackwardSupport)
            AddSupportPowers(backwardTransition);
    }

    protected override void OnFirstForward(Tensor<T> input)
    {
        ValidateSequence(input);
        ResolveShapes([_numNodes, input.Shape[1], _inputSize], [_numNodes, input.Shape[1], _hiddenSize]);
    }

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        ValidateSequence(input);

        int sequenceLength = input.Shape[1];
        var state = Tensor<T>.CreateDefault([_numNodes, _hiddenSize], NumOps.Zero);
        var states = new List<Tensor<T>>(sequenceLength);

        for (int step = 0; step < sequenceLength; step++)
        {
            var stepInput = Engine.Reshape(
                Engine.TensorNarrow(input, 1, step, 1),
                [_numNodes, _inputSize]);
            state = ForwardStep(stepInput, state);
            states.Add(state);
        }

        LastState = state;
        return Engine.Reshape(
            Engine.TensorConcatenate(states.ToArray(), axis: 1),
            [_numNodes, sequenceLength, _hiddenSize]);
    }

    /// <summary>Runs one paper DCGRU recurrence from an explicit hidden state.</summary>
    internal Tensor<T> ForwardStep(Tensor<T> input, Tensor<T> state)
    {
        if (input.Rank != 2 || input.Shape[0] != _numNodes || input.Shape[1] != _inputSize)
            throw new ArgumentException(
                $"DCGRU step input must be [{_numNodes}, {_inputSize}].", nameof(input));
        if (state.Rank != 2 || state.Shape[0] != _numNodes || state.Shape[1] != _hiddenSize)
            throw new ArgumentException(
                $"DCGRU state must be [{_numNodes}, {_hiddenSize}].", nameof(state));

        // Official DCRNN cell: r,u = sigmoid(W_G *_G [x,h] + 1).
        var inputAndState = Engine.TensorConcatenate([input, state], axis: 1);
        var gateLogits = _gateProjection.Forward(BuildDiffusionBasis(inputAndState));
        var gateBias = Tensor<T>.CreateDefault(gateLogits._shape, NumOps.One);
        var gates = Engine.Sigmoid(Engine.TensorAdd(gateLogits, gateBias));
        var reset = Engine.TensorNarrow(gates, 1, 0, _hiddenSize);
        var update = Engine.TensorNarrow(gates, 1, _hiddenSize, _hiddenSize);

        // Candidate: c = tanh(W_C *_G [x, r .* h]).
        var resetState = Engine.TensorMultiply(reset, state);
        var candidateInput = Engine.TensorConcatenate([input, resetState], axis: 1);
        var candidate = Engine.Tanh(_candidateProjection.Forward(BuildDiffusionBasis(candidateInput)));

        if (_ones is null || !_ones._shape.AsSpan().SequenceEqual(state._shape))
            _ones = Tensor<T>.CreateDefault(state._shape, NumOps.One);

        // H = u .* H_prev + (1 - u) .* C (Li et al., Eq. 3).
        var oneMinusUpdate = Engine.TensorSubtract(_ones, update);
        return Engine.TensorAdd(
            Engine.TensorMultiply(update, state),
            Engine.TensorMultiply(oneMinusUpdate, candidate));
    }

    private Tensor<T> BuildDiffusionBasis(Tensor<T> features)
    {
        var basis = new List<Tensor<T>>(1 + _supports.Count) { features };
        foreach (var support in _supports)
            basis.Add(Engine.TensorMatMul(support, features));
        return Engine.TensorConcatenate(basis.ToArray(), axis: 1);
    }

    private void AddSupportPowers(double[,] transition)
    {
        var power = (double[,])transition.Clone();
        for (int step = 1; step <= _maxDiffusionStep; step++)
        {
            _supports.Add(ToTensor(power));
            power = Multiply(power, transition);
        }
    }

    private Tensor<T> ToTensor(double[,] matrix)
    {
        var tensor = new Tensor<T>([_numNodes, _numNodes]);
        for (int row = 0; row < _numNodes; row++)
            for (int column = 0; column < _numNodes; column++)
                tensor[row, column] = NumOps.FromDouble(matrix[row, column]);
        return tensor;
    }

    private void ValidateSequence(Tensor<T> input)
    {
        if (input.Rank != 3
            || input.Shape[0] != _numNodes
            || input.Shape[2] != _inputSize)
        {
            throw new ArgumentException(
                $"DCGRU sequence input must be [{_numNodes}, time, {_inputSize}].", nameof(input));
        }
    }

    private void ValidateTransition(double[,] transition, string parameterName)
    {
        if (transition.GetLength(0) != _numNodes || transition.GetLength(1) != _numNodes)
            throw new ArgumentException(
                $"Transition matrix must be [{_numNodes}, {_numNodes}].", parameterName);
    }

    private static double[,] CreateIdentity(int size)
    {
        var identity = new double[size, size];
        for (int i = 0; i < size; i++) identity[i, i] = 1.0;
        return identity;
    }

    private static double[,] Multiply(double[,] left, double[,] right)
    {
        int size = left.GetLength(0);
        var result = new double[size, size];
        for (int row = 0; row < size; row++)
            for (int column = 0; column < size; column++)
                for (int k = 0; k < size; k++)
                    result[row, column] += left[row, k] * right[k, column];
        return result;
    }

    public override void UpdateParameters(T learningRate)
    {
        _gateProjection.UpdateParameters(learningRate);
        _candidateProjection.UpdateParameters(learningRate);
    }

    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        _gateProjection.SetTrainingMode(isTraining);
        _candidateProjection.SetTrainingMode(isTraining);
    }

    public override void ResetState()
    {
        LastState = null;
        _ones = null;
        _gateProjection.ResetState();
        _candidateProjection.ResetState();
    }
}
