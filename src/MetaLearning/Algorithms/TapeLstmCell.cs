using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// An LSTM cell (Hochreiter &amp; Schmidhuber 1997) written in engine tensor ops, so a live gradient tape
/// differentiates it, over weights unpacked from one flat parameter vector.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Gates in the order input, forget, cell, output. With input <c>x</c> and recurrent inputs <c>r_j</c>:
/// <c>z_k = x W_k' + sum_j r_j U_kj' + b_k</c>; <c>i = sigmoid(z_0)</c>, <c>f = sigmoid(z_1)</c>,
/// <c>g = tanh(z_2)</c>, <c>o = sigmoid(z_3)</c>; <c>c' = f * c + i * g</c> and <c>h' = o * tanh(c')</c>.
/// Separate weights per recurrent input are the same map as one weight matrix over their concatenation - the
/// attention LSTM of Matching Networks feeds <c>[h, r]</c>.
/// </para>
/// <para>
/// The flat layout, for each gate in that order: the input weights <c>[hidden, input]</c>, then each recurrent
/// input's weights <c>[hidden, hidden]</c>, then the bias <c>[hidden]</c>. Each block is its own tape leaf, so
/// the gradient of the flat vector is the leaves' gradients in the same order.
/// </para>
/// </remarks>
internal sealed class TapeLstmCell<T>
{
    private const int Gates = 4;
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private readonly Tensor<T>[] _inputWeights = new Tensor<T>[Gates];
    private readonly Tensor<T>[][] _recurrentWeights = new Tensor<T>[Gates][];
    private readonly Tensor<T>[] _biases = new Tensor<T>[Gates];
    private readonly List<Tensor<T>> _leaves = new List<Tensor<T>>();

    /// <summary>Unpacks a cell's weights from <paramref name="packed"/>, starting at <paramref name="offset"/>.</summary>
    /// <exception cref="ArgumentException">The vector is too short for a cell of this size.</exception>
    internal TapeLstmCell(Vector<T> packed, int offset, int input, int hidden, int recurrentInputs)
    {
        if (packed is null) throw new ArgumentNullException(nameof(packed));
        int count = ParameterCount(input, hidden, recurrentInputs);
        if (offset < 0 || offset + count > packed.Length)
        {
            throw new ArgumentException(
                $"An LSTM cell with input {input}, hidden {hidden} and {recurrentInputs} recurrent inputs needs "
                + $"{count} weights from offset {offset}, but the vector holds {packed.Length}.", nameof(packed));
        }

        Input = input;
        Hidden = hidden;
        RecurrentInputs = recurrentInputs;
        int position = offset;
        for (int gate = 0; gate < Gates; gate++)
        {
            _inputWeights[gate] = Take(packed, ref position, hidden, input);
            _recurrentWeights[gate] = new Tensor<T>[recurrentInputs];
            for (int j = 0; j < recurrentInputs; j++)
            {
                _recurrentWeights[gate][j] = Take(packed, ref position, hidden, hidden);
            }

            _biases[gate] = Take(packed, ref position, hidden);
        }
    }

    /// <summary>Width of the input.</summary>
    internal int Input { get; }

    /// <summary>Width of the output and the cell.</summary>
    internal int Hidden { get; }

    /// <summary>Number of recurrent inputs, each <see cref="Hidden"/> wide.</summary>
    internal int RecurrentInputs { get; }

    /// <summary>The weight blocks, in the flat layout's order: the tensors a tape differentiates against.</summary>
    internal IReadOnlyList<Tensor<T>> Leaves => _leaves;

    /// <summary>Number of weights one cell reads from the flat vector.</summary>
    internal static int ParameterCount(int input, int hidden, int recurrentInputs)
        => Gates * (hidden * input + recurrentInputs * hidden * hidden + hidden);

    /// <summary>
    /// Fills <paramref name="count"/> weights with U(-1/sqrt(hidden), 1/sqrt(hidden)), the default initialisation of
    /// PyTorch's <c>nn.LSTM</c>.
    /// </summary>
    internal static void InitializeUniform(Vector<T> packed, int offset, int count, int hidden, Random random)
    {
        double bound = 1.0 / Math.Sqrt(hidden);
        for (int i = 0; i < count; i++)
        {
            packed[offset + i] = Ops.FromDouble((2.0 * random.NextDouble() - 1.0) * bound);
        }
    }

    /// <summary>One step: the new output and cell from the input, the recurrent inputs and the previous cell.</summary>
    /// <param name="input"><c>[rows, Input]</c>.</param>
    /// <param name="recurrent"><see cref="RecurrentInputs"/> tensors, each <c>[rows, Hidden]</c>.</param>
    /// <param name="cell">The previous cell, <c>[rows, Hidden]</c>.</param>
    internal (Tensor<T> Output, Tensor<T> Cell) Step(Tensor<T> input, IReadOnlyList<Tensor<T>> recurrent, Tensor<T> cell)
    {
        if (recurrent.Count != RecurrentInputs)
        {
            throw new ArgumentException(
                $"This cell takes {RecurrentInputs} recurrent inputs, not {recurrent.Count}.", nameof(recurrent));
        }

        var engine = AiDotNetEngine.Current;
        Tensor<T> Gate(int gate)
        {
            var z = engine.TensorMatMul(input, engine.TensorTranspose(_inputWeights[gate]));
            for (int j = 0; j < RecurrentInputs; j++)
            {
                z = engine.TensorAdd(z, engine.TensorMatMul(recurrent[j], engine.TensorTranspose(_recurrentWeights[gate][j])));
            }

            return engine.TensorAdd(z, engine.Reshape(_biases[gate], new[] { 1, Hidden }));
        }

        var inputGate = engine.Sigmoid(Gate(0));
        var forgetGate = engine.Sigmoid(Gate(1));
        var candidate = engine.Tanh(Gate(2));
        var outputGate = engine.Sigmoid(Gate(3));
        var nextCell = engine.TensorAdd(engine.TensorMultiply(forgetGate, cell), engine.TensorMultiply(inputGate, candidate));
        var output = engine.TensorMultiply(outputGate, engine.Tanh(nextCell));
        return (output, nextCell);
    }

    /// <summary>
    /// Writes the leaves' gradients into <paramref name="into"/> at <paramref name="offset"/>, in the flat layout;
    /// a leaf the tape never reached contributes zeros.
    /// </summary>
    internal void CopyGradients(Dictionary<Tensor<T>, Tensor<T>> gradients, Vector<T> into, int offset)
    {
        int position = offset;
        foreach (var leaf in _leaves)
        {
            if (gradients.TryGetValue(leaf, out var gradient))
            {
                for (int i = 0; i < leaf.Length; i++) into[position + i] = gradient[i];
            }

            position += leaf.Length;
        }
    }

    private Tensor<T> Take(Vector<T> packed, ref int position, params int[] shape)
    {
        var block = new Tensor<T>(shape);
        for (int i = 0; i < block.Length; i++) block[i] = packed[position + i];
        position += block.Length;
        _leaves.Add(block);
        return block;
    }
}
