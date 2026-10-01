// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Extensions;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Piecewise linear encoding (PLE) of numerical features, from Gorishniy, Rubachev and Babenko, "On Embeddings for
/// Numerical Features in Tabular Deep Learning" (NeurIPS 2022).
/// </summary>
/// <remarks>
/// <para>
/// Each feature x is split into T bins by edges b_0 &lt; b_1 &lt; ... &lt; b_T, and becomes the T values
/// e_t = clamp((x - b_{t-1}) / (b_t - b_{t-1})): 0 below the bin, 1 above it, linear inside. The first bin is not
/// clamped below and the last is not clamped above, exactly as the paper defines it, so a value outside the edges
/// still moves the encoding. The output is [batch, features * T].
/// </para>
/// <para>
/// The paper builds the edges from quantiles of the training data and keeps them fixed; call
/// <see cref="FitBoundaries"/> with the training features to do the same. Until then the edges are evenly spaced
/// over [-2, 2], which suits standardized inputs. The edges are persisted state, not trainable parameters.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this like creating "bins" for each number. A value fully passes every bin below
/// it (1), sits partway through its own bin, and has not reached the bins above it (0), so nearby values get nearby
/// encodings while the model can still treat different ranges differently.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
// Rank 2 only: the forward consumes [batch, features].
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class PiecewiseLinearEncodingLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _numFeatures;
    private readonly int _numBins;

    /// <summary>The bin edges, [numFeatures, numBins + 1], strictly increasing per feature.</summary>
    [Buffer(Name = "bin_edges", Role = PersistentTensorRole.Constant)]
    private Tensor<T> _binEdges;

    /// <summary>
    /// Gets the output dimension (numFeatures * numBins).
    /// </summary>
    public int OutputDimension => _numFeatures * _numBins;

    /// <summary>The number of bins each feature is split into.</summary>
    public int NumBins => _numBins;

    /// <inheritdoc />
    /// <remarks>
    /// The feature axis is replaced, not carried: every scalar feature widens into <c>_numBins</c> values, and the
    /// layer uses its own <c>_numFeatures</c>, so the width is <see cref="OutputDimension"/> whatever arrives.
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 2 || _numFeatures <= 0 || _numBins <= 0) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(OutputDimension)),
        };
    }

    /// <inheritdoc/>
    /// <remarks>The encoding has no trainable parameters; gradients pass through it to its input.</remarks>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Initializes piecewise linear encoding with evenly spaced edges over [-2, 2].
    /// </summary>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="numBins">Number of bins per feature (T in the paper).</param>
    public PiecewiseLinearEncodingLayer(int numFeatures, int numBins = 16)
        : base([numFeatures], [numFeatures * numBins])
    {
        if (numFeatures < 1)
            throw new ArgumentException("Must have at least 1 feature", nameof(numFeatures));
        if (numBins < 2)
            throw new ArgumentException("Must have at least 2 bins", nameof(numBins));

        _numFeatures = numFeatures;
        _numBins = numBins;
        _binEdges = new Tensor<T>([numFeatures, numBins + 1]);
        for (int f = 0; f < numFeatures; f++)
        {
            for (int t = 0; t <= numBins; t++)
            {
                _binEdges[f * (numBins + 1) + t] = NumOps.FromDouble(-2.0 + 4.0 * t / numBins);
            }
        }
    }

    /// <summary>
    /// Sets each feature's edges to the quantiles of <paramref name="features"/> at 0, 1/T, ..., 1, as the paper does.
    /// </summary>
    /// <param name="features">Training features, [samples, numFeatures].</param>
    /// <remarks>
    /// Repeated quantiles (a feature with few distinct values) would make a bin of zero width; such an edge is moved
    /// just past the previous one so every bin keeps a positive width.
    /// </remarks>
    public void FitBoundaries(Tensor<T> features)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));
        if (features.Rank != 2 || features.Shape[1] != _numFeatures)
            throw new ArgumentException($"Expected [samples, {_numFeatures}] features.", nameof(features));
        int samples = features.Shape[0];
        if (samples < 1) throw new ArgumentException("Need at least one sample.", nameof(features));

        var column = new double[samples];
        for (int f = 0; f < _numFeatures; f++)
        {
            for (int s = 0; s < samples; s++)
            {
                double value = NumOps.ToDouble(features[s * _numFeatures + f]);
                if (double.IsNaN(value) || double.IsInfinity(value))
                    throw new ArgumentException($"Feature {f} has a non-finite value at sample {s}.", nameof(features));
                column[s] = value;
            }
            Array.Sort(column);

            double previous = double.NegativeInfinity;
            for (int t = 0; t <= _numBins; t++)
            {
                // Linearly interpolated quantile at t / T.
                double position = (double)t / _numBins * (samples - 1);
                int lower = (int)Math.Floor(position);
                int upper = Math.Min(lower + 1, samples - 1);
                double edge = column[lower] + (column[upper] - column[lower]) * (position - lower);
                if (edge <= previous)
                    edge = previous + Math.Max(Math.Abs(previous), 1.0) * 1e-6;
                _binEdges[f * (_numBins + 1) + t] = NumOps.FromDouble(edge);
                previous = edge;
            }
        }
    }

    /// <summary>
    /// Encodes [batch, numFeatures] into [batch, numFeatures * numBins].
    /// </summary>
    /// <remarks>
    /// Built from engine ops, so a gradient tape records it: the derivative 1 / (b_t - b_{t-1}) reaches the input
    /// wherever the clamp is not active. (It used to be a scalar loop writing into a rented tensor, which no tape
    /// saw, computing a symmetric "triangle" that is not the paper's encoding.)
    /// </remarks>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 2 || input.Shape[1] != _numFeatures)
            throw new ArgumentException($"Expected [batch, {_numFeatures}] features.", nameof(input));
        int batch = input.Shape[0];
        int bins = _numBins;
        var shape = new[] { batch, _numFeatures, bins };

        // Per-bin constants: lower edge, width, and the clamp bounds (no lower clamp on the first bin, no upper clamp
        // on the last). None of them needs a gradient.
        var lowerEdge = new Tensor<T>([1, _numFeatures, bins]);
        var width = new Tensor<T>([1, _numFeatures, bins]);
        var floor = new Tensor<T>([1, _numFeatures, bins]);
        var ceiling = new Tensor<T>([1, _numFeatures, bins]);
        for (int f = 0; f < _numFeatures; f++)
        {
            for (int t = 0; t < bins; t++)
            {
                int i = f * bins + t;
                var lo = _binEdges[f * (bins + 1) + t];
                lowerEdge[i] = lo;
                width[i] = NumOps.Subtract(_binEdges[f * (bins + 1) + t + 1], lo);
                floor[i] = t == 0 ? NumOps.FromDouble(double.NegativeInfinity) : NumOps.Zero;
                ceiling[i] = t == bins - 1 ? NumOps.FromDouble(double.PositiveInfinity) : NumOps.One;
            }
        }

        var x = Engine.TensorBroadcastTo(Engine.Reshape(input, new[] { batch, _numFeatures, 1 }), shape);
        var raw = Engine.TensorDivide(
            Engine.TensorSubtract(x, Engine.TensorBroadcastTo(lowerEdge, shape)),
            Engine.TensorBroadcastTo(width, shape));
        var encoded = Engine.TensorClampTensor(raw,
            Engine.TensorBroadcastTo(floor, shape),
            Engine.TensorBroadcastTo(ceiling, shape));
        return Engine.Reshape(encoded, new[] { batch, _numFeatures * bins });
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        // No trainable parameters: the edges are fixed once fitted, as in the paper.
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
    }
}
