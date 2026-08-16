using AiDotNet.Autodiff;
// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.PointCloud.Layers;

/// <summary>
/// Implements global max pooling for point clouds to extract global features.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Max pooling takes the maximum value across all points for each feature channel.
///
/// How it works:
/// - Input: N points, each with C features [N, C]
/// - Operation: For each feature channel, find the maximum value across all N points
/// - Output: A single vector of C features [1, C]
///
/// Why it's useful:
/// - Creates a global representation of the entire point cloud
/// - Achieves permutation invariance (order of points doesn't matter)
/// - Reduces dimensionality from many points to one feature vector
///
/// Example:
/// - Input: 1024 points with 64 features each = [1024, 64]
/// - Max pooling across points
/// - Output: 1 global feature vector with 64 features = [1, 64]
///
/// This is a key component in PointNet for making the network invariant to point order.
/// </remarks>
// Rank 2 only: ForwardTraced reads input.Shape[0] as the point count and reduces axis 0, so there is
// no batch axis and no unbatched variant to be optional about. [numPoints, numFeatures] is named
// [Length, Features] to match TNetLayer, the sibling that feeds this layer and declares the same
// layout - Length rather than Time because a point cloud's ordering carries no meaning (that
// permutation invariance is the whole point of pooling here).
[TensorLayout(TensorAxis.Length, TensorAxis.Features, Direction = TensorLayoutDirection.Input,
    Note = "Point cloud as [numPoints, numFeatures].")]
[TensorLayout(TensorAxis.Length, TensorAxis.Features, Direction = TensorLayoutDirection.Output,
    Note = "Global feature vector as [1, numFeatures]; the point axis survives, collapsed to one.")]
public partial class MaxPoolingLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Straight off <c>ForwardTraced</c>: <c>Engine.ReduceMax(input, [0], true, out _)</c>. The
    /// reduction is over axis 0 with keepDims = true, so the point axis survives at extent 1 - the
    /// <c>[1, numFeatures]</c> output shape the constructor also declares via
    /// <c>base([0, numFeatures], [1, numFeatures])</c>. The 1 is structural to a keep-dims reduction,
    /// not a configured size, which is why it is the one literal in this contract.
    /// </para>
    /// <para>
    /// THE FEATURE AXIS IS <c>Same</c>, NOT <c>Fixed(_numFeatures)</c>. <c>ReduceMax</c> touches only
    /// axis 0 and hands back whatever width arrived; nothing in the forward pass ever compares
    /// <c>input.Shape[1]</c> against <c>_numFeatures</c>, which is used solely to declare the base
    /// shapes. Declaring the field would state a constraint the layer does not enforce and would be
    /// wrong for any caller that pools a differently-sized cloud.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 2) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Length, AxisRelation.Fixed(1)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Same(TensorAxis.Features)),
        };
    }

    private readonly int _numFeatures;
    private int[]? _maxIndices; // Store indices of max values for backward pass
    private int _numPoints;

    /// <summary>
    /// Initializes a new instance of the MaxPoolingLayer class.
    /// </summary>
    /// <param name="numFeatures">Number of feature channels to pool.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Creates a max pooling layer for point cloud global feature extraction.
    ///
    /// The number of features determines the output size:
    /// - If input is [N, 64], output will be [1, 64]
    /// - If input is [N, 128], output will be [1, 128]
    ///
    /// This layer has no trainable parameters - it's a fixed operation that
    /// selects the maximum value for each feature across all points.
    /// </remarks>
    public MaxPoolingLayer(int numFeatures)
        : base([0, numFeatures], [1, numFeatures])
    {
        _numFeatures = numFeatures;
        Parameters = Vector<T>.Empty(); // No trainable parameters
    }

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        _numPoints = input.Shape[0];

        // Use ReduceMax to select the discrete winning point per feature, then express the exact
        // selected value as detached-mask * input followed by a tape-tracked sum. This preserves
        // max-pooling forward semantics and avoids the package ReduceMax backward's incorrect
        // routing for the point-cloud reduction shapes exercised by DGCNN finite differences.
        _ = Engine.ReduceMax(input, [0], true, out int[] maxIndices);
        _maxIndices = maxIndices;

        int numFeatures = input.Shape[1];
        var maxMask = new Tensor<T>([_numPoints, numFeatures]);
        for (int feature = 0; feature < numFeatures; feature++)
        {
            // ReduceMax returns the winning element's flat index in the SOURCE
            // [point, feature] tensor. Convert it back to a point coordinate;
            // using it directly made every winner except the earliest features
            // fail the bounds check and silently wrote an all-zero mask.
            int selectedPoint = maxIndices[feature] / numFeatures;
            if ((uint)selectedPoint < (uint)_numPoints)
                maxMask[selectedPoint, feature] = NumOps.One;
        }

        var pooledOutput = Engine.ReduceSum(
            Engine.TensorMultiply(input, maxMask), [0], keepDims: true);

        return pooledOutput;
    }

    public override void ClearGradients()
    {
        // No gradients to clear
    }

    public override void ResetState()
    {
        _maxIndices = null;
        _numPoints = 0;
    }

    public override bool SupportsTraining => false; // No parameters to update; still participates in backprop
}
