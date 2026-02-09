namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Implements the Sparsemax activation function, which projects input onto the probability simplex
/// with sparse outputs (many exact zeros).
/// </summary>
/// <remarks>
/// <para>
/// Sparsemax is an alternative to softmax that produces sparse probability distributions.
/// Unlike softmax, which always produces non-zero probabilities for all inputs, sparsemax
/// can produce exact zeros, making it ideal for feature selection in attention mechanisms.
/// </para>
/// <para>
/// <b>For Beginners:</b> Sparsemax is like softmax but produces cleaner, more focused attention.
///
/// Imagine you're selecting which features to pay attention to:
/// - Softmax says "pay a little attention to everything"
/// - Sparsemax says "focus only on the important features, ignore the rest completely"
///
/// This makes neural networks more interpretable because you can clearly see
/// which features the model considers important (non-zero values) versus
/// which ones it ignores (exact zeros).
///
/// In TabNet, sparsemax is used to select which features to use at each decision step,
/// providing built-in interpretability about feature importance.
/// </para>
/// <para>
/// <b>Mathematical Background:</b>
/// Sparsemax solves: argmin_{p ∈ Δ^K} ||p - z||²
/// where Δ^K is the (K-1)-dimensional probability simplex.
///
/// Reference: "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
/// by André F. T. Martins and Ramón Fernandez Astudillo (ICML 2016)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class Sparsemax<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Sparsemax class.
    /// </summary>
    public Sparsemax()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Applies the sparsemax function to the input tensor along the specified axis.
    /// </summary>
    /// <param name="input">The input tensor of logits.</param>
    /// <param name="axis">The axis along which to apply sparsemax. Default is -1 (last axis).</param>
    /// <returns>A tensor with sparsemax-normalized values (sparse probability distribution).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This transforms raw scores into a sparse probability distribution.
    /// The output values sum to 1 (like probabilities), but many values will be exactly 0.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input, int axis = -1)
    {
        // Handle negative axis
        if (axis < 0)
        {
            axis = input.Rank + axis;
        }

        // Get the dimension size along the specified axis
        int dimSize = input.Shape[axis];

        // Create output tensor with same shape
        var output = new Tensor<T>(input.Shape);

        // Process each slice along the axis
        ProcessAlongAxis(input, output, axis, dimSize);

        return output;
    }

    /// <summary>
    /// Computes the gradient of the sparsemax function for backpropagation.
    /// </summary>
    /// <param name="gradOutput">The gradient flowing back from the next layer.</param>
    /// <param name="sparsemaxOutput">The output from the forward pass.</param>
    /// <param name="axis">The axis along which sparsemax was applied.</param>
    /// <returns>The gradient with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// The Jacobian of sparsemax is: ∂sparsemax/∂z = diag(s) - s s^T / ||s||₁
    /// where s is the support (non-zero elements) indicator.
    /// </para>
    /// </remarks>
    public Tensor<T> Backward(Tensor<T> gradOutput, Tensor<T> sparsemaxOutput, int axis = -1)
    {
        // Handle negative axis
        if (axis < 0)
        {
            axis = gradOutput.Rank + axis;
        }

        // Create gradient tensor
        var gradInput = new Tensor<T>(gradOutput.Shape);

        // Process each slice along the axis
        ProcessGradientAlongAxis(gradOutput, sparsemaxOutput, gradInput, axis);

        return gradInput;
    }

    /// <summary>
    /// Applies sparsemax to a 1D vector (single slice).
    /// </summary>
    private Vector<T> SparsemaxVector(Vector<T> z)
    {
        int n = z.Length;

        // Sort z in descending order
        var sorted = new T[n];
        var indices = new int[n];
        for (int i = 0; i < n; i++)
        {
            sorted[i] = z[i];
            indices[i] = i;
        }

        // Sort descending
        Array.Sort(sorted, indices, Comparer<T>.Create((a, b) =>
            _numOps.GreaterThan(a, b) ? -1 : (_numOps.LessThan(a, b) ? 1 : 0)));

        // Find the threshold tau
        var cumSum = _numOps.Zero;
        var threshold = _numOps.Zero;
        int k = 0;

        for (int i = 0; i < n; i++)
        {
            cumSum = _numOps.Add(cumSum, sorted[i]);
            // Check if 1 + (i+1) * z_i > cumsum
            var test = _numOps.Add(_numOps.One, _numOps.Multiply(_numOps.FromDouble(i + 1), sorted[i]));
            if (_numOps.GreaterThan(test, cumSum))
            {
                k = i + 1;
                threshold = _numOps.Divide(
                    _numOps.Subtract(cumSum, _numOps.One),
                    _numOps.FromDouble(k));
            }
        }

        // Compute sparsemax output: max(z - tau, 0)
        var result = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            var diff = _numOps.Subtract(z[i], threshold);
            result[i] = _numOps.GreaterThan(diff, _numOps.Zero) ? diff : _numOps.Zero;
        }

        return result;
    }

    /// <summary>
    /// Computes the gradient for a single sparsemax slice.
    /// </summary>
    private Vector<T> SparsemaxGradientVector(Vector<T> gradOut, Vector<T> sparsemaxOut)
    {
        int n = gradOut.Length;

        // Find support (non-zero elements in sparsemax output)
        var support = new bool[n];
        int supportSize = 0;
        for (int i = 0; i < n; i++)
        {
            support[i] = _numOps.GreaterThan(sparsemaxOut[i], _numOps.Zero);
            if (support[i]) supportSize++;
        }

        if (supportSize == 0)
        {
            return new Vector<T>(n); // Return zeros
        }

        // Compute sum of gradients on support
        var gradSum = _numOps.Zero;
        for (int i = 0; i < n; i++)
        {
            if (support[i])
            {
                gradSum = _numOps.Add(gradSum, gradOut[i]);
            }
        }

        // Divide by support size
        var gradMean = _numOps.Divide(gradSum, _numOps.FromDouble(supportSize));

        // Gradient is: gradOut - gradMean on support, 0 elsewhere
        var result = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            if (support[i])
            {
                result[i] = _numOps.Subtract(gradOut[i], gradMean);
            }
            // else result[i] stays zero
        }

        return result;
    }

    /// <summary>
    /// Processes the input tensor along the specified axis to apply sparsemax.
    /// </summary>
    private void ProcessAlongAxis(Tensor<T> input, Tensor<T> output, int axis, int dimSize)
    {
        // For simplicity, handle 2D case (batch x features) which is most common for tabular data
        if (input.Rank == 2)
        {
            if (axis == 1 || axis == -1)
            {
                // Apply sparsemax to each row
                int batchSize = input.Shape[0];
                int features = input.Shape[1];

                for (int b = 0; b < batchSize; b++)
                {
                    // Extract row
                    var row = new Vector<T>(features);
                    for (int f = 0; f < features; f++)
                    {
                        row[f] = input[b * features + f];
                    }

                    // Apply sparsemax
                    var result = SparsemaxVector(row);

                    // Copy back
                    for (int f = 0; f < features; f++)
                    {
                        output[b * features + f] = result[f];
                    }
                }
            }
            else if (axis == 0)
            {
                // Apply sparsemax to each column
                int batchSize = input.Shape[0];
                int features = input.Shape[1];

                for (int f = 0; f < features; f++)
                {
                    // Extract column
                    var col = new Vector<T>(batchSize);
                    for (int b = 0; b < batchSize; b++)
                    {
                        col[b] = input[b * features + f];
                    }

                    // Apply sparsemax
                    var result = SparsemaxVector(col);

                    // Copy back
                    for (int b = 0; b < batchSize; b++)
                    {
                        output[b * features + f] = result[b];
                    }
                }
            }
        }
        else if (input.Rank == 1)
        {
            // Direct 1D case
            var vec = new Vector<T>(input.Length);
            for (int i = 0; i < input.Length; i++)
            {
                vec[i] = input[i];
            }

            var result = SparsemaxVector(vec);

            for (int i = 0; i < input.Length; i++)
            {
                output[i] = result[i];
            }
        }
        else
        {
            // For higher-dimensional tensors, flatten all dims except the axis dim
            // and process as 2D, then reshape back
            ProcessHigherDimensionalAxis(input, output, axis);
        }
    }

    /// <summary>
    /// Processes gradient along the specified axis.
    /// </summary>
    private void ProcessGradientAlongAxis(Tensor<T> gradOutput, Tensor<T> sparsemaxOutput, Tensor<T> gradInput, int axis)
    {
        // Handle 2D case (most common for tabular)
        if (gradOutput.Rank == 2)
        {
            if (axis == 1 || axis == -1)
            {
                int batchSize = gradOutput.Shape[0];
                int features = gradOutput.Shape[1];

                for (int b = 0; b < batchSize; b++)
                {
                    var gradRow = new Vector<T>(features);
                    var outRow = new Vector<T>(features);

                    for (int f = 0; f < features; f++)
                    {
                        gradRow[f] = gradOutput[b * features + f];
                        outRow[f] = sparsemaxOutput[b * features + f];
                    }

                    var result = SparsemaxGradientVector(gradRow, outRow);

                    for (int f = 0; f < features; f++)
                    {
                        gradInput[b * features + f] = result[f];
                    }
                }
            }
            else if (axis == 0)
            {
                int batchSize = gradOutput.Shape[0];
                int features = gradOutput.Shape[1];

                for (int f = 0; f < features; f++)
                {
                    var gradCol = new Vector<T>(batchSize);
                    var outCol = new Vector<T>(batchSize);

                    for (int b = 0; b < batchSize; b++)
                    {
                        gradCol[b] = gradOutput[b * features + f];
                        outCol[b] = sparsemaxOutput[b * features + f];
                    }

                    var result = SparsemaxGradientVector(gradCol, outCol);

                    for (int b = 0; b < batchSize; b++)
                    {
                        gradInput[b * features + f] = result[b];
                    }
                }
            }
        }
        else if (gradOutput.Rank == 1)
        {
            var gradVec = new Vector<T>(gradOutput.Length);
            var outVec = new Vector<T>(sparsemaxOutput.Length);

            for (int i = 0; i < gradOutput.Length; i++)
            {
                gradVec[i] = gradOutput[i];
                outVec[i] = sparsemaxOutput[i];
            }

            var result = SparsemaxGradientVector(gradVec, outVec);

            for (int i = 0; i < gradInput.Length; i++)
            {
                gradInput[i] = result[i];
            }
        }
        else
        {
            ProcessHigherDimensionalGradient(gradOutput, sparsemaxOutput, gradInput, axis);
        }
    }

    /// <summary>
    /// Handles sparsemax for tensors with rank > 2.
    /// </summary>
    private void ProcessHigherDimensionalAxis(Tensor<T> input, Tensor<T> output, int axis)
    {
        // Calculate strides for iteration
        int[] shape = input.Shape;
        int rank = shape.Length;

        // Calculate the total number of slices
        int numSlices = 1;
        for (int i = 0; i < rank; i++)
        {
            if (i != axis)
            {
                numSlices *= shape[i];
            }
        }

        int sliceLength = shape[axis];

        // Process each slice
        for (int sliceIdx = 0; sliceIdx < numSlices; sliceIdx++)
        {
            var slice = new Vector<T>(sliceLength);

            // Extract slice - compute indices
            int[] indices = new int[rank];
            int temp = sliceIdx;
            for (int d = rank - 1; d >= 0; d--)
            {
                if (d != axis)
                {
                    indices[d] = temp % shape[d];
                    temp /= shape[d];
                }
            }

            // Extract values along axis
            for (int i = 0; i < sliceLength; i++)
            {
                indices[axis] = i;
                int flatIdx = ComputeFlatIndex(indices, shape);
                slice[i] = input[flatIdx];
            }

            // Apply sparsemax
            var result = SparsemaxVector(slice);

            // Write back
            for (int i = 0; i < sliceLength; i++)
            {
                indices[axis] = i;
                int flatIdx = ComputeFlatIndex(indices, shape);
                output[flatIdx] = result[i];
            }
        }
    }

    /// <summary>
    /// Handles gradient computation for tensors with rank > 2.
    /// </summary>
    private void ProcessHigherDimensionalGradient(Tensor<T> gradOutput, Tensor<T> sparsemaxOutput, Tensor<T> gradInput, int axis)
    {
        int[] shape = gradOutput.Shape;
        int rank = shape.Length;

        int numSlices = 1;
        for (int i = 0; i < rank; i++)
        {
            if (i != axis)
            {
                numSlices *= shape[i];
            }
        }

        int sliceLength = shape[axis];

        for (int sliceIdx = 0; sliceIdx < numSlices; sliceIdx++)
        {
            var gradSlice = new Vector<T>(sliceLength);
            var outSlice = new Vector<T>(sliceLength);

            int[] indices = new int[rank];
            int temp = sliceIdx;
            for (int d = rank - 1; d >= 0; d--)
            {
                if (d != axis)
                {
                    indices[d] = temp % shape[d];
                    temp /= shape[d];
                }
            }

            for (int i = 0; i < sliceLength; i++)
            {
                indices[axis] = i;
                int flatIdx = ComputeFlatIndex(indices, shape);
                gradSlice[i] = gradOutput[flatIdx];
                outSlice[i] = sparsemaxOutput[flatIdx];
            }

            var result = SparsemaxGradientVector(gradSlice, outSlice);

            for (int i = 0; i < sliceLength; i++)
            {
                indices[axis] = i;
                int flatIdx = ComputeFlatIndex(indices, shape);
                gradInput[flatIdx] = result[i];
            }
        }
    }

    /// <summary>
    /// Computes flat array index from multi-dimensional indices.
    /// </summary>
    private static int ComputeFlatIndex(int[] indices, int[] shape)
    {
        int flatIdx = 0;
        int stride = 1;
        for (int d = shape.Length - 1; d >= 0; d--)
        {
            flatIdx += indices[d] * stride;
            stride *= shape[d];
        }
        return flatIdx;
    }
}
