#nullable disable
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Operators;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Provides CPU implementations of mesh convolution operations for 3D surface learning.
/// </summary>
/// <remarks>
/// <para>
/// This class implements vectorized mesh convolution algorithms including:
/// - Spiral convolution for irregular mesh connectivity
/// - Diffusion convolution using heat equation on surfaces
/// - Mesh Laplacian computation for geometry-aware processing
/// </para>
/// <para>
/// All operations are optimized using SIMD vectorization via System.Numerics.Vector
/// where applicable, and parallel processing for batch operations.
/// </para>
/// </remarks>
public static class MeshConvolutionOperations
{
    #region Spiral Convolution

    /// <summary>
    /// Performs spiral convolution on mesh vertex features.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="vertexFeatures">Input vertex features [numVertices, inputChannels].</param>
    /// <param name="spiralIndices">Spiral neighbor indices [numVertices, spiralLength].</param>
    /// <param name="weights">Convolution weights [outputChannels, inputChannels * spiralLength].</param>
    /// <param name="biases">Bias values [outputChannels].</param>
    /// <returns>Output vertex features [numVertices, outputChannels].</returns>
    /// <remarks>
    /// <para>
    /// The algorithm:
    /// 1. For each vertex, gather features from neighbors in spiral order
    /// 2. Concatenate gathered features into a single vector
    /// 3. Apply linear transformation (matrix multiply with weights)
    /// 4. Add bias
    /// </para>
    /// <para>
    /// Uses SIMD vectorization for the matrix multiplication and parallel processing
    /// across vertices for optimal performance.
    /// </para>
    /// </remarks>
    public static Tensor<T> SpiralConv<T>(
        Tensor<T> vertexFeatures,
        Tensor<int> spiralIndices,
        Tensor<T> weights,
        Tensor<T> biases)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int numVertices = vertexFeatures.Shape[0];
        int inputChannels = vertexFeatures.Shape[1];
        int spiralLength = spiralIndices.Shape[1];
        int outputChannels = weights.Shape[0];
        int gatheredSize = inputChannels * spiralLength;

        var output = new T[numVertices * outputChannels];
        var vertexData = vertexFeatures.ToArray();
        var indicesData = spiralIndices.ToArray();
        var weightsData = weights.ToArray();
        var biasData = biases.ToArray();

        // Process vertices in parallel
        Parallel.For(0, numVertices, v =>
        {
            // Step 1: Gather features from spiral neighbors
            var gathered = new T[gatheredSize];

            for (int s = 0; s < spiralLength; s++)
            {
                int neighborIdx = indicesData[v * spiralLength + s];
                int gatherOffset = s * inputChannels;

                if (neighborIdx >= 0 && neighborIdx < numVertices)
                {
                    // Copy neighbor features
                    for (int c = 0; c < inputChannels; c++)
                    {
                        gathered[gatherOffset + c] = vertexData[neighborIdx * inputChannels + c];
                    }
                }
                // Invalid indices default to zero (already initialized)
            }

            // Step 2: Apply linear transformation with SIMD
            for (int oc = 0; oc < outputChannels; oc++)
            {
                T sum = biasData[oc];
                int weightRowOffset = oc * gatheredSize;

                // Use SIMD for inner product
                int simdWidth = System.Numerics.Vector<float>.Count;
                int i = 0;

                if (typeof(T) == typeof(float))
                {
                    var sumVec = System.Numerics.Vector<float>.Zero;
                    float[] gatheredFloat = (float[])(object)gathered;
                    float[] weightsFloat = (float[])(object)weightsData;

                    for (; i <= gatheredSize - simdWidth; i += simdWidth)
                    {
                        var gVec = new System.Numerics.Vector<float>(gatheredFloat, i);
                        var wVec = new System.Numerics.Vector<float>(weightsFloat, weightRowOffset + i);
                        sumVec += gVec * wVec;
                    }

                    float scalarSum = 0f;
                    for (int j = 0; j < simdWidth; j++)
                    {
                        scalarSum += sumVec[j];
                    }
                    sum = numOps.Add(sum, numOps.FromDouble(scalarSum));
                }
                else if (typeof(T) == typeof(double))
                {
                    var sumVec = System.Numerics.Vector<double>.Zero;
                    double[] gatheredDouble = (double[])(object)gathered;
                    double[] weightsDouble = (double[])(object)weightsData;
                    int simdWidthDouble = System.Numerics.Vector<double>.Count;

                    for (; i <= gatheredSize - simdWidthDouble; i += simdWidthDouble)
                    {
                        var gVec = new System.Numerics.Vector<double>(gatheredDouble, i);
                        var wVec = new System.Numerics.Vector<double>(weightsDouble, weightRowOffset + i);
                        sumVec += gVec * wVec;
                    }

                    double scalarSum = 0.0;
                    for (int j = 0; j < simdWidthDouble; j++)
                    {
                        scalarSum += sumVec[j];
                    }
                    sum = numOps.Add(sum, numOps.FromDouble(scalarSum));
                }

                // Handle remaining elements
                for (; i < gatheredSize; i++)
                {
                    sum = numOps.Add(sum, numOps.Multiply(gathered[i], weightsData[weightRowOffset + i]));
                }

                output[v * outputChannels + oc] = sum;
            }
        });

        return new Tensor<T>(output, [numVertices, outputChannels]);
    }

    /// <summary>
    /// Computes the backward pass for spiral convolution with respect to input features.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="outputGradient">Gradient with respect to output [numVertices, outputChannels].</param>
    /// <param name="spiralIndices">Spiral neighbor indices [numVertices, spiralLength].</param>
    /// <param name="weights">Convolution weights [outputChannels, inputChannels * spiralLength].</param>
    /// <param name="inputChannels">Number of input channels.</param>
    /// <returns>Gradient with respect to input features [numVertices, inputChannels].</returns>
    public static Tensor<T> SpiralConvBackwardInput<T>(
        Tensor<T> outputGradient,
        Tensor<int> spiralIndices,
        Tensor<T> weights,
        int inputChannels)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int numVertices = outputGradient.Shape[0];
        int outputChannels = outputGradient.Shape[1];
        int spiralLength = spiralIndices.Shape[1];
        int gatheredSize = inputChannels * spiralLength;

        var inputGrad = new T[numVertices * inputChannels];
        var gradLocks = new object[numVertices];
        for (int i = 0; i < numVertices; i++) gradLocks[i] = new object();

        var gradData = outputGradient.ToArray();
        var indicesData = spiralIndices.ToArray();
        var weightsData = weights.ToArray();

        // Process vertices in parallel
        Parallel.For(0, numVertices, v =>
        {
            // Compute gradient for gathered features first
            var gatheredGrad = new T[gatheredSize];

            for (int oc = 0; oc < outputChannels; oc++)
            {
                T grad = gradData[v * outputChannels + oc];
                int weightRowOffset = oc * gatheredSize;

                // Use SIMD for gradient computation
                int simdWidth = System.Numerics.Vector<float>.Count;
                int i = 0;

                if (typeof(T) == typeof(float))
                {
                    float gradFloat = (float)(object)grad;
                    var gradVec = new System.Numerics.Vector<float>(gradFloat);
                    float[] gatheredGradFloat = (float[])(object)gatheredGrad;
                    float[] weightsFloat = (float[])(object)weightsData;

                    for (; i <= gatheredSize - simdWidth; i += simdWidth)
                    {
                        var wVec = new System.Numerics.Vector<float>(weightsFloat, weightRowOffset + i);
                        var existingVec = new System.Numerics.Vector<float>(gatheredGradFloat, i);
                        var resultVec = existingVec + gradVec * wVec;
                        resultVec.CopyTo(gatheredGradFloat, i);
                    }
                }
                else if (typeof(T) == typeof(double))
                {
                    double gradDouble = (double)(object)grad;
                    var gradVec = new System.Numerics.Vector<double>(gradDouble);
                    double[] gatheredGradDouble = (double[])(object)gatheredGrad;
                    double[] weightsDouble = (double[])(object)weightsData;
                    int simdWidthDouble = System.Numerics.Vector<double>.Count;

                    for (; i <= gatheredSize - simdWidthDouble; i += simdWidthDouble)
                    {
                        var wVec = new System.Numerics.Vector<double>(weightsDouble, weightRowOffset + i);
                        var existingVec = new System.Numerics.Vector<double>(gatheredGradDouble, i);
                        var resultVec = existingVec + gradVec * wVec;
                        resultVec.CopyTo(gatheredGradDouble, i);
                    }
                }

                // Handle remaining elements
                for (; i < gatheredSize; i++)
                {
                    gatheredGrad[i] = numOps.Add(gatheredGrad[i],
                        numOps.Multiply(grad, weightsData[weightRowOffset + i]));
                }
            }

            // Scatter gradients back to input according to spiral indices
            for (int s = 0; s < spiralLength; s++)
            {
                int neighborIdx = indicesData[v * spiralLength + s];
                int gatherOffset = s * inputChannels;

                if (neighborIdx >= 0 && neighborIdx < numVertices)
                {
                    lock (gradLocks[neighborIdx])
                    {
                        for (int c = 0; c < inputChannels; c++)
                        {
                            int inputIdx = neighborIdx * inputChannels + c;
                            inputGrad[inputIdx] = numOps.Add(inputGrad[inputIdx],
                                gatheredGrad[gatherOffset + c]);
                        }
                    }
                }
            }
        });

        return new Tensor<T>(inputGrad, [numVertices, inputChannels]);
    }

    /// <summary>
    /// Computes the backward pass for spiral convolution with respect to weights.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="outputGradient">Gradient with respect to output [numVertices, outputChannels].</param>
    /// <param name="vertexFeatures">Input vertex features from forward pass [numVertices, inputChannels].</param>
    /// <param name="spiralIndices">Spiral neighbor indices [numVertices, spiralLength].</param>
    /// <returns>Gradient with respect to weights [outputChannels, inputChannels * spiralLength].</returns>
    public static Tensor<T> SpiralConvBackwardWeights<T>(
        Tensor<T> outputGradient,
        Tensor<T> vertexFeatures,
        Tensor<int> spiralIndices)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int numVertices = outputGradient.Shape[0];
        int outputChannels = outputGradient.Shape[1];
        int inputChannels = vertexFeatures.Shape[1];
        int spiralLength = spiralIndices.Shape[1];
        int gatheredSize = inputChannels * spiralLength;

        var weightGrad = new T[outputChannels * gatheredSize];
        var weightLocks = new object[outputChannels];
        for (int i = 0; i < outputChannels; i++) weightLocks[i] = new object();

        var gradData = outputGradient.ToArray();
        var vertexData = vertexFeatures.ToArray();
        var indicesData = spiralIndices.ToArray();

        // Process vertices in parallel
        Parallel.For(0, numVertices, v =>
        {
            // Gather features from spiral neighbors
            var gathered = new T[gatheredSize];

            for (int s = 0; s < spiralLength; s++)
            {
                int neighborIdx = indicesData[v * spiralLength + s];
                int gatherOffset = s * inputChannels;

                if (neighborIdx >= 0 && neighborIdx < numVertices)
                {
                    for (int c = 0; c < inputChannels; c++)
                    {
                        gathered[gatherOffset + c] = vertexData[neighborIdx * inputChannels + c];
                    }
                }
            }

            // Compute weight gradient contribution from this vertex
            for (int oc = 0; oc < outputChannels; oc++)
            {
                T grad = gradData[v * outputChannels + oc];
                int weightRowOffset = oc * gatheredSize;

                lock (weightLocks[oc])
                {
                    // Use SIMD for gradient accumulation
                    int simdWidth = System.Numerics.Vector<float>.Count;
                    int i = 0;

                    if (typeof(T) == typeof(float))
                    {
                        float gradFloat = (float)(object)grad;
                        var gradVec = new System.Numerics.Vector<float>(gradFloat);
                        float[] gatheredFloat = (float[])(object)gathered;
                        float[] weightGradFloat = (float[])(object)weightGrad;

                        for (; i <= gatheredSize - simdWidth; i += simdWidth)
                        {
                            var gVec = new System.Numerics.Vector<float>(gatheredFloat, i);
                            var existingVec = new System.Numerics.Vector<float>(weightGradFloat, weightRowOffset + i);
                            var resultVec = existingVec + gradVec * gVec;
                            resultVec.CopyTo(weightGradFloat, weightRowOffset + i);
                        }
                    }
                    else if (typeof(T) == typeof(double))
                    {
                        double gradDouble = (double)(object)grad;
                        var gradVec = new System.Numerics.Vector<double>(gradDouble);
                        double[] gatheredDouble = (double[])(object)gathered;
                        double[] weightGradDouble = (double[])(object)weightGrad;
                        int simdWidthDouble = System.Numerics.Vector<double>.Count;

                        for (; i <= gatheredSize - simdWidthDouble; i += simdWidthDouble)
                        {
                            var gVec = new System.Numerics.Vector<double>(gatheredDouble, i);
                            var existingVec = new System.Numerics.Vector<double>(weightGradDouble, weightRowOffset + i);
                            var resultVec = existingVec + gradVec * gVec;
                            resultVec.CopyTo(weightGradDouble, weightRowOffset + i);
                        }
                    }

                    // Handle remaining elements
                    for (; i < gatheredSize; i++)
                    {
                        weightGrad[weightRowOffset + i] = numOps.Add(weightGrad[weightRowOffset + i],
                            numOps.Multiply(grad, gathered[i]));
                    }
                }
            }
        });

        return new Tensor<T>(weightGrad, [outputChannels, gatheredSize]);
    }

    /// <summary>
    /// Computes the backward pass for spiral convolution with respect to biases.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="outputGradient">Gradient with respect to output [numVertices, outputChannels].</param>
    /// <returns>Gradient with respect to biases [outputChannels].</returns>
    public static Tensor<T> SpiralConvBackwardBias<T>(Tensor<T> outputGradient)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int numVertices = outputGradient.Shape[0];
        int outputChannels = outputGradient.Shape[1];

        var biasGrad = new T[outputChannels];
        var gradData = outputGradient.ToArray();

        // Sum gradients across all vertices for each output channel
        // Using SIMD for the reduction
        Parallel.For(0, outputChannels, oc =>
        {
            T sum = numOps.Zero;

            int simdWidth = System.Numerics.Vector<float>.Count;
            int v = 0;

            if (typeof(T) == typeof(float))
            {
                var sumVec = System.Numerics.Vector<float>.Zero;
                float[] gradFloat = (float[])(object)gradData;

                for (; v <= numVertices - simdWidth; v += simdWidth)
                {
                    // Gather elements at stride outputChannels
                    var values = new float[simdWidth];
                    for (int j = 0; j < simdWidth; j++)
                    {
                        values[j] = gradFloat[(v + j) * outputChannels + oc];
                    }
                    sumVec += new System.Numerics.Vector<float>(values);
                }

                float scalarSum = 0f;
                for (int j = 0; j < simdWidth; j++)
                {
                    scalarSum += sumVec[j];
                }
                sum = numOps.FromDouble(scalarSum);
            }

            // Handle remaining elements
            for (; v < numVertices; v++)
            {
                sum = numOps.Add(sum, gradData[v * outputChannels + oc]);
            }

            biasGrad[oc] = sum;
        });

        return new Tensor<T>(biasGrad, [outputChannels]);
    }

    #endregion

    #region Diffusion Convolution

    /// <summary>
    /// Performs diffusion convolution on mesh vertex features using the Laplacian operator.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="vertexFeatures">Input vertex features [numVertices, inputChannels].</param>
    /// <param name="laplacian">Mesh Laplacian matrix [numVertices, numVertices].</param>
    /// <param name="weights">Diffusion weights [outputChannels, inputChannels].</param>
    /// <param name="biases">Bias values [outputChannels].</param>
    /// <param name="diffusionTime">Diffusion time parameter controlling spatial extent.</param>
    /// <returns>Output vertex features [numVertices, outputChannels].</returns>
    /// <remarks>
    /// <para>
    /// Implements diffusion convolution using Taylor series approximation of exp(-t*L):
    /// exp(-t*L) ≈ I - t*L + (t²/2)*L² - (t³/6)*L³ + ...
    /// 
    /// Using 4 terms provides good accuracy for reasonable diffusion times.
    /// </para>
    /// </remarks>
    public static Tensor<T> DiffusionConv<T>(
        Tensor<T> vertexFeatures,
        Tensor<T> laplacian,
        Tensor<T> weights,
        Tensor<T> biases,
        T diffusionTime)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int numVertices = vertexFeatures.Shape[0];
        int inputChannels = vertexFeatures.Shape[1];
        int outputChannels = weights.Shape[0];

        double t = numOps.ToDouble(diffusionTime);
        double t2 = t * t / 2.0;
        double t3 = t * t * t / 6.0;
        double t4 = t * t * t * t / 24.0;

        var vertexData = vertexFeatures.ToArray();
        var lapData = laplacian.ToArray();
        var weightsData = weights.ToArray();
        var biasData = biases.ToArray();

        // Extract sparse structure once (O(E) where E = edges) to avoid O(V²) in each multiply
        var sparseStructure = ExtractSparseStructure(lapData, numVertices, numOps);

        // Step 1: Apply diffusion using Taylor series
        var diffused = new T[numVertices * inputChannels];

        // Compute L*x, L²*x, L³*x, L⁴*x using sparse multiplication
        var Lx = new T[numVertices * inputChannels];
        var L2x = new T[numVertices * inputChannels];
        var L3x = new T[numVertices * inputChannels];
        var L4x = new T[numVertices * inputChannels];

        // L*x
        ComputeSparseLaplacianProduct(sparseStructure, vertexData, Lx, numVertices, inputChannels, numOps);

        // L²*x = L*(L*x)
        ComputeSparseLaplacianProduct(sparseStructure, Lx, L2x, numVertices, inputChannels, numOps);

        // L³*x = L*(L²*x)
        ComputeSparseLaplacianProduct(sparseStructure, L2x, L3x, numVertices, inputChannels, numOps);

        // L⁴*x = L*(L³*x)
        ComputeSparseLaplacianProduct(sparseStructure, L3x, L4x, numVertices, inputChannels, numOps);

        // Combine: exp(-t*L)*x ≈ x - t*L*x + (t²/2)*L²*x - (t³/6)*L³*x + (t⁴/24)*L⁴*x
        Parallel.For(0, numVertices * inputChannels, i =>
        {
            double val = numOps.ToDouble(vertexData[i]);
            val -= t * numOps.ToDouble(Lx[i]);
            val += t2 * numOps.ToDouble(L2x[i]);
            val -= t3 * numOps.ToDouble(L3x[i]);
            val += t4 * numOps.ToDouble(L4x[i]);
            diffused[i] = numOps.FromDouble(val);
        });

        // Step 2: Apply linear transformation
        var output = new T[numVertices * outputChannels];

        Parallel.For(0, numVertices, v =>
        {
            for (int oc = 0; oc < outputChannels; oc++)
            {
                T sum = biasData[oc];

                for (int ic = 0; ic < inputChannels; ic++)
                {
                    sum = numOps.Add(sum, numOps.Multiply(
                        diffused[v * inputChannels + ic],
                        weightsData[oc * inputChannels + ic]));
                }

                output[v * outputChannels + oc] = sum;
            }
        });

        return new Tensor<T>(output, [numVertices, outputChannels]);
    }

    /// <summary>
    /// Extracts sparse structure from a dense Laplacian matrix.
    /// The sparse structure stores only non-zero entries for each row.
    /// </summary>
    /// <returns>Array of neighbor lists, where each list contains (index, weight) tuples.</returns>
    private static List<(int index, double weight)>[] ExtractSparseStructure<T>(
        T[] laplacian, int numVertices, INumericOperations<T> numOps)
    {
        var neighbors = new List<(int index, double weight)>[numVertices];

        Parallel.For(0, numVertices, v =>
        {
            var vertexNeighbors = new List<(int index, double weight)>();
            for (int j = 0; j < numVertices; j++)
            {
                double w = numOps.ToDouble(laplacian[v * numVertices + j]);
                if (Math.Abs(w) > 1e-15)
                {
                    vertexNeighbors.Add((j, w));
                }
            }
            neighbors[v] = vertexNeighbors;
        });

        return neighbors;
    }

    /// <summary>
    /// Computes sparse matrix-vector product using pre-computed sparse structure.
    /// Complexity is O(E*C) where E is edges and C is channels, instead of O(V²*C).
    /// </summary>
    private static void ComputeSparseLaplacianProduct<T>(
        List<(int index, double weight)>[] sparseStructure,
        T[] input, T[] output,
        int numVertices, int channels,
        INumericOperations<T> numOps)
    {
        Parallel.For(0, numVertices, v =>
        {
            var vertexNeighbors = sparseStructure[v];

            for (int c = 0; c < channels; c++)
            {
                double sum = 0.0;

                foreach (var (j, w) in vertexNeighbors)
                {
                    sum += w * numOps.ToDouble(input[j * channels + c]);
                }

                output[v * channels + c] = numOps.FromDouble(sum);
            }
        });
    }

    /// <summary>
    /// Computes the product of Laplacian matrix with a feature matrix.
    /// Uses sparse-aware multiplication that skips zero entries for O(E) complexity
    /// instead of O(V²) where E is the number of edges.
    /// </summary>
    private static void ComputeLaplacianProduct<T>(
        T[] laplacian, T[] input, T[] output,
        int numVertices, int channels,
        INumericOperations<T> numOps)
    {
        // Extract sparse structure and compute product
        var sparseStructure = ExtractSparseStructure(laplacian, numVertices, numOps);
        ComputeSparseLaplacianProduct(sparseStructure, input, output, numVertices, channels, numOps);
    }

    /// <summary>
    /// Computes the backward pass for diffusion convolution.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="outputGradient">Gradient with respect to output [numVertices, outputChannels].</param>
    /// <param name="vertexFeatures">Input vertex features from forward pass.</param>
    /// <param name="laplacian">Mesh Laplacian matrix from forward pass.</param>
    /// <param name="weights">Diffusion weights from forward pass.</param>
    /// <param name="diffusionTime">Diffusion time from forward pass.</param>
    /// <returns>Tuple of (input gradient, weight gradient, bias gradient).</returns>
    public static (Tensor<T> inputGrad, Tensor<T> weightGrad, Tensor<T> biasGrad) DiffusionConvBackward<T>(
        Tensor<T> outputGradient,
        Tensor<T> vertexFeatures,
        Tensor<T> laplacian,
        Tensor<T> weights,
        T diffusionTime)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int numVertices = outputGradient.Shape[0];
        int outputChannels = outputGradient.Shape[1];
        int inputChannels = vertexFeatures.Shape[1];

        double t = numOps.ToDouble(diffusionTime);
        double t2 = t * t / 2.0;
        double t3 = t * t * t / 6.0;
        double t4 = t * t * t * t / 24.0;

        var gradData = outputGradient.ToArray();
        var vertexData = vertexFeatures.ToArray();
        var lapData = laplacian.ToArray();
        var weightsData = weights.ToArray();

        // Compute diffused features for weight gradient
        var Lx = new T[numVertices * inputChannels];
        var L2x = new T[numVertices * inputChannels];
        var L3x = new T[numVertices * inputChannels];
        var L4x = new T[numVertices * inputChannels];
        var diffused = new T[numVertices * inputChannels];

        ComputeLaplacianProduct(lapData, vertexData, Lx, numVertices, inputChannels, numOps);
        ComputeLaplacianProduct(lapData, Lx, L2x, numVertices, inputChannels, numOps);
        ComputeLaplacianProduct(lapData, L2x, L3x, numVertices, inputChannels, numOps);
        ComputeLaplacianProduct(lapData, L3x, L4x, numVertices, inputChannels, numOps);

        Parallel.For(0, numVertices * inputChannels, i =>
        {
            double val = numOps.ToDouble(vertexData[i]);
            val -= t * numOps.ToDouble(Lx[i]);
            val += t2 * numOps.ToDouble(L2x[i]);
            val -= t3 * numOps.ToDouble(L3x[i]);
            val += t4 * numOps.ToDouble(L4x[i]);
            diffused[i] = numOps.FromDouble(val);
        });

        // Compute bias gradient
        var biasGrad = new T[outputChannels];
        Parallel.For(0, outputChannels, oc =>
        {
            T sum = numOps.Zero;
            for (int v = 0; v < numVertices; v++)
            {
                sum = numOps.Add(sum, gradData[v * outputChannels + oc]);
            }
            biasGrad[oc] = sum;
        });

        // Compute weight gradient
        var weightGrad = new T[outputChannels * inputChannels];
        Parallel.For(0, outputChannels, oc =>
        {
            for (int ic = 0; ic < inputChannels; ic++)
            {
                T sum = numOps.Zero;
                for (int v = 0; v < numVertices; v++)
                {
                    sum = numOps.Add(sum, numOps.Multiply(
                        gradData[v * outputChannels + oc],
                        diffused[v * inputChannels + ic]));
                }
                weightGrad[oc * inputChannels + ic] = sum;
            }
        });

        // Compute input gradient: backprop through linear then through diffusion
        var linearGrad = new T[numVertices * inputChannels];
        Parallel.For(0, numVertices, v =>
        {
            for (int ic = 0; ic < inputChannels; ic++)
            {
                T sum = numOps.Zero;
                for (int oc = 0; oc < outputChannels; oc++)
                {
                    sum = numOps.Add(sum, numOps.Multiply(
                        gradData[v * outputChannels + oc],
                        weightsData[oc * inputChannels + ic]));
                }
                linearGrad[v * inputChannels + ic] = sum;
            }
        });

        // Backprop through diffusion (exp(-t*L) is symmetric for symmetric L)
        var inputGrad = new T[numVertices * inputChannels];
        var LTg = new T[numVertices * inputChannels];
        var L2Tg = new T[numVertices * inputChannels];
        var L3Tg = new T[numVertices * inputChannels];
        var L4Tg = new T[numVertices * inputChannels];

        // L^T = L for symmetric Laplacian
        ComputeLaplacianProduct(lapData, linearGrad, LTg, numVertices, inputChannels, numOps);
        ComputeLaplacianProduct(lapData, LTg, L2Tg, numVertices, inputChannels, numOps);
        ComputeLaplacianProduct(lapData, L2Tg, L3Tg, numVertices, inputChannels, numOps);
        ComputeLaplacianProduct(lapData, L3Tg, L4Tg, numVertices, inputChannels, numOps);

        Parallel.For(0, numVertices * inputChannels, i =>
        {
            double val = numOps.ToDouble(linearGrad[i]);
            val -= t * numOps.ToDouble(LTg[i]);
            val += t2 * numOps.ToDouble(L2Tg[i]);
            val -= t3 * numOps.ToDouble(L3Tg[i]);
            val += t4 * numOps.ToDouble(L4Tg[i]);
            inputGrad[i] = numOps.FromDouble(val);
        });

        return (
            new Tensor<T>(inputGrad, [numVertices, inputChannels]),
            new Tensor<T>(weightGrad, [outputChannels, inputChannels]),
            new Tensor<T>(biasGrad, [outputChannels])
        );
    }

    #endregion

    #region Mesh Laplacian

    /// <summary>
    /// Computes the mesh Laplacian matrix from vertex positions and face indices.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="vertices">Vertex positions [numVertices, 3].</param>
    /// <param name="faces">Face indices [numFaces, 3] for triangular mesh.</param>
    /// <param name="laplacianType">Type of Laplacian operator to compute.</param>
    /// <returns>Laplacian matrix [numVertices, numVertices].</returns>
    public static Tensor<T> ComputeMeshLaplacian<T>(
        Tensor<T> vertices,
        Tensor<int> faces,
        LaplacianType laplacianType = LaplacianType.Cotangent)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int numVertices = vertices.Shape[0];
        int numFaces = faces.Shape[0];

        var vertexData = vertices.ToArray();
        var faceData = faces.ToArray();
        var laplacian = new double[numVertices * numVertices];
        var lapLocks = new object[numVertices];
        for (int i = 0; i < numVertices; i++) lapLocks[i] = new object();

        bool useCotangent = laplacianType != LaplacianType.Uniform;

        // Process faces in parallel
        Parallel.For(0, numFaces, f =>
        {
            int v0 = faceData[f * 3 + 0];
            int v1 = faceData[f * 3 + 1];
            int v2 = faceData[f * 3 + 2];

            if (useCotangent)
            {
                // Get vertex positions
                double x0 = numOps.ToDouble(vertexData[v0 * 3 + 0]);
                double y0 = numOps.ToDouble(vertexData[v0 * 3 + 1]);
                double z0 = numOps.ToDouble(vertexData[v0 * 3 + 2]);
                double x1 = numOps.ToDouble(vertexData[v1 * 3 + 0]);
                double y1 = numOps.ToDouble(vertexData[v1 * 3 + 1]);
                double z1 = numOps.ToDouble(vertexData[v1 * 3 + 2]);
                double x2 = numOps.ToDouble(vertexData[v2 * 3 + 0]);
                double y2 = numOps.ToDouble(vertexData[v2 * 3 + 1]);
                double z2 = numOps.ToDouble(vertexData[v2 * 3 + 2]);

                // Compute cotangent weights
                double cot0 = ComputeCotangent(x1 - x0, y1 - y0, z1 - z0, x2 - x0, y2 - y0, z2 - z0);
                double cot1 = ComputeCotangent(x0 - x1, y0 - y1, z0 - z1, x2 - x1, y2 - y1, z2 - z1);
                double cot2 = ComputeCotangent(x0 - x2, y0 - y2, z0 - z2, x1 - x2, y1 - y2, z1 - z2);

                // Add cotangent weights (symmetric, negative off-diagonal)
                AddSymmetricWeight(laplacian, lapLocks, v1, v2, -0.5 * cot0, numVertices);
                AddSymmetricWeight(laplacian, lapLocks, v0, v2, -0.5 * cot1, numVertices);
                AddSymmetricWeight(laplacian, lapLocks, v0, v1, -0.5 * cot2, numVertices);
            }
            else
            {
                // Uniform Laplacian: just mark adjacency
                AddSymmetricWeight(laplacian, lapLocks, v0, v1, -1.0, numVertices);
                AddSymmetricWeight(laplacian, lapLocks, v1, v2, -1.0, numVertices);
                AddSymmetricWeight(laplacian, lapLocks, v2, v0, -1.0, numVertices);
            }
        });

        // Set diagonal to negative sum of row
        Parallel.For(0, numVertices, v =>
        {
            double sum = 0.0;
            for (int j = 0; j < numVertices; j++)
            {
                if (j != v)
                {
                    sum += laplacian[v * numVertices + j];
                }
            }
            laplacian[v * numVertices + v] = -sum;
        });

        // Normalize if requested
        if (laplacianType == LaplacianType.Normalized)
        {
            // Compute D^(-1/2) * L * D^(-1/2) where D is diagonal of -L
            var diagInvSqrt = new double[numVertices];
            for (int v = 0; v < numVertices; v++)
            {
                double d = -laplacian[v * numVertices + v];
                diagInvSqrt[v] = d > 1e-10 ? 1.0 / Math.Sqrt(d) : 0.0;
            }

            Parallel.For(0, numVertices, i =>
            {
                for (int j = 0; j < numVertices; j++)
                {
                    laplacian[i * numVertices + j] *= diagInvSqrt[i] * diagInvSqrt[j];
                }
            });
        }

        // Convert to tensor
        var result = new T[numVertices * numVertices];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = numOps.FromDouble(laplacian[i]);
        }

        return new Tensor<T>(result, [numVertices, numVertices]);
    }

    /// <summary>
    /// Computes the cotangent of the angle between two edge vectors.
    /// </summary>
    private static double ComputeCotangent(
        double e1x, double e1y, double e1z,
        double e2x, double e2y, double e2z)
    {
        double dot = e1x * e2x + e1y * e2y + e1z * e2z;
        double crossX = e1y * e2z - e1z * e2y;
        double crossY = e1z * e2x - e1x * e2z;
        double crossZ = e1x * e2y - e1y * e2x;
        double crossMag = Math.Sqrt(crossX * crossX + crossY * crossY + crossZ * crossZ);

        return crossMag > 1e-10 ? dot / crossMag : 0.0;
    }

    /// <summary>
    /// Adds a symmetric weight to the Laplacian matrix with thread-safe locking.
    /// </summary>
    private static void AddSymmetricWeight(
        double[] laplacian, object[] locks,
        int i, int j, double weight, int numVertices)
    {
        // Handle degenerate case where i == j (prevents self-deadlock)
        if (i == j)
        {
            lock (locks[i])
            {
                // For diagonal elements in a symmetric matrix update
                laplacian[i * numVertices + i] += weight;
            }
            return;
        }

        // Use consistent lock ordering (min before max) to prevent deadlock
        int minIdx = Math.Min(i, j);
        int maxIdx = Math.Max(i, j);

        lock (locks[minIdx])
        {
            lock (locks[maxIdx])
            {
                laplacian[i * numVertices + j] += weight;
                laplacian[j * numVertices + i] += weight;
            }
        }
    }

    #endregion

    #region Spiral Index Generation

    /// <summary>
    /// Generates spiral indices for mesh vertices based on connectivity.
    /// </summary>
    /// <typeparam name="T">The numeric type for vertex positions.</typeparam>
    /// <param name="vertices">Vertex positions [numVertices, 3].</param>
    /// <param name="faces">Face indices [numFaces, 3].</param>
    /// <param name="spiralLength">Number of neighbors in each spiral.</param>
    /// <returns>Spiral indices [numVertices, spiralLength].</returns>
    public static Tensor<int> GenerateSpiralIndices<T>(
        Tensor<T> vertices,
        Tensor<int> faces,
        int spiralLength)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int numVertices = vertices.Shape[0];
        int numFaces = faces.Shape[0];

        var vertexData = vertices.ToArray();
        var faceData = faces.ToArray();

        // Build adjacency list using HashSet for O(1) lookups during construction
        var adjacencySet = new HashSet<int>[numVertices];
        for (int i = 0; i < numVertices; i++)
        {
            adjacencySet[i] = new HashSet<int>();
        }

        for (int f = 0; f < numFaces; f++)
        {
            int v0 = faceData[f * 3 + 0];
            int v1 = faceData[f * 3 + 1];
            int v2 = faceData[f * 3 + 2];

            adjacencySet[v0].Add(v1);
            adjacencySet[v0].Add(v2);
            adjacencySet[v1].Add(v0);
            adjacencySet[v1].Add(v2);
            adjacencySet[v2].Add(v0);
            adjacencySet[v2].Add(v1);
        }

        // Convert to List for spiral generation (needs ordering)
        var adjacency = new List<int>[numVertices];
        for (int i = 0; i < numVertices; i++)
        {
            adjacency[i] = new List<int>(adjacencySet[i]);
        }

        var spiralIndices = new int[numVertices * spiralLength];

        // Generate spiral for each vertex
        Parallel.For(0, numVertices, v =>
        {
            var spiral = GenerateVertexSpiral(v, adjacency, vertexData, spiralLength, numOps);

            for (int s = 0; s < spiralLength; s++)
            {
                spiralIndices[v * spiralLength + s] = spiral[s];
            }
        });

        return new Tensor<int>(spiralIndices, [numVertices, spiralLength]);
    }

    /// <summary>
    /// Generates spiral ordering for a single vertex.
    /// </summary>
    private static int[] GenerateVertexSpiral<T>(
        int centerVertex,
        List<int>[] adjacency,
        T[] vertexData,
        int spiralLength,
        INumericOperations<T> numOps)
    {
        var spiral = new int[spiralLength];
        var visited = new HashSet<int> { centerVertex };
        var currentRing = new List<int>(adjacency[centerVertex]);
        int spiralIdx = 0;

        // Get center position
        double cx = numOps.ToDouble(vertexData[centerVertex * 3 + 0]);
        double cy = numOps.ToDouble(vertexData[centerVertex * 3 + 1]);
        double cz = numOps.ToDouble(vertexData[centerVertex * 3 + 2]);

        // Sort first ring by angle (spiral ordering)
        if (currentRing.Count > 0)
        {
            // Use first neighbor as reference
            int refNeighbor = currentRing[0];
            double rx = numOps.ToDouble(vertexData[refNeighbor * 3 + 0]) - cx;
            double ry = numOps.ToDouble(vertexData[refNeighbor * 3 + 1]) - cy;
            double rz = numOps.ToDouble(vertexData[refNeighbor * 3 + 2]) - cz;

            // Sort by angle around normal
            currentRing.Sort((a, b) =>
            {
                double ax = numOps.ToDouble(vertexData[a * 3 + 0]) - cx;
                double ay = numOps.ToDouble(vertexData[a * 3 + 1]) - cy;
                double az = numOps.ToDouble(vertexData[a * 3 + 2]) - cz;
                double bx = numOps.ToDouble(vertexData[b * 3 + 0]) - cx;
                double by = numOps.ToDouble(vertexData[b * 3 + 1]) - cy;
                double bz = numOps.ToDouble(vertexData[b * 3 + 2]) - cz;

                double angleA = Math.Atan2(
                    ax * ry - ay * rx,
                    ax * rx + ay * ry + az * rz);
                double angleB = Math.Atan2(
                    bx * ry - by * rx,
                    bx * rx + by * ry + bz * rz);

                return angleA.CompareTo(angleB);
            });
        }

        // Expand spiral outward
        while (spiralIdx < spiralLength && currentRing.Count > 0)
        {
            var nextRing = new List<int>();
            var nextRingSet = new HashSet<int>();  // O(1) lookup for deduplication

            foreach (int neighbor in currentRing)
            {
                if (spiralIdx >= spiralLength) break;
                if (visited.Contains(neighbor)) continue;

                spiral[spiralIdx++] = neighbor;
                visited.Add(neighbor);

                // Add neighbor's neighbors to next ring
                foreach (int nn in adjacency[neighbor])
                {
                    if (!visited.Contains(nn) && !nextRingSet.Contains(nn))
                    {
                        nextRing.Add(nn);
                        nextRingSet.Add(nn);
                    }
                }
            }

            currentRing = nextRing;
        }

        // Fill remaining with -1 (invalid/padding)
        for (; spiralIdx < spiralLength; spiralIdx++)
        {
            spiral[spiralIdx] = -1;
        }

        return spiral;
    }

    #endregion
}
