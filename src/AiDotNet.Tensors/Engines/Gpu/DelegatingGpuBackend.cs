using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Base class that delegates all IDirectGpuBackend calls to an inner backend.
/// Subclasses can override specific methods to add behavior (e.g., recording, profiling).
/// </summary>
/// <remarks>
/// This implements the Decorator pattern, allowing transparent interception of GPU operations
/// without modifying the underlying backend implementations.
/// </remarks>
public class DelegatingGpuBackend : IDirectGpuBackend
{
    /// <summary>
    /// The inner backend that performs actual GPU operations.
    /// </summary>
    protected readonly IDirectGpuBackend Inner;

    /// <summary>
    /// Creates a new delegating backend wrapper.
    /// </summary>
    /// <param name="inner">The backend to delegate to.</param>
    public DelegatingGpuBackend(IDirectGpuBackend inner)
    {
        Inner = inner ?? throw new ArgumentNullException(nameof(inner));
    }

    #region Properties

    /// <inheritdoc/>
    public virtual bool IsAvailable => Inner.IsAvailable;

    /// <inheritdoc/>
    public virtual string BackendName => Inner.BackendName;

    /// <inheritdoc/>
    public virtual string DeviceName => Inner.DeviceName;

    /// <inheritdoc/>
    public virtual string DeviceVendor => Inner.DeviceVendor;

    /// <inheritdoc/>
    public virtual int ComputeUnits => Inner.ComputeUnits;

    /// <inheritdoc/>
    public virtual long GlobalMemoryBytes => Inner.GlobalMemoryBytes;

    /// <inheritdoc/>
    public virtual long LocalMemoryBytes => Inner.LocalMemoryBytes;

    #endregion

    #region Memory Management

    /// <inheritdoc/>
    public virtual IGpuBuffer AllocateBuffer(float[] data) => Inner.AllocateBuffer(data);

    /// <inheritdoc/>
    public virtual IGpuBuffer AllocateBuffer(int size) => Inner.AllocateBuffer(size);

    /// <inheritdoc/>
    public virtual float[] DownloadBuffer(IGpuBuffer buffer) => Inner.DownloadBuffer(buffer);

    /// <inheritdoc/>
    public virtual void DownloadBuffer(IGpuBuffer buffer, float[] destination) => Inner.DownloadBuffer(buffer, destination);

    #endregion

    #region GEMM Operations

    /// <inheritdoc/>
    public virtual void Gemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
        => Inner.Gemm(A, B, C, M, N, K, alpha, beta);

    /// <inheritdoc/>
    public virtual IGpuBuffer MatMul(IGpuBuffer A, IGpuBuffer B, int M, int N, int K)
        => Inner.MatMul(A, B, M, N, K);

    /// <inheritdoc/>
    public virtual void BatchedGemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, int batchCount, float alpha = 1.0f, float beta = 0.0f)
        => Inner.BatchedGemm(A, B, C, M, N, K, batchCount, alpha, beta);

    #endregion

    #region Fused Operations

    /// <inheritdoc/>
    public virtual IGpuBuffer GemmBiasRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => Inner.GemmBiasRelu(A, B, bias, M, N, K);

    /// <inheritdoc/>
    public virtual IGpuBuffer GemmBiasGelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => Inner.GemmBiasGelu(A, B, bias, M, N, K);

    /// <inheritdoc/>
    public virtual IGpuBuffer GemmBiasSigmoid(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => Inner.GemmBiasSigmoid(A, B, bias, M, N, K);

    /// <inheritdoc/>
    public virtual IGpuBuffer GemmBiasTanh(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => Inner.GemmBiasTanh(A, B, bias, M, N, K);

    /// <inheritdoc/>
    public virtual IGpuBuffer GemmBias(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => Inner.GemmBias(A, B, bias, M, N, K);

    #endregion

    #region Broadcast Operations

    /// <inheritdoc/>
    public virtual void BiasAdd(IGpuBuffer A, IGpuBuffer bias, IGpuBuffer C, int M, int N)
        => Inner.BiasAdd(A, bias, C, M, N);

    public virtual void Conv2DBiasAdd(IGpuBuffer output, IGpuBuffer bias, int batch, int channels, int spatialSize)
        => Inner.Conv2DBiasAdd(output, bias, batch, channels, spatialSize);

    #endregion

    #region Element-wise Operations

    /// <inheritdoc/>
    public virtual void Add(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => Inner.Add(A, B, C, size);

    /// <inheritdoc/>
    public virtual void Subtract(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => Inner.Subtract(A, B, C, size);

    /// <inheritdoc/>
    public virtual void Multiply(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => Inner.Multiply(A, B, C, size);

    /// <inheritdoc/>
    public virtual void Divide(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => Inner.Divide(A, B, C, size);

    /// <inheritdoc/>
    public virtual void Min(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => Inner.Min(A, B, C, size);

    /// <inheritdoc/>
    public virtual void Max(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => Inner.Max(A, B, C, size);

    /// <inheritdoc/>
    public virtual void Scale(IGpuBuffer A, IGpuBuffer B, float scalar, int size) => Inner.Scale(A, B, scalar, size);

    /// <inheritdoc/>
    public virtual void Power(IGpuBuffer A, IGpuBuffer B, float exponent, int size) => Inner.Power(A, B, exponent, size);

    /// <inheritdoc/>
    public virtual void Abs(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Abs(A, B, size);

    /// <inheritdoc/>
    public virtual void Exp(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Exp(A, B, size);

    /// <inheritdoc/>
    public virtual void Exp2(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Exp2(A, B, size);

    /// <inheritdoc/>
    public virtual void Exp10(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Exp10(A, B, size);

    /// <inheritdoc/>
    public virtual void ExpM1(IGpuBuffer A, IGpuBuffer B, int size) => Inner.ExpM1(A, B, size);

    /// <inheritdoc/>
    public virtual void Log(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Log(A, B, size);

    /// <inheritdoc/>
    public virtual void Log2(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Log2(A, B, size);

    /// <inheritdoc/>
    public virtual void Log1P(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Log1P(A, B, size);

    /// <inheritdoc/>
    public virtual void Sqrt(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Sqrt(A, B, size);

    /// <inheritdoc/>
    public virtual void Sign(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Sign(A, B, size);

    /// <inheritdoc/>
    public virtual void Relu(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Relu(A, B, size);

    /// <inheritdoc/>
    public virtual void Sigmoid(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Sigmoid(A, B, size);

    /// <inheritdoc/>
    public virtual void Tanh(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Tanh(A, B, size);

    /// <inheritdoc/>
    public virtual void Gelu(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Gelu(A, B, size);

    /// <inheritdoc/>
    public virtual void Softmax(IGpuBuffer A, IGpuBuffer B, int batchSize, int features) => Inner.Softmax(A, B, batchSize, features);

    /// <inheritdoc/>
    public virtual void Squash(IGpuBuffer input, IGpuBuffer output, int numCapsules, int capsuleDim, float epsilon)
        => Inner.Squash(input, output, numCapsules, capsuleDim, epsilon);

    /// <inheritdoc/>
    public virtual void SquashBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int numCapsules, int capsuleDim, float epsilon)
        => Inner.SquashBackward(gradOutput, input, gradInput, numCapsules, capsuleDim, epsilon);

    /// <inheritdoc/>
    public virtual void CapsulePredictions(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int batchSize, int inputCapsules, int inputDim, int outputCapsules, int outputDim)
        => Inner.CapsulePredictions(input, weights, output, batchSize, inputCapsules, inputDim, outputCapsules, outputDim);

    /// <inheritdoc/>
    public virtual void CapsuleTransform(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer output,
        int batchSize, int inputCapsules, int inputDim, int numCapsules, int capsuleDim)
        => Inner.CapsuleTransform(input, weights, output, batchSize, inputCapsules, inputDim, numCapsules, capsuleDim);

    /// <inheritdoc/>
    public virtual void CapsuleWeightedSum(IGpuBuffer coupling, IGpuBuffer predictions, IGpuBuffer output,
        int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
        => Inner.CapsuleWeightedSum(coupling, predictions, output, batchSize, inputCapsules, outputCapsules, capsuleDim);

    /// <inheritdoc/>
    public virtual void CapsuleAgreement(IGpuBuffer predictions, IGpuBuffer output, IGpuBuffer agreement,
        int batchSize, int inputCapsules, int outputCapsules, int capsuleDim)
        => Inner.CapsuleAgreement(predictions, output, agreement, batchSize, inputCapsules, outputCapsules, capsuleDim);

    /// <inheritdoc/>
    public virtual void TileBatch(IGpuBuffer input, IGpuBuffer output, int repeats, int innerSize)
        => Inner.TileBatch(input, output, repeats, innerSize);

    /// <inheritdoc/>
    public virtual void TileAxis(IGpuBuffer input, IGpuBuffer output, int outerSize, int axisSize, int innerSize, int repeats)
        => Inner.TileAxis(input, output, outerSize, axisSize, innerSize, repeats);

    #endregion

    #region Trigonometric Operations

    /// <inheritdoc/>
    public virtual void Sin(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Sin(A, B, size);

    /// <inheritdoc/>
    public virtual void Cos(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Cos(A, B, size);

    /// <inheritdoc/>
    public virtual void Tan(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Tan(A, B, size);

    /// <inheritdoc/>
    public virtual void Asin(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Asin(A, B, size);

    /// <inheritdoc/>
    public virtual void Acos(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Acos(A, B, size);

    /// <inheritdoc/>
    public virtual void Atan(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Atan(A, B, size);

    #endregion

    #region Hyperbolic Operations

    /// <inheritdoc/>
    public virtual void Sinh(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Sinh(A, B, size);

    /// <inheritdoc/>
    public virtual void Cosh(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Cosh(A, B, size);

    /// <inheritdoc/>
    public virtual void Asinh(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Asinh(A, B, size);

    /// <inheritdoc/>
    public virtual void Acosh(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Acosh(A, B, size);

    /// <inheritdoc/>
    public virtual void Atanh(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Atanh(A, B, size);

    #endregion

    #region Additional Unary Operations

    /// <inheritdoc/>
    public virtual void Reciprocal(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Reciprocal(A, B, size);

    /// <inheritdoc/>
    public virtual void Cbrt(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Cbrt(A, B, size);

    /// <inheritdoc/>
    public virtual void Log10(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Log10(A, B, size);

    /// <inheritdoc/>
    public virtual void Negate(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Negate(A, B, size);

    /// <inheritdoc/>
    public virtual void Floor(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Floor(A, B, size);

    /// <inheritdoc/>
    public virtual void Ceiling(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Ceiling(A, B, size);

    /// <inheritdoc/>
    public virtual void Round(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Round(A, B, size);

    /// <inheritdoc/>
    public virtual void Truncate(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Truncate(A, B, size);

    #endregion

    #region Sparse Operations

    /// <inheritdoc/>
    public virtual void Enforce2x4Sparsity(IGpuBuffer denseInput, IGpuBuffer sparseValues, IGpuBuffer sparseIndices, int M, int K)
        => Inner.Enforce2x4Sparsity(denseInput, sparseValues, sparseIndices, M, K);

    /// <inheritdoc/>
    public virtual void Decompress2x4Sparse(IGpuBuffer sparseValues, IGpuBuffer sparseIndices, IGpuBuffer denseOutput, int M, int K)
        => Inner.Decompress2x4Sparse(sparseValues, sparseIndices, denseOutput, M, K);

    /// <inheritdoc/>
    public virtual void SparseGemm(IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices, IGpuBuffer B, IGpuBuffer C, int M, int N, int K, float alpha = 1.0f, float beta = 0.0f)
        => Inner.SparseGemm(sparseAValues, sparseAIndices, B, C, M, N, K, alpha, beta);

    /// <inheritdoc/>
    public virtual IGpuBuffer SparseGemmBiasRelu(IGpuBuffer sparseAValues, IGpuBuffer sparseAIndices, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => Inner.SparseGemmBiasRelu(sparseAValues, sparseAIndices, B, bias, M, N, K);

    /// <inheritdoc/>
    public virtual void CsrSpMM(IGpuBuffer csrValues, IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers, IGpuBuffer denseB, IGpuBuffer output, int M, int K, int N, int nnz)
        => Inner.CsrSpMM(csrValues, csrColIndices, csrRowPointers, denseB, output, M, K, N, nnz);

    /// <inheritdoc/>
    public virtual void CsrSpMMBias(IGpuBuffer csrValues, IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers, IGpuBuffer denseB, IGpuBuffer bias, IGpuBuffer output, int M, int K, int N, int nnz)
        => Inner.CsrSpMMBias(csrValues, csrColIndices, csrRowPointers, denseB, bias, output, M, K, N, nnz);

    /// <inheritdoc/>
    public virtual void ScatterAddEdges(IGpuBuffer input, IGpuBuffer sourceIndices, IGpuBuffer targetIndices, IGpuBuffer? edgeValues, IGpuBuffer output, int numNodes, int numEdges, int features)
        => Inner.ScatterAddEdges(input, sourceIndices, targetIndices, edgeValues, output, numNodes, numEdges, features);

    /// <inheritdoc/>
    public virtual IGpuBuffer AllocateByteBuffer(int size) => Inner.AllocateByteBuffer(size);

    /// <inheritdoc/>
    public virtual void CsrSegmentedMax(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer input, IGpuBuffer output, int M, int K, int N)
        => Inner.CsrSegmentedMax(csrColIndices, csrRowPointers, input, output, M, K, N);

    /// <inheritdoc/>
    public virtual void CsrSegmentedMin(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer input, IGpuBuffer output, int M, int K, int N)
        => Inner.CsrSegmentedMin(csrColIndices, csrRowPointers, input, output, M, K, N);

    /// <inheritdoc/>
    public virtual void CsrSegmentedStdDev(IGpuBuffer csrColIndices, IGpuBuffer csrRowPointers,
        IGpuBuffer input, IGpuBuffer output, int M, int K, int N, float epsilon = 1e-8f)
        => Inner.CsrSegmentedStdDev(csrColIndices, csrRowPointers, input, output, M, K, N, epsilon);

    #endregion

    #region Reduction Operations

    /// <inheritdoc/>
    public virtual float Sum(IGpuBuffer A, int size) => Inner.Sum(A, size);

    /// <inheritdoc/>
    public virtual float Max(IGpuBuffer A, int size) => Inner.Max(A, size);

    /// <inheritdoc/>
    public virtual void SumAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
        => Inner.SumAxis(A, B, outerSize, reduceSize);

    #endregion

    #region Synchronization

    /// <inheritdoc/>
    public virtual void Synchronize() => Inner.Synchronize();

    #endregion

    #region Convolution Operations

    /// <inheritdoc/>
    public virtual void Conv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
        => Inner.Conv2D(input, kernel, output, batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);

    /// <inheritdoc/>
    public virtual void Conv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
        => Inner.Conv2DBackwardInput(gradOutput, kernel, gradInput, batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);

    /// <inheritdoc/>
    public virtual void Conv2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
        => Inner.Conv2DBackwardKernel(input, gradOutput, gradKernel, batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);

    /// <inheritdoc/>
    public virtual void Conv3D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inDepth, int inHeight, int inWidth,
        int outChannels, int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW,
        int padD, int padH, int padW,
        int dilationD, int dilationH, int dilationW)
        => Inner.Conv3D(input, kernel, output, batch, inChannels, inDepth, inHeight, inWidth,
            outChannels, outDepth, outHeight, outWidth, kernelD, kernelH, kernelW,
            strideD, strideH, strideW, padD, padH, padW, dilationD, dilationH, dilationW);

    /// <inheritdoc/>
    public virtual void DepthwiseConv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
        => Inner.DepthwiseConv2D(input, kernel, output, batch, channels, inHeight, inWidth,
            outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW);

    /// <inheritdoc/>
    public virtual void ConvTranspose2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
        => Inner.ConvTranspose2D(input, kernel, output, batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, outputPadH, outputPadW);

    /// <inheritdoc/>
    public virtual void LocallyConnectedConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer? bias, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
        => Inner.LocallyConnectedConv2D(input, weights, bias, output, batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW);

    /// <inheritdoc/>
    public virtual void LocallyConnectedConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
        => Inner.LocallyConnectedConv2DBackwardInput(gradOutput, weights, gradInput, batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW);

    /// <inheritdoc/>
    public virtual void LocallyConnectedConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
        => Inner.LocallyConnectedConv2DBackwardWeights(gradOutput, input, gradWeights, batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW);

    /// <inheritdoc/>
    public virtual void LocallyConnectedConv2DBackwardBias(IGpuBuffer gradOutput, IGpuBuffer gradBias,
        int batch, int outChannels, int outHeight, int outWidth)
        => Inner.LocallyConnectedConv2DBackwardBias(gradOutput, gradBias, batch, outChannels, outHeight, outWidth);

    /// <inheritdoc/>
    public virtual void DeformableConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
        => Inner.DeformableConv2D(input, weights, offsets, mask, output, batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW,
            dilationH, dilationW, groups, deformGroups);

    /// <inheritdoc/>
    public virtual void DeformableConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
        => Inner.DeformableConv2DBackwardInput(gradOutput, weights, offsets, mask, gradInput, batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW,
            dilationH, dilationW, groups, deformGroups);

    /// <inheritdoc/>
    public virtual void DeformableConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
        => Inner.DeformableConv2DBackwardWeights(gradOutput, input, offsets, mask, gradWeights, batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW,
            dilationH, dilationW, groups, deformGroups);

    /// <inheritdoc/>
    public virtual void DeformableConv2DBackwardOffset(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradOffsets,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
        => Inner.DeformableConv2DBackwardOffset(gradOutput, input, weights, offsets, mask, gradOffsets, batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW,
            dilationH, dilationW, groups, deformGroups);

    /// <inheritdoc/>
    public virtual void DeformableConv2DBackwardMask(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer gradMask,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
        => Inner.DeformableConv2DBackwardMask(gradOutput, input, weights, offsets, gradMask, batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW,
            dilationH, dilationW, groups, deformGroups);

    #endregion

    #region Pooling Operations

    /// <inheritdoc/>
    public virtual void MaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
        => Inner.MaxPool2D(input, output, indices, batch, channels, inHeight, inWidth,
            outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW);

    /// <inheritdoc/>
    public virtual void MaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
        => Inner.MaxPool2DBackward(gradOutput, indices, gradInput, batch, channels, inHeight, inWidth,
            outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW);

    /// <inheritdoc/>
    public virtual void AvgPool2D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad)
        => Inner.AvgPool2D(input, output, batch, channels, inHeight, inWidth,
            outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, countIncludePad);

    /// <inheritdoc/>
    public virtual void AvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad)
        => Inner.AvgPool2DBackward(gradOutput, gradInput, batch, channels, inHeight, inWidth,
            outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, countIncludePad);

    /// <inheritdoc/>
    public virtual void GlobalAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
        => Inner.GlobalAvgPool2D(input, output, batch, channels, height, width);

    /// <inheritdoc/>
    public virtual void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
        => Inner.GlobalMaxPool2D(input, output, batch, channels, height, width);

    /// <inheritdoc/>
    public virtual void AdaptiveAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth)
        => Inner.AdaptiveAvgPool2D(input, output, batch, channels, inHeight, inWidth, outHeight, outWidth);

    /// <inheritdoc/>
    public virtual void MaxPool3D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW)
        => Inner.MaxPool3D(input, output, indices, batch, channels, inDepth, inHeight, inWidth,
            outDepth, outHeight, outWidth, kernelD, kernelH, kernelW, strideD, strideH, strideW);

    /// <inheritdoc/>
    public virtual void MaxPool3DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth)
        => Inner.MaxPool3DBackward(gradOutput, indices, gradInput, batch, channels,
            inDepth, inHeight, inWidth, outDepth, outHeight, outWidth);

    /// <inheritdoc/>
    public virtual void NearestNeighborUpsample3D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW)
        => Inner.NearestNeighborUpsample3D(input, output, batch, channels, inDepth, inHeight, inWidth, scaleD, scaleH, scaleW);

    /// <inheritdoc/>
    public virtual void NearestNeighborUpsample3DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW)
        => Inner.NearestNeighborUpsample3DBackward(gradOutput, gradInput, batch, channels, inDepth, inHeight, inWidth, scaleD, scaleH, scaleW);

    #endregion

    #region Spatial Transformer Operations

    /// <inheritdoc/>
    public virtual void AffineGrid(IGpuBuffer theta, IGpuBuffer grid, int batch, int outputHeight, int outputWidth)
        => Inner.AffineGrid(theta, grid, batch, outputHeight, outputWidth);

    /// <inheritdoc/>
    public virtual void GridSample(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode = 0, bool alignCorners = false)
        => Inner.GridSample(input, grid, output, batch, channels, inHeight, inWidth, outHeight, outWidth, paddingMode, alignCorners);

    /// <inheritdoc/>
    public virtual void GridSampleBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer grid,
        IGpuBuffer gradInput, IGpuBuffer gradGrid,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode = 0, bool alignCorners = false)
        => Inner.GridSampleBackward(gradOutput, input, grid, gradInput, gradGrid, batch, channels, inHeight, inWidth, outHeight, outWidth, paddingMode, alignCorners);

    #endregion

    #region Normalization Operations

    /// <inheritdoc/>
    public virtual void BatchNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar, IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        int batch, int channels, int spatialSize, float epsilon, float momentum, bool training)
        => Inner.BatchNorm(input, output, gamma, beta, runningMean, runningVar, saveMean, saveInvVar,
            batch, channels, spatialSize, epsilon, momentum, training);

    /// <inheritdoc/>
    public virtual void BatchNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
        => Inner.BatchNormBackward(gradOutput, input, gamma, saveMean, saveInvVar, gradInput, gradGamma, gradBeta,
            batch, channels, spatialSize, epsilon);

    /// <inheritdoc/>
    public virtual void LayerNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batchSize, int normalizedSize, float epsilon)
        => Inner.LayerNorm(input, output, gamma, beta, saveMean, saveInvVar, batchSize, normalizedSize, epsilon);

    /// <inheritdoc/>
    public virtual void LayerNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batchSize, int normalizedSize, float epsilon)
        => Inner.LayerNormBackward(gradOutput, input, gamma, saveMean, saveInvVar, gradInput, gradGamma, gradBeta,
            batchSize, normalizedSize, epsilon);

    /// <inheritdoc/>
    public virtual void GroupNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int numGroups, int channels, int spatialSize, float epsilon)
        => Inner.GroupNorm(input, output, gamma, beta, saveMean, saveInvVar, batch, numGroups, channels, spatialSize, epsilon);

    /// <inheritdoc/>
    public virtual void InstanceNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int channels, int spatialSize, float epsilon)
        => Inner.InstanceNorm(input, output, gamma, beta, saveMean, saveInvVar, batch, channels, spatialSize, epsilon);

    /// <inheritdoc/>
    public virtual void RmsNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer saveRms,
        int batchSize, int normalizedSize, float epsilon)
        => Inner.RmsNorm(input, output, gamma, saveRms, batchSize, normalizedSize, epsilon);

    /// <inheritdoc/>
    public virtual void RmsNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveRms,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, int batchSize, int normalizedSize, float epsilon)
        => Inner.RmsNormBackward(gradOutput, input, gamma, saveRms, gradInput, gradGamma, batchSize, normalizedSize, epsilon);

    #endregion

    #region Dropout and Regularization

    /// <inheritdoc/>
    public virtual void Dropout(IGpuBuffer input, IGpuBuffer output, IGpuBuffer mask, int size, float dropoutRate, ulong seed, bool training)
        => Inner.Dropout(input, output, mask, size, dropoutRate, seed, training);

    /// <inheritdoc/>
    public virtual void DropoutBackward(IGpuBuffer gradOutput, IGpuBuffer mask, IGpuBuffer gradInput, int size, float dropoutRate)
        => Inner.DropoutBackward(gradOutput, mask, gradInput, size, dropoutRate);

    #endregion

    #region Embedding Operations

    /// <inheritdoc/>
    public virtual void Embedding(IGpuBuffer indices, IGpuBuffer embeddingTable, IGpuBuffer output, int numIndices, int embeddingDim)
        => Inner.Embedding(indices, embeddingTable, output, numIndices, embeddingDim);

    /// <inheritdoc/>
    public virtual void EmbeddingBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradEmbedding, int numIndices, int embeddingDim, int vocabSize)
        => Inner.EmbeddingBackward(gradOutput, indices, gradEmbedding, numIndices, embeddingDim, vocabSize);

    /// <inheritdoc/>
    public virtual IGpuBuffer AllocateIntBuffer(int size) => Inner.AllocateIntBuffer(size);

    /// <inheritdoc/>
    public virtual IGpuBuffer AllocateIntBuffer(int[] data) => Inner.AllocateIntBuffer(data);

    #endregion

    #region Attention Operations

    /// <inheritdoc/>
    public virtual void ScaledDotProductAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights, IGpuBuffer? mask,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
        => Inner.ScaledDotProductAttention(query, key, value, output, attentionWeights, mask,
            batch, numHeads, seqLen, headDim, scale, isCausal);

    /// <inheritdoc/>
    public virtual void ScaledDotProductAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights, IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
        => Inner.ScaledDotProductAttentionBackward(gradOutput, query, key, value, attentionWeights,
            gradQuery, gradKey, gradValue, batch, numHeads, seqLen, headDim, scale, isCausal);

    /// <inheritdoc/>
    public virtual void FlashAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? mask, int batch, int numHeads, int seqLen, int headDim, float scale, bool isCausal)
        => Inner.FlashAttention(query, key, value, output, mask, batch, numHeads, seqLen, headDim, scale, isCausal);

    /// <inheritdoc/>
    public virtual void FlashAttentionV2(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
        => Inner.FlashAttentionV2(query, key, value, output, softmaxStats, batch, numHeads, seqQ, seqK, headDim, scale, isCausal);

    /// <inheritdoc/>
    public virtual void FlashAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer softmaxStats,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
        => Inner.FlashAttentionBackward(gradOutput, query, key, value, output, softmaxStats,
            gradQuery, gradKey, gradValue, batch, numHeads, seqQ, seqK, headDim, scale, isCausal);

    /// <inheritdoc/>
    public virtual void GroupedQueryAttention(IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer output, IGpuBuffer? attentionWeights,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale, bool isCausal)
        => Inner.GroupedQueryAttention(query, key, value, output, attentionWeights,
            batch, numQHeads, numKVHeads, seqQ, seqK, headDim, scale, isCausal);

    /// <inheritdoc/>
    public virtual void GroupedQueryAttentionBackward(IGpuBuffer gradOutput, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value,
        IGpuBuffer attentionWeights,
        IGpuBuffer gradQuery, IGpuBuffer gradKey, IGpuBuffer gradValue,
        int batch, int numQHeads, int numKVHeads, int seqQ, int seqK, int headDim, float scale)
        => Inner.GroupedQueryAttentionBackward(gradOutput, query, key, value, attentionWeights,
            gradQuery, gradKey, gradValue, batch, numQHeads, numKVHeads, seqQ, seqK, headDim, scale);

    #endregion

    #region Transpose and Reshape

    /// <inheritdoc/>
    public virtual void Transpose(IGpuBuffer A, IGpuBuffer B, int rows, int cols) => Inner.Transpose(A, B, rows, cols);

    /// <inheritdoc/>
    public virtual void BatchedTranspose(IGpuBuffer A, IGpuBuffer B, int batch, int rows, int cols)
        => Inner.BatchedTranspose(A, B, batch, rows, cols);

    /// <inheritdoc/>
    public virtual void Permute(IGpuBuffer input, IGpuBuffer output, int[] shape, int[] permutation)
        => Inner.Permute(input, output, shape, permutation);

    /// <inheritdoc/>
    public virtual void Copy(IGpuBuffer source, IGpuBuffer destination, int size) => Inner.Copy(source, destination, size);

    /// <inheritdoc/>
    public virtual void Copy(IGpuBuffer source, int sourceOffset, IGpuBuffer destination, int destinationOffset, int length)
        => Inner.Copy(source, sourceOffset, destination, destinationOffset, length);

    /// <inheritdoc/>
    public virtual void Copy2DStrided(IGpuBuffer source, IGpuBuffer destination, int numRows,
        int srcCols, int destTotalCols, int destColOffset)
        => Inner.Copy2DStrided(source, destination, numRows, srcCols, destTotalCols, destColOffset);

    /// <inheritdoc/>
    public virtual void NearestNeighborUpsample(IGpuBuffer input, IGpuBuffer output, int batchChannels, int height, int width, int scaleFactor)
        => Inner.NearestNeighborUpsample(input, output, batchChannels, height, width, scaleFactor);

    /// <inheritdoc/>
    public virtual void Fill(IGpuBuffer buffer, float value, int size) => Inner.Fill(buffer, value, size);

    #endregion

    #region Random Number Generation

    /// <inheritdoc/>
    public virtual void GenerateRandomUniform(IGpuBuffer output, int size, float min, float max, ulong seed)
        => Inner.GenerateRandomUniform(output, size, min, max, seed);

    /// <inheritdoc/>
    public virtual void GenerateRandomNormal(IGpuBuffer output, int size, float mean, float stdDev, ulong seed)
        => Inner.GenerateRandomNormal(output, size, mean, stdDev, seed);

    #endregion

    #region Specialized Layer Operations

    /// <inheritdoc/>
    public virtual void RbfForward(IGpuBuffer input, IGpuBuffer centers, IGpuBuffer epsilons, IGpuBuffer output, int batchSize, int numCenters, int inputDim)
        => Inner.RbfForward(input, centers, epsilons, output, batchSize, numCenters, inputDim);

    /// <inheritdoc/>
    public virtual void StdpUpdate(IGpuBuffer weights, IGpuBuffer preTrace, IGpuBuffer postTrace, IGpuBuffer preSpike, IGpuBuffer postSpike,
        float ltpRate, float ltdRate, float homeostasisRate, float minWeight, float maxWeight, int numPre, int numPost)
        => Inner.StdpUpdate(weights, preTrace, postTrace, preSpike, postSpike, ltpRate, ltdRate, homeostasisRate, minWeight, maxWeight, numPre, numPost);

    /// <inheritdoc/>
    public virtual void UpdateTraces(IGpuBuffer traces, IGpuBuffer spikes, IGpuBuffer input, float decay, float threshold, int size)
        => Inner.UpdateTraces(traces, spikes, input, decay, threshold, size);

    #endregion

    #region Hyperbolic Geometry Operations

    /// <inheritdoc/>
    public virtual void PoincareProject(IGpuBuffer input, IGpuBuffer output, int batchSize, int dim, float curvature, float epsilon = 1e-5f)
        => Inner.PoincareProject(input, output, batchSize, dim, curvature, epsilon);

    /// <inheritdoc/>
    public virtual void MobiusAdd(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batchSize, int dim, float curvature)
        => Inner.MobiusAdd(x, y, output, batchSize, dim, curvature);

    /// <inheritdoc/>
    public virtual void PoincareExpMap(IGpuBuffer basePoint, IGpuBuffer tangentVec, IGpuBuffer output, int batchSize, int dim, float curvature)
        => Inner.PoincareExpMap(basePoint, tangentVec, output, batchSize, dim, curvature);

    /// <inheritdoc/>
    public virtual void PoincareDistance(IGpuBuffer x, IGpuBuffer y, IGpuBuffer output, int batchSize, int dim, float curvature)
        => Inner.PoincareDistance(x, y, output, batchSize, dim, curvature);

    /// <inheritdoc/>
    public virtual void HyperbolicLinearForward(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures, float curvature, float epsilon)
        => Inner.HyperbolicLinearForward(input, weights, biases, output, batchSize, inputFeatures, outputFeatures, curvature, epsilon);

    #endregion

    #region Octonion Algebra Operations

    /// <inheritdoc/>
    public virtual void OctonionMultiply(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
        => Inner.OctonionMultiply(a, b, output, count);

    /// <inheritdoc/>
    public virtual void OctonionAdd(IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int count)
        => Inner.OctonionAdd(a, b, output, count);

    /// <inheritdoc/>
    public virtual void OctonionLinearForward(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer biases, IGpuBuffer output,
        int batchSize, int inputFeatures, int outputFeatures)
        => Inner.OctonionLinearForward(input, weights, biases, output, batchSize, inputFeatures, outputFeatures);

    #endregion

    #region Quantum Computing Operations

    /// <inheritdoc/>
    public virtual void QuantumMeasurement(IGpuBuffer realPart, IGpuBuffer imagPart, IGpuBuffer probabilities, int batchSize, int stateSize)
        => Inner.QuantumMeasurement(realPart, imagPart, probabilities, batchSize, stateSize);

    /// <inheritdoc/>
    public virtual void NormalizeProbabilities(IGpuBuffer probabilities, int batchSize, int stateSize)
        => Inner.NormalizeProbabilities(probabilities, batchSize, stateSize);

    /// <inheritdoc/>
    public virtual void ComplexMatVec(IGpuBuffer matReal, IGpuBuffer matImag, IGpuBuffer vecReal, IGpuBuffer vecImag,
        IGpuBuffer outReal, IGpuBuffer outImag, int batchSize, int dim)
        => Inner.ComplexMatVec(matReal, matImag, vecReal, vecImag, outReal, outImag, batchSize, dim);

    /// <inheritdoc/>
    public virtual void QuantumRotation(IGpuBuffer stateReal, IGpuBuffer stateImag, IGpuBuffer outReal, IGpuBuffer outImag,
        IGpuBuffer angles, int numQubits, int batchSize)
        => Inner.QuantumRotation(stateReal, stateImag, outReal, outImag, angles, numQubits, batchSize);

    /// <inheritdoc/>
    public virtual void MeasurementForward(IGpuBuffer input, IGpuBuffer output, int batchSize, int stateSize)
        => Inner.MeasurementForward(input, output, batchSize, stateSize);

    #endregion

    #region Activation Gradients

    /// <inheritdoc/>
    public virtual void ReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => Inner.ReluBackward(gradOutput, input, gradInput, size);

    /// <inheritdoc/>
    public virtual void SigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
        => Inner.SigmoidBackward(gradOutput, output, gradInput, size);

    /// <inheritdoc/>
    public virtual void TanhBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int size)
        => Inner.TanhBackward(gradOutput, output, gradInput, size);

    /// <inheritdoc/>
    public virtual void GeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => Inner.GeluBackward(gradOutput, input, gradInput, size);

    /// <inheritdoc/>
    public virtual void SoftmaxBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int batchSize, int features)
        => Inner.SoftmaxBackward(gradOutput, output, gradInput, batchSize, features);

    /// <inheritdoc/>
    public virtual void LeakyRelu(IGpuBuffer A, IGpuBuffer B, float alpha, int size) => Inner.LeakyRelu(A, B, alpha, size);

    /// <inheritdoc/>
    public virtual void LeakyReluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, int size)
        => Inner.LeakyReluBackward(gradOutput, input, gradInput, alpha, size);

    /// <inheritdoc/>
    public virtual void Elu(IGpuBuffer A, IGpuBuffer B, float alpha, int size) => Inner.Elu(A, B, alpha, size);

    /// <inheritdoc/>
    public virtual void EluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer output, IGpuBuffer gradInput, float alpha, int size)
        => Inner.EluBackward(gradOutput, input, output, gradInput, alpha, size);

    /// <inheritdoc/>
    public virtual void Swish(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Swish(A, B, size);

    /// <inheritdoc/>
    public virtual void SwishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => Inner.SwishBackward(gradOutput, input, gradInput, size);

    /// <inheritdoc/>
    public virtual void Silu(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Silu(A, B, size);

    /// <inheritdoc/>
    public virtual void Mish(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Mish(A, B, size);

    /// <inheritdoc/>
    public virtual void Softplus(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Softplus(A, B, size);

    /// <inheritdoc/>
    public virtual void Hardswish(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Hardswish(A, B, size);

    /// <inheritdoc/>
    public virtual void Selu(IGpuBuffer A, IGpuBuffer B, float alpha, float scale, int size) => Inner.Selu(A, B, alpha, scale, size);

    /// <inheritdoc/>
    public virtual void Hardsigmoid(IGpuBuffer A, IGpuBuffer B, int size) => Inner.Hardsigmoid(A, B, size);

    /// <inheritdoc/>
    public virtual void Hardtanh(IGpuBuffer A, IGpuBuffer B, float minVal, float maxVal, int size)
        => Inner.Hardtanh(A, B, minVal, maxVal, size);

    /// <inheritdoc/>
    public virtual void SiluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => Inner.SiluBackward(gradOutput, input, gradInput, size);

    /// <inheritdoc/>
    public virtual void MishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => Inner.MishBackward(gradOutput, input, gradInput, size);

    /// <inheritdoc/>
    public virtual void SoftplusBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => Inner.SoftplusBackward(gradOutput, input, gradInput, size);

    /// <inheritdoc/>
    public virtual void HardswishBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => Inner.HardswishBackward(gradOutput, input, gradInput, size);

    /// <inheritdoc/>
    public virtual void SeluBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float alpha, float scale, int size)
        => Inner.SeluBackward(gradOutput, input, gradInput, alpha, scale, size);

    /// <inheritdoc/>
    public virtual void HardsigmoidBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int size)
        => Inner.HardsigmoidBackward(gradOutput, input, gradInput, size);

    /// <inheritdoc/>
    public virtual void HardtanhBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, float minVal, float maxVal, int size)
        => Inner.HardtanhBackward(gradOutput, input, gradInput, minVal, maxVal, size);

    #endregion

    #region Loss Functions

    /// <inheritdoc/>
    public virtual float CrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int batchSize, int numClasses)
        => Inner.CrossEntropyLoss(predictions, targets, batchSize, numClasses);

    /// <inheritdoc/>
    public virtual void CrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int batchSize, int numClasses)
        => Inner.CrossEntropyBackward(predictions, targets, gradInput, batchSize, numClasses);

    /// <inheritdoc/>
    public virtual float BinaryCrossEntropyLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => Inner.BinaryCrossEntropyLoss(predictions, targets, size);

    /// <inheritdoc/>
    public virtual void BinaryCrossEntropyBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => Inner.BinaryCrossEntropyBackward(predictions, targets, gradInput, size);

    /// <inheritdoc/>
    public virtual float MseLoss(IGpuBuffer predictions, IGpuBuffer targets, int size)
        => Inner.MseLoss(predictions, targets, size);

    /// <inheritdoc/>
    public virtual void MseBackward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size)
        => Inner.MseBackward(predictions, targets, gradInput, size);

    /// <inheritdoc/>
    public virtual float SmoothL1Loss(IGpuBuffer predictions, IGpuBuffer targets, int size, float beta)
        => Inner.SmoothL1Loss(predictions, targets, size, beta);

    /// <inheritdoc/>
    public virtual void SmoothL1Backward(IGpuBuffer predictions, IGpuBuffer targets, IGpuBuffer gradInput, int size, float beta)
        => Inner.SmoothL1Backward(predictions, targets, gradInput, size, beta);

    #endregion

    #region Gradient Clipping and Utility

    /// <inheritdoc/>
    public virtual void Clamp(IGpuBuffer A, IGpuBuffer B, float min, float max, int size) => Inner.Clamp(A, B, min, max, size);

    /// <inheritdoc/>
    public virtual float L2Norm(IGpuBuffer A, int size) => Inner.L2Norm(A, size);

    /// <inheritdoc/>
    public virtual void ClipByValue(IGpuBuffer A, IGpuBuffer B, float clipValue, int size)
        => Inner.ClipByValue(A, B, clipValue, size);

    /// <inheritdoc/>
    public virtual void ClipByNorm(IGpuBuffer A, IGpuBuffer B, float maxNorm, int size)
        => Inner.ClipByNorm(A, B, maxNorm, size);

    /// <inheritdoc/>
    public virtual void Fma(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, IGpuBuffer D, int size)
        => Inner.Fma(A, B, C, D, size);

    /// <inheritdoc/>
    public virtual void ScatterAdd(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer destination, int sourceSize, int destSize)
        => Inner.ScatterAdd(source, indices, destination, sourceSize, destSize);

    /// <inheritdoc/>
    public virtual void ScatterAddBackward(IGpuBuffer gradDestination, IGpuBuffer indices, IGpuBuffer gradSource, int numIndices, int featureSize)
        => Inner.ScatterAddBackward(gradDestination, indices, gradSource, numIndices, featureSize);

    /// <inheritdoc/>
    public virtual void Gather(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, int numIndices, int featureSize)
        => Inner.Gather(source, indices, output, numIndices, featureSize);

    #endregion

    #region Comparison Operations

    /// <inheritdoc/>
    public virtual void GreaterThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => Inner.GreaterThan(A, B, C, size);

    /// <inheritdoc/>
    public virtual void LessThan(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => Inner.LessThan(A, B, C, size);

    /// <inheritdoc/>
    public virtual void Equal(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => Inner.Equal(A, B, C, size);

    /// <inheritdoc/>
    public virtual void Where(IGpuBuffer condition, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
        => Inner.Where(condition, A, B, C, size);

    /// <inheritdoc/>
    public virtual void NotEqualScalar(IGpuBuffer A, IGpuBuffer C, float scalar, int size)
        => Inner.NotEqualScalar(A, C, scalar, size);

    #endregion

    #region Statistics

    /// <inheritdoc/>
    public virtual void MeanAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
        => Inner.MeanAxis(A, B, outerSize, reduceSize);

    /// <inheritdoc/>
    public virtual void VarAxis(IGpuBuffer A, IGpuBuffer mean, IGpuBuffer variance, int outerSize, int reduceSize)
        => Inner.VarAxis(A, mean, variance, outerSize, reduceSize);

    /// <inheritdoc/>
    public virtual void ArgMax(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
        => Inner.ArgMax(A, indices, outerSize, reduceSize);

    /// <inheritdoc/>
    public virtual void ArgMaxAxis(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
        => Inner.ArgMaxAxis(A, indices, outerSize, reduceSize);

    /// <inheritdoc/>
    public virtual void ArgMin(IGpuBuffer A, IGpuBuffer indices, int outerSize, int reduceSize)
        => Inner.ArgMin(A, indices, outerSize, reduceSize);

    /// <inheritdoc/>
    public virtual void MaxAxis(IGpuBuffer A, IGpuBuffer B, int outerSize, int reduceSize)
        => Inner.MaxAxis(A, B, outerSize, reduceSize);

    /// <inheritdoc/>
    public virtual void TopK(IGpuBuffer A, IGpuBuffer values, IGpuBuffer indices, int outerSize, int reduceSize, int k, bool sorted = true)
        => Inner.TopK(A, values, indices, outerSize, reduceSize, k, sorted);

    /// <inheritdoc/>
    public virtual void BroadcastMultiplyLastAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
        => Inner.BroadcastMultiplyLastAxis(A, B, C, outerSize, innerSize);

    /// <inheritdoc/>
    public virtual void BroadcastMultiplyFirstAxis(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int outerSize, int innerSize)
        => Inner.BroadcastMultiplyFirstAxis(A, B, C, outerSize, innerSize);

    #endregion

    #region Optimizer Operations

    /// <inheritdoc/>
    public virtual void SgdMomentumUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
        => Inner.SgdMomentumUpdate(param, gradient, velocity, learningRate, momentum, weightDecay, size);

    /// <inheritdoc/>
    public virtual void AdamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
        => Inner.AdamUpdate(param, gradient, m, v, learningRate, beta1, beta2, epsilon, weightDecay, step, size);

    /// <inheritdoc/>
    public virtual void AdamWUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
        => Inner.AdamWUpdate(param, gradient, m, v, learningRate, beta1, beta2, epsilon, weightDecay, step, size);

    /// <inheritdoc/>
    public virtual void RmspropUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer squaredAvg,
        float learningRate, float rho, float epsilon, float weightDecay, int size)
        => Inner.RmspropUpdate(param, gradient, squaredAvg, learningRate, rho, epsilon, weightDecay, size);

    /// <inheritdoc/>
    public virtual void AdagradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumulatedGrad,
        float learningRate, float epsilon, float weightDecay, int size)
        => Inner.AdagradUpdate(param, gradient, accumulatedGrad, learningRate, epsilon, weightDecay, size);

    /// <inheritdoc/>
    public virtual void NagUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, int size)
        => Inner.NagUpdate(param, gradient, velocity, learningRate, momentum, weightDecay, size);

    /// <inheritdoc/>
    public virtual void LarsUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer velocity,
        float learningRate, float momentum, float weightDecay, float trustCoeff, int size)
        => Inner.LarsUpdate(param, gradient, velocity, learningRate, momentum, weightDecay, trustCoeff, size);

    /// <inheritdoc/>
    public virtual void LambUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
        => Inner.LambUpdate(param, gradient, m, v, learningRate, beta1, beta2, epsilon, weightDecay, step, size);

    /// <inheritdoc/>
    public virtual void SgdUpdate(IGpuBuffer param, IGpuBuffer gradient,
        float learningRate, float weightDecay, int size)
        => Inner.SgdUpdate(param, gradient, learningRate, weightDecay, size);

    /// <inheritdoc/>
    public virtual void AdadeltaUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer accumGrad, IGpuBuffer accumUpdate,
        float rho, float epsilon, float weightDecay, int size)
        => Inner.AdadeltaUpdate(param, gradient, accumGrad, accumUpdate, rho, epsilon, weightDecay, size);

    /// <inheritdoc/>
    public virtual void AmsgradUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
        => Inner.AmsgradUpdate(param, gradient, m, v, vMax, learningRate, beta1, beta2, epsilon, weightDecay, step, size);

    /// <inheritdoc/>
    public virtual void AdamaxUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer u,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
        => Inner.AdamaxUpdate(param, gradient, m, u, learningRate, beta1, beta2, epsilon, weightDecay, step, size);

    /// <inheritdoc/>
    public virtual void LionUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m,
        float learningRate, float beta1, float beta2, float weightDecay, int size)
        => Inner.LionUpdate(param, gradient, m, learningRate, beta1, beta2, weightDecay, size);

    /// <inheritdoc/>
    public virtual void NadamUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer m, IGpuBuffer v,
        float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step, int size)
        => Inner.NadamUpdate(param, gradient, m, v, learningRate, beta1, beta2, epsilon, weightDecay, step, size);

    /// <inheritdoc/>
    public virtual void FtrlUpdate(IGpuBuffer param, IGpuBuffer gradient, IGpuBuffer z, IGpuBuffer n,
        float learningRate, float l1Reg, float l2Reg, float beta, int size)
        => Inner.FtrlUpdate(param, gradient, z, n, learningRate, l1Reg, l2Reg, beta, size);

    /// <inheritdoc/>
    public virtual void ConvertToFp16(IGpuBuffer input, IGpuBuffer output, int size)
        => Inner.ConvertToFp16(input, output, size);

    /// <inheritdoc/>
    public virtual void ConvertToFp32(IGpuBuffer input, IGpuBuffer output, int size)
        => Inner.ConvertToFp32(input, output, size);

    #endregion

    #region FFT and Signal Processing

    /// <inheritdoc/>
    public virtual void FFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag, int n, bool inverse)
        => Inner.FFT(inputReal, inputImag, outputReal, outputImag, n, inverse);

    /// <inheritdoc/>
    public virtual void RFFT(IGpuBuffer input, IGpuBuffer outputReal, IGpuBuffer outputImag, int n)
        => Inner.RFFT(input, outputReal, outputImag, n);

    /// <inheritdoc/>
    public virtual void IRFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer output, int n)
        => Inner.IRFFT(inputReal, inputImag, output, n);

    /// <inheritdoc/>
    public virtual void BatchedFFT(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag,
        int batch, int n, bool inverse)
        => Inner.BatchedFFT(inputReal, inputImag, outputReal, outputImag, batch, n, inverse);

    /// <inheritdoc/>
    public virtual void FFT2D(IGpuBuffer inputReal, IGpuBuffer inputImag, IGpuBuffer outputReal, IGpuBuffer outputImag,
        int height, int width, bool inverse)
        => Inner.FFT2D(inputReal, inputImag, outputReal, outputImag, height, width, inverse);

    /// <inheritdoc/>
    public virtual void ApplyWindow(IGpuBuffer input, IGpuBuffer window, IGpuBuffer output, int n)
        => Inner.ApplyWindow(input, window, output, n);

    /// <inheritdoc/>
    public virtual void ComplexMagnitude(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer magnitude, int n)
        => Inner.ComplexMagnitude(real, imag, magnitude, n);

    /// <inheritdoc/>
    public virtual void ComplexPhase(IGpuBuffer real, IGpuBuffer imag, IGpuBuffer phase, int n)
        => Inner.ComplexPhase(real, imag, phase, n);

    /// <inheritdoc/>
    public virtual void PolarToComplex(IGpuBuffer magnitude, IGpuBuffer phase, IGpuBuffer real, IGpuBuffer imag, int n)
        => Inner.PolarToComplex(magnitude, phase, real, imag, n);

    /// <inheritdoc/>
    public virtual void ApplyMelFilterbank(IGpuBuffer powerSpec, IGpuBuffer filterbank, IGpuBuffer melSpec,
        int numFrames, int numFreqs, int nMels)
        => Inner.ApplyMelFilterbank(powerSpec, filterbank, melSpec, numFrames, numFreqs, nMels);

    /// <inheritdoc/>
    public virtual void PowerToDb(IGpuBuffer power, IGpuBuffer db, int n, float refValue, float minDb)
        => Inner.PowerToDb(power, db, n, refValue, minDb);

    /// <inheritdoc/>
    public virtual void DbToPower(IGpuBuffer db, IGpuBuffer power, int n, float refValue)
        => Inner.DbToPower(db, power, n, refValue);

    #endregion

    #region IDisposable

    private bool _disposed;

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes resources.
    /// </summary>
    /// <param name="disposing">True if disposing managed resources.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;

        if (disposing)
        {
            // Note: We don't dispose Inner by default - the creator owns it
            // Subclasses can override to dispose if they own the inner backend
        }

        _disposed = true;
    }

    #endregion
}
