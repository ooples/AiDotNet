#if !NET462
using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// cuDNN P/Invoke bindings for high-performance deep learning primitives.
/// Target: Match PyTorch/TensorFlow performance for convolutions, pooling, batch norm, attention.
/// </summary>
/// <remarks>
/// <para><b>Performance Targets:</b></para>
/// <list type="bullet">
/// <item>Convolution: ~15,000 GFLOPS (cuDNN optimized algorithms)</item>
/// <item>BatchNorm: Near memory-bandwidth limited</item>
/// <item>Pooling: Near memory-bandwidth limited</item>
/// <item>Attention: Tensor core accelerated (Volta+)</item>
/// </list>
/// <para><b>cuDNN Frontend Reference:</b> https://github.com/NVIDIA/cudnn-frontend</para>
/// </remarks>
public static class CuDnnNative
{
    // cuDNN library name - try newer version first, fall back to older
    private const string CudnnLibrary = "cudnn64_9.dll";
    private const string CudnnLibraryFallback = "cudnn64_8.dll";

    #region Enums

    /// <summary>cuDNN status codes.</summary>
    public enum CudnnStatus
    {
        Success = 0,
        NotInitialized = 1,
        AllocFailed = 2,
        BadParam = 3,
        InternalError = 4,
        InvalidValue = 5,
        ArchMismatch = 6,
        MappingError = 7,
        ExecutionFailed = 8,
        NotSupported = 9,
        LicenseError = 10,
        RuntimePrerequisiteMissing = 11,
        RuntimeInProgress = 12,
        RuntimeFpOverflow = 13,
        VersionMismatch = 14
    }

    /// <summary>cuDNN data types.</summary>
    public enum CudnnDataType
    {
        Float = 0,
        Double = 1,
        Half = 2,
        Int8 = 3,
        Int32 = 4,
        Int8x4 = 5,
        Uint8 = 6,
        Uint8x4 = 7,
        Int8x32 = 8,
        BFloat16 = 9,
        Int64 = 10,
        Boolean = 11,
        Fp8E4M3 = 12,
        Fp8E5M2 = 13,
        FastFloatForFp8 = 14
    }

    /// <summary>cuDNN tensor formats.</summary>
    public enum CudnnTensorFormat
    {
        NCHW = 0,      // Row major (image, feature maps, rows, columns)
        NHWC = 1,      // Feature maps interleaved
        NCHWVectC = 2  // Vectorized for int8
    }

    /// <summary>cuDNN convolution modes.</summary>
    public enum CudnnConvolutionMode
    {
        Convolution = 0,
        CrossCorrelation = 1
    }

    /// <summary>cuDNN convolution forward algorithms.</summary>
    public enum CudnnConvolutionFwdAlgo
    {
        ImplicitGemm = 0,
        ImplicitPrecompGemm = 1,
        Gemm = 2,
        Direct = 3,
        Fft = 4,
        FftTiling = 5,
        Winograd = 6,
        WinogradNonfused = 7,
        Count = 8
    }

    /// <summary>cuDNN convolution backward data algorithms.</summary>
    public enum CudnnConvolutionBwdDataAlgo
    {
        Algo0 = 0,
        Algo1 = 1,
        Fft = 2,
        FftTiling = 3,
        Winograd = 4,
        WinogradNonfused = 5,
        Count = 6
    }

    /// <summary>cuDNN convolution backward filter algorithms.</summary>
    public enum CudnnConvolutionBwdFilterAlgo
    {
        Algo0 = 0,
        Algo1 = 1,
        Fft = 2,
        Algo3 = 3,
        Winograd = 4,
        WinogradNonfused = 5,
        FftTiling = 6,
        Count = 7
    }

    /// <summary>cuDNN pooling modes.</summary>
    public enum CudnnPoolingMode
    {
        Max = 0,
        AverageCountIncludePadding = 1,
        AverageCountExcludePadding = 2,
        MaxDeterministic = 3
    }

    /// <summary>cuDNN activation modes.</summary>
    public enum CudnnActivationMode
    {
        Sigmoid = 0,
        ReLU = 1,
        Tanh = 2,
        ClippedReLU = 3,
        Elu = 4,
        Identity = 5,
        Swish = 6
    }

    /// <summary>cuDNN softmax algorithms.</summary>
    public enum CudnnSoftmaxAlgorithm
    {
        Fast = 0,
        Accurate = 1,
        Log = 2
    }

    /// <summary>cuDNN softmax modes.</summary>
    public enum CudnnSoftmaxMode
    {
        Instance = 0,  // Compute softmax over all C, H, W for each N
        Channel = 1    // Compute softmax over all C for each N, H, W
    }

    /// <summary>cuDNN batch normalization modes.</summary>
    public enum CudnnBatchNormMode
    {
        PerActivation = 0,  // Normalize per activation (C*H*W elements)
        Spatial = 1,        // Normalize per channel (recommended for conv layers)
        SpatialPersistent = 2  // Faster spatial mode with reduced workspace
    }

    /// <summary>cuDNN NaN propagation.</summary>
    public enum CudnnNanPropagation
    {
        NotPropagateNan = 0,
        PropagateNan = 1
    }

    /// <summary>cuDNN reduce tensor operations.</summary>
    public enum CudnnReduceTensorOp
    {
        Add = 0,
        Mul = 1,
        Min = 2,
        Max = 3,
        AMax = 4,
        Avg = 5,
        Norm1 = 6,
        Norm2 = 7,
        MulNoZeros = 8
    }

    /// <summary>cuDNN math types for tensor cores.</summary>
    public enum CudnnMathType
    {
        Default = 0,
        TensorOp = 1,           // Allow tensor core operations
        TensorOpAllowConversion = 2,  // Allow data type conversion for tensor cores
        Fma = 3                 // Force FMA operations
    }

    /// <summary>cuDNN RNN modes.</summary>
    public enum CudnnRnnMode
    {
        ReLU = 0,
        Tanh = 1,
        LSTM = 2,
        GRU = 3
    }

    /// <summary>cuDNN RNN directions.</summary>
    public enum CudnnDirectionMode
    {
        Unidirectional = 0,
        Bidirectional = 1
    }

    /// <summary>cuDNN RNN input modes.</summary>
    public enum CudnnRnnInputMode
    {
        LinearInput = 0,
        SkipInput = 1
    }

    /// <summary>cuDNN attention query map modes.</summary>
    public enum CudnnMultiHeadAttnWeightKind
    {
        QWeights = 0,
        KWeights = 1,
        VWeights = 2,
        OWeights = 3,
        QBiases = 4,
        KBiases = 5,
        VBiases = 6,
        OBiases = 7
    }

    #endregion

    #region Handle Management

    /// <summary>Creates a cuDNN handle.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnCreate")]
    public static extern CudnnStatus cudnnCreate(out IntPtr handle);

    /// <summary>Destroys a cuDNN handle.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnDestroy")]
    public static extern CudnnStatus cudnnDestroy(IntPtr handle);

    /// <summary>Sets the CUDA stream for cuDNN operations.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnSetStream")]
    public static extern CudnnStatus cudnnSetStream(IntPtr handle, IntPtr stream);

    /// <summary>Gets the cuDNN version.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnGetVersion")]
    public static extern ulong cudnnGetVersion();

    /// <summary>Gets error string for status code.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnGetErrorString")]
    public static extern IntPtr cudnnGetErrorString(CudnnStatus status);

    #endregion

    #region Tensor Descriptors

    /// <summary>Creates a tensor descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnCreateTensorDescriptor")]
    public static extern CudnnStatus cudnnCreateTensorDescriptor(out IntPtr tensorDesc);

    /// <summary>Destroys a tensor descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnDestroyTensorDescriptor")]
    public static extern CudnnStatus cudnnDestroyTensorDescriptor(IntPtr tensorDesc);

    /// <summary>Sets a 4D tensor descriptor (NCHW).</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnSetTensor4dDescriptor")]
    public static extern CudnnStatus cudnnSetTensor4dDescriptor(
        IntPtr tensorDesc,
        CudnnTensorFormat format,
        CudnnDataType dataType,
        int n, int c, int h, int w);

    /// <summary>Sets a 4D tensor descriptor with explicit strides.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnSetTensor4dDescriptorEx")]
    public static extern CudnnStatus cudnnSetTensor4dDescriptorEx(
        IntPtr tensorDesc,
        CudnnDataType dataType,
        int n, int c, int h, int w,
        int nStride, int cStride, int hStride, int wStride);

    /// <summary>Sets an N-dimensional tensor descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnSetTensorNdDescriptor")]
    public static extern CudnnStatus cudnnSetTensorNdDescriptor(
        IntPtr tensorDesc,
        CudnnDataType dataType,
        int nbDims,
        int[] dimA,
        int[] strideA);

    /// <summary>Adds a tensor to another tensor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnAddTensor")]
    public static extern CudnnStatus cudnnAddTensor(
        IntPtr handle,
        ref float alpha,
        IntPtr aDesc, IntPtr A,
        ref float beta,
        IntPtr cDesc, IntPtr C);

    /// <summary>Sets all elements of a tensor to a scalar value.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnSetTensor")]
    public static extern CudnnStatus cudnnSetTensor(
        IntPtr handle,
        IntPtr yDesc,
        IntPtr y,
        IntPtr valuePtr);

    /// <summary>Scales a tensor by a scalar value.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnScaleTensor")]
    public static extern CudnnStatus cudnnScaleTensor(
        IntPtr handle,
        IntPtr yDesc,
        IntPtr y,
        ref float alpha);

    #endregion

    #region Filter Descriptors

    /// <summary>Creates a filter descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnCreateFilterDescriptor")]
    public static extern CudnnStatus cudnnCreateFilterDescriptor(out IntPtr filterDesc);

    /// <summary>Destroys a filter descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnDestroyFilterDescriptor")]
    public static extern CudnnStatus cudnnDestroyFilterDescriptor(IntPtr filterDesc);

    /// <summary>Sets a 4D filter descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnSetFilter4dDescriptor")]
    public static extern CudnnStatus cudnnSetFilter4dDescriptor(
        IntPtr filterDesc,
        CudnnDataType dataType,
        CudnnTensorFormat format,
        int k, int c, int h, int w);

    /// <summary>Sets an N-dimensional filter descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnSetFilterNdDescriptor")]
    public static extern CudnnStatus cudnnSetFilterNdDescriptor(
        IntPtr filterDesc,
        CudnnDataType dataType,
        CudnnTensorFormat format,
        int nbDims,
        int[] filterDimA);

    #endregion

    #region Convolution Descriptors

    /// <summary>Creates a convolution descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnCreateConvolutionDescriptor")]
    public static extern CudnnStatus cudnnCreateConvolutionDescriptor(out IntPtr convDesc);

    /// <summary>Destroys a convolution descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnDestroyConvolutionDescriptor")]
    public static extern CudnnStatus cudnnDestroyConvolutionDescriptor(IntPtr convDesc);

    /// <summary>Sets a 2D convolution descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnSetConvolution2dDescriptor")]
    public static extern CudnnStatus cudnnSetConvolution2dDescriptor(
        IntPtr convDesc,
        int padH, int padW,
        int strideH, int strideW,
        int dilationH, int dilationW,
        CudnnConvolutionMode mode,
        CudnnDataType computeType);

    /// <summary>Sets an N-dimensional convolution descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnSetConvolutionNdDescriptor")]
    public static extern CudnnStatus cudnnSetConvolutionNdDescriptor(
        IntPtr convDesc,
        int arrayLength,
        int[] padA,
        int[] strideA,
        int[] dilationA,
        CudnnConvolutionMode mode,
        CudnnDataType computeType);

    /// <summary>Sets the math type for convolution operations (enables tensor cores).</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnSetConvolutionMathType")]
    public static extern CudnnStatus cudnnSetConvolutionMathType(
        IntPtr convDesc,
        CudnnMathType mathType);

    /// <summary>Sets the group count for grouped convolutions.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnSetConvolutionGroupCount")]
    public static extern CudnnStatus cudnnSetConvolutionGroupCount(
        IntPtr convDesc,
        int groupCount);

    /// <summary>Gets the output dimensions for a convolution.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnGetConvolution2dForwardOutputDim")]
    public static extern CudnnStatus cudnnGetConvolution2dForwardOutputDim(
        IntPtr convDesc,
        IntPtr inputTensorDesc,
        IntPtr filterDesc,
        out int n, out int c, out int h, out int w);

    #endregion

    #region Convolution Forward

    /// <summary>Finds the best convolution forward algorithm.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnGetConvolutionForwardAlgorithm_v7")]
    public static extern CudnnStatus cudnnGetConvolutionForwardAlgorithm_v7(
        IntPtr handle,
        IntPtr xDesc,
        IntPtr wDesc,
        IntPtr convDesc,
        IntPtr yDesc,
        int requestedAlgoCount,
        out int returnedAlgoCount,
        IntPtr perfResults);

    /// <summary>Gets workspace size for convolution forward.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnGetConvolutionForwardWorkspaceSize")]
    public static extern CudnnStatus cudnnGetConvolutionForwardWorkspaceSize(
        IntPtr handle,
        IntPtr xDesc,
        IntPtr wDesc,
        IntPtr convDesc,
        IntPtr yDesc,
        CudnnConvolutionFwdAlgo algo,
        out ulong sizeInBytes);

    /// <summary>Executes the convolution forward operation.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnConvolutionForward")]
    public static extern CudnnStatus cudnnConvolutionForward(
        IntPtr handle,
        ref float alpha,
        IntPtr xDesc, IntPtr x,
        IntPtr wDesc, IntPtr w,
        IntPtr convDesc,
        CudnnConvolutionFwdAlgo algo,
        IntPtr workSpace, ulong workSpaceSizeInBytes,
        ref float beta,
        IntPtr yDesc, IntPtr y);

    /// <summary>Executes convolution forward with double precision.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnConvolutionForward")]
    public static extern CudnnStatus cudnnConvolutionForwardDouble(
        IntPtr handle,
        ref double alpha,
        IntPtr xDesc, IntPtr x,
        IntPtr wDesc, IntPtr w,
        IntPtr convDesc,
        CudnnConvolutionFwdAlgo algo,
        IntPtr workSpace, ulong workSpaceSizeInBytes,
        ref double beta,
        IntPtr yDesc, IntPtr y);

    #endregion

    #region Convolution Backward

    /// <summary>Gets workspace size for backward data.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnGetConvolutionBackwardDataWorkspaceSize")]
    public static extern CudnnStatus cudnnGetConvolutionBackwardDataWorkspaceSize(
        IntPtr handle,
        IntPtr wDesc,
        IntPtr dyDesc,
        IntPtr convDesc,
        IntPtr dxDesc,
        CudnnConvolutionBwdDataAlgo algo,
        out ulong sizeInBytes);

    /// <summary>Executes backward data convolution.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnConvolutionBackwardData")]
    public static extern CudnnStatus cudnnConvolutionBackwardData(
        IntPtr handle,
        ref float alpha,
        IntPtr wDesc, IntPtr w,
        IntPtr dyDesc, IntPtr dy,
        IntPtr convDesc,
        CudnnConvolutionBwdDataAlgo algo,
        IntPtr workSpace, ulong workSpaceSizeInBytes,
        ref float beta,
        IntPtr dxDesc, IntPtr dx);

    /// <summary>Gets workspace size for backward filter.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnGetConvolutionBackwardFilterWorkspaceSize")]
    public static extern CudnnStatus cudnnGetConvolutionBackwardFilterWorkspaceSize(
        IntPtr handle,
        IntPtr xDesc,
        IntPtr dyDesc,
        IntPtr convDesc,
        IntPtr dwDesc,
        CudnnConvolutionBwdFilterAlgo algo,
        out ulong sizeInBytes);

    /// <summary>Executes backward filter convolution.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnConvolutionBackwardFilter")]
    public static extern CudnnStatus cudnnConvolutionBackwardFilter(
        IntPtr handle,
        ref float alpha,
        IntPtr xDesc, IntPtr x,
        IntPtr dyDesc, IntPtr dy,
        IntPtr convDesc,
        CudnnConvolutionBwdFilterAlgo algo,
        IntPtr workSpace, ulong workSpaceSizeInBytes,
        ref float beta,
        IntPtr dwDesc, IntPtr dw);

    /// <summary>Computes bias gradients.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnConvolutionBackwardBias")]
    public static extern CudnnStatus cudnnConvolutionBackwardBias(
        IntPtr handle,
        ref float alpha,
        IntPtr dyDesc, IntPtr dy,
        ref float beta,
        IntPtr dbDesc, IntPtr db);

    #endregion

    #region Pooling

    /// <summary>Creates a pooling descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnCreatePoolingDescriptor")]
    public static extern CudnnStatus cudnnCreatePoolingDescriptor(out IntPtr poolingDesc);

    /// <summary>Destroys a pooling descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnDestroyPoolingDescriptor")]
    public static extern CudnnStatus cudnnDestroyPoolingDescriptor(IntPtr poolingDesc);

    /// <summary>Sets a 2D pooling descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnSetPooling2dDescriptor")]
    public static extern CudnnStatus cudnnSetPooling2dDescriptor(
        IntPtr poolingDesc,
        CudnnPoolingMode mode,
        CudnnNanPropagation maxPoolNanOpt,
        int windowHeight, int windowWidth,
        int verticalPadding, int horizontalPadding,
        int verticalStride, int horizontalStride);

    /// <summary>Sets an N-dimensional pooling descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnSetPoolingNdDescriptor")]
    public static extern CudnnStatus cudnnSetPoolingNdDescriptor(
        IntPtr poolingDesc,
        CudnnPoolingMode mode,
        CudnnNanPropagation maxPoolNanOpt,
        int nbDims,
        int[] windowDimA,
        int[] paddingA,
        int[] strideA);

    /// <summary>Gets the output dimensions for pooling.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnGetPooling2dForwardOutputDim")]
    public static extern CudnnStatus cudnnGetPooling2dForwardOutputDim(
        IntPtr poolingDesc,
        IntPtr inputTensorDesc,
        out int n, out int c, out int h, out int w);

    /// <summary>Executes pooling forward.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnPoolingForward")]
    public static extern CudnnStatus cudnnPoolingForward(
        IntPtr handle,
        IntPtr poolingDesc,
        ref float alpha,
        IntPtr xDesc, IntPtr x,
        ref float beta,
        IntPtr yDesc, IntPtr y);

    /// <summary>Executes pooling backward.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnPoolingBackward")]
    public static extern CudnnStatus cudnnPoolingBackward(
        IntPtr handle,
        IntPtr poolingDesc,
        ref float alpha,
        IntPtr yDesc, IntPtr y,
        IntPtr dyDesc, IntPtr dy,
        IntPtr xDesc, IntPtr x,
        ref float beta,
        IntPtr dxDesc, IntPtr dx);

    #endregion

    #region Activation

    /// <summary>Creates an activation descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnCreateActivationDescriptor")]
    public static extern CudnnStatus cudnnCreateActivationDescriptor(out IntPtr activationDesc);

    /// <summary>Destroys an activation descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnDestroyActivationDescriptor")]
    public static extern CudnnStatus cudnnDestroyActivationDescriptor(IntPtr activationDesc);

    /// <summary>Sets an activation descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnSetActivationDescriptor")]
    public static extern CudnnStatus cudnnSetActivationDescriptor(
        IntPtr activationDesc,
        CudnnActivationMode mode,
        CudnnNanPropagation reluNanOpt,
        double coef);  // For clipped ReLU, this is the ceiling

    /// <summary>Executes activation forward.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnActivationForward")]
    public static extern CudnnStatus cudnnActivationForward(
        IntPtr handle,
        IntPtr activationDesc,
        ref float alpha,
        IntPtr xDesc, IntPtr x,
        ref float beta,
        IntPtr yDesc, IntPtr y);

    /// <summary>Executes activation backward.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnActivationBackward")]
    public static extern CudnnStatus cudnnActivationBackward(
        IntPtr handle,
        IntPtr activationDesc,
        ref float alpha,
        IntPtr yDesc, IntPtr y,
        IntPtr dyDesc, IntPtr dy,
        IntPtr xDesc, IntPtr x,
        ref float beta,
        IntPtr dxDesc, IntPtr dx);

    #endregion

    #region Softmax

    /// <summary>Executes softmax forward.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnSoftmaxForward")]
    public static extern CudnnStatus cudnnSoftmaxForward(
        IntPtr handle,
        CudnnSoftmaxAlgorithm algo,
        CudnnSoftmaxMode mode,
        ref float alpha,
        IntPtr xDesc, IntPtr x,
        ref float beta,
        IntPtr yDesc, IntPtr y);

    /// <summary>Executes softmax backward.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnSoftmaxBackward")]
    public static extern CudnnStatus cudnnSoftmaxBackward(
        IntPtr handle,
        CudnnSoftmaxAlgorithm algo,
        CudnnSoftmaxMode mode,
        ref float alpha,
        IntPtr yDesc, IntPtr y,
        IntPtr dyDesc, IntPtr dy,
        ref float beta,
        IntPtr dxDesc, IntPtr dx);

    #endregion

    #region Batch Normalization

    /// <summary>Derives batch normalization parameters tensor descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnDeriveBNTensorDescriptor")]
    public static extern CudnnStatus cudnnDeriveBNTensorDescriptor(
        IntPtr derivedBnDesc,
        IntPtr xDesc,
        CudnnBatchNormMode mode);

    /// <summary>Executes batch normalization forward during training.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnBatchNormalizationForwardTraining")]
    public static extern CudnnStatus cudnnBatchNormalizationForwardTraining(
        IntPtr handle,
        CudnnBatchNormMode mode,
        ref float alpha,
        ref float beta,
        IntPtr xDesc, IntPtr x,
        IntPtr yDesc, IntPtr y,
        IntPtr bnScaleBiasMeanVarDesc,
        IntPtr bnScale,
        IntPtr bnBias,
        double exponentialAverageFactor,
        IntPtr resultRunningMean,
        IntPtr resultRunningVariance,
        double epsilon,
        IntPtr resultSaveMean,
        IntPtr resultSaveInvVariance);

    /// <summary>Executes batch normalization forward during inference.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnBatchNormalizationForwardInference")]
    public static extern CudnnStatus cudnnBatchNormalizationForwardInference(
        IntPtr handle,
        CudnnBatchNormMode mode,
        ref float alpha,
        ref float beta,
        IntPtr xDesc, IntPtr x,
        IntPtr yDesc, IntPtr y,
        IntPtr bnScaleBiasMeanVarDesc,
        IntPtr bnScale,
        IntPtr bnBias,
        IntPtr estimatedMean,
        IntPtr estimatedVariance,
        double epsilon);

    /// <summary>Executes batch normalization backward.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnBatchNormalizationBackward")]
    public static extern CudnnStatus cudnnBatchNormalizationBackward(
        IntPtr handle,
        CudnnBatchNormMode mode,
        ref float alphaDataDiff,
        ref float betaDataDiff,
        ref float alphaParamDiff,
        ref float betaParamDiff,
        IntPtr xDesc, IntPtr x,
        IntPtr dyDesc, IntPtr dy,
        IntPtr dxDesc, IntPtr dx,
        IntPtr bnScaleBiasDiffDesc,
        IntPtr bnScale,
        IntPtr resultBnScaleDiff,
        IntPtr resultBnBiasDiff,
        double epsilon,
        IntPtr savedMean,
        IntPtr savedInvVariance);

    #endregion

    #region Dropout

    /// <summary>Creates a dropout descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnCreateDropoutDescriptor")]
    public static extern CudnnStatus cudnnCreateDropoutDescriptor(out IntPtr dropoutDesc);

    /// <summary>Destroys a dropout descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnDestroyDropoutDescriptor")]
    public static extern CudnnStatus cudnnDestroyDropoutDescriptor(IntPtr dropoutDesc);

    /// <summary>Gets the size of dropout state.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnDropoutGetStatesSize")]
    public static extern CudnnStatus cudnnDropoutGetStatesSize(
        IntPtr handle,
        out ulong sizeInBytes);

    /// <summary>Gets the size of dropout reserve space.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnDropoutGetReserveSpaceSize")]
    public static extern CudnnStatus cudnnDropoutGetReserveSpaceSize(
        IntPtr xDesc,
        out ulong sizeInBytes);

    /// <summary>Sets the dropout descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnSetDropoutDescriptor")]
    public static extern CudnnStatus cudnnSetDropoutDescriptor(
        IntPtr dropoutDesc,
        IntPtr handle,
        float dropout,
        IntPtr states,
        ulong stateSizeInBytes,
        ulong seed);

    /// <summary>Executes dropout forward.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnDropoutForward")]
    public static extern CudnnStatus cudnnDropoutForward(
        IntPtr handle,
        IntPtr dropoutDesc,
        IntPtr xDesc, IntPtr x,
        IntPtr yDesc, IntPtr y,
        IntPtr reserveSpace,
        ulong reserveSpaceSizeInBytes);

    /// <summary>Executes dropout backward.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnDropoutBackward")]
    public static extern CudnnStatus cudnnDropoutBackward(
        IntPtr handle,
        IntPtr dropoutDesc,
        IntPtr dyDesc, IntPtr dy,
        IntPtr dxDesc, IntPtr dx,
        IntPtr reserveSpace,
        ulong reserveSpaceSizeInBytes);

    #endregion

    #region Reduce Tensor

    /// <summary>Creates a reduce tensor descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnCreateReduceTensorDescriptor")]
    public static extern CudnnStatus cudnnCreateReduceTensorDescriptor(out IntPtr reduceTensorDesc);

    /// <summary>Destroys a reduce tensor descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnDestroyReduceTensorDescriptor")]
    public static extern CudnnStatus cudnnDestroyReduceTensorDescriptor(IntPtr reduceTensorDesc);

    /// <summary>Sets a reduce tensor descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnSetReduceTensorDescriptor")]
    public static extern CudnnStatus cudnnSetReduceTensorDescriptor(
        IntPtr reduceTensorDesc,
        CudnnReduceTensorOp reduceTensorOp,
        CudnnDataType reduceTensorCompType,
        CudnnNanPropagation reduceTensorNanOpt,
        int reduceTensorIndices,  // CUDNN_REDUCE_TENSOR_NO_INDICES = 0
        int reduceTensorIndicesType);  // CUDNN_32BIT_INDICES = 0

    /// <summary>Gets the workspace size for reduce tensor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnGetReductionWorkspaceSize")]
    public static extern CudnnStatus cudnnGetReductionWorkspaceSize(
        IntPtr handle,
        IntPtr reduceTensorDesc,
        IntPtr aDesc,
        IntPtr cDesc,
        out ulong sizeInBytes);

    /// <summary>Executes reduce tensor operation.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnReduceTensor")]
    public static extern CudnnStatus cudnnReduceTensor(
        IntPtr handle,
        IntPtr reduceTensorDesc,
        IntPtr indices, ulong indicesSizeInBytes,
        IntPtr workspace, ulong workspaceSizeInBytes,
        ref float alpha,
        IntPtr aDesc, IntPtr A,
        ref float beta,
        IntPtr cDesc, IntPtr C);

    #endregion

    #region RNN

    /// <summary>Creates an RNN descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnCreateRNNDescriptor")]
    public static extern CudnnStatus cudnnCreateRNNDescriptor(out IntPtr rnnDesc);

    /// <summary>Destroys an RNN descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnDestroyRNNDescriptor")]
    public static extern CudnnStatus cudnnDestroyRNNDescriptor(IntPtr rnnDesc);

    /// <summary>Sets RNN descriptor (v8 API).</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnSetRNNDescriptor_v8")]
    public static extern CudnnStatus cudnnSetRNNDescriptor_v8(
        IntPtr rnnDesc,
        int algo,       // CUDNN_RNN_ALGO_STANDARD = 0
        CudnnRnnMode cellMode,
        int biasMode,   // CUDNN_RNN_DOUBLE_BIAS = 0
        CudnnDirectionMode dirMode,
        CudnnRnnInputMode inputMode,
        CudnnDataType dataType,
        CudnnDataType mathPrec,
        CudnnMathType mathType,
        int inputSize,
        int hiddenSize,
        int projSize,
        int numLayers,
        IntPtr dropoutDesc,  // Can be IntPtr.Zero
        int auxFlags);

    /// <summary>Gets RNN workspace and reserve space sizes.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnGetRNNTempSpaceSizes")]
    public static extern CudnnStatus cudnnGetRNNTempSpaceSizes(
        IntPtr handle,
        IntPtr rnnDesc,
        int fMode,  // CUDNN_FWD_MODE_TRAINING = 1
        IntPtr xDesc,
        out ulong workSpaceSize,
        out ulong reserveSpaceSize);

    /// <summary>Gets RNN weight space size.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnGetRNNWeightSpaceSize")]
    public static extern CudnnStatus cudnnGetRNNWeightSpaceSize(
        IntPtr handle,
        IntPtr rnnDesc,
        out ulong weightSpaceSize);

    #endregion

    #region Multi-Head Attention

    /// <summary>Creates an attention descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnCreateAttnDescriptor")]
    public static extern CudnnStatus cudnnCreateAttnDescriptor(out IntPtr attnDesc);

    /// <summary>Destroys an attention descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnDestroyAttnDescriptor")]
    public static extern CudnnStatus cudnnDestroyAttnDescriptor(IntPtr attnDesc);

    /// <summary>Sets an attention descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnSetAttnDescriptor")]
    public static extern CudnnStatus cudnnSetAttnDescriptor(
        IntPtr attnDesc,
        uint attnMode,   // CUDNN_ATTN_QUERYMAP_ALL_TO_ONE, etc.
        int nHeads,
        double smScaler,
        CudnnDataType dataType,
        CudnnDataType computePrec,
        CudnnMathType mathType,
        IntPtr attnDropoutDesc,  // Can be IntPtr.Zero
        IntPtr postDropoutDesc,  // Can be IntPtr.Zero
        int qSize, int kSize, int vSize,
        int qProjSize, int kProjSize, int vProjSize, int oProjSize,
        int qoMaxSeqLength, int kvMaxSeqLength,
        int maxBatchSize,
        int maxBeamSize);

    /// <summary>Gets attention buffer sizes.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnGetMultiHeadAttnBuffers")]
    public static extern CudnnStatus cudnnGetMultiHeadAttnBuffers(
        IntPtr handle,
        IntPtr attnDesc,
        out ulong weightSizeInBytes,
        out ulong workSpaceSizeInBytes,
        out ulong reserveSpaceSizeInBytes);

    /// <summary>Gets attention weight pointers.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnGetMultiHeadAttnWeights")]
    public static extern CudnnStatus cudnnGetMultiHeadAttnWeights(
        IntPtr handle,
        IntPtr attnDesc,
        CudnnMultiHeadAttnWeightKind wKind,
        ulong weightSizeInBytes,
        IntPtr weights,
        IntPtr wDesc,
        out IntPtr wAddr);

    /// <summary>Executes multi-head attention forward.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnMultiHeadAttnForward")]
    public static extern CudnnStatus cudnnMultiHeadAttnForward(
        IntPtr handle,
        IntPtr attnDesc,
        int currIdx,
        int[] loWinIdx,
        int[] hiWinIdx,
        int[] devSeqLengthsQO,
        int[] devSeqLengthsKV,
        IntPtr qDesc, IntPtr queries, IntPtr residuals,
        IntPtr kDesc, IntPtr keys,
        IntPtr vDesc, IntPtr values,
        IntPtr oDesc, IntPtr output,
        ulong weightSizeInBytes, IntPtr weights,
        ulong workSpaceSizeInBytes, IntPtr workSpace,
        ulong reserveSpaceSizeInBytes, IntPtr reserveSpace);

    #endregion

    #region Op Tensor

    /// <summary>Creates an op tensor descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnCreateOpTensorDescriptor")]
    public static extern CudnnStatus cudnnCreateOpTensorDescriptor(out IntPtr opTensorDesc);

    /// <summary>Destroys an op tensor descriptor.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnDestroyOpTensorDescriptor")]
    public static extern CudnnStatus cudnnDestroyOpTensorDescriptor(IntPtr opTensorDesc);

    /// <summary>Executes element-wise tensor operation.</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnOpTensor")]
    public static extern CudnnStatus cudnnOpTensor(
        IntPtr handle,
        IntPtr opTensorDesc,
        ref float alpha1,
        IntPtr aDesc, IntPtr A,
        ref float alpha2,
        IntPtr bDesc, IntPtr B,
        ref float beta,
        IntPtr cDesc, IntPtr C);

    #endregion

    #region Transform Tensor

    /// <summary>Transforms tensor format (e.g., NCHW to NHWC).</summary>
    [DllImport(CudnnLibrary, EntryPoint = "cudnnTransformTensor")]
    public static extern CudnnStatus cudnnTransformTensor(
        IntPtr handle,
        ref float alpha,
        IntPtr xDesc, IntPtr x,
        ref float beta,
        IntPtr yDesc, IntPtr y);

    #endregion
}

/// <summary>
/// High-level cuDNN context wrapper with automatic resource management.
/// Provides simplified access to cuDNN operations for neural network layers.
/// </summary>
public sealed class CuDnnContext : IDisposable
{
    private IntPtr _cudnnHandle;
    private CudaBlasContext? _blasContext;
    private bool _disposed;
    private static bool _isAvailable;
    private static bool _checkedAvailability;
    private static readonly object _lock = new object();

    /// <summary>Gets whether cuDNN is available on this system.</summary>
    public static bool IsAvailable
    {
        get
        {
            if (!_checkedAvailability)
            {
                lock (_lock)
                {
                    if (!_checkedAvailability)
                    {
                        _isAvailable = CheckAvailability();
                        _checkedAvailability = true;
                    }
                }
            }
            return _isAvailable;
        }
    }

    /// <summary>Gets the cuDNN version number.</summary>
    public static ulong Version
    {
        get
        {
            try
            {
                return CuDnnNative.cudnnGetVersion();
            }
            catch
            {
                return 0;
            }
        }
    }

    /// <summary>Gets the native cuDNN handle.</summary>
    public IntPtr Handle => _cudnnHandle;

    /// <summary>Gets the CUDA/cuBLAS context for memory operations.</summary>
    public CudaBlasContext BlasContext => _blasContext ?? throw new ObjectDisposedException(nameof(CuDnnContext));

    /// <summary>Creates a new cuDNN context.</summary>
    public CuDnnContext()
    {
        if (!IsAvailable)
            throw new InvalidOperationException("cuDNN is not available on this system.");

        // Create CudaBlasContext which handles CUDA initialization
        _blasContext = new CudaBlasContext();

        // Create cuDNN handle
        var status = CuDnnNative.cudnnCreate(out _cudnnHandle);
        if (status != CuDnnNative.CudnnStatus.Success)
        {
            _blasContext.Dispose();
            _blasContext = null;
            throw new InvalidOperationException($"Failed to create cuDNN handle: {status}");
        }
    }

    /// <summary>Allocates device memory.</summary>
    public CudaDeviceMemory<T> Allocate<T>(long count) where T : unmanaged
    {
        return BlasContext.Allocate<T>(count);
    }

    /// <summary>Copies data from host to device.</summary>
    public void CopyToDevice<T>(CudaDeviceMemory<T> dst, T[] src) where T : unmanaged
    {
        BlasContext.CopyToDevice(dst, src);
    }

    /// <summary>Copies data from device to host.</summary>
    public void CopyToHost<T>(T[] dst, CudaDeviceMemory<T> src) where T : unmanaged
    {
        BlasContext.CopyToHost(dst, src);
    }

    private static bool CheckAvailability()
    {
        try
        {
            // Try to get version to check if library is loaded
            var version = CuDnnNative.cudnnGetVersion();
            return version > 0;
        }
        catch (DllNotFoundException)
        {
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>Gets error message for cuDNN status.</summary>
    public static string GetErrorString(CuDnnNative.CudnnStatus status)
    {
        try
        {
            var ptr = CuDnnNative.cudnnGetErrorString(status);
            return ptr != IntPtr.Zero ? Marshal.PtrToStringAnsi(ptr) ?? status.ToString() : status.ToString();
        }
        catch
        {
            return status.ToString();
        }
    }

    /// <summary>Throws if status indicates an error.</summary>
    public static void CheckStatus(CuDnnNative.CudnnStatus status, string operation)
    {
        if (status != CuDnnNative.CudnnStatus.Success)
        {
            throw new InvalidOperationException($"cuDNN {operation} failed: {GetErrorString(status)}");
        }
    }

    /// <summary>Disposes the cuDNN context.</summary>
    public void Dispose()
    {
        if (_disposed) return;

        if (_cudnnHandle != IntPtr.Zero)
        {
            CuDnnNative.cudnnDestroy(_cudnnHandle);
            _cudnnHandle = IntPtr.Zero;
        }

        _blasContext?.Dispose();
        _blasContext = null;

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~CuDnnContext()
    {
        Dispose();
    }
}

/// <summary>
/// cuDNN tensor descriptor wrapper with automatic resource management.
/// </summary>
public sealed class CuDnnTensorDescriptor : IDisposable
{
    private IntPtr _descriptor;
    private bool _disposed;

    /// <summary>Gets the native descriptor handle.</summary>
    public IntPtr Handle => _descriptor;

    /// <summary>Creates a new tensor descriptor.</summary>
    public CuDnnTensorDescriptor()
    {
        var status = CuDnnNative.cudnnCreateTensorDescriptor(out _descriptor);
        CuDnnContext.CheckStatus(status, "CreateTensorDescriptor");
    }

    /// <summary>Sets the descriptor for a 4D tensor in NCHW format.</summary>
    public void Set4D(CuDnnNative.CudnnDataType dataType, int n, int c, int h, int w)
    {
        var status = CuDnnNative.cudnnSetTensor4dDescriptor(
            _descriptor,
            CuDnnNative.CudnnTensorFormat.NCHW,
            dataType,
            n, c, h, w);
        CuDnnContext.CheckStatus(status, "SetTensor4dDescriptor");
    }

    /// <summary>Sets the descriptor for a 4D tensor with specific format.</summary>
    public void Set4D(CuDnnNative.CudnnTensorFormat format, CuDnnNative.CudnnDataType dataType, int n, int c, int h, int w)
    {
        var status = CuDnnNative.cudnnSetTensor4dDescriptor(
            _descriptor,
            format,
            dataType,
            n, c, h, w);
        CuDnnContext.CheckStatus(status, "SetTensor4dDescriptor");
    }

    /// <summary>Sets the descriptor for an N-dimensional tensor.</summary>
    public void SetNd(CuDnnNative.CudnnDataType dataType, int[] dims, int[] strides)
    {
        var status = CuDnnNative.cudnnSetTensorNdDescriptor(
            _descriptor,
            dataType,
            dims.Length,
            dims,
            strides);
        CuDnnContext.CheckStatus(status, "SetTensorNdDescriptor");
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_descriptor != IntPtr.Zero)
        {
            CuDnnNative.cudnnDestroyTensorDescriptor(_descriptor);
            _descriptor = IntPtr.Zero;
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~CuDnnTensorDescriptor()
    {
        Dispose();
    }
}

/// <summary>
/// cuDNN filter descriptor wrapper with automatic resource management.
/// </summary>
public sealed class CuDnnFilterDescriptor : IDisposable
{
    private IntPtr _descriptor;
    private bool _disposed;

    /// <summary>Gets the native descriptor handle.</summary>
    public IntPtr Handle => _descriptor;

    /// <summary>Creates a new filter descriptor.</summary>
    public CuDnnFilterDescriptor()
    {
        var status = CuDnnNative.cudnnCreateFilterDescriptor(out _descriptor);
        CuDnnContext.CheckStatus(status, "CreateFilterDescriptor");
    }

    /// <summary>Sets the descriptor for a 4D filter (K output channels, C input channels, H, W).</summary>
    public void Set4D(CuDnnNative.CudnnDataType dataType, int k, int c, int h, int w)
    {
        var status = CuDnnNative.cudnnSetFilter4dDescriptor(
            _descriptor,
            dataType,
            CuDnnNative.CudnnTensorFormat.NCHW,
            k, c, h, w);
        CuDnnContext.CheckStatus(status, "SetFilter4dDescriptor");
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_descriptor != IntPtr.Zero)
        {
            CuDnnNative.cudnnDestroyFilterDescriptor(_descriptor);
            _descriptor = IntPtr.Zero;
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~CuDnnFilterDescriptor()
    {
        Dispose();
    }
}

/// <summary>
/// cuDNN convolution descriptor wrapper with automatic resource management.
/// </summary>
public sealed class CuDnnConvolutionDescriptor : IDisposable
{
    private IntPtr _descriptor;
    private bool _disposed;

    /// <summary>Gets the native descriptor handle.</summary>
    public IntPtr Handle => _descriptor;

    /// <summary>Creates a new convolution descriptor.</summary>
    public CuDnnConvolutionDescriptor()
    {
        var status = CuDnnNative.cudnnCreateConvolutionDescriptor(out _descriptor);
        CuDnnContext.CheckStatus(status, "CreateConvolutionDescriptor");
    }

    /// <summary>Sets the descriptor for a 2D convolution.</summary>
    public void Set2D(int padH, int padW, int strideH, int strideW, int dilationH, int dilationW,
        CuDnnNative.CudnnDataType computeType, bool useTensorCores = true)
    {
        var status = CuDnnNative.cudnnSetConvolution2dDescriptor(
            _descriptor,
            padH, padW,
            strideH, strideW,
            dilationH, dilationW,
            CuDnnNative.CudnnConvolutionMode.CrossCorrelation,
            computeType);
        CuDnnContext.CheckStatus(status, "SetConvolution2dDescriptor");

        // Enable tensor cores if requested
        if (useTensorCores)
        {
            status = CuDnnNative.cudnnSetConvolutionMathType(
                _descriptor,
                CuDnnNative.CudnnMathType.TensorOp);
            // Ignore failure - tensor cores may not be available
        }
    }

    /// <summary>Sets the group count for grouped/depthwise convolutions.</summary>
    public void SetGroupCount(int groupCount)
    {
        var status = CuDnnNative.cudnnSetConvolutionGroupCount(_descriptor, groupCount);
        CuDnnContext.CheckStatus(status, "SetConvolutionGroupCount");
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_descriptor != IntPtr.Zero)
        {
            CuDnnNative.cudnnDestroyConvolutionDescriptor(_descriptor);
            _descriptor = IntPtr.Zero;
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~CuDnnConvolutionDescriptor()
    {
        Dispose();
    }
}

/// <summary>
/// cuDNN pooling descriptor wrapper with automatic resource management.
/// </summary>
public sealed class CuDnnPoolingDescriptor : IDisposable
{
    private IntPtr _descriptor;
    private bool _disposed;

    /// <summary>Gets the native descriptor handle.</summary>
    public IntPtr Handle => _descriptor;

    /// <summary>Creates a new pooling descriptor.</summary>
    public CuDnnPoolingDescriptor()
    {
        var status = CuDnnNative.cudnnCreatePoolingDescriptor(out _descriptor);
        CuDnnContext.CheckStatus(status, "CreatePoolingDescriptor");
    }

    /// <summary>Sets the descriptor for 2D pooling.</summary>
    public void Set2D(CuDnnNative.CudnnPoolingMode mode, int windowH, int windowW,
        int padH, int padW, int strideH, int strideW)
    {
        var status = CuDnnNative.cudnnSetPooling2dDescriptor(
            _descriptor,
            mode,
            CuDnnNative.CudnnNanPropagation.NotPropagateNan,
            windowH, windowW,
            padH, padW,
            strideH, strideW);
        CuDnnContext.CheckStatus(status, "SetPooling2dDescriptor");
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_descriptor != IntPtr.Zero)
        {
            CuDnnNative.cudnnDestroyPoolingDescriptor(_descriptor);
            _descriptor = IntPtr.Zero;
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~CuDnnPoolingDescriptor()
    {
        Dispose();
    }
}

/// <summary>
/// cuDNN activation descriptor wrapper with automatic resource management.
/// </summary>
public sealed class CuDnnActivationDescriptor : IDisposable
{
    private IntPtr _descriptor;
    private bool _disposed;

    /// <summary>Gets the native descriptor handle.</summary>
    public IntPtr Handle => _descriptor;

    /// <summary>Creates a new activation descriptor.</summary>
    public CuDnnActivationDescriptor()
    {
        var status = CuDnnNative.cudnnCreateActivationDescriptor(out _descriptor);
        CuDnnContext.CheckStatus(status, "CreateActivationDescriptor");
    }

    /// <summary>Sets the activation mode.</summary>
    public void Set(CuDnnNative.CudnnActivationMode mode, double coef = 0.0)
    {
        var status = CuDnnNative.cudnnSetActivationDescriptor(
            _descriptor,
            mode,
            CuDnnNative.CudnnNanPropagation.NotPropagateNan,
            coef);
        CuDnnContext.CheckStatus(status, "SetActivationDescriptor");
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_descriptor != IntPtr.Zero)
        {
            CuDnnNative.cudnnDestroyActivationDescriptor(_descriptor);
            _descriptor = IntPtr.Zero;
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~CuDnnActivationDescriptor()
    {
        Dispose();
    }
}

/// <summary>
/// High-level cuDNN convolution operations helper.
/// Provides simplified API for neural network convolutional layers.
/// </summary>
public sealed class CuDnnConvolution : IDisposable
{
    private readonly CuDnnContext _context;
    private readonly bool _ownsContext;
    private IntPtr _workspace;
    private ulong _workspaceSize;
    private bool _disposed;

    /// <summary>Creates a new cuDNN convolution helper with shared context.</summary>
    public CuDnnConvolution(CuDnnContext context)
    {
        _context = context ?? throw new ArgumentNullException(nameof(context));
        _ownsContext = false;
    }

    /// <summary>Creates a new cuDNN convolution helper with its own context.</summary>
    public CuDnnConvolution()
    {
        _context = new CuDnnContext();
        _ownsContext = true;
    }

    /// <summary>Gets whether cuDNN convolution is available.</summary>
    public static bool IsAvailable => CuDnnContext.IsAvailable;

    /// <summary>
    /// Performs 2D convolution forward pass.
    /// </summary>
    /// <param name="input">Input tensor [N, C, H, W]</param>
    /// <param name="filter">Filter tensor [K, C, FH, FW]</param>
    /// <param name="n">Batch size</param>
    /// <param name="c">Input channels</param>
    /// <param name="h">Input height</param>
    /// <param name="w">Input width</param>
    /// <param name="k">Output channels (number of filters)</param>
    /// <param name="filterH">Filter height</param>
    /// <param name="filterW">Filter width</param>
    /// <param name="padH">Padding height</param>
    /// <param name="padW">Padding width</param>
    /// <param name="strideH">Stride height</param>
    /// <param name="strideW">Stride width</param>
    /// <returns>Output tensor [N, K, outH, outW]</returns>
    public float[]? Conv2DForward(
        float[] input, float[] filter,
        int n, int c, int h, int w,
        int k, int filterH, int filterW,
        int padH = 0, int padW = 0,
        int strideH = 1, int strideW = 1)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CuDnnConvolution));

        try
        {
            // Create descriptors
            using var inputDesc = new CuDnnTensorDescriptor();
            using var filterDesc = new CuDnnFilterDescriptor();
            using var convDesc = new CuDnnConvolutionDescriptor();
            using var outputDesc = new CuDnnTensorDescriptor();

            inputDesc.Set4D(CuDnnNative.CudnnDataType.Float, n, c, h, w);
            filterDesc.Set4D(CuDnnNative.CudnnDataType.Float, k, c, filterH, filterW);
            convDesc.Set2D(padH, padW, strideH, strideW, 1, 1, CuDnnNative.CudnnDataType.Float);

            // Get output dimensions
            int outN, outC, outH, outW;
            var status = CuDnnNative.cudnnGetConvolution2dForwardOutputDim(
                convDesc.Handle, inputDesc.Handle, filterDesc.Handle,
                out outN, out outC, out outH, out outW);
            CuDnnContext.CheckStatus(status, "GetConvolution2dForwardOutputDim");

            outputDesc.Set4D(CuDnnNative.CudnnDataType.Float, outN, outC, outH, outW);

            // Get workspace size (use implicit GEMM algorithm)
            var algo = CuDnnNative.CudnnConvolutionFwdAlgo.ImplicitPrecompGemm;
            ulong workspaceSize;
            status = CuDnnNative.cudnnGetConvolutionForwardWorkspaceSize(
                _context.Handle, inputDesc.Handle, filterDesc.Handle,
                convDesc.Handle, outputDesc.Handle, algo, out workspaceSize);

            if (status != CuDnnNative.CudnnStatus.Success)
            {
                // Fall back to GEMM algorithm
                algo = CuDnnNative.CudnnConvolutionFwdAlgo.Gemm;
                status = CuDnnNative.cudnnGetConvolutionForwardWorkspaceSize(
                    _context.Handle, inputDesc.Handle, filterDesc.Handle,
                    convDesc.Handle, outputDesc.Handle, algo, out workspaceSize);
                CuDnnContext.CheckStatus(status, "GetConvolutionForwardWorkspaceSize");
            }

            // Allocate GPU memory
            int inputSize = n * c * h * w;
            int filterSize = k * c * filterH * filterW;
            int outputSize = outN * outC * outH * outW;

            using var gpuInput = _context.Allocate<float>(inputSize);
            using var gpuFilter = _context.Allocate<float>(filterSize);
            using var gpuOutput = _context.Allocate<float>(outputSize);

            _context.CopyToDevice(gpuInput, input);
            _context.CopyToDevice(gpuFilter, filter);

            // Allocate workspace if needed
            IntPtr workspace = IntPtr.Zero;
            if (workspaceSize > 0)
            {
                EnsureWorkspace(workspaceSize);
                workspace = _workspace;
            }

            // Execute convolution
            float alpha = 1.0f;
            float beta = 0.0f;

            status = CuDnnNative.cudnnConvolutionForward(
                _context.Handle,
                ref alpha,
                inputDesc.Handle, gpuInput.DevicePtr,
                filterDesc.Handle, gpuFilter.DevicePtr,
                convDesc.Handle,
                algo,
                workspace, workspaceSize,
                ref beta,
                outputDesc.Handle, gpuOutput.DevicePtr);
            CuDnnContext.CheckStatus(status, "ConvolutionForward");

            // Copy result back
            var output = new float[outputSize];
            _context.CopyToHost(output, gpuOutput);

            return output;
        }
        catch (Exception)
        {
            return null;
        }
    }

    private void EnsureWorkspace(ulong requiredSize)
    {
        if (_workspaceSize >= requiredSize) return;

        // Free old workspace
        if (_workspace != IntPtr.Zero)
        {
            CuBlasNative.cuMemFree(_workspace);
            _workspace = IntPtr.Zero;
            _workspaceSize = 0;
        }

        // Allocate new workspace
        var result = CuBlasNative.cuMemAlloc(out _workspace, requiredSize);
        if (result != CudaResult.Success)
        {
            throw new InvalidOperationException($"Failed to allocate cuDNN workspace: {result}");
        }
        _workspaceSize = requiredSize;
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_workspace != IntPtr.Zero)
        {
            CuBlasNative.cuMemFree(_workspace);
            _workspace = IntPtr.Zero;
        }

        if (_ownsContext)
        {
            _context.Dispose();
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~CuDnnConvolution()
    {
        Dispose();
    }
}

/// <summary>
/// High-level cuDNN pooling operations helper.
/// </summary>
public sealed class CuDnnPooling : IDisposable
{
    private readonly CuDnnContext _context;
    private readonly bool _ownsContext;
    private bool _disposed;

    public CuDnnPooling(CuDnnContext context)
    {
        _context = context ?? throw new ArgumentNullException(nameof(context));
        _ownsContext = false;
    }

    public CuDnnPooling()
    {
        _context = new CuDnnContext();
        _ownsContext = true;
    }

    public static bool IsAvailable => CuDnnContext.IsAvailable;

    /// <summary>
    /// Performs 2D max pooling forward pass.
    /// </summary>
    public float[]? MaxPool2DForward(
        float[] input,
        int n, int c, int h, int w,
        int windowH, int windowW,
        int padH = 0, int padW = 0,
        int strideH = 2, int strideW = 2)
    {
        return Pool2DForward(input, n, c, h, w, windowH, windowW, padH, padW, strideH, strideW,
            CuDnnNative.CudnnPoolingMode.Max);
    }

    /// <summary>
    /// Performs 2D average pooling forward pass.
    /// </summary>
    public float[]? AvgPool2DForward(
        float[] input,
        int n, int c, int h, int w,
        int windowH, int windowW,
        int padH = 0, int padW = 0,
        int strideH = 2, int strideW = 2)
    {
        return Pool2DForward(input, n, c, h, w, windowH, windowW, padH, padW, strideH, strideW,
            CuDnnNative.CudnnPoolingMode.AverageCountExcludePadding);
    }

    private float[]? Pool2DForward(
        float[] input,
        int n, int c, int h, int w,
        int windowH, int windowW,
        int padH, int padW,
        int strideH, int strideW,
        CuDnnNative.CudnnPoolingMode mode)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CuDnnPooling));

        try
        {
            using var inputDesc = new CuDnnTensorDescriptor();
            using var poolDesc = new CuDnnPoolingDescriptor();
            using var outputDesc = new CuDnnTensorDescriptor();

            inputDesc.Set4D(CuDnnNative.CudnnDataType.Float, n, c, h, w);
            poolDesc.Set2D(mode, windowH, windowW, padH, padW, strideH, strideW);

            // Get output dimensions
            int outN, outC, outH, outW;
            var status = CuDnnNative.cudnnGetPooling2dForwardOutputDim(
                poolDesc.Handle, inputDesc.Handle,
                out outN, out outC, out outH, out outW);
            CuDnnContext.CheckStatus(status, "GetPooling2dForwardOutputDim");

            outputDesc.Set4D(CuDnnNative.CudnnDataType.Float, outN, outC, outH, outW);

            // Allocate GPU memory
            int inputSize = n * c * h * w;
            int outputSize = outN * outC * outH * outW;

            using var gpuInput = _context.Allocate<float>(inputSize);
            using var gpuOutput = _context.Allocate<float>(outputSize);

            _context.CopyToDevice(gpuInput, input);

            float alpha = 1.0f;
            float beta = 0.0f;

            status = CuDnnNative.cudnnPoolingForward(
                _context.Handle,
                poolDesc.Handle,
                ref alpha,
                inputDesc.Handle, gpuInput.DevicePtr,
                ref beta,
                outputDesc.Handle, gpuOutput.DevicePtr);
            CuDnnContext.CheckStatus(status, "PoolingForward");

            var output = new float[outputSize];
            _context.CopyToHost(output, gpuOutput);

            return output;
        }
        catch (Exception)
        {
            return null;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_ownsContext)
        {
            _context.Dispose();
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~CuDnnPooling()
    {
        Dispose();
    }
}

/// <summary>
/// High-level cuDNN batch normalization operations helper.
/// </summary>
public sealed class CuDnnBatchNorm : IDisposable
{
    private readonly CuDnnContext _context;
    private readonly bool _ownsContext;
    private bool _disposed;

    public CuDnnBatchNorm(CuDnnContext context)
    {
        _context = context ?? throw new ArgumentNullException(nameof(context));
        _ownsContext = false;
    }

    public CuDnnBatchNorm()
    {
        _context = new CuDnnContext();
        _ownsContext = true;
    }

    public static bool IsAvailable => CuDnnContext.IsAvailable;

    /// <summary>
    /// Performs batch normalization inference (no running statistics update).
    /// </summary>
    public float[]? ForwardInference(
        float[] input,
        float[] scale, float[] bias,
        float[] runningMean, float[] runningVariance,
        int n, int c, int h, int w,
        double epsilon = 1e-5)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CuDnnBatchNorm));

        try
        {
            using var inputDesc = new CuDnnTensorDescriptor();
            using var outputDesc = new CuDnnTensorDescriptor();
            using var bnDesc = new CuDnnTensorDescriptor();

            inputDesc.Set4D(CuDnnNative.CudnnDataType.Float, n, c, h, w);
            outputDesc.Set4D(CuDnnNative.CudnnDataType.Float, n, c, h, w);

            // Derive BN tensor descriptor
            var status = CuDnnNative.cudnnDeriveBNTensorDescriptor(
                bnDesc.Handle, inputDesc.Handle,
                CuDnnNative.CudnnBatchNormMode.Spatial);
            CuDnnContext.CheckStatus(status, "DeriveBNTensorDescriptor");

            int tensorSize = n * c * h * w;

            using var gpuInput = _context.Allocate<float>(tensorSize);
            using var gpuOutput = _context.Allocate<float>(tensorSize);
            using var gpuScale = _context.Allocate<float>(c);
            using var gpuBias = _context.Allocate<float>(c);
            using var gpuMean = _context.Allocate<float>(c);
            using var gpuVariance = _context.Allocate<float>(c);

            _context.CopyToDevice(gpuInput, input);
            _context.CopyToDevice(gpuScale, scale);
            _context.CopyToDevice(gpuBias, bias);
            _context.CopyToDevice(gpuMean, runningMean);
            _context.CopyToDevice(gpuVariance, runningVariance);

            float alpha = 1.0f;
            float beta = 0.0f;

            status = CuDnnNative.cudnnBatchNormalizationForwardInference(
                _context.Handle,
                CuDnnNative.CudnnBatchNormMode.Spatial,
                ref alpha,
                ref beta,
                inputDesc.Handle, gpuInput.DevicePtr,
                outputDesc.Handle, gpuOutput.DevicePtr,
                bnDesc.Handle,
                gpuScale.DevicePtr,
                gpuBias.DevicePtr,
                gpuMean.DevicePtr,
                gpuVariance.DevicePtr,
                epsilon);
            CuDnnContext.CheckStatus(status, "BatchNormalizationForwardInference");

            var output = new float[tensorSize];
            _context.CopyToHost(output, gpuOutput);

            return output;
        }
        catch (Exception)
        {
            return null;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_ownsContext)
        {
            _context.Dispose();
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~CuDnnBatchNorm()
    {
        Dispose();
    }
}

/// <summary>
/// High-level cuDNN softmax operations helper.
/// </summary>
public sealed class CuDnnSoftmax : IDisposable
{
    private readonly CuDnnContext _context;
    private readonly bool _ownsContext;
    private bool _disposed;

    public CuDnnSoftmax(CuDnnContext context)
    {
        _context = context ?? throw new ArgumentNullException(nameof(context));
        _ownsContext = false;
    }

    public CuDnnSoftmax()
    {
        _context = new CuDnnContext();
        _ownsContext = true;
    }

    public static bool IsAvailable => CuDnnContext.IsAvailable;

    /// <summary>
    /// Performs softmax forward pass over the channel dimension.
    /// </summary>
    public float[]? Forward(float[] input, int n, int c, int h = 1, int w = 1)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CuDnnSoftmax));

        try
        {
            using var inputDesc = new CuDnnTensorDescriptor();
            using var outputDesc = new CuDnnTensorDescriptor();

            inputDesc.Set4D(CuDnnNative.CudnnDataType.Float, n, c, h, w);
            outputDesc.Set4D(CuDnnNative.CudnnDataType.Float, n, c, h, w);

            int tensorSize = n * c * h * w;

            using var gpuInput = _context.Allocate<float>(tensorSize);
            using var gpuOutput = _context.Allocate<float>(tensorSize);

            _context.CopyToDevice(gpuInput, input);

            float alpha = 1.0f;
            float beta = 0.0f;

            var status = CuDnnNative.cudnnSoftmaxForward(
                _context.Handle,
                CuDnnNative.CudnnSoftmaxAlgorithm.Accurate,
                CuDnnNative.CudnnSoftmaxMode.Channel,
                ref alpha,
                inputDesc.Handle, gpuInput.DevicePtr,
                ref beta,
                outputDesc.Handle, gpuOutput.DevicePtr);
            CuDnnContext.CheckStatus(status, "SoftmaxForward");

            var output = new float[tensorSize];
            _context.CopyToHost(output, gpuOutput);

            return output;
        }
        catch (Exception)
        {
            return null;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;

        if (_ownsContext)
        {
            _context.Dispose();
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    ~CuDnnSoftmax()
    {
        Dispose();
    }
}
#endif
