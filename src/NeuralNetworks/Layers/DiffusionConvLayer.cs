using AiDotNet.Autodiff;
using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements diffusion convolution for mesh surface processing using the heat diffusion equation.
/// </summary>
/// <remarks>
/// <para>
/// DiffusionConvLayer applies learned diffusion kernels on mesh surfaces to aggregate
/// information across geodesic neighborhoods. Instead of using fixed spatial neighborhoods,
/// it leverages the heat diffusion equation to create adaptive receptive fields that
/// respect the underlying geometry.
/// </para>
/// <para><b>For Beginners:</b> Think of heat spreading on a surface:
/// 
/// - Place a heat source at each vertex
/// - Let the heat diffuse across the mesh surface
/// - After some time, nearby vertices (in geodesic distance) will share heat
/// - Use this diffusion pattern to aggregate features from neighbors
/// 
/// Key advantages:
/// - Respects the true surface geometry (not just mesh connectivity)
/// - Adaptive receptive field based on diffusion time
/// - Robust to mesh discretization and vertex density
/// 
/// Applications:
/// - Shape classification and segmentation
/// - Surface analysis (curvature, normals, features)
/// - Correspondence between shapes
/// - Texture synthesis on surfaces
/// </para>
/// <para>
/// Reference: "DiffusionNet: Discretization Agnostic Learning on Surfaces" by Sharp et al.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class DiffusionConvLayer<T> : LayerBase<T>
{
    #region Properties

    /// <summary>
    /// Gets the number of input feature channels per vertex.
    /// </summary>
    public int InputChannels { get; private set; }

    /// <summary>
    /// Gets the number of output feature channels per vertex.
    /// </summary>
    public int OutputChannels { get; private set; }

    /// <summary>
    /// Gets the number of diffusion time scales to use.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Multiple time scales capture different spatial extents. Small times capture
    /// local features while large times capture global structure. Common choices
    /// are 4-8 time scales.
    /// </para>
    /// </remarks>
    public int NumTimeScales { get; private set; }

    /// <summary>
    /// Gets the learned diffusion time values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are learnable parameters that determine the spatial extent of diffusion.
    /// Typically initialized as log-spaced values from small (local) to large (global).
    /// </para>
    /// </remarks>
    public T[] DiffusionTimes { get; private set; }

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU-accelerated forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// DiffusionConvLayer supports GPU execution for spectral diffusion operations.
    /// GPU training is supported when an eigenbasis is available.
    /// </para>
    /// </remarks>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU training.
    /// </summary>
    public override bool SupportsGpuTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    public override bool SupportsJitCompilation => _weights != null && _biases != null && CanActivationBeJitted();

    #endregion

    #region Private Fields

    /// <summary>
    /// Learnable weights [OutputChannels, InputChannels * NumTimeScales].
    /// </summary>
    private Tensor<T> _weights;

    /// <summary>
    /// Learnable bias values [OutputChannels].
    /// </summary>
    private Tensor<T> _biases;

    /// <summary>
    /// Cached weight gradients from backward pass.
    /// </summary>
    private Tensor<T>? _weightsGradient;

    /// <summary>
    /// Cached bias gradients from backward pass.
    /// </summary>
    private Tensor<T>? _biasesGradient;

    /// <summary>
    /// Cached input from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached pre-activation output from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastPreActivation;

    /// <summary>
    /// Cached output from the last forward pass.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Cached diffused features for backward pass [numVertices, InputChannels * NumTimeScales].
    /// </summary>
    private Tensor<T>? _diffusedFeatures;

    /// <summary>
    /// Laplacian matrix for the current mesh [numVertices, numVertices].
    /// </summary>
    private Tensor<T>? _laplacian;

    /// <summary>
    /// Mass matrix (vertex areas) for the current mesh [numVertices].
    /// </summary>
    private Tensor<T>? _massMatrix;

    /// <summary>
    /// Eigenvalues of the Laplacian [numEigenvalues].
    /// </summary>
    private T[]? _eigenvalues;

    /// <summary>
    /// Eigenvectors of the Laplacian [numVertices, numEigenvalues].
    /// </summary>
    private Tensor<T>? _eigenvectors;

    /// <summary>
    /// Number of eigenvectors to use for spectral acceleration.
    /// </summary>
    private readonly int _numEigenvectors;

    /// <summary>
    /// Controls automatic eigenbasis computation for CPU execution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// null = automatic (compute when missing), true = always compute, false = use direct
    /// Laplacian method when eigenbasis is missing.
    /// </para>
    /// </remarks>
    private readonly bool? _preferSpectralDiffusion;

    /// <summary>
    /// Synchronizes automatic eigenbasis computation.
    /// </summary>
    private readonly object _eigenbasisLock = new();

    /// <summary>
    /// Cached gradients for diffusion time parameters from backward pass.      
    /// </summary>
    private T[]? _diffusionTimesGradient;

    /// <summary>
    /// Cached GPU input from the last forward pass.
    /// </summary>
    private IGpuTensor<T>? _gpuInput;

    /// <summary>
    /// Cached GPU input shape from the last forward pass.
    /// </summary>
    private int[]? _gpuInputShape;

    /// <summary>
    /// Cached GPU diffused features for backward pass.
    /// </summary>
    private IGpuTensor<T>? _gpuDiffusedFeatures;

    /// <summary>
    /// Cached GPU pre-activation output for backward pass.
    /// </summary>
    private IGpuTensor<T>? _gpuPreActivation;

    /// <summary>
    /// Cached GPU activated output for backward pass.
    /// </summary>
    private IGpuTensor<T>? _gpuOutput;

    /// <summary>
    /// GPU weight tensor.
    /// </summary>
    private GpuTensor<T>? _gpuWeights;

    /// <summary>
    /// GPU bias tensor.
    /// </summary>
    private GpuTensor<T>? _gpuBiases;

    /// <summary>
    /// GPU diffusion time tensor.
    /// </summary>
    private GpuTensor<T>? _gpuDiffusionTimes;

    /// <summary>
    /// GPU weight gradients.
    /// </summary>
    private GpuTensor<T>? _gpuWeightsGradient;

    /// <summary>
    /// GPU bias gradients.
    /// </summary>
    private GpuTensor<T>? _gpuBiasesGradient;

    /// <summary>
    /// GPU diffusion time gradients.
    /// </summary>
    private GpuTensor<T>? _gpuDiffusionTimesGradient;

    /// <summary>
    /// GPU optimizer state for weights.
    /// </summary>
    private GpuTensor<T>? _gpuWeightsVelocity;
    private GpuTensor<T>? _gpuWeightsM;
    private GpuTensor<T>? _gpuWeightsV;

    /// <summary>
    /// GPU optimizer state for biases.
    /// </summary>
    private GpuTensor<T>? _gpuBiasesVelocity;
    private GpuTensor<T>? _gpuBiasesM;
    private GpuTensor<T>? _gpuBiasesV;

    /// <summary>
    /// GPU optimizer state for diffusion times.
    /// </summary>
    private GpuTensor<T>? _gpuDiffusionTimesVelocity;
    private GpuTensor<T>? _gpuDiffusionTimesM;
    private GpuTensor<T>? _gpuDiffusionTimesV;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the <see cref="DiffusionConvLayer{T}"/> class.
    /// </summary>
    /// <param name="inputChannels">Number of input feature channels per vertex.</param>
    /// <param name="outputChannels">Number of output feature channels per vertex.</param>
    /// <param name="numTimeScales">Number of diffusion time scales to use (default: 4).</param>
    /// <param name="numEigenvectors">Number of eigenvectors for spectral acceleration (default: 128).</param>
    /// <param name="numVertices">Expected number of vertices (for shape calculation).</param>
    /// <param name="activation">Activation function to apply. Defaults to ReLU.</param>
    /// <param name="preferSpectralDiffusion">
    /// Controls CPU execution when eigenbasis is missing. null = automatic (compute),
    /// true = always compute, false = use direct Laplacian method.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when parameters are non-positive.</exception>
    public DiffusionConvLayer(
        int inputChannels,
        int outputChannels,
        int numTimeScales = 4,
        int numEigenvectors = 128,
        int numVertices = 1,
        IActivationFunction<T>? activation = null,
        bool? preferSpectralDiffusion = null)
        : base(
            [numVertices, inputChannels],
            [numVertices, outputChannels],
            activation ?? new ReLUActivation<T>())
    {
        ValidateParameters(inputChannels, outputChannels, numTimeScales, numEigenvectors);

        InputChannels = inputChannels;
        OutputChannels = outputChannels;
        NumTimeScales = numTimeScales;
        _numEigenvectors = numEigenvectors;
        _preferSpectralDiffusion = preferSpectralDiffusion;

        DiffusionTimes = InitializeDiffusionTimes(numTimeScales);

        int weightSize = inputChannels * numTimeScales;
        _weights = new Tensor<T>([outputChannels, weightSize]);
        _biases = new Tensor<T>([outputChannels]);

        InitializeWeights();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DiffusionConvLayer{T}"/> class with vector activation.
    /// </summary>
    /// <param name="inputChannels">Number of input feature channels per vertex.</param>
    /// <param name="outputChannels">Number of output feature channels per vertex.</param>
    /// <param name="numTimeScales">Number of diffusion time scales to use.</param>
    /// <param name="numEigenvectors">Number of eigenvectors for spectral acceleration.</param>
    /// <param name="numVertices">Expected number of vertices (for shape calculation).</param>
    /// <param name="vectorActivation">Vector activation function to apply.</param>
    /// <param name="preferSpectralDiffusion">
    /// Controls CPU execution when eigenbasis is missing. null = automatic (compute),
    /// true = always compute, false = use direct Laplacian method.
    /// </param>
    public DiffusionConvLayer(
        int inputChannels,
        int outputChannels,
        int numTimeScales = 4,
        int numEigenvectors = 128,
        int numVertices = 1,
        IVectorActivationFunction<T>? vectorActivation = null,
        bool? preferSpectralDiffusion = null)
        : base(
            [numVertices, inputChannels],
            [numVertices, outputChannels],
            vectorActivation ?? new ReLUActivation<T>())
    {
        ValidateParameters(inputChannels, outputChannels, numTimeScales, numEigenvectors);

        InputChannels = inputChannels;
        OutputChannels = outputChannels;
        NumTimeScales = numTimeScales;
        _numEigenvectors = numEigenvectors;
        _preferSpectralDiffusion = preferSpectralDiffusion;

        DiffusionTimes = InitializeDiffusionTimes(numTimeScales);

        int weightSize = inputChannels * numTimeScales;
        _weights = new Tensor<T>([outputChannels, weightSize]);
        _biases = new Tensor<T>([outputChannels]);

        InitializeWeights();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Validates constructor parameters.
    /// </summary>
    private static void ValidateParameters(int inputChannels, int outputChannels, int numTimeScales, int numEigenvectors)
    {
        if (inputChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputChannels), "Input channels must be positive.");
        if (outputChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputChannels), "Output channels must be positive.");
        if (numTimeScales <= 0)
            throw new ArgumentOutOfRangeException(nameof(numTimeScales), "Number of time scales must be positive.");
        if (numEigenvectors <= 0)
            throw new ArgumentOutOfRangeException(nameof(numEigenvectors), "Number of eigenvectors must be positive.");
    }

    /// <summary>
    /// Initializes diffusion times as log-spaced values.
    /// </summary>
    /// <param name="numTimeScales">Number of time scales.</param>
    /// <returns>Array of diffusion time values.</returns>
    private T[] InitializeDiffusionTimes(int numTimeScales)
    {
        var times = new T[numTimeScales];
        double minTime = 0.01;
        double maxTime = 1.0;
        double logMin = Math.Log(minTime);
        double logMax = Math.Log(maxTime);

        // Handle edge case where numTimeScales is 1
        if (numTimeScales == 1)
        {
            // Use geometric mean of min and max times
            times[0] = NumOps.FromDouble(Math.Sqrt(minTime * maxTime));
            return times;
        }

        for (int i = 0; i < numTimeScales; i++)
        {
            double logT = logMin + (logMax - logMin) * i / (numTimeScales - 1);
            times[i] = NumOps.FromDouble(Math.Exp(logT));
        }

        return times;
    }

    /// <summary>
    /// Initializes weights using He initialization.
    /// </summary>
    private void InitializeWeights()
    {
        int fanIn = InputChannels * NumTimeScales;
        // === Vectorized: He initialization using TensorRandomUniformRange (Phase C: New IEngine methods) ===
        T scale = NumOps.Sqrt(NumericalStabilityHelper.SafeDiv(
            NumOps.FromDouble(2.0),
            NumOps.FromDouble(fanIn)));

        // Initialize weights in [-scale, scale] range
        _weights = Engine.TensorRandomUniformRange<T>(_weights.Shape, NumOps.Negate(scale), scale);

        // Initialize biases to zero
        _biases = new Tensor<T>(_biases.Shape);
        Engine.TensorFill(_biases, NumOps.Zero);
    }

    #endregion

    #region Mesh Configuration

    /// <summary>
    /// Sets the Laplacian eigenbasis for the current mesh.
    /// </summary>
    /// <param name="eigenvalues">Eigenvalues of the mesh Laplacian.</param>
    /// <param name="eigenvectors">Eigenvectors of the mesh Laplacian [numVertices, numEigenvalues].</param>
    /// <param name="massMatrix">Vertex area weights (optional).</param>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> The eigenbasis allows efficient computation of diffusion
    /// on the mesh surface. You can precompute this from the mesh Laplacian matrix.</para>
    /// </remarks>
    public void SetEigenbasis(T[] eigenvalues, Tensor<T> eigenvectors, Tensor<T>? massMatrix = null)
    {
        if (eigenvalues == null)
            throw new ArgumentNullException(nameof(eigenvalues));
        if (eigenvectors == null)
            throw new ArgumentNullException(nameof(eigenvectors));

        _eigenvalues = eigenvalues;
        _eigenvectors = eigenvectors;
        _massMatrix = massMatrix;
    }

    /// <summary>
    /// Sets the Laplacian matrix directly for the current mesh.
    /// </summary>
    /// <param name="laplacian">The mesh Laplacian matrix [numVertices, numVertices].</param>
    /// <param name="massMatrix">Vertex area weights (optional).</param>
    /// <remarks>
    /// <para>
    /// If the eigenbasis is not precomputed, it will be derived from the Laplacian
    /// automatically unless the layer is configured to prefer the direct method.
    /// Automatic eigenbasis computation ignores the mass matrix; provide an explicit
    /// eigenbasis when massMatrix is required.
    /// </para>
    /// </remarks>
    public void SetLaplacian(Tensor<T> laplacian, Tensor<T>? massMatrix = null)
    {
        _laplacian = laplacian ?? throw new ArgumentNullException(nameof(laplacian));
        _massMatrix = massMatrix;
        _eigenvalues = null;
        _eigenvectors = null;
    }

    #endregion

    #region Forward Pass

    /// <summary>
    /// Performs the forward pass of diffusion convolution.
    /// </summary>
    /// <param name="input">
    /// Vertex features tensor with shape [numVertices, InputChannels] or
    /// [batch, numVertices, InputChannels].
    /// </param>
    /// <returns>
    /// Output tensor with shape [numVertices, OutputChannels] or
    /// [batch, numVertices, OutputChannels].
    /// </returns>
    /// <exception cref="InvalidOperationException">Thrown when eigenbasis/laplacian is not set.</exception>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (_eigenvalues == null && _laplacian == null)
        {
            throw new InvalidOperationException(
                "Eigenbasis or Laplacian must be set via SetEigenbasis or SetLaplacian before calling Forward.");
        }

        _lastInput = input;

        bool hasBatch = input.Rank == 3;
        int numVertices = hasBatch ? input.Shape[1] : input.Shape[0];
        int inputChannels = hasBatch ? input.Shape[2] : input.Shape[1];

        if (inputChannels != InputChannels)
        {
            throw new ArgumentException(
                $"Input channels ({inputChannels}) must match layer InputChannels ({InputChannels}).",
                nameof(input));
        }

        Tensor<T> output;

        if (hasBatch)
        {
            int batchSize = input.Shape[0];
            output = ProcessBatched(input, batchSize, numVertices);
        }
        else
        {
            output = ProcessSingle(input, numVertices);
        }

        _lastPreActivation = output;
        var activated = ApplyActivation(output);
        _lastOutput = activated;

        return activated;
    }

    /// <summary>
    /// Processes a single (non-batched) input tensor.
    /// </summary>
    private Tensor<T> ProcessSingle(Tensor<T> input, int numVertices)
    {
        var diffused = ComputeDiffusedFeatures(input, numVertices);
        _diffusedFeatures = diffused;

        var transposedWeights = Engine.TensorTranspose(_weights);
        var output = Engine.TensorMatMul(diffused, transposedWeights);
        output = AddBiases(output, numVertices);

        return output;
    }

    /// <summary>
    /// Processes a batched input tensor.
    /// </summary>
    private Tensor<T> ProcessBatched(Tensor<T> input, int batchSize, int numVertices)
    {
        var outputData = new T[batchSize * numVertices * OutputChannels];
        if (_preferSpectralDiffusion != false && (_eigenvalues == null || _eigenvectors == null))
        {
            EnsureEigenbasisForExecution();
        }

        Parallel.For(0, batchSize, b =>
        {
            var singleInput = ExtractBatchSlice(input, b, numVertices);
            var singleOutput = ProcessSingle(singleInput, numVertices);
            var singleData = singleOutput.ToArray();

            int offset = b * numVertices * OutputChannels;
            Array.Copy(singleData, 0, outputData, offset, singleData.Length);
        });

        return new Tensor<T>(outputData, [batchSize, numVertices, OutputChannels]);
    }

    /// <summary>
    /// Extracts a single sample from a batched tensor.
    /// </summary>
    private Tensor<T> ExtractBatchSlice(Tensor<T> batched, int batchIndex, int numVertices)
    {
        int channels = batched.Shape[2];
        var data = new T[numVertices * channels];

        for (int v = 0; v < numVertices; v++)
        {
            for (int c = 0; c < channels; c++)
            {
                data[v * channels + c] = batched[batchIndex, v, c];
            }
        }

        return new Tensor<T>(data, [numVertices, channels]);
    }

    /// <summary>
    /// Computes diffused features at multiple time scales.
    /// </summary>
    /// <param name="input">Input features [numVertices, InputChannels].</param>
    /// <param name="numVertices">Number of vertices.</param>
    /// <returns>Diffused features [numVertices, InputChannels * NumTimeScales].</returns>
    private Tensor<T> ComputeDiffusedFeatures(Tensor<T> input, int numVertices)
    {
        int diffusedSize = InputChannels * NumTimeScales;
        var diffused = new T[numVertices * diffusedSize];

        if (_eigenvalues == null || _eigenvectors == null)
        {
            if (_preferSpectralDiffusion != false)
            {
                EnsureEigenbasisForExecution();
            }
        }

        if (_eigenvalues != null && _eigenvectors != null)
        {
            ComputeDiffusionSpectral(input, diffused, numVertices);
        }
        else if (_laplacian != null)
        {
            ComputeDiffusionDirect(input, diffused, numVertices);
        }

        return new Tensor<T>(diffused, [numVertices, diffusedSize]);
    }

    /// <summary>
    /// Computes diffusion using spectral method (eigenbasis) with vectorized operations.
    /// </summary>
    /// <param name="input">Input features.</param>
    /// <param name="diffused">Output buffer for diffused features.</param>
    /// <param name="numVertices">Number of vertices.</param>
    /// <remarks>
    /// <para>
    /// The spectral diffusion is computed as:
    /// diffused = Phi @ diag(exp(-lambda * t)) @ Phi^T @ input
    /// where Phi is the eigenvector matrix and lambda are eigenvalues.
    /// </para>
    /// <para>
    /// We use vectorized operations:
    /// 1. Project input to spectral domain: coeffs = Phi^T @ input
    /// 2. Apply heat kernel: coeffs = coeffs * exp(-lambda * t)
    /// 3. Project back: output = Phi @ coeffs
    /// </para>
    /// </remarks>
    private void ComputeDiffusionSpectral(Tensor<T> input, T[] diffused, int numVertices)
    {
        if (_eigenvalues == null || _eigenvectors == null)
            return;

        int numEig = Math.Min(_numEigenvectors, _eigenvalues.Length);

        // Transpose eigenvectors for spectral projection: [numEig, numVertices]
        var eigenvectorsTransposed = Engine.TensorTranspose(_eigenvectors);

        // Create eigenvalue tensor for vectorized decay computation
        // Use explicit array copy instead of slice syntax for net471 compatibility
        var eigenvalueArray = new T[numEig];
        Array.Copy(_eigenvalues, eigenvalueArray, numEig);
        var eigenvalueTensor = new Tensor<T>(eigenvalueArray, [numEig]);

        for (int t = 0; t < NumTimeScales; t++)
        {
            T time = DiffusionTimes[t];

            // Vectorized decay factor computation: exp(-eigenvalue * time)
            var negEigTimesTime = Engine.TensorMultiplyScalar(Engine.TensorNegate(eigenvalueTensor), time);
            var decayVector = Engine.TensorExp(negEigTimesTime);
            var decayTensor = decayVector.Reshape([numEig, 1]);

            // Step 1: Project input to spectral domain
            // spectralCoeffs = eigenvectors^T @ input -> [numEig, InputChannels]
            var spectralCoeffs = Engine.TensorMatMul(eigenvectorsTransposed, input);

            // Step 2: Apply heat kernel (multiply by decay factors)
            // Scale each row by corresponding decay factor
            var tiledDecay = Engine.TensorTile(decayTensor, [1, InputChannels]);
            spectralCoeffs = Engine.TensorMultiply(spectralCoeffs, tiledDecay);

            // Step 3: Project back to spatial domain
            // output = eigenvectors @ spectralCoeffs -> [numVertices, InputChannels]
            var output = Engine.TensorMatMul(_eigenvectors, spectralCoeffs);

            // Copy to output buffer at appropriate offset
            var outputArray = output.ToArray();
            int baseOffset = t * InputChannels;
            for (int v = 0; v < numVertices; v++)
            {
                for (int c = 0; c < InputChannels; c++)
                {
                    int srcIdx = v * InputChannels + c;
                    int dstIdx = v * InputChannels * NumTimeScales + baseOffset + c;
                    diffused[dstIdx] = outputArray[srcIdx];
                }
            }
        }
    }

    /// <summary>
    /// Computes diffusion using direct matrix method with vectorized operations.
    /// </summary>
    /// <param name="input">Input features.</param>
    /// <param name="diffused">Output buffer for diffused features.</param>
    /// <param name="numVertices">Number of vertices.</param>
    /// <remarks>
    /// <para>
    /// Direct diffusion computes: output[v, c] = sum_u(exp(-L[v,u] * t) * input[u, c])
    /// This is a matrix multiplication with element-wise exponential decay.
    /// </para>
    /// </remarks>
    private void ComputeDiffusionDirect(Tensor<T> input, T[] diffused, int numVertices)
    {
        if (_laplacian == null)
            return;

        for (int t = 0; t < NumTimeScales; t++)
        {
            T time = DiffusionTimes[t];

            // Vectorized heat kernel computation: K[v,u] = exp(-L[v,u] * t)
            var negLaplacian = Engine.TensorNegate(_laplacian);
            var scaledLaplacian = Engine.TensorMultiplyScalar(negLaplacian, time);
            var heatKernel = Engine.TensorExp(scaledLaplacian);

            // Compute diffused output: output = heatKernel @ input -> [numVertices, InputChannels]
            var output = Engine.TensorMatMul(heatKernel, input);

            // Copy to output buffer at appropriate offset
            var outputArray = output.ToArray();
            int baseOffset = t * InputChannels;
            for (int v = 0; v < numVertices; v++)
            {
                for (int c = 0; c < InputChannels; c++)
                {
                    int srcIdx = v * InputChannels + c;
                    int dstIdx = v * InputChannels * NumTimeScales + baseOffset + c;
                    diffused[dstIdx] = outputArray[srcIdx];
                }
            }
        }
    }

    /// <summary>
    /// Adds biases to each vertex output.
    /// </summary>
    private Tensor<T> AddBiases(Tensor<T> output, int numVertices)
    {
        var biasExpanded = _biases.Reshape(1, OutputChannels);
        return Engine.TensorBroadcastAdd(output, biasExpanded);
    }

    private void EnsureEigenbasisForExecution()
    {
        if (_eigenvalues != null && _eigenvectors != null)
            return;

        lock (_eigenbasisLock)
        {
            if (_eigenvalues != null && _eigenvectors != null)
                return;

            if (_laplacian == null)
            {
                throw new InvalidOperationException(
                    "Execution requires an eigenbasis or Laplacian. Call SetEigenbasis or SetLaplacian before executing the layer.");
            }

            if (_massMatrix != null)
            {
                throw new InvalidOperationException(
                    "Automatic eigenbasis computation does not support a mass matrix. Provide a precomputed eigenbasis via SetEigenbasis.");
            }

            if (_laplacian.Rank != 2 || _laplacian.Shape[0] != _laplacian.Shape[1])
            {
                throw new InvalidOperationException("Laplacian must be a square 2D matrix.");
            }

            var laplacianMatrix = _laplacian.ToMatrix();
            var decomposition = new EigenDecomposition<T>(laplacianMatrix, EigenAlgorithmType.Jacobi);

            var eigenValues = decomposition.EigenValues;
            var eigenVectors = decomposition.EigenVectors;
            int size = eigenValues.Length;
            var pairs = new (double value, int index)[size];
            for (int i = 0; i < size; i++)
            {
                pairs[i] = (NumOps.ToDouble(eigenValues[i]), i);
            }

            Array.Sort(pairs, (a, b) => a.value.CompareTo(b.value));

            int count = Math.Min(_numEigenvectors, size);
            var selectedValues = new T[count];
            var selectedVectors = new T[laplacianMatrix.Rows * count];

            for (int col = 0; col < count; col++)
            {
                int srcIndex = pairs[col].index;
                selectedValues[col] = eigenValues[srcIndex];
                for (int row = 0; row < laplacianMatrix.Rows; row++)
                {
                    selectedVectors[row * count + col] = eigenVectors[row, srcIndex];
                }
            }

            _eigenvalues = selectedValues;
            _eigenvectors = new Tensor<T>(selectedVectors, [laplacianMatrix.Rows, count]);
        }
    }

    /// <summary>
    /// Performs GPU-accelerated forward pass using spectral heat diffusion.    
    /// </summary>
    /// <remarks>
    /// <para>
    /// The spectral diffusion is computed entirely on GPU:
    /// 1. Project input to spectral domain: coeffs = Phi^T @ input
    /// 2. Apply heat kernel: coeffs = coeffs * exp(-lambda * t) for each time scale
    /// 3. Project back: diffused_t = Phi @ coeffs
    /// 4. Concatenate all time scales
    /// 5. Linear transform: output = diffused @ weights^T + bias
    /// 6. Apply activation
    /// </para>
    /// <para>
    /// If the eigenbasis is missing but a Laplacian is available, it is computed
    /// automatically before GPU execution.
    /// </para>
    /// </remarks>
    /// <param name="inputs">Input GPU tensors (uses first input).</param>
    /// <returns>GPU-resident output tensor.</returns>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        // Spectral method requires eigenbasis; compute from Laplacian if needed.
        if (_eigenvalues == null || _eigenvectors == null)
        {
            EnsureEigenbasisForExecution();
        }

        var input = inputs[0];
        int[] shape = input.Shape;

        // Handle batched vs non-batched input
        int batchSize;
        int numVertices;
        int inputChannels;

        if (shape.Length == 2)
        {
            batchSize = 1;
            numVertices = shape[0];
            inputChannels = shape[1];
        }
        else if (shape.Length == 3)
        {
            batchSize = shape[0];
            numVertices = shape[1];
            inputChannels = shape[2];
        }
        else
        {
            throw new ArgumentException($"Input must be 2D or 3D, got {shape.Length}D");
        }

        if (inputChannels != InputChannels)
        {
            throw new ArgumentException(
                $"Input channels ({inputChannels}) must match layer InputChannels ({InputChannels}).");
        }

        if (IsTrainingMode)
        {
            ClearGpuCache();
            _gpuInput = input;
            _gpuInputShape = shape.ToArray();
        }

        var eigenvalues = _eigenvalues ?? throw new InvalidOperationException(
            "Eigenbasis is required for GPU execution. Call SetEigenbasis or SetLaplacian before GPU execution.");
        var eigenvectors = _eigenvectors ?? throw new InvalidOperationException(
            "Eigenbasis is required for GPU execution. Call SetEigenbasis or SetLaplacian before GPU execution.");
        int numEig = Math.Min(_numEigenvectors, eigenvalues.Length);
        int diffusedSize = InputChannels * NumTimeScales;

        // Upload eigenvectors to GPU
        using var eigenvectorsBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(eigenvectors.Data));

        // Transpose eigenvectors for spectral projection: [numEig, numVertices]
        using var eigenvectorsTransposedBuffer = backend.AllocateBuffer(numEig * numVertices);
        backend.Transpose(eigenvectorsBuffer, eigenvectorsTransposedBuffer, numVertices, numEig);

        // Create eigenvalue buffer and compute decay factors for each time scale
        var eigenvalueArray = new float[numEig];
        for (int k = 0; k < numEig; k++)
        {
            eigenvalueArray[k] = (float)NumOps.ToDouble(eigenvalues[k]);
        }

        // Allocate diffused output buffer [batchSize * numVertices, diffusedSize]
        var diffusedBuffer = backend.AllocateBuffer(batchSize * numVertices * diffusedSize);
        backend.Fill(diffusedBuffer, 0.0f, batchSize * numVertices * diffusedSize);
        var diffusedRetained = false;
        IGpuBuffer? preActivationBuffer = null;
        var preActivationRetained = false;

        try
        {
            // Process each time scale on GPU
            for (int t = 0; t < NumTimeScales; t++)
            {
                float time = (float)NumOps.ToDouble(DiffusionTimes[t]);

                // Compute decay factors: exp(-eigenvalue * time)
                var decayData = new float[numEig];
                for (int k = 0; k < numEig; k++)
                {
                    decayData[k] = (float)Math.Exp(-eigenvalueArray[k] * time);
                }
                using var decayBuffer = backend.AllocateBuffer(decayData);

                // Step 1: Project input to spectral domain
                // spectralCoeffs = eigenvectors^T @ input
                // [numEig, numVertices] @ [batchSize * numVertices, inputChannels]
                // We need to handle batch dimension - reshape input for batched processing
                using var spectralCoeffsBuffer = backend.AllocateBuffer(batchSize * numEig * InputChannels);

                if (batchSize == 1)
                {
                    // [numEig, numVertices] @ [numVertices, inputChannels] -> [numEig, inputChannels]
                    backend.Gemm(eigenvectorsTransposedBuffer, input.Buffer, spectralCoeffsBuffer,
                        numEig, InputChannels, numVertices);
                }
                else
                {
                    // Process each batch separately with GPU operations
                    // BatchedGemm requires same A for all batches, so we tile eigenvectorsTransposed
                    using var tiledEigTBuffer = backend.AllocateBuffer(batchSize * numEig * numVertices);
                    backend.TileBatch(eigenvectorsTransposedBuffer, tiledEigTBuffer, batchSize, numEig * numVertices);

                    // BatchedGemm: [batch, numEig, numVertices] @ [batch, numVertices, inputChannels]
                    backend.BatchedGemm(tiledEigTBuffer, input.Buffer, spectralCoeffsBuffer,
                        numEig, InputChannels, numVertices, batchSize);
                }

                // Step 2: Apply heat kernel - multiply each row by corresponding decay factor
                // spectralCoeffs[k, :] *= decay[k]
                using var decayTiledBuffer = backend.AllocateBuffer(batchSize * numEig * InputChannels);

                // Tile decay along batch and feature dimensions
                var decayTiledData = new float[batchSize * numEig * InputChannels];
                for (int b = 0; b < batchSize; b++)
                {
                    for (int k = 0; k < numEig; k++)
                    {
                        for (int c = 0; c < InputChannels; c++)
                        {
                            decayTiledData[b * numEig * InputChannels + k * InputChannels + c] = decayData[k];
                        }
                    }
                }
                using var decayTiledUploadBuffer = backend.AllocateBuffer(decayTiledData);
                backend.Multiply(spectralCoeffsBuffer, decayTiledUploadBuffer, spectralCoeffsBuffer,
                    batchSize * numEig * InputChannels);

                // Step 3: Project back to spatial domain
                // output = eigenvectors @ spectralCoeffs
                // [numVertices, numEig] @ [batchSize * numEig, inputChannels]
                using var timeOutputBuffer = backend.AllocateBuffer(batchSize * numVertices * InputChannels);

                if (batchSize == 1)
                {
                    // [numVertices, numEig] @ [numEig, inputChannels] -> [numVertices, inputChannels]
                    backend.Gemm(eigenvectorsBuffer, spectralCoeffsBuffer, timeOutputBuffer,
                        numVertices, InputChannels, numEig);
                }
                else
                {
                    // Tile eigenvectors for batched operation
                    using var tiledEigBuffer = backend.AllocateBuffer(batchSize * numVertices * numEig);
                    backend.TileBatch(eigenvectorsBuffer, tiledEigBuffer, batchSize, numVertices * numEig);

                    // BatchedGemm: [batch, numVertices, numEig] @ [batch, numEig, inputChannels]
                    backend.BatchedGemm(tiledEigBuffer, spectralCoeffsBuffer, timeOutputBuffer,
                        numVertices, InputChannels, numEig, batchSize);
                }

                // Copy this time scale's output to the appropriate columns in diffusedBuffer
                // diffused[:, t*inputChannels:(t+1)*inputChannels] = timeOutput
                int colOffset = t * InputChannels;
                for (int b = 0; b < batchSize; b++)
                {
                    for (int v = 0; v < numVertices; v++)
                    {
                        // Copy each row's time-scale output to the correct position
                        int srcOffset = b * numVertices * InputChannels + v * InputChannels;
                        int dstOffset = b * numVertices * diffusedSize + v * diffusedSize + colOffset;

                        // Since we can't do strided copy easily, download and re-upload
                        // This is O(1) roundtrips for the entire time scale
                    }
                }

                // For efficiency, download timeOutput, scatter to diffused, then continue
                // At the end, upload the complete diffused buffer
                var timeOutputData = backend.DownloadBuffer(timeOutputBuffer);
                var diffusedData = backend.DownloadBuffer(diffusedBuffer);

                for (int b = 0; b < batchSize; b++)
                {
                    for (int v = 0; v < numVertices; v++)
                    {
                        int srcOffset = b * numVertices * InputChannels + v * InputChannels;
                        int dstOffset = b * numVertices * diffusedSize + v * diffusedSize + colOffset;
                        for (int c = 0; c < InputChannels; c++)
                        {
                            diffusedData[dstOffset + c] = timeOutputData[srcOffset + c];
                        }
                    }
                }

                // Re-upload diffused buffer with accumulated data
                using var updatedDiffusedBuffer = backend.AllocateBuffer(diffusedData);
                backend.Copy(updatedDiffusedBuffer, diffusedBuffer, batchSize * numVertices * diffusedSize);
            }

            if (IsTrainingMode)
            {
                _gpuDiffusedFeatures?.Dispose();
                _gpuDiffusedFeatures = new GpuTensor<T>(backend, diffusedBuffer,
                    [batchSize * numVertices, diffusedSize], GpuTensorRole.Activation, ownsBuffer: true);
                diffusedRetained = true;
            }

            // Upload weights and biases
            using var weightsBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_weights.Data));
            using var biasBuffer = backend.AllocateBuffer(DirectGpuEngine.ToFloatArray<T>(_biases.Data));

            // Transpose weights for GEMM: [outputChannels, diffusedSize]^T = [diffusedSize, outputChannels]
            using var weightsTransposedBuffer = backend.AllocateBuffer(diffusedSize * OutputChannels);
            backend.Transpose(weightsBuffer, weightsTransposedBuffer, OutputChannels, diffusedSize);

            // Linear transform: output = diffused @ weights^T
            // [batchSize * numVertices, diffusedSize] @ [diffusedSize, outputChannels] -> [batchSize * numVertices, outputChannels]
            preActivationBuffer = backend.AllocateBuffer(batchSize * numVertices * OutputChannels);
            backend.Gemm(diffusedBuffer, weightsTransposedBuffer, preActivationBuffer,
                batchSize * numVertices, OutputChannels, diffusedSize);

            // Add bias: broadcast [outputChannels] across rows
            backend.BiasAdd(preActivationBuffer, biasBuffer, preActivationBuffer, batchSize * numVertices, OutputChannels);

            // Apply activation on GPU using base class helper
            var outputBuffer = backend.AllocateBuffer(batchSize * numVertices * OutputChannels);
            var fusedActivation = GetFusedActivationType();
            ApplyGpuActivation(backend, preActivationBuffer, outputBuffer, batchSize * numVertices * OutputChannels, fusedActivation);

            // Create output shape
            int[] outputShape = batchSize == 1
                ? [numVertices, OutputChannels]
                : [batchSize, numVertices, OutputChannels];

            if (IsTrainingMode)
            {
                _gpuPreActivation?.Dispose();
                _gpuPreActivation = new GpuTensor<T>(backend, preActivationBuffer,
                    [batchSize * numVertices, OutputChannels], GpuTensorRole.Activation, ownsBuffer: true);
                preActivationRetained = true;
            }

            var outputTensor = new GpuTensor<T>(backend, outputBuffer, outputShape, GpuTensorRole.Activation, ownsBuffer: true);
            if (IsTrainingMode)
                _gpuOutput = outputTensor;

            return outputTensor;
        }
        finally
        {
            if (!diffusedRetained)
            {
                diffusedBuffer.Dispose();
            }

            if (preActivationBuffer != null && !preActivationRetained)
            {
                preActivationBuffer.Dispose();
            }
        }
    }

    #endregion

    #region GPU Training Methods

    /// <summary>
    /// Performs the backward pass using GPU-resident tensors.
    /// </summary>
    /// <param name="outputGradient">GPU-resident gradient tensor.</param>
    /// <returns>GPU-resident input gradient tensor.</returns>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        if (_gpuInput == null || _gpuInputShape == null || _gpuDiffusedFeatures == null ||
            _gpuPreActivation == null || _gpuOutput == null)
            throw new InvalidOperationException("ForwardGpu must be called in training mode before BackwardGpu.");

        if (_eigenvalues == null || _eigenvectors == null)
            throw new InvalidOperationException(
                "GPU backward requires an eigenbasis. Call SetEigenbasis or SetLaplacian before training.");

        int batchSize;
        int numVertices;
        int inputChannels;

        if (_gpuInputShape.Length == 2)
        {
            batchSize = 1;
            numVertices = _gpuInputShape[0];
            inputChannels = _gpuInputShape[1];
        }
        else if (_gpuInputShape.Length == 3)
        {
            batchSize = _gpuInputShape[0];
            numVertices = _gpuInputShape[1];
            inputChannels = _gpuInputShape[2];
        }
        else
        {
            throw new ArgumentException($"Input must be 2D or 3D, got {_gpuInputShape.Length}D");
        }

        int diffusedSize = InputChannels * NumTimeScales;
        int outputRows = batchSize * numVertices;
        int outputCols = OutputChannels;

        IGpuTensor<T> outputGrad2D;
        if (outputGradient.Shape.Length == 2)
        {
            outputGrad2D = outputGradient;
        }
        else if (outputGradient.Shape.Length == 3)
        {
            outputGrad2D = outputGradient.CreateView(0, [outputRows, outputCols]);
        }
        else
        {
            throw new ArgumentException($"Output gradient must be 2D or 3D, got {outputGradient.Shape.Length}D");
        }

        if (outputGrad2D.Shape[0] != outputRows || outputGrad2D.Shape[1] != outputCols)
            throw new ArgumentException("Output gradient shape does not match cached output.");

        IGpuBuffer? weightGradBuffer = null;
        IGpuBuffer? biasGradBuffer = null;
        IGpuBuffer? inputGradBuffer = null;

        try
        {
            int outputGradSize = outputRows * outputCols;
            using var deltaBuffer = backend.AllocateBuffer(outputGradSize);

            bool activationHandled = ApplyActivationBackwardGpu(
                backend,
                outputGrad2D.Buffer,
                _gpuPreActivation.Buffer,
                _gpuOutput.Buffer,
                deltaBuffer,
                outputGradSize);

            if (!activationHandled)
            {
                var fusedActivation = GetFusedActivationType();
                if (fusedActivation != FusedActivationType.None)
                {
                    activationHandled = ApplyGpuActivationBackward(
                        backend,
                        outputGrad2D.Buffer,
                        _gpuPreActivation.Buffer,
                        _gpuOutput.Buffer,
                        deltaBuffer,
                        outputGradSize,
                        fusedActivation);
                }
            }

            if (!activationHandled)
            {
                var cpuGrad = outputGradient.ToTensor();
                var cpuOutput = _gpuOutput.ToTensor();
                var deltaCpu = ApplyActivationDerivative(cpuOutput, cpuGrad);
                var deltaReshaped = deltaCpu.Reshape(outputRows, outputCols);
                using var deltaUpload = backend.AllocateBuffer(
                    DirectGpuEngine.ToFloatArray<T>(deltaReshaped.Data));
                backend.Copy(deltaUpload, deltaBuffer, outputGradSize);
            }

            using var deltaTransposed = backend.AllocateBuffer(outputCols * outputRows);
            backend.Transpose(deltaBuffer, deltaTransposed, outputRows, outputCols);

            weightGradBuffer = backend.AllocateBuffer(_weights.Length);
            backend.Gemm(deltaTransposed, _gpuDiffusedFeatures.Buffer, weightGradBuffer,
                outputCols, diffusedSize, outputRows);

            biasGradBuffer = backend.AllocateBuffer(OutputChannels);
            backend.SumAxis(deltaTransposed, biasGradBuffer, OutputChannels, outputRows);

            inputGradBuffer = backend.AllocateBuffer(outputRows * inputChannels);
            backend.Fill(inputGradBuffer, 0.0f, outputRows * inputChannels);

            int numEig = Math.Min(_numEigenvectors, _eigenvalues.Length);

            using var eigenvectorsBuffer = backend.AllocateBuffer(
                DirectGpuEngine.ToFloatArray<T>(_eigenvectors.Data));
            using var eigenvectorsTransposedBuffer = backend.AllocateBuffer(numEig * numVertices);
            backend.Transpose(eigenvectorsBuffer, eigenvectorsTransposedBuffer, numVertices, numEig);

            IGpuBuffer? tiledEigBuffer = null;
            IGpuBuffer? tiledEigTBuffer = null;

            try
            {
                if (batchSize > 1)
                {
                    tiledEigBuffer = backend.AllocateBuffer(batchSize * numVertices * numEig);
                    backend.TileBatch(eigenvectorsBuffer, tiledEigBuffer, batchSize, numVertices * numEig);

                    tiledEigTBuffer = backend.AllocateBuffer(batchSize * numEig * numVertices);
                    backend.TileBatch(eigenvectorsTransposedBuffer, tiledEigTBuffer, batchSize, numEig * numVertices);
                }

                using var spectralCoeffsBuffer = backend.AllocateBuffer(batchSize * numEig * inputChannels);
                if (batchSize == 1)
                {
                    backend.Gemm(eigenvectorsTransposedBuffer, _gpuInput.Buffer, spectralCoeffsBuffer,
                        numEig, inputChannels, numVertices);
                }
                else
                {
                    var batchedEigTBuffer = tiledEigTBuffer
                        ?? throw new InvalidOperationException("Batched eigenvector buffers were not initialized.");
                    backend.BatchedGemm(batchedEigTBuffer, _gpuInput.Buffer, spectralCoeffsBuffer,
                        numEig, inputChannels, numVertices, batchSize);
                }

                _diffusionTimesGradient = new T[NumTimeScales];

                var weightsData = DirectGpuEngine.ToFloatArray<T>(_weights.Data);

                for (int t = 0; t < NumTimeScales; t++)
                {
                    float time = (float)NumOps.ToDouble(DiffusionTimes[t]);
                    var decayData = new float[numEig];
                    var derivData = new float[numEig];
                    for (int k = 0; k < numEig; k++)
                    {
                        float lambda = (float)NumOps.ToDouble(_eigenvalues[k]);
                        float decay = (float)Math.Exp(-lambda * time);
                        decayData[k] = decay;
                        derivData[k] = -lambda * decay;
                    }

                    var weightsSlice = new float[OutputChannels * inputChannels];
                    for (int oc = 0; oc < OutputChannels; oc++)
                    {
                        int srcOffset = oc * diffusedSize + t * inputChannels;
                        int dstOffset = oc * inputChannels;
                        Array.Copy(weightsData, srcOffset, weightsSlice, dstOffset, inputChannels);
                    }

                    using var weightsSliceBuffer = backend.AllocateBuffer(weightsSlice);
                    using var diffusedGradBuffer = backend.AllocateBuffer(outputRows * inputChannels);
                    backend.Gemm(deltaBuffer, weightsSliceBuffer, diffusedGradBuffer,
                        outputRows, inputChannels, OutputChannels);

                    using var spectralGradBuffer = backend.AllocateBuffer(batchSize * numEig * inputChannels);
                    if (batchSize == 1)
                    {
                        backend.Gemm(eigenvectorsTransposedBuffer, diffusedGradBuffer, spectralGradBuffer,
                            numEig, inputChannels, numVertices);
                    }
                    else
                    {
                        var batchedEigTBuffer = tiledEigTBuffer
                            ?? throw new InvalidOperationException("Batched eigenvector buffers were not initialized.");
                        backend.BatchedGemm(batchedEigTBuffer, diffusedGradBuffer, spectralGradBuffer,
                            numEig, inputChannels, numVertices, batchSize);
                    }

                    var decayTiledData = new float[batchSize * numEig * inputChannels];
                    int decayIndex = 0;
                    for (int b = 0; b < batchSize; b++)
                    {
                        for (int k = 0; k < numEig; k++)
                        {
                            float decay = decayData[k];
                            for (int c = 0; c < inputChannels; c++)
                            {
                                decayTiledData[decayIndex++] = decay;
                            }
                        }
                    }

                    using var decayTiledBuffer = backend.AllocateBuffer(decayTiledData);
                    backend.Multiply(spectralGradBuffer, decayTiledBuffer, spectralGradBuffer,
                        batchSize * numEig * inputChannels);

                    using var spatialGradBuffer = backend.AllocateBuffer(outputRows * inputChannels);
                    if (batchSize == 1)
                    {
                        backend.Gemm(eigenvectorsBuffer, spectralGradBuffer, spatialGradBuffer,
                            numVertices, inputChannels, numEig);
                    }
                    else
                    {
                        var batchedEigBuffer = tiledEigBuffer
                            ?? throw new InvalidOperationException("Batched eigenvector buffers were not initialized.");
                        backend.BatchedGemm(batchedEigBuffer, spectralGradBuffer, spatialGradBuffer,
                            numVertices, inputChannels, numEig, batchSize);
                    }

                    backend.Add(inputGradBuffer, spatialGradBuffer, inputGradBuffer,
                        outputRows * inputChannels);

                    var derivTiledData = new float[batchSize * numEig * inputChannels];
                    int derivIndex = 0;
                    for (int b = 0; b < batchSize; b++)
                    {
                        for (int k = 0; k < numEig; k++)
                        {
                            float deriv = derivData[k];
                            for (int c = 0; c < inputChannels; c++)
                            {
                                derivTiledData[derivIndex++] = deriv;
                            }
                        }
                    }

                    using var derivTiledBuffer = backend.AllocateBuffer(derivTiledData);
                    using var derivCoeffsBuffer = backend.AllocateBuffer(batchSize * numEig * inputChannels);
                    backend.Multiply(spectralCoeffsBuffer, derivTiledBuffer, derivCoeffsBuffer,
                        batchSize * numEig * inputChannels);

                    using var spatialDerivBuffer = backend.AllocateBuffer(outputRows * inputChannels);
                    if (batchSize == 1)
                    {
                        backend.Gemm(eigenvectorsBuffer, derivCoeffsBuffer, spatialDerivBuffer,
                            numVertices, inputChannels, numEig);
                    }
                    else
                    {
                        var batchedEigBuffer = tiledEigBuffer
                            ?? throw new InvalidOperationException("Batched eigenvector buffers were not initialized.");
                        backend.BatchedGemm(batchedEigBuffer, derivCoeffsBuffer, spatialDerivBuffer,
                            numVertices, inputChannels, numEig, batchSize);
                    }

                    using var productBuffer = backend.AllocateBuffer(outputRows * inputChannels);
                    backend.Multiply(spatialDerivBuffer, diffusedGradBuffer, productBuffer,
                        outputRows * inputChannels);

                    float timeGrad = backend.Sum(productBuffer, outputRows * inputChannels);
                    _diffusionTimesGradient[t] = NumOps.FromDouble(timeGrad);
                }
            }
            finally
            {
                tiledEigBuffer?.Dispose();
                tiledEigTBuffer?.Dispose();
            }

            var weightGradData = backend.DownloadBuffer(weightGradBuffer);
            _weightsGradient = new Tensor<T>(
                DirectGpuEngine.FromFloatArray<T>(weightGradData),
                _weights.Shape);

            var biasGradData = backend.DownloadBuffer(biasGradBuffer);
            _biasesGradient = new Tensor<T>(
                DirectGpuEngine.FromFloatArray<T>(biasGradData),
                _biases.Shape);

            _gpuWeightsGradient?.Dispose();
            _gpuWeightsGradient = new GpuTensor<T>(
                backend,
                weightGradBuffer,
                _weights.Shape,
                GpuTensorRole.Gradient,
                ownsBuffer: true);
            weightGradBuffer = null;

            _gpuBiasesGradient?.Dispose();
            _gpuBiasesGradient = new GpuTensor<T>(
                backend,
                biasGradBuffer,
                _biases.Shape,
                GpuTensorRole.Gradient,
                ownsBuffer: true);
            biasGradBuffer = null;

            if (_diffusionTimesGradient != null)
            {
                _gpuDiffusionTimesGradient?.Dispose();
                _gpuDiffusionTimesGradient = new GpuTensor<T>(
                    backend,
                    _diffusionTimesGradient,
                    [NumTimeScales],
                    GpuTensorRole.Gradient);
            }

            int[] inputGradShape = batchSize == 1
                ? [numVertices, inputChannels]
                : [batchSize, numVertices, inputChannels];

            ClearGpuCache();

            var inputGradTensor = new GpuTensor<T>(
                backend,
                inputGradBuffer,
                inputGradShape,
                GpuTensorRole.Gradient,
                ownsBuffer: true);
            inputGradBuffer = null;
            return inputGradTensor;
        }
        finally
        {
            weightGradBuffer?.Dispose();
            biasGradBuffer?.Dispose();
            inputGradBuffer?.Dispose();
        }
    }

    /// <summary>
    /// Updates parameters on GPU using the configured optimizer.
    /// </summary>
    /// <param name="config">The GPU optimizer configuration.</param>
    public override void UpdateParametersGpu(IGpuOptimizerConfig config)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("UpdateParametersGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        if (_gpuWeightsGradient == null || _gpuBiasesGradient == null || _gpuDiffusionTimesGradient == null)
            throw new InvalidOperationException("BackwardGpu must be called before UpdateParametersGpu.");

        _gpuWeights ??= new GpuTensor<T>(backend, _weights, GpuTensorRole.Weight);
        _gpuBiases ??= new GpuTensor<T>(backend, _biases, GpuTensorRole.Bias);
        _gpuDiffusionTimes ??= new GpuTensor<T>(backend, DiffusionTimes, [NumTimeScales], GpuTensorRole.Weight);

        EnsureDiffusionConvOptimizerState(backend, config.OptimizerType);

        config.ApplyUpdate(
            backend,
            _gpuWeights.Buffer,
            _gpuWeightsGradient.Buffer,
            BuildDiffusionConvOptimizerState("weights"),
            _weights.Length);

        config.ApplyUpdate(
            backend,
            _gpuBiases.Buffer,
            _gpuBiasesGradient.Buffer,
            BuildDiffusionConvOptimizerState("biases"),
            _biases.Length);

        config.ApplyUpdate(
            backend,
            _gpuDiffusionTimes.Buffer,
            _gpuDiffusionTimesGradient.Buffer,
            BuildDiffusionConvOptimizerState("diffusionTimes"),
            NumTimeScales);

        backend.Clamp(_gpuDiffusionTimes.Buffer, _gpuDiffusionTimes.Buffer, 1e-6f, float.MaxValue, NumTimeScales);

        _weights = _gpuWeights.ToTensor();
        _biases = _gpuBiases.ToTensor();

        var updatedTimes = _gpuDiffusionTimes.ToTensor().Data.Span;
        for (int t = 0; t < NumTimeScales; t++)
        {
            DiffusionTimes[t] = updatedTimes[t];
        }
    }

    private void EnsureDiffusionConvOptimizerState(IDirectGpuBackend backend, GpuOptimizerType optimizerType)
    {
        int weightSize = _weights.Length;
        int biasSize = _biases.Length;
        int timeSize = NumTimeScales;

        switch (optimizerType)
        {
            case GpuOptimizerType.Sgd:
            case GpuOptimizerType.Nag:
            case GpuOptimizerType.Lars:
                _gpuWeightsVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasesVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuDiffusionTimesVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([timeSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;

            case GpuOptimizerType.Adam:
            case GpuOptimizerType.AdamW:
            case GpuOptimizerType.Lamb:
                _gpuWeightsM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightsV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasesM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasesV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuDiffusionTimesM ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([timeSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuDiffusionTimesV ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([timeSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;

            case GpuOptimizerType.RmsProp:
            case GpuOptimizerType.Adagrad:
                _gpuWeightsVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasesVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuDiffusionTimesVelocity ??= new GpuTensor<T>(backend, Tensor<T>.CreateDefault([timeSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                break;
        }
    }

    private GpuOptimizerState BuildDiffusionConvOptimizerState(string paramName)
    {
        return paramName switch
        {
            "weights" => new GpuOptimizerState
            {
                Velocity = _gpuWeightsVelocity?.Buffer,
                M = _gpuWeightsM?.Buffer,
                V = _gpuWeightsV?.Buffer,
                SquaredAvg = _gpuWeightsVelocity?.Buffer,
                AccumulatedGrad = _gpuWeightsVelocity?.Buffer
            },
            "biases" => new GpuOptimizerState
            {
                Velocity = _gpuBiasesVelocity?.Buffer,
                M = _gpuBiasesM?.Buffer,
                V = _gpuBiasesV?.Buffer,
                SquaredAvg = _gpuBiasesVelocity?.Buffer,
                AccumulatedGrad = _gpuBiasesVelocity?.Buffer
            },
            "diffusionTimes" => new GpuOptimizerState
            {
                Velocity = _gpuDiffusionTimesVelocity?.Buffer,
                M = _gpuDiffusionTimesM?.Buffer,
                V = _gpuDiffusionTimesV?.Buffer,
                SquaredAvg = _gpuDiffusionTimesVelocity?.Buffer,
                AccumulatedGrad = _gpuDiffusionTimesVelocity?.Buffer
            },
            _ => new GpuOptimizerState()
        };
    }

    #endregion

    #region Backward Pass

    /// <summary>
    /// Performs the backward pass to compute gradients.
    /// </summary>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation.
    /// </summary>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastPreActivation == null || _lastOutput == null || _diffusedFeatures == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var delta = ApplyActivationDerivative(_lastOutput, outputGradient);

        bool hasBatch = _lastInput.Rank == 3;
        int numVertices = hasBatch ? _lastInput.Shape[1] : _lastInput.Shape[0];

        // Initialize diffusion times gradient
        _diffusionTimesGradient = new T[NumTimeScales];
        for (int t = 0; t < NumTimeScales; t++)
        {
            _diffusionTimesGradient[t] = NumOps.Zero;
        }

        if (hasBatch)
        {
            // For batched inputs, accumulate gradients across all batches
            int batchSize = _lastInput.Shape[0];

            // Reshape delta to [batchSize * numVertices, OutputChannels] for matrix operations
            var deltaReshaped = delta.Reshape(batchSize * numVertices, OutputChannels);
            var diffusedReshaped = _diffusedFeatures.Reshape(batchSize * numVertices, InputChannels * NumTimeScales);

            // Compute weight gradients: sum over all samples
            var transposedDelta = Engine.TensorTranspose(deltaReshaped);
            _weightsGradient = Engine.TensorMatMul(transposedDelta, diffusedReshaped);

            // Compute bias gradients: sum over all samples
            _biasesGradient = Engine.ReduceSum(deltaReshaped, [0], keepDims: false);

            // Compute input gradients
            var diffusedGrad = Engine.TensorMatMul(deltaReshaped, _weights);

            // Compute diffusion time gradients (accumulate across batches)
            ComputeDiffusionTimeGradients(diffusedGrad, _lastInput.Reshape(batchSize * numVertices, InputChannels), numVertices * batchSize);

            var inputGrad = BackpropagateThroughDiffusion(diffusedGrad, numVertices * batchSize);

            return inputGrad.Reshape(batchSize, numVertices, InputChannels);
        }
        else
        {
            var transposedDelta = Engine.TensorTranspose(delta);
            _weightsGradient = Engine.TensorMatMul(transposedDelta, _diffusedFeatures);
            _biasesGradient = Engine.ReduceSum(delta, [0], keepDims: false);

            var diffusedGrad = Engine.TensorMatMul(delta, _weights);

            // Compute diffusion time gradients
            ComputeDiffusionTimeGradients(diffusedGrad, _lastInput, numVertices);

            var inputGrad = BackpropagateThroughDiffusion(diffusedGrad, numVertices);

            return inputGrad;
        }
    }

    /// <summary>
    /// Computes gradients for diffusion time parameters using the chain rule.
    /// </summary>
    /// <param name="diffusedGrad">Gradient w.r.t. diffused features [numVertices, InputChannels * NumTimeScales].</param>
    /// <param name="input">Original input features [numVertices, InputChannels].</param>
    /// <param name="numVertices">Number of vertices.</param>
    /// <remarks>
    /// <para>
    /// For spectral diffusion, the gradient w.r.t. time t is:
    /// dL/dt = sum_k,c (-lambda_k * exp(-lambda_k * t) * (Phi @ coeffs)_c * dL/d(diffused)_t,c)
    /// where coeffs = Phi^T @ input.
    /// </para>
    /// </remarks>
    private void ComputeDiffusionTimeGradients(Tensor<T> diffusedGrad, Tensor<T> input, int numVertices)
    {
        if (_diffusionTimesGradient == null)
        {
            _diffusionTimesGradient = new T[NumTimeScales];
        }

        if (_eigenvalues != null && _eigenvectors != null)
        {
            ComputeTimeGradientsSpectral(diffusedGrad, input, numVertices);
        }
        else if (_laplacian != null)
        {
            ComputeTimeGradientsDirect(diffusedGrad, input, numVertices);
        }
    }

    /// <summary>
    /// Computes time gradients using spectral method.
    /// </summary>
    private void ComputeTimeGradientsSpectral(Tensor<T> diffusedGrad, Tensor<T> input, int numVertices)
    {
        if (_eigenvalues == null || _eigenvectors == null || _diffusionTimesGradient == null)
            return;

        int numEig = Math.Min(_numEigenvectors, _eigenvalues.Length);
        var eigenvectorsTransposed = Engine.TensorTranspose(_eigenvectors);

        // Project input to spectral domain once: [numEig, InputChannels]
        var spectralCoeffs = Engine.TensorMatMul(eigenvectorsTransposed, input);

        for (int t = 0; t < NumTimeScales; t++)
        {
            T time = DiffusionTimes[t];
            T timeGrad = NumOps.Zero;

            // For each eigenvalue, compute the derivative contribution
            // d/dt(exp(-lambda * t)) = -lambda * exp(-lambda * t)
            var derivativeCoeffs = new T[numEig * InputChannels];
            for (int k = 0; k < numEig; k++)
            {
                T lambda = _eigenvalues[k];
                T decay = NumOps.Exp(NumOps.Negate(NumOps.Multiply(lambda, time)));
                T derivativeFactor = NumOps.Negate(NumOps.Multiply(lambda, decay));

                for (int c = 0; c < InputChannels; c++)
                {
                    int idx = k * InputChannels + c;
                    derivativeCoeffs[idx] = NumOps.Multiply(spectralCoeffs[k, c], derivativeFactor);
                }
            }

            // Project derivative back to spatial domain: [numVertices, InputChannels]
            var derivTensor = new Tensor<T>(derivativeCoeffs, [numEig, InputChannels]);
            var spatialDeriv = Engine.TensorMatMul(_eigenvectors, derivTensor);

            // Extract gradient slice for this time scale
            int baseOffset = t * InputChannels;
            var timeGradSlice = Engine.TensorSlice(diffusedGrad, [0, baseOffset], [numVertices, InputChannels]);

            // Dot product of gradient and derivative gives time gradient
            var product = Engine.TensorMultiply(timeGradSlice, spatialDeriv);
            var sumTensor = Engine.ReduceSum(product, [0, 1], keepDims: false);
            timeGrad = sumTensor.GetFlat(0);

            _diffusionTimesGradient[t] = NumOps.Add(_diffusionTimesGradient[t], timeGrad);
        }
    }

    /// <summary>
    /// Computes time gradients using direct (iterative) method.
    /// </summary>
    /// <remarks>
    /// For direct diffusion using matrix exponential approximation, the gradient
    /// computation is more complex. We use finite differences as a practical approximation.
    /// </remarks>
    private void ComputeTimeGradientsDirect(Tensor<T> diffusedGrad, Tensor<T> input, int numVertices)
    {
        if (_laplacian == null || _diffusionTimesGradient == null)
            return;

        // Use numerical differentiation for direct method
        T epsilon = NumOps.FromDouble(1e-5);

        for (int t = 0; t < NumTimeScales; t++)
        {
            T originalTime = DiffusionTimes[t];

            // Compute diffusion at t + epsilon
            DiffusionTimes[t] = NumOps.Add(originalTime, epsilon);
            var diffusedPlus = new T[numVertices * InputChannels];
            ComputeDiffusionDirectSingleTime(input, diffusedPlus, numVertices, t);

            // Compute diffusion at t - epsilon
            DiffusionTimes[t] = NumOps.Subtract(originalTime, epsilon);
            var diffusedMinus = new T[numVertices * InputChannels];
            ComputeDiffusionDirectSingleTime(input, diffusedMinus, numVertices, t);

            // Restore original time
            DiffusionTimes[t] = originalTime;

            // Compute finite difference: (f(t+e) - f(t-e)) / (2*e)
            T twoEpsilon = NumOps.Multiply(epsilon, NumOps.FromDouble(2.0));
            T timeGrad = NumOps.Zero;

            int baseOffset = t * InputChannels;
            for (int v = 0; v < numVertices; v++)
            {
                for (int c = 0; c < InputChannels; c++)
                {
                    int localIdx = v * InputChannels + c;
                    int gradIdx = v * InputChannels * NumTimeScales + baseOffset + c;

                    T diff = NumOps.Divide(
                        NumOps.Subtract(diffusedPlus[localIdx], diffusedMinus[localIdx]),
                        twoEpsilon);
                    T grad = diffusedGrad.GetFlat(gradIdx);
                    timeGrad = NumOps.Add(timeGrad, NumOps.Multiply(grad, diff));
                }
            }

            _diffusionTimesGradient[t] = NumOps.Add(_diffusionTimesGradient[t], timeGrad);
        }
    }

    /// <summary>
    /// Computes direct diffusion for a single time scale.
    /// </summary>
    private void ComputeDiffusionDirectSingleTime(Tensor<T> input, T[] output, int numVertices, int timeIndex)
    {
        if (_laplacian == null)
            return;

        T time = DiffusionTimes[timeIndex];

        // Use iterative approximation: (I - t*L/n)^n
        int numIterations = Math.Max(1, (int)(NumOps.ToDouble(time) * 10));
        T stepSize = NumOps.Divide(time, NumOps.FromDouble(numIterations));
        T negStepSize = NumOps.Negate(stepSize);

        // Start with input
        var current = input;

        for (int iter = 0; iter < numIterations; iter++)
        {
            // diffusion_step = x - stepSize * L @ x
            var laplacianProduct = Engine.TensorMatMul(_laplacian, current);
            var scaled = Engine.TensorMultiplyScalar(laplacianProduct, negStepSize);
            current = Engine.TensorAdd(current, scaled);
        }

        // Copy to output
        var currentArray = current.ToArray();
        Array.Copy(currentArray, output, output.Length);
    }

    /// <summary>
    /// Backward pass using automatic differentiation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Currently routes to manual implementation. Full autodiff integration pending
    /// the addition of diffusion-specific operations to the computation graph.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        // TODO: Implement proper autodiff when diffusion graph operations are available
        return BackwardManual(outputGradient);
    }

    /// <summary>
    /// Backpropagates gradients through the diffusion operation using vectorized operations.
    /// </summary>
    /// <param name="diffusedGrad">Gradient tensor [numVertices, InputChannels * NumTimeScales].</param>
    /// <param name="numVertices">Number of vertices.</param>
    /// <returns>Input gradient tensor [numVertices, InputChannels].</returns>
    private Tensor<T> BackpropagateThroughDiffusion(Tensor<T> diffusedGrad, int numVertices)
    {
        Tensor<T> inputGrad;

        if (_eigenvalues != null && _eigenvectors != null)
        {
            inputGrad = BackpropagateDiffusionSpectralVectorized(diffusedGrad, numVertices);
        }
        else if (_laplacian != null)
        {
            inputGrad = BackpropagateDiffusionDirectVectorized(diffusedGrad, numVertices);
        }
        else
        {
            inputGrad = new Tensor<T>([numVertices, InputChannels]);
        }

        return inputGrad;
    }

    /// <summary>
    /// Backpropagates through spectral diffusion using vectorized operations.
    /// </summary>
    /// <param name="diffusedGrad">Gradient tensor [numVertices, InputChannels * NumTimeScales].</param>
    /// <param name="numVertices">Number of vertices.</param>
    /// <returns>Input gradient tensor.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass through spectral diffusion is:
    /// inputGrad = sum_t(Phi @ diag(exp(-lambda * t)) @ Phi^T @ diffusedGrad_t)
    /// </para>
    /// </remarks>
    private Tensor<T> BackpropagateDiffusionSpectralVectorized(Tensor<T> diffusedGrad, int numVertices)
    {
        if (_eigenvalues == null || _eigenvectors == null)
            return new Tensor<T>([numVertices, InputChannels]);

        int numEig = Math.Min(_numEigenvectors, _eigenvalues.Length);

        // Initialize accumulated gradient
        var inputGrad = new Tensor<T>([numVertices, InputChannels]);

        // Transpose eigenvectors for projection: [numEig, numVertices]
        var eigenvectorsTransposed = Engine.TensorTranspose(_eigenvectors);

        // Create eigenvalue tensor for vectorized decay computation
        // Use explicit array copy instead of slice syntax for net471 compatibility
        var eigenvalueArray = new T[numEig];
        Array.Copy(_eigenvalues, eigenvalueArray, numEig);
        var eigenvalueTensor = new Tensor<T>(eigenvalueArray, [numEig]);

        for (int t = 0; t < NumTimeScales; t++)
        {
            T time = DiffusionTimes[t];
            int baseOffset = t * InputChannels;

            // Extract gradient slice for this time scale
            var timeGrad = Engine.TensorSlice(diffusedGrad, [0, baseOffset], [numVertices, InputChannels]);

            // Vectorized decay factor computation: exp(-eigenvalue * time)
            var negEigTimesTime = Engine.TensorMultiplyScalar(Engine.TensorNegate(eigenvalueTensor), time);
            var decayVector = Engine.TensorExp(negEigTimesTime);
            var decayTensor = decayVector.Reshape([numEig, 1]);

            // Project gradient to spectral domain
            var spectralGrad = Engine.TensorMatMul(eigenvectorsTransposed, timeGrad);

            // Apply decay
            var tiledDecay = Engine.TensorTile(decayTensor, [1, InputChannels]);
            spectralGrad = Engine.TensorMultiply(spectralGrad, tiledDecay);

            // Project back and accumulate
            var contribution = Engine.TensorMatMul(_eigenvectors, spectralGrad);
            inputGrad = Engine.TensorAdd(inputGrad, contribution);
        }

        return inputGrad;
    }

    /// <summary>
    /// Backpropagates through direct diffusion using vectorized operations.
    /// </summary>
    /// <param name="diffusedGrad">Gradient tensor [numVertices, InputChannels * NumTimeScales].</param>
    /// <param name="numVertices">Number of vertices.</param>
    /// <returns>Input gradient tensor.</returns>
    private Tensor<T> BackpropagateDiffusionDirectVectorized(Tensor<T> diffusedGrad, int numVertices)
    {
        if (_laplacian == null)
            return new Tensor<T>([numVertices, InputChannels]);

        // Initialize accumulated gradient
        var inputGrad = new Tensor<T>([numVertices, InputChannels]);

        for (int t = 0; t < NumTimeScales; t++)
        {
            T time = DiffusionTimes[t];
            int baseOffset = t * InputChannels;

            // Extract gradient slice for this time scale
            var timeGrad = Engine.TensorSlice(diffusedGrad, [0, baseOffset], [numVertices, InputChannels]);

            // Vectorized heat kernel computation: K[v,u] = exp(-L[v,u] * t)
            // For symmetric Laplacian, K^T = K
            var negLaplacian = Engine.TensorNegate(_laplacian);
            var scaledLaplacian = Engine.TensorMultiplyScalar(negLaplacian, time);
            var heatKernel = Engine.TensorExp(scaledLaplacian);

            // For backward pass, we need K^T @ grad
            var heatKernelTransposed = Engine.TensorTranspose(heatKernel);
            var contribution = Engine.TensorMatMul(heatKernelTransposed, timeGrad);
            inputGrad = Engine.TensorAdd(inputGrad, contribution);
        }

        return inputGrad;
    }

    #endregion

    #region Parameter Management

    /// <summary>
    /// Updates layer parameters using computed gradients.
    /// </summary>
    /// <param name="learningRate">Learning rate for gradient descent step.</param>
    /// <remarks>
    /// <para>
    /// Updates weights, biases, and diffusion time parameters using gradient descent.
    /// Diffusion times are constrained to remain positive after update.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGradient == null || _biasesGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        // Update weights
        var scaledWeightGrad = Engine.TensorMultiplyScalar(_weightsGradient, learningRate);
        _weights = Engine.TensorSubtract(_weights, scaledWeightGrad);

        // Update biases
        var scaledBiasGrad = Engine.TensorMultiplyScalar(_biasesGradient, learningRate);
        _biases = Engine.TensorSubtract(_biases, scaledBiasGrad);

        // Update diffusion times if gradients are available
        if (_diffusionTimesGradient != null)
        {
            T minTime = NumOps.FromDouble(1e-6); // Minimum allowed diffusion time
            for (int t = 0; t < NumTimeScales; t++)
            {
                T update = NumOps.Multiply(learningRate, _diffusionTimesGradient[t]);
                DiffusionTimes[t] = NumOps.Subtract(DiffusionTimes[t], update);

                // Clamp to minimum positive value to ensure valid diffusion
                if (NumOps.ToDouble(DiffusionTimes[t]) < NumOps.ToDouble(minTime))
                {
                    DiffusionTimes[t] = minTime;
                }
            }
        }
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        var timeParams = new Vector<T>(DiffusionTimes);
        return Vector<T>.Concatenate(
            new Vector<T>(_weights.ToArray()),
            new Vector<T>(_biases.ToArray()),
            timeParams);
    }

    /// <summary>
    /// Sets all trainable parameters from a vector.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        int expected = _weights.Length + _biases.Length + DiffusionTimes.Length;
        if (parameters.Length != expected)
            throw new ArgumentException($"Expected {expected} parameters, got {parameters.Length}.");

        int idx = 0;
        _weights = new Tensor<T>(_weights.Shape, parameters.Slice(idx, _weights.Length));
        idx += _weights.Length;
        _biases = new Tensor<T>(_biases.Shape, parameters.Slice(idx, _biases.Length));
        idx += _biases.Length;

        for (int i = 0; i < DiffusionTimes.Length; i++)
        {
            DiffusionTimes[i] = parameters[idx + i];
        }
    }

    /// <summary>
    /// Gets the weight tensor.
    /// </summary>
    public override Tensor<T> GetWeights() => _weights;

    /// <summary>
    /// Gets the bias tensor.
    /// </summary>
    public override Tensor<T> GetBiases() => _biases;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount => _weights.Length + _biases.Length + DiffusionTimes.Length;

    /// <summary>
    /// Creates a deep copy of this layer.
    /// </summary>
    public override LayerBase<T> Clone()
    {
        DiffusionConvLayer<T> copy;

        if (UsingVectorActivation)
        {
            copy = new DiffusionConvLayer<T>(
                InputChannels, OutputChannels, NumTimeScales, _numEigenvectors, InputShape[0], VectorActivation, _preferSpectralDiffusion);
        }
        else
        {
            copy = new DiffusionConvLayer<T>(
                InputChannels, OutputChannels, NumTimeScales, _numEigenvectors, InputShape[0], ScalarActivation, _preferSpectralDiffusion);
        }

        copy.SetParameters(GetParameters());

        if (_eigenvalues != null && _eigenvectors != null)
        {
            copy.SetEigenbasis(_eigenvalues, _eigenvectors, _massMatrix);
        }
        else if (_laplacian != null)
        {
            copy.SetLaplacian(_laplacian, _massMatrix);
        }

        return copy;
    }

    #endregion

    #region State Management

    /// <summary>
    /// Resets cached state from forward/backward passes.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _lastPreActivation = null;
        _lastOutput = null;
        _diffusedFeatures = null;
        _weightsGradient = null;
        _biasesGradient = null;
        ClearGpuCache();
    }

    private void ClearGpuCache()
    {
        _gpuDiffusedFeatures?.Dispose();
        _gpuDiffusedFeatures = null;
        _gpuPreActivation?.Dispose();
        _gpuPreActivation = null;
        _gpuInput = null;
        _gpuInputShape = null;
        _gpuOutput = null;
    }

    #endregion

    #region Serialization

    /// <summary>
    /// Serializes the layer to a binary stream.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Saves all learnable parameters and mesh configuration including:
    /// - Layer configuration (channels, time scales, eigenvector count)
    /// - Weights and biases
    /// - Diffusion time parameters
    /// - Eigenvalues and eigenvectors (if available)
    /// - Laplacian and mass matrices (if available)
    /// </para>
    /// </remarks>
    /// <param name="writer">Binary writer to serialize to.</param>
    public override void Serialize(BinaryWriter writer)
    {
        base.Serialize(writer);
        writer.Write(InputChannels);
        writer.Write(OutputChannels);
        writer.Write(NumTimeScales);
        writer.Write(_numEigenvectors);

        // Serialize weights
        var weightArray = _weights.ToArray();
        for (int i = 0; i < weightArray.Length; i++)
        {
            writer.Write(NumOps.ToDouble(weightArray[i]));
        }

        // Serialize biases
        var biasArray = _biases.ToArray();
        for (int i = 0; i < biasArray.Length; i++)
        {
            writer.Write(NumOps.ToDouble(biasArray[i]));
        }

        // Serialize diffusion times
        for (int i = 0; i < DiffusionTimes.Length; i++)
        {
            writer.Write(NumOps.ToDouble(DiffusionTimes[i]));
        }

        // Serialize mesh configuration
        // Flag indicating which mesh data is available
        byte meshFlags = 0;
        if (_eigenvalues != null && _eigenvectors != null) meshFlags |= 0x01;
        if (_laplacian != null) meshFlags |= 0x02;
        if (_massMatrix != null) meshFlags |= 0x04;
        writer.Write(meshFlags);

        // Serialize eigenvalues and eigenvectors
        if (_eigenvalues != null && _eigenvectors != null)
        {
            writer.Write(_eigenvalues.Length);
            for (int i = 0; i < _eigenvalues.Length; i++)
            {
                writer.Write(NumOps.ToDouble(_eigenvalues[i]));
            }

            // Write eigenvector shape and data
            writer.Write(_eigenvectors.Shape[0]); // numVertices
            writer.Write(_eigenvectors.Shape[1]); // numEigenvalues
            var eigenvectorArray = _eigenvectors.ToArray();
            for (int i = 0; i < eigenvectorArray.Length; i++)
            {
                writer.Write(NumOps.ToDouble(eigenvectorArray[i]));
            }
        }

        // Serialize Laplacian
        if (_laplacian != null)
        {
            writer.Write(_laplacian.Shape[0]); // numVertices (square matrix)
            var laplacianArray = _laplacian.ToArray();
            for (int i = 0; i < laplacianArray.Length; i++)
            {
                writer.Write(NumOps.ToDouble(laplacianArray[i]));
            }
        }

        // Serialize mass matrix
        if (_massMatrix != null)
        {
            writer.Write(_massMatrix.Length);
            var massArray = _massMatrix.ToArray();
            for (int i = 0; i < massArray.Length; i++)
            {
                writer.Write(NumOps.ToDouble(massArray[i]));
            }
        }
    }

    /// <summary>
    /// Deserializes the layer from a binary stream.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Restores all learnable parameters and mesh configuration.
    /// After deserialization, the layer is ready for inference without
    /// needing to call SetEigenbasis or SetLaplacian.
    /// </para>
    /// </remarks>
    /// <param name="reader">Binary reader to deserialize from.</param>
    public override void Deserialize(BinaryReader reader)
    {
        base.Deserialize(reader);
        InputChannels = reader.ReadInt32();
        OutputChannels = reader.ReadInt32();
        NumTimeScales = reader.ReadInt32();
        int numEigenvectors = reader.ReadInt32();

        int weightSize = InputChannels * NumTimeScales;
        _weights = new Tensor<T>([OutputChannels, weightSize]);
        var weightArray = new T[_weights.Length];
        for (int i = 0; i < weightArray.Length; i++)
        {
            weightArray[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        _weights = new Tensor<T>(weightArray, _weights.Shape);

        _biases = new Tensor<T>([OutputChannels]);
        var biasArray = new T[_biases.Length];
        for (int i = 0; i < biasArray.Length; i++)
        {
            biasArray[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        _biases = new Tensor<T>(biasArray, _biases.Shape);

        DiffusionTimes = new T[NumTimeScales];
        for (int i = 0; i < NumTimeScales; i++)
        {
            DiffusionTimes[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Deserialize mesh configuration
        byte meshFlags = reader.ReadByte();

        // Deserialize eigenvalues and eigenvectors
        if ((meshFlags & 0x01) != 0)
        {
            int numEig = reader.ReadInt32();
            _eigenvalues = new T[numEig];
            for (int i = 0; i < numEig; i++)
            {
                _eigenvalues[i] = NumOps.FromDouble(reader.ReadDouble());
            }

            int numVertices = reader.ReadInt32();
            int eigCount = reader.ReadInt32();
            var eigenvectorArray = new T[numVertices * eigCount];
            for (int i = 0; i < eigenvectorArray.Length; i++)
            {
                eigenvectorArray[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            _eigenvectors = new Tensor<T>(eigenvectorArray, [numVertices, eigCount]);
        }

        // Deserialize Laplacian
        if ((meshFlags & 0x02) != 0)
        {
            int numVertices = reader.ReadInt32();
            var laplacianArray = new T[numVertices * numVertices];
            for (int i = 0; i < laplacianArray.Length; i++)
            {
                laplacianArray[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            _laplacian = new Tensor<T>(laplacianArray, [numVertices, numVertices]);
        }

        // Deserialize mass matrix
        if ((meshFlags & 0x04) != 0)
        {
            int numVertices = reader.ReadInt32();
            var massArray = new T[numVertices];
            for (int i = 0; i < massArray.Length; i++)
            {
                massArray[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            _massMatrix = new Tensor<T>(massArray, [numVertices]);
        }
    }

    #endregion

    #region JIT Compilation

    /// <summary>
    /// Exports the layer as a computation graph for JIT compilation.
    /// </summary>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (_weights == null || _biases == null)
            throw new InvalidOperationException("Layer weights not initialized.");

        var symbolicInput = new Tensor<T>(InputShape);
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "diffusion_conv_input");
        inputNodes.Add(inputNode);

        var weightNode = TensorOperations<T>.Constant(_weights, "diffusion_conv_weights");
        var biasNode = TensorOperations<T>.Constant(_biases, "diffusion_conv_bias");

        var transposedWeights = TensorOperations<T>.Transpose(weightNode);
        var matmulNode = TensorOperations<T>.MatrixMultiply(inputNode, transposedWeights);
        var biasedNode = TensorOperations<T>.Add(matmulNode, biasNode);

        var activatedOutput = ApplyActivationToGraph(biasedNode);
        return activatedOutput;
    }

    #endregion
}
