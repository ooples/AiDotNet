using AiDotNet.Autodiff;
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
    /// <exception cref="ArgumentOutOfRangeException">Thrown when parameters are non-positive.</exception>
    public DiffusionConvLayer(
        int inputChannels,
        int outputChannels,
        int numTimeScales = 4,
        int numEigenvectors = 128,
        int numVertices = 1,
        IActivationFunction<T>? activation = null)
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
    public DiffusionConvLayer(
        int inputChannels,
        int outputChannels,
        int numTimeScales = 4,
        int numEigenvectors = 128,
        int numVertices = 1,
        IVectorActivationFunction<T>? vectorActivation = null)
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
        T scale = NumOps.Sqrt(NumericalStabilityHelper.SafeDiv(
            NumOps.FromDouble(2.0),
            NumOps.FromDouble(fanIn)));
        double scaleDouble = NumOps.ToDouble(scale);

        var random = RandomHelper.CreateSecureRandom();
        var weightData = new T[_weights.Length];

        for (int i = 0; i < weightData.Length; i++)
        {
            weightData[i] = NumOps.FromDouble((random.NextDouble() * 2.0 - 1.0) * scaleDouble);
        }
        _weights = new Tensor<T>(weightData, _weights.Shape);

        var biasData = new T[OutputChannels];
        for (int i = 0; i < biasData.Length; i++)
        {
            biasData[i] = NumOps.Zero;
        }
        _biases = new Tensor<T>(biasData, _biases.Shape);
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
    /// If the eigenbasis is not precomputed, diffusion will be computed using
    /// matrix exponentiation (slower but more accurate).
    /// </para>
    /// </remarks>
    public void SetLaplacian(Tensor<T> laplacian, Tensor<T>? massMatrix = null)
    {
        _laplacian = laplacian ?? throw new ArgumentNullException(nameof(laplacian));
        _massMatrix = massMatrix;
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
    /// Computes diffusion using spectral method (eigenbasis).
    /// </summary>
    /// <param name="input">Input features.</param>
    /// <param name="diffused">Output buffer for diffused features.</param>
    /// <param name="numVertices">Number of vertices.</param>
    private void ComputeDiffusionSpectral(Tensor<T> input, T[] diffused, int numVertices)
    {
        if (_eigenvalues == null || _eigenvectors == null)
            return;

        int numEig = Math.Min(_numEigenvectors, _eigenvalues.Length);

        Parallel.For(0, NumTimeScales, t =>
        {
            T time = DiffusionTimes[t];

            for (int c = 0; c < InputChannels; c++)
            {
                var spectralCoeffs = new T[numEig];
                for (int k = 0; k < numEig; k++)
                {
                    T coeff = NumOps.Zero;
                    for (int v = 0; v < numVertices; v++)
                    {
                        coeff = NumOps.Add(coeff, NumOps.Multiply(
                            _eigenvectors[v, k],
                            input[v, c]));
                    }

                    T decay = NumOps.Exp(NumOps.Negate(NumOps.Multiply(_eigenvalues[k], time)));
                    spectralCoeffs[k] = NumOps.Multiply(coeff, decay);
                }

                for (int v = 0; v < numVertices; v++)
                {
                    T result = NumOps.Zero;
                    for (int k = 0; k < numEig; k++)
                    {
                        result = NumOps.Add(result, NumOps.Multiply(
                            _eigenvectors[v, k],
                            spectralCoeffs[k]));
                    }

                    int outIdx = v * InputChannels * NumTimeScales + t * InputChannels + c;
                    diffused[outIdx] = result;
                }
            }
        });
    }

    /// <summary>
    /// Computes diffusion using direct matrix method.
    /// </summary>
    /// <param name="input">Input features.</param>
    /// <param name="diffused">Output buffer for diffused features.</param>
    /// <param name="numVertices">Number of vertices.</param>
    private void ComputeDiffusionDirect(Tensor<T> input, T[] diffused, int numVertices)
    {
        if (_laplacian == null)
            return;

        for (int t = 0; t < NumTimeScales; t++)
        {
            T time = DiffusionTimes[t];

            Parallel.For(0, numVertices, v =>
            {
                for (int c = 0; c < InputChannels; c++)
                {
                    T result = NumOps.Zero;

                    for (int u = 0; u < numVertices; u++)
                    {
                        T laplacianVal = _laplacian[v, u];
                        T inputVal = input[u, c];
                        T decayedVal = NumOps.Multiply(inputVal,
                            NumOps.Exp(NumOps.Multiply(NumOps.Negate(laplacianVal), time)));
                        result = NumOps.Add(result, decayedVal);
                    }

                    int outIdx = v * InputChannels * NumTimeScales + t * InputChannels + c;
                    diffused[outIdx] = result;
                }
            });
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

        var transposedDelta = Engine.TensorTranspose(delta);
        _weightsGradient = Engine.TensorMatMul(transposedDelta, _diffusedFeatures);
        _biasesGradient = Engine.ReduceSum(delta, [0], keepDims: false);

        var diffusedGrad = Engine.TensorMatMul(delta, _weights);
        var inputGrad = BackpropagateThroughDiffusion(diffusedGrad, numVertices);

        if (hasBatch)
        {
            int batchSize = _lastInput.Shape[0];
            inputGrad = inputGrad.Reshape(batchSize, numVertices, InputChannels);
        }

        return inputGrad;
    }

    /// <summary>
    /// Backward pass using automatic differentiation.
    /// </summary>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        return BackwardManual(outputGradient);
    }

    /// <summary>
    /// Backpropagates gradients through the diffusion operation.
    /// </summary>
    private Tensor<T> BackpropagateThroughDiffusion(Tensor<T> diffusedGrad, int numVertices)
    {
        var inputGrad = new T[numVertices * InputChannels];

        if (_eigenvalues != null && _eigenvectors != null)
        {
            BackpropagateDiffusionSpectral(diffusedGrad, inputGrad, numVertices);
        }
        else if (_laplacian != null)
        {
            BackpropagateDiffusionDirect(diffusedGrad, inputGrad, numVertices);
        }

        return new Tensor<T>(inputGrad, [numVertices, InputChannels]);
    }

    /// <summary>
    /// Backpropagates through spectral diffusion.
    /// </summary>
    private void BackpropagateDiffusionSpectral(Tensor<T> diffusedGrad, T[] inputGrad, int numVertices)
    {
        if (_eigenvalues == null || _eigenvectors == null)
            return;

        int numEig = Math.Min(_numEigenvectors, _eigenvalues.Length);

        Parallel.For(0, InputChannels, c =>
        {
            for (int v = 0; v < numVertices; v++)
            {
                T grad = NumOps.Zero;

                for (int t = 0; t < NumTimeScales; t++)
                {
                    T time = DiffusionTimes[t];
                    int gradIdx = v * InputChannels * NumTimeScales + t * InputChannels + c;
                    T diffGrad = diffusedGrad[v, t * InputChannels + c];

                    for (int k = 0; k < numEig; k++)
                    {
                        T decay = NumOps.Exp(NumOps.Negate(NumOps.Multiply(_eigenvalues[k], time)));
                        T eigVec = _eigenvectors[v, k];
                        grad = NumOps.Add(grad, NumOps.Multiply(NumOps.Multiply(diffGrad, eigVec), decay));
                    }
                }

                int outIdx = v * InputChannels + c;
                inputGrad[outIdx] = grad;
            }
        });
    }

    /// <summary>
    /// Backpropagates through direct diffusion.
    /// </summary>
    private void BackpropagateDiffusionDirect(Tensor<T> diffusedGrad, T[] inputGrad, int numVertices)
    {
        if (_laplacian == null)
            return;

        Parallel.For(0, numVertices, v =>
        {
            for (int c = 0; c < InputChannels; c++)
            {
                T grad = NumOps.Zero;

                for (int t = 0; t < NumTimeScales; t++)
                {
                    T time = DiffusionTimes[t];

                    for (int u = 0; u < numVertices; u++)
                    {
                        T laplacianVal = _laplacian[u, v];
                        T diffGrad = diffusedGrad[u, t * InputChannels + c];
                        T decay = NumOps.Exp(NumOps.Multiply(NumOps.Negate(laplacianVal), time));
                        grad = NumOps.Add(grad, NumOps.Multiply(diffGrad, decay));
                    }
                }

                int outIdx = v * InputChannels + c;
                inputGrad[outIdx] = grad;
            }
        });
    }

    #endregion

    #region Parameter Management

    /// <summary>
    /// Updates layer parameters using computed gradients.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGradient == null || _biasesGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        var scaledWeightGrad = Engine.TensorMultiplyScalar(_weightsGradient, learningRate);
        _weights = Engine.TensorSubtract(_weights, scaledWeightGrad);

        var scaledBiasGrad = Engine.TensorMultiplyScalar(_biasesGradient, learningRate);
        _biases = Engine.TensorSubtract(_biases, scaledBiasGrad);
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
                InputChannels, OutputChannels, NumTimeScales, _numEigenvectors, InputShape[0], VectorActivation);
        }
        else
        {
            copy = new DiffusionConvLayer<T>(
                InputChannels, OutputChannels, NumTimeScales, _numEigenvectors, InputShape[0], ScalarActivation);
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
    }

    #endregion

    #region Serialization

    /// <summary>
    /// Serializes the layer to a binary stream.
    /// </summary>
    public override void Serialize(BinaryWriter writer)
    {
        base.Serialize(writer);
        writer.Write(InputChannels);
        writer.Write(OutputChannels);
        writer.Write(NumTimeScales);
        writer.Write(_numEigenvectors);

        var weightArray = _weights.ToArray();
        for (int i = 0; i < weightArray.Length; i++)
        {
            writer.Write(NumOps.ToDouble(weightArray[i]));
        }

        var biasArray = _biases.ToArray();
        for (int i = 0; i < biasArray.Length; i++)
        {
            writer.Write(NumOps.ToDouble(biasArray[i]));
        }

        for (int i = 0; i < DiffusionTimes.Length; i++)
        {
            writer.Write(NumOps.ToDouble(DiffusionTimes[i]));
        }
    }

    /// <summary>
    /// Deserializes the layer from a binary stream.
    /// </summary>
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
