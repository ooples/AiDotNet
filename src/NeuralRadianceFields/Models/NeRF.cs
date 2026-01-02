using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralRadianceFields.Interfaces;

namespace AiDotNet.NeuralRadianceFields.Models;

/// <summary>
/// Implements Neural Radiance Fields (NeRF) for novel view synthesis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> NeRF is a groundbreaking method for creating photorealistic 3D scenes from 2D images.
/// </para>
/// <para>
/// What NeRF does:
/// - Input: Collection of photos of a scene from different angles
/// - Training: Learn a neural network that represents the 3D scene
/// - Output: Ability to render the scene from any new viewpoint
/// </para>
/// <para>
/// Key innovation:
/// - Represents entire 3D scene as a continuous 5D function
/// - Input: (x, y, z, θ, φ) - position and viewing direction
/// - Output: (r, g, b, σ) - color and volume density
/// </para>
/// <para>
/// Architecture:
/// 1. Positional encoding: Transform (x,y,z) to higher-dimensional space
///    - Why: Helps network learn high-frequency details
///    - Example: (x,y,z) → [sin(x), cos(x), sin(2x), cos(2x), ..., sin(2^L*x), cos(2^L*x)]
///    - Similar encoding for direction (θ, φ)
///
/// 2. Coarse network (8 layers, 256 units):
///    - Input: Encoded position
///    - Output: Density + intermediate features
///    - Input: Intermediate features + encoded direction
///    - Output: RGB color
///
/// 3. Fine network (same structure):
///    - Resamples based on coarse network predictions
///    - Focuses samples where density is high
///    - Produces final high-quality output
/// </para>
/// <para>
/// Why positional encoding matters:
/// - Neural networks naturally learn low-frequency functions (smooth, blurry)
/// - Real scenes have high-frequency details (sharp edges, textures)
/// - Positional encoding enables learning high-frequency details
/// - Without it: Blurry reconstructions
/// - With it: Sharp, detailed reconstructions
/// </para>
/// <para>
/// Training process:
/// 1. Sample random rays from training images
/// 2. Sample points along each ray
/// 3. Query network at each sample point
/// 4. Render ray using volume rendering
/// 5. Compare rendered color to actual pixel color
/// 6. Backpropagate error and update network weights
/// 7. Repeat for thousands of iterations
/// </para>
/// <para>
/// Hierarchical sampling:
/// - Coarse sampling: Uniform samples along ray
/// - Analyze coarse results: Where is density high?
/// - Fine sampling: More samples where density is high (near surfaces)
/// - Final rendering: Use both coarse and fine samples
/// - Result: Better quality with fewer total samples
/// </para>
/// <para>
/// Rendering equation (volume rendering):
/// C(r) = Σ T(t_i) * (1 - exp(-σ_i * δ_i)) * c_i
/// where:
/// - C(r): Final color of ray r
/// - T(t_i): Transmittance (how much light reaches point i)
/// - σ_i: Density at sample point i
/// - δ_i: Distance between sample points
/// - c_i: Color at sample point i
/// - T(t_i) = exp(-Σ(j&lt;i) σ_j * δ_j)
/// </para>
/// <para>
/// Applications:
/// - Virtual reality: Create immersive 3D environments from photos
/// - Film industry: Digitize real locations for CGI
/// - Real estate: Virtual property tours
/// - Cultural heritage: Preserve historical sites digitally
/// - Robotics: Build 3D maps for navigation
/// - Medical imaging: Reconstruct 3D anatomy from scans
/// </para>
/// <para>
/// Limitations of original NeRF:
/// - Slow training: Hours to days per scene
/// - Slow rendering: Seconds per image
/// - Scene-specific: Must retrain for each new scene
/// - Static only: Can't handle moving objects
///
/// These limitations led to many improved variants:
/// - Instant-NGP: 100x faster training and rendering
/// - Plenoxels: No neural network, faster optimization
/// - TensoRF: Tensor decomposition for efficiency
/// - Dynamic NeRF: Handle time-varying scenes
/// - Mip-NeRF: Better handling of scale/blur
/// </para>
/// <para>
/// Reference: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
/// by Mildenhall et al., ECCV 2020
/// </para>
/// </remarks>
public class NeRF<T> : NeuralNetworkBase<T>, IRadianceField<T>
{
    #region Model Architecture Parameters

    /// <summary>
    /// Number of frequency levels for position encoding (L in paper, typically 10).
    /// </summary>
    private readonly int _positionEncodingLevels;

    /// <summary>
    /// Number of frequency levels for direction encoding (L' in paper, typically 4).
    /// </summary>
    private readonly int _directionEncodingLevels;

    /// <summary>
    /// Hidden layer dimension (typically 256).
    /// </summary>
    private readonly int _hiddenDim;

    /// <summary>
    /// Number of MLP layers (typically 8).
    /// </summary>
    private readonly int _numLayers;

    /// <summary>
    /// Hidden dimension for color prediction network.
    /// </summary>
    private readonly int _colorHiddenDim;

    /// <summary>
    /// Number of layers in color prediction network.
    /// </summary>
    private readonly int _colorNumLayers;

    /// <summary>
    /// Number of samples per ray for rendering.
    /// </summary>
    private readonly int _renderSamples;

    /// <summary>
    /// Number of additional samples for hierarchical sampling.
    /// </summary>
    private readonly int _hierarchicalSamples;

    /// <summary>
    /// Layer index for skip connection.
    /// </summary>
    private readonly int _skipConnectionLayer;

    /// <summary>
    /// Whether to use hierarchical (coarse-to-fine) sampling.
    /// </summary>
    private readonly bool _useHierarchicalSampling;

    /// <summary>
    /// Near bound for ray sampling.
    /// </summary>
    private readonly T _renderNearBound;

    /// <summary>
    /// Far bound for ray sampling.
    /// </summary>
    private readonly T _renderFarBound;

    /// <summary>
    /// Learning rate for training.
    /// </summary>
    private readonly T _learningRate;

    #endregion

    #region Network Layers

    /// <summary>
    /// Position encoding MLP layers.
    /// </summary>
    private readonly List<DenseLayer<T>> _positionLayers = [];

    /// <summary>
    /// Density prediction layer.
    /// </summary>
    private DenseLayer<T>? _densityLayer;

    /// <summary>
    /// Feature extraction layer (before color prediction).
    /// </summary>
    private DenseLayer<T>? _featureLayer;

    /// <summary>
    /// Color prediction MLP layers.
    /// </summary>
    private readonly List<DenseLayer<T>> _colorLayers = [];

    /// <summary>
    /// Final RGB output layer.
    /// </summary>
    private DenseLayer<T>? _colorOutputLayer;

    #endregion

    #region Training State

    /// <summary>
    /// Cached positions from last forward pass (for backpropagation).
    /// </summary>
    private Tensor<T>? _lastPositions;

    /// <summary>
    /// Cached directions from last forward pass (for backpropagation).
    /// </summary>
    private Tensor<T>? _lastDirections;

    /// <summary>
    /// Cached position encoding from last forward pass.
    /// </summary>
    private Tensor<T>? _lastPositionEncoding;

    /// <summary>
    /// Cached direction encoding from last forward pass.
    /// </summary>
    private Tensor<T>? _lastDirectionEncoding;

    /// <summary>
    /// Cached raw density output from last forward pass.
    /// </summary>
    private Tensor<T>? _lastDensityRaw;

    /// <summary>
    /// Cached raw RGB output from last forward pass.
    /// </summary>
    private Tensor<T>? _lastRgbRaw;

    /// <summary>
    /// Loss function for training.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

    #endregion

    #region Public Properties

    /// <summary>
    /// Gets whether this network supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    #endregion

    #region Constructor

    /// <summary>
    /// Creates a new NeRF model for 3D scene representation and novel view synthesis.
    /// </summary>
    /// <param name="positionEncodingLevels">Number of frequency levels for position encoding.
    /// Higher values enable more high-frequency details but are harder to optimize. Default is 10.</param>
    /// <param name="directionEncodingLevels">Number of frequency levels for direction encoding.
    /// Lower than position (view dependence is smoother than geometry). Default is 4.</param>
    /// <param name="hiddenDim">Size of hidden layers. Larger values have more capacity but are slower. Default is 256.</param>
    /// <param name="numLayers">Depth of network. More layers can learn more complex functions. Default is 8.</param>
    /// <param name="colorHiddenDim">Hidden dimension for color prediction network. Default is 128.</param>
    /// <param name="colorNumLayers">Number of layers in color prediction network. Default is 1.</param>
    /// <param name="useHierarchicalSampling">Whether to use two-stage rendering (coarse + fine).
    /// True gives better quality but is slower. Default is true.</param>
    /// <param name="renderSamples">Number of samples per ray for rendering. Default is 64.</param>
    /// <param name="hierarchicalSamples">Additional samples for hierarchical sampling. Default is 128.</param>
    /// <param name="renderNearBound">Near bound for ray sampling. Default is 2.0.</param>
    /// <param name="renderFarBound">Far bound for ray sampling. Default is 6.0.</param>
    /// <param name="learningRate">Learning rate for training. Default is 5e-4.</param>
    /// <param name="lossFunction">Loss function for training. If null, MSE loss is used.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a NeRF model for 3D scene representation.
    ///
    /// Parameters explained:
    /// - positionEncodingLevels: How many frequencies for position encoding
    ///   - Higher = more high-frequency details (but harder to optimize)
    ///   - Typical: 10 (produces 60-dimensional encoding from 3D position)
    ///   - Formula: 3 * 2 * L = 60 for L=10
    ///
    /// - directionEncodingLevels: Frequencies for viewing direction encoding
    ///   - Lower than position (view dependence is smoother than geometry)
    ///   - Typical: 4 (produces 24-dimensional encoding from 2D direction)
    ///   - Formula: 3 * 2 * L' = 24 for L'=4
    ///
    /// - hiddenDim: Size of hidden layers
    ///   - Larger = more capacity (can represent more complex scenes)
    ///   - Larger = slower and needs more memory
    ///   - Typical: 256
    ///
    /// - numLayers: Depth of network
    ///   - More layers = can learn more complex functions
    ///   - More layers = slower and harder to train
    ///   - Typical: 8
    ///
    /// - useHierarchicalSampling: Two-stage rendering (coarse + fine)
    ///   - True: Better quality, slower (recommended)
    ///   - False: Faster, lower quality
    ///
    /// Standard NeRF configuration:
    /// <code>
    /// var nerf = new NeRF&lt;float&gt;(
    ///     positionEncodingLevels: 10,
    ///     directionEncodingLevels: 4,
    ///     hiddenDim: 256,
    ///     numLayers: 8,
    ///     useHierarchicalSampling: true);
    /// </code>
    /// </para>
    /// </remarks>
    public NeRF(
        int positionEncodingLevels = 10,
        int directionEncodingLevels = 4,
        int hiddenDim = 256,
        int numLayers = 8,
        int colorHiddenDim = 128,
        int colorNumLayers = 1,
        bool useHierarchicalSampling = true,
        int renderSamples = 64,
        int hierarchicalSamples = 128,
        double renderNearBound = 2.0,
        double renderFarBound = 6.0,
        double learningRate = 5e-4,
        ILossFunction<T>? lossFunction = null)
        : base(CreateArchitecture(hiddenDim), lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        // Validate parameters
        if (positionEncodingLevels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(positionEncodingLevels), "Position encoding levels must be positive.");
        }
        if (directionEncodingLevels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(directionEncodingLevels), "Direction encoding levels must be positive.");
        }
        if (hiddenDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenDim), "Hidden dimension must be positive.");
        }
        if (numLayers <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numLayers), "Number of layers must be positive.");
        }
        if (colorHiddenDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(colorHiddenDim), "Color hidden dimension must be positive.");
        }
        if (colorNumLayers < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(colorNumLayers), "Color layer count cannot be negative.");
        }
        if (renderSamples <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(renderSamples), "Render samples must be positive.");
        }
        if (hierarchicalSamples < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(hierarchicalSamples), "Hierarchical samples cannot be negative.");
        }

        // Store parameters
        _positionEncodingLevels = positionEncodingLevels;
        _directionEncodingLevels = directionEncodingLevels;
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _colorHiddenDim = colorHiddenDim;
        _colorNumLayers = colorNumLayers;
        _useHierarchicalSampling = useHierarchicalSampling;
        _renderSamples = renderSamples;
        _hierarchicalSamples = hierarchicalSamples;
        _renderNearBound = NumOps.FromDouble(renderNearBound);
        _renderFarBound = NumOps.FromDouble(renderFarBound);
        _learningRate = NumOps.FromDouble(learningRate);
        _skipConnectionLayer = numLayers >= 4 ? Math.Min(numLayers / 2, numLayers - 1) : -1;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        // Initialize network layers
        InitializeLayers();
    }

    #endregion

    #region Layer Initialization

    private static NeuralNetworkArchitecture<T> CreateArchitecture(int hiddenDim)
    {
        return new NeuralNetworkArchitecture<T>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputHeight: 1,
            inputWidth: 1,
            inputDepth: 6,
            outputSize: 4);
    }

    /// <summary>
    /// Initializes the neural network layers.
    /// </summary>
    protected override void InitializeLayers()
    {
        ClearLayers();
        _positionLayers.Clear();
        _colorLayers.Clear();

        int positionDim = 3 * 2 * _positionEncodingLevels;
        int directionDim = 3 * 2 * _directionEncodingLevels;

        // Position MLP layers
        for (int i = 0; i < _numLayers; i++)
        {
            int inputDim = i == 0 ? positionDim : _hiddenDim;
            if (_skipConnectionLayer >= 0 && i == _skipConnectionLayer)
            {
                inputDim = _hiddenDim + positionDim;
            }

            var layer = new DenseLayer<T>(inputDim, _hiddenDim, activationFunction: new ReLUActivation<T>());
            _positionLayers.Add(layer);
            AddLayerToCollection(layer);
        }

        // Density and feature layers
        _densityLayer = new DenseLayer<T>(_hiddenDim, 1, activationFunction: new IdentityActivation<T>());
        AddLayerToCollection(_densityLayer);

        _featureLayer = new DenseLayer<T>(_hiddenDim, _hiddenDim, activationFunction: new IdentityActivation<T>());
        AddLayerToCollection(_featureLayer);

        // Color MLP layers
        int colorInputDim = _hiddenDim + directionDim;
        int colorHiddenDim = _colorNumLayers > 0 ? _colorHiddenDim : colorInputDim;
        for (int i = 0; i < _colorNumLayers; i++)
        {
            int inputDim = i == 0 ? colorInputDim : colorHiddenDim;
            var layer = new DenseLayer<T>(inputDim, colorHiddenDim, activationFunction: new ReLUActivation<T>());
            _colorLayers.Add(layer);
            AddLayerToCollection(layer);
        }

        _colorOutputLayer = new DenseLayer<T>(colorHiddenDim, 3, activationFunction: new IdentityActivation<T>());
        AddLayerToCollection(_colorOutputLayer);
    }

    #endregion

    #region IRadianceField Implementation

    /// <summary>
    /// Queries the radiance field at given positions and viewing directions.
    /// </summary>
    /// <param name="positions">3D positions tensor of shape [N, 3].</param>
    /// <param name="viewingDirections">Viewing direction vectors of shape [N, 3].</param>
    /// <returns>RGB colors and volume densities for each query point.</returns>
    public (Tensor<T> rgb, Tensor<T> density) QueryField(Tensor<T> positions, Tensor<T> viewingDirections)
    {
        if (positions.Shape.Length != 2 || positions.Shape[1] != 3)
        {
            throw new ArgumentException("Positions must have shape [N, 3].", nameof(positions));
        }
        if (viewingDirections.Shape.Length != 2 || viewingDirections.Shape[1] != 3)
        {
            throw new ArgumentException("Viewing directions must have shape [N, 3].", nameof(viewingDirections));
        }

        var positionEncoding = PositionalEncoding(positions, _positionEncodingLevels);
        var directionEncoding = PositionalEncoding(NormalizeDirections(viewingDirections), _directionEncodingLevels);

        Tensor<T> x = positionEncoding;
        for (int i = 0; i < _positionLayers.Count; i++)
        {
            if (_skipConnectionLayer >= 0 && i == _skipConnectionLayer)
            {
                x = Engine.TensorConcatenate(new[] { x, positionEncoding }, axis: 1);
            }

            x = _positionLayers[i].Forward(x);
        }

        if (_densityLayer == null || _featureLayer == null || _colorOutputLayer == null)
        {
            throw new InvalidOperationException("NeRF layers are not initialized.");
        }

        var densityRaw = _densityLayer.Forward(x);
        var features = _featureLayer.Forward(x);
        var density = ApplySoftplus(densityRaw);

        var colorInput = Engine.TensorConcatenate(new[] { features, directionEncoding }, axis: 1);
        Tensor<T> color = colorInput;
        for (int i = 0; i < _colorLayers.Count; i++)
        {
            color = _colorLayers[i].Forward(color);
        }

        var rgbRaw = _colorOutputLayer.Forward(color);
        var rgb = ApplySigmoid(rgbRaw);

        if (IsTrainingMode)
        {
            _lastPositions = positions;
            _lastDirections = viewingDirections;
            _lastPositionEncoding = positionEncoding;
            _lastDirectionEncoding = directionEncoding;
            _lastDensityRaw = densityRaw;
            _lastRgbRaw = rgbRaw;
        }

        return (rgb, density);
    }

    #endregion

    #region Rendering Methods

    /// <summary>
    /// Renders an image from a camera viewpoint.
    /// </summary>
    /// <param name="cameraPosition">Camera position in world coordinates.</param>
    /// <param name="cameraRotation">Camera rotation matrix (3x3).</param>
    /// <param name="imageWidth">Output image width in pixels.</param>
    /// <param name="imageHeight">Output image height in pixels.</param>
    /// <param name="focalLength">Camera focal length.</param>
    /// <returns>Rendered image tensor of shape [height, width, 3].</returns>
    public Tensor<T> RenderImage(
        Vector<T> cameraPosition,
        Matrix<T> cameraRotation,
        int imageWidth,
        int imageHeight,
        T focalLength)
    {
        // Generate rays for each pixel
        var (rayOrigins, rayDirections) = GenerateCameraRays(
            cameraPosition, cameraRotation, imageWidth, imageHeight, focalLength);

        // Render rays
        var renderedColors = RenderRays(rayOrigins, rayDirections, _renderSamples, _renderNearBound, _renderFarBound);

        // Reshape to image
        return renderedColors.Reshape(imageHeight, imageWidth, 3);
    }

    /// <summary>
    /// Renders colors for a batch of rays.
    /// </summary>
    /// <param name="rayOrigins">Ray origin positions [N, 3].</param>
    /// <param name="rayDirections">Ray direction vectors [N, 3].</param>
    /// <param name="numSamples">Number of samples per ray.</param>
    /// <param name="nearBound">Near clipping plane.</param>
    /// <param name="farBound">Far clipping plane.</param>
    /// <returns>Rendered colors for each ray [N, 3].</returns>
    public Tensor<T> RenderRays(
        Tensor<T> rayOrigins,
        Tensor<T> rayDirections,
        int numSamples,
        T nearBound,
        T farBound)
    {
        int numRays = rayOrigins.Shape[0];

        // Sample points along rays
        var (samplePositions, sampleDirections, sampleTs, rayNear, rayFar) = SamplePointsAlongRays(
            rayOrigins, rayDirections, numSamples, nearBound, farBound, stratified: true);

        // Query radiance field
        var (rgb, density) = QueryField(samplePositions, sampleDirections);

        if (_useHierarchicalSampling && _hierarchicalSamples > 0 && numSamples > 1)
        {
            var weights = ComputeSampleWeights(density, numRays, numSamples, rayNear, rayFar, sampleTs);
            var fineTs = SampleImportance(sampleTs, weights, numRays, numSamples, _hierarchicalSamples);
            var mergedTs = MergeSampleTs(sampleTs, fineTs, numRays, numSamples, _hierarchicalSamples);
            var (mergedPositions, mergedDirections) = BuildSamplePositions(rayOrigins, rayDirections, mergedTs);
            var (mergedRgb, mergedDensity) = QueryField(mergedPositions, mergedDirections);
            return VolumeRendering(mergedRgb, mergedDensity, numRays, numSamples + _hierarchicalSamples, rayNear, rayFar, mergedTs);
        }

        // Perform volume rendering
        return VolumeRendering(rgb, density, numRays, numSamples, rayNear, rayFar, sampleTs);
    }

    #endregion

    #region Positional Encoding

    /// <summary>
    /// Computes positional encoding for Neural Radiance Fields using vectorized Engine operations.
    /// </summary>
    /// <param name="input">Input tensor of shape [N, D] where N is number of points and D is input dimension.</param>
    /// <param name="numLevels">Number of frequency levels for encoding.</param>
    /// <returns>Encoded tensor of shape [N, D * 2 * numLevels].</returns>
    private Tensor<T> PositionalEncoding(Tensor<T> input, int numLevels)
    {
        // Use vectorized Engine implementation for CPU/GPU acceleration
        return Engine.PositionalEncoding(input, numLevels);
    }

    /// <summary>
    /// Computes the backward pass for positional encoding using vectorized Engine operations.
    /// </summary>
    /// <param name="input">Original input tensor of shape [N, D].</param>
    /// <param name="numLevels">Number of frequency levels used in encoding.</param>
    /// <param name="encodedGradient">Gradient from downstream of shape [N, D * 2 * numLevels].</param>
    /// <returns>Gradient with respect to input of shape [N, D].</returns>
    private Tensor<T> PositionalEncodingBackward(Tensor<T> input, int numLevels, Tensor<T> encodedGradient)
    {
        // Use vectorized Engine implementation for CPU/GPU acceleration
        return Engine.PositionalEncodingBackward(input, encodedGradient, numLevels);
    }

    #endregion

    #region Helper Methods

    private Tensor<T> NormalizeDirections(Tensor<T> directions)
    {
        var norm = Engine.TensorNorm(directions, axis: 1, keepDims: true);
        norm = Engine.TensorAddScalar(norm, NumOps.FromDouble(1e-8));
        var normBroadcast = Engine.TensorTile(norm, new[] { 1, directions.Shape[1] });
        return Engine.TensorDivide(directions, normBroadcast);
    }

    private Tensor<T> ApplySoftplus(Tensor<T> input)
    {
        var exp = Engine.TensorExp(input);
        var expPlus = Engine.TensorAddScalar(exp, NumOps.One);
        return Engine.TensorLog(expPlus);
    }

    private Tensor<T> ApplySigmoid(Tensor<T> input)
    {
        return Engine.Sigmoid(input);
    }

    private Tensor<T> ApplySigmoidGradient(Tensor<T> raw, Tensor<T> gradient)
    {
        var numOps = NumOps;
        var data = new T[gradient.Length];
        for (int i = 0; i < gradient.Length; i++)
        {
            double rawVal = numOps.ToDouble(raw.Data[i]);
            double sig = 1.0 / (1.0 + Math.Exp(-rawVal));
            double grad = numOps.ToDouble(gradient.Data[i]);
            data[i] = numOps.FromDouble(grad * sig * (1.0 - sig));
        }

        return new Tensor<T>(data, gradient.Shape);
    }

    private Tensor<T> ApplySoftplusGradient(Tensor<T> raw, Tensor<T> gradient)
    {
        var numOps = NumOps;
        var data = new T[gradient.Length];
        for (int i = 0; i < gradient.Length; i++)
        {
            double rawVal = numOps.ToDouble(raw.Data[i]);
            double sigmoid = 1.0 / (1.0 + Math.Exp(-rawVal));
            double grad = numOps.ToDouble(gradient.Data[i]);
            data[i] = numOps.FromDouble(grad * sigmoid);
        }

        return new Tensor<T>(data, gradient.Shape);
    }

    private Tensor<T> AddTensors(Tensor<T> left, Tensor<T> right)
    {
        if (left.Length != right.Length)
        {
            throw new ArgumentException("Tensor lengths must match.");
        }

        var numOps = NumOps;
        var data = new T[left.Length];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = numOps.Add(left.Data[i], right.Data[i]);
        }

        return new Tensor<T>(data, left.Shape);
    }

    private (Tensor<T> origins, Tensor<T> directions) GenerateCameraRays(
        Vector<T> cameraPosition,
        Matrix<T> cameraRotation,
        int width,
        int height,
        T focalLength)
    {
        int numRays = width * height;
        var origins = new T[numRays * 3];
        var directions = new T[numRays * 3];
        double f = NumOps.ToDouble(focalLength);
        double cx = (width - 1) * 0.5;
        double cy = (height - 1) * 0.5;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double px = (x - cx) / f;
                double py = (y - cy) / f;
                double pz = 1.0;

                double dx = NumOps.ToDouble(cameraRotation[0, 0]) * px +
                            NumOps.ToDouble(cameraRotation[0, 1]) * py +
                            NumOps.ToDouble(cameraRotation[0, 2]) * pz;
                double dy = NumOps.ToDouble(cameraRotation[1, 0]) * px +
                            NumOps.ToDouble(cameraRotation[1, 1]) * py +
                            NumOps.ToDouble(cameraRotation[1, 2]) * pz;
                double dz = NumOps.ToDouble(cameraRotation[2, 0]) * px +
                            NumOps.ToDouble(cameraRotation[2, 1]) * py +
                            NumOps.ToDouble(cameraRotation[2, 2]) * pz;

                double norm = Math.Sqrt(dx * dx + dy * dy + dz * dz);
                if (norm > 0.0)
                {
                    double inv = 1.0 / norm;
                    dx *= inv;
                    dy *= inv;
                    dz *= inv;
                }

                int idx = (y * width + x) * 3;
                origins[idx] = cameraPosition[0];
                origins[idx + 1] = cameraPosition[1];
                origins[idx + 2] = cameraPosition[2];
                directions[idx] = NumOps.FromDouble(dx);
                directions[idx + 1] = NumOps.FromDouble(dy);
                directions[idx + 2] = NumOps.FromDouble(dz);
            }
        }

        return (new Tensor<T>(origins, [numRays, 3]), new Tensor<T>(directions, [numRays, 3]));
    }

    private (Tensor<T> positions, Tensor<T> directions, double[] sampleTs, double[] rayNear, double[] rayFar)
        SamplePointsAlongRays(
            Tensor<T> rayOrigins,
            Tensor<T> rayDirections,
            int numSamples,
            T nearBound,
            T farBound,
            bool stratified)
    {
        int numRays = rayOrigins.Shape[0];
        var numOps = NumOps;

        var positions = new T[numRays * numSamples * 3];
        var directions = new T[numRays * numSamples * 3];
        var sampleTs = new double[numRays * numSamples];
        var rayNear = new double[numRays];
        var rayFar = new double[numRays];

        double near = numOps.ToDouble(nearBound);
        double far = numOps.ToDouble(farBound);
        double step = numSamples > 0 ? (far - near) / numSamples : 0.0;
        var random = Random;

        for (int r = 0; r < numRays; r++)
        {
            rayNear[r] = near;
            rayFar[r] = far;

            for (int s = 0; s < numSamples; s++)
            {
                double t;
                if (numSamples == 1)
                {
                    t = 0.5 * (near + far);
                }
                else if (stratified)
                {
                    double t0 = near + step * s;
                    double t1 = t0 + step;
                    t = t0 + random.NextDouble() * (t1 - t0);
                }
                else
                {
                    t = near + step * (s + 0.5);
                }

                sampleTs[r * numSamples + s] = t;
                var tValue = numOps.FromDouble(t);

                for (int d = 0; d < 3; d++)
                {
                    var origin = rayOrigins.Data[r * 3 + d];
                    var dir = rayDirections.Data[r * 3 + d];
                    positions[(r * numSamples + s) * 3 + d] = numOps.Add(origin, numOps.Multiply(tValue, dir));
                    directions[(r * numSamples + s) * 3 + d] = dir;
                }
            }
        }

        return (new Tensor<T>(positions, [numRays * numSamples, 3]),
            new Tensor<T>(directions, [numRays * numSamples, 3]),
            sampleTs,
            rayNear,
            rayFar);
    }

    private (Tensor<T> positions, Tensor<T> directions) BuildSamplePositions(
        Tensor<T> rayOrigins,
        Tensor<T> rayDirections,
        double[] sampleTs)
    {
        int numRays = rayOrigins.Shape[0];
        int numSamples = sampleTs.Length / numRays;
        var positions = new T[numRays * numSamples * 3];
        var directions = new T[numRays * numSamples * 3];
        var numOps = NumOps;

        for (int r = 0; r < numRays; r++)
        {
            for (int s = 0; s < numSamples; s++)
            {
                double t = sampleTs[r * numSamples + s];
                var tValue = numOps.FromDouble(t);
                for (int d = 0; d < 3; d++)
                {
                    var origin = rayOrigins.Data[r * 3 + d];
                    var dir = rayDirections.Data[r * 3 + d];
                    positions[(r * numSamples + s) * 3 + d] = numOps.Add(origin, numOps.Multiply(tValue, dir));
                    directions[(r * numSamples + s) * 3 + d] = dir;
                }
            }
        }

        return (new Tensor<T>(positions, [numRays * numSamples, 3]),
            new Tensor<T>(directions, [numRays * numSamples, 3]));
    }

    private Tensor<T> VolumeRendering(
        Tensor<T> rgb,
        Tensor<T> density,
        int numRays,
        int numSamples,
        double[] rayNear,
        double[] rayFar,
        double[]? sampleTs = null)
    {
        var colors = new T[numRays * 3];
        var numOps = NumOps;
        var rgbData = rgb.Data;
        var densityData = density.Data;

        for (int r = 0; r < numRays; r++)
        {
            double transmittance = 1.0;
            double accumR = 0.0;
            double accumG = 0.0;
            double accumB = 0.0;
            double uniformDeltaT = numSamples > 0 ? (rayFar[r] - rayNear[r]) / numSamples : 0.0;

            if (uniformDeltaT < 0.0)
            {
                uniformDeltaT = 0.0;
            }

            for (int s = 0; s < numSamples; s++)
            {
                int idx = r * numSamples + s;
                double deltaT = uniformDeltaT;
                if (sampleTs != null)
                {
                    double t0 = sampleTs[idx];
                    double t1 = s + 1 < numSamples ? sampleTs[idx + 1] : rayFar[r];
                    if (t1 <= t0)
                    {
                        t1 = rayFar[r];
                    }

                    deltaT = t1 - t0;
                    if (deltaT < 0.0)
                    {
                        deltaT = 0.0;
                    }
                }

                double sigma = numOps.ToDouble(densityData[idx]);
                double alpha = 1.0 - Math.Exp(-sigma * deltaT);
                if (alpha <= 0.0)
                {
                    continue;
                }

                int rgbIdx = idx * 3;
                accumR += transmittance * alpha * numOps.ToDouble(rgbData[rgbIdx]);
                accumG += transmittance * alpha * numOps.ToDouble(rgbData[rgbIdx + 1]);
                accumB += transmittance * alpha * numOps.ToDouble(rgbData[rgbIdx + 2]);

                transmittance *= (1.0 - alpha);
                if (transmittance < 1e-4)
                {
                    break;
                }
            }

            int outIdx = r * 3;
            colors[outIdx] = numOps.FromDouble(accumR);
            colors[outIdx + 1] = numOps.FromDouble(accumG);
            colors[outIdx + 2] = numOps.FromDouble(accumB);
        }

        return new Tensor<T>(colors, [numRays, 3]);
    }

    private double[] ComputeSampleWeights(
        Tensor<T> density,
        int numRays,
        int numSamples,
        double[] rayNear,
        double[] rayFar,
        double[] sampleTs)
    {
        var weights = new double[numRays * numSamples];
        var densityData = density.Data;
        var numOps = NumOps;

        for (int r = 0; r < numRays; r++)
        {
            double transmittance = 1.0;
            for (int s = 0; s < numSamples; s++)
            {
                int idx = r * numSamples + s;
                double t0 = sampleTs[idx];
                double t1 = s + 1 < numSamples ? sampleTs[idx + 1] : rayFar[r];
                if (t1 < t0)
                {
                    t1 = rayFar[r];
                }

                double deltaT = t1 - t0;
                if (deltaT < 0.0)
                {
                    deltaT = 0.0;
                }

                double sigma = numOps.ToDouble(densityData[idx]);
                double alpha = 1.0 - Math.Exp(-sigma * deltaT);
                weights[idx] = transmittance * alpha;
                transmittance *= (1.0 - alpha);
                if (transmittance < 1e-4)
                {
                    break;
                }
            }
        }

        return weights;
    }

    private double[] SampleImportance(
        double[] sampleTs,
        double[] weights,
        int numRays,
        int numSamples,
        int numFineSamples)
    {
        var fineTs = new double[numRays * numFineSamples];
        var random = Random;

        for (int r = 0; r < numRays; r++)
        {
            int coarseOffset = r * numSamples;
            int fineOffset = r * numFineSamples;
            double weightSum = 0.0;
            for (int s = 0; s < numSamples; s++)
            {
                weightSum += weights[coarseOffset + s];
            }

            if (weightSum <= 1e-8)
            {
                double minT = sampleTs[coarseOffset];
                double maxT = sampleTs[coarseOffset + numSamples - 1];
                for (int i = 0; i < numFineSamples; i++)
                {
                    double u = (i + random.NextDouble()) / numFineSamples;
                    fineTs[fineOffset + i] = minT + u * (maxT - minT);
                }
                continue;
            }

            var cdf = new double[numSamples];
            double accum = 0.0;
            for (int s = 0; s < numSamples; s++)
            {
                accum += weights[coarseOffset + s] / weightSum;
                cdf[s] = accum;
            }

            for (int i = 0; i < numFineSamples; i++)
            {
                double u = (i + random.NextDouble()) / numFineSamples;
                int idx = 0;
                while (idx < numSamples - 1 && u > cdf[idx])
                {
                    idx++;
                }

                double cdfPrev = idx > 0 ? cdf[idx - 1] : 0.0;
                double cdfNext = cdf[idx];
                double t0 = sampleTs[coarseOffset + Math.Max(idx - 1, 0)];
                double t1 = sampleTs[coarseOffset + idx];
                double denom = cdfNext - cdfPrev;
                double t = denom > 0.0 ? t0 + (u - cdfPrev) / denom * (t1 - t0) : t0;
                fineTs[fineOffset + i] = t;
            }
        }

        return fineTs;
    }

    private double[] MergeSampleTs(
        double[] coarseTs,
        double[] fineTs,
        int numRays,
        int numCoarse,
        int numFine)
    {
        int total = numCoarse + numFine;
        var merged = new double[numRays * total];
        var buffer = new double[total];

        for (int r = 0; r < numRays; r++)
        {
            int coarseOffset = r * numCoarse;
            int fineOffset = r * numFine;

            Array.Copy(coarseTs, coarseOffset, buffer, 0, numCoarse);
            Array.Copy(fineTs, fineOffset, buffer, numCoarse, numFine);
            Array.Sort(buffer);

            Array.Copy(buffer, 0, merged, r * total, total);
        }

        return merged;
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <summary>
    /// Performs forward pass with memory for backpropagation.
    /// </summary>
    public override Tensor<T> ForwardWithMemory(Tensor<T> input)
    {
        if (input.Shape.Length != 2 || input.Shape[1] != 6)
        {
            throw new ArgumentException("Input must have shape [N, 6] (position + direction).", nameof(input));
        }

        // Input should be [N, 6] with positions and directions concatenated
        // Split and query
        int numPoints = input.Shape[0];
        var positions = new T[numPoints * 3];
        var directions = new T[numPoints * 3];

        for (int i = 0; i < numPoints; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                positions[i * 3 + j] = input.Data[i * 6 + j];
                directions[i * 3 + j] = input.Data[i * 6 + 3 + j];
            }
        }

        var posT = new Tensor<T>(positions, [numPoints, 3]);
        var dirT = new Tensor<T>(directions, [numPoints, 3]);

        var (rgb, density) = QueryField(posT, dirT);

        // Concatenate rgb and density for output
        var output = new T[numPoints * 4];
        for (int i = 0; i < numPoints; i++)
        {
            output[i * 4] = rgb.Data[i * 3];
            output[i * 4 + 1] = rgb.Data[i * 3 + 1];
            output[i * 4 + 2] = rgb.Data[i * 3 + 2];
            output[i * 4 + 3] = density.Data[i];
        }

        return new Tensor<T>(output, [numPoints, 4]);
    }

    /// <summary>
    /// Performs backpropagation to compute gradients.
    /// </summary>
    public override Tensor<T> Backpropagate(Tensor<T> outputGradient)
    {
        if (_lastPositionEncoding == null || _lastDirectionEncoding == null || _lastDensityRaw == null || _lastRgbRaw == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backpropagation.");
        }
        if (_densityLayer == null || _featureLayer == null || _colorOutputLayer == null)
        {
            throw new InvalidOperationException("NeRF layers are not initialized.");
        }
        if (outputGradient.Shape.Length != 2 || outputGradient.Shape[1] != 4)
        {
            throw new ArgumentException("Output gradient must have shape [N, 4].", nameof(outputGradient));
        }

        int numPoints = outputGradient.Shape[0];
        int posDim = _lastPositionEncoding.Shape[1];
        int dirDim = _lastDirectionEncoding.Shape[1];

        var rgbGrad = new T[numPoints * 3];
        var densityGrad = new T[numPoints];
        for (int i = 0; i < numPoints; i++)
        {
            int baseIdx = i * 4;
            rgbGrad[i * 3] = outputGradient.Data[baseIdx];
            rgbGrad[i * 3 + 1] = outputGradient.Data[baseIdx + 1];
            rgbGrad[i * 3 + 2] = outputGradient.Data[baseIdx + 2];
            densityGrad[i] = outputGradient.Data[baseIdx + 3];
        }

        var rgbGradTensor = new Tensor<T>(rgbGrad, [numPoints, 3]);
        var densityGradTensor = new Tensor<T>(densityGrad, [numPoints, 1]);
        var rgbRawGrad = ApplySigmoidGradient(_lastRgbRaw, rgbGradTensor);
        var densityRawGrad = ApplySoftplusGradient(_lastDensityRaw, densityGradTensor);

        Tensor<T> gradColor = _colorOutputLayer.Backward(rgbRawGrad);
        for (int i = _colorLayers.Count - 1; i >= 0; i--)
        {
            gradColor = _colorLayers[i].Backward(gradColor);
        }

        var gradFeatures = new T[numPoints * _hiddenDim];
        var gradDirEncoded = new T[numPoints * dirDim];
        int colorStride = _hiddenDim + dirDim;
        for (int i = 0; i < numPoints; i++)
        {
            int colorBase = i * colorStride;
            int featureBase = i * _hiddenDim;
            int dirBase = i * dirDim;
            for (int f = 0; f < _hiddenDim; f++)
            {
                gradFeatures[featureBase + f] = gradColor.Data[colorBase + f];
            }
            for (int d = 0; d < dirDim; d++)
            {
                gradDirEncoded[dirBase + d] = gradColor.Data[colorBase + _hiddenDim + d];
            }
        }

        var gradFeatureTensor = new Tensor<T>(gradFeatures, [numPoints, _hiddenDim]);
        var gradDirEncodedTensor = new Tensor<T>(gradDirEncoded, [numPoints, dirDim]);

        var gradFromFeatures = _featureLayer.Backward(gradFeatureTensor);
        var gradFromDensity = _densityLayer.Backward(densityRawGrad);
        var gradBase = AddTensors(gradFromFeatures, gradFromDensity);

        var posEncodingGrad = new T[numPoints * posDim];
        Tensor<T> grad = gradBase;
        var numOps = NumOps;

        for (int i = _positionLayers.Count - 1; i >= 0; i--)
        {
            if (_skipConnectionLayer >= 0 && i == _skipConnectionLayer)
            {
                var gradConcat = _positionLayers[i].Backward(grad);
                var gradHidden = new T[numPoints * _hiddenDim];
                var gradSkip = new T[numPoints * posDim];
                for (int n = 0; n < numPoints; n++)
                {
                    int concatBase = n * (_hiddenDim + posDim);
                    int hiddenBase = n * _hiddenDim;
                    int posBase = n * posDim;
                    for (int h = 0; h < _hiddenDim; h++)
                    {
                        gradHidden[hiddenBase + h] = gradConcat.Data[concatBase + h];
                    }
                    for (int p = 0; p < posDim; p++)
                    {
                        gradSkip[posBase + p] = gradConcat.Data[concatBase + _hiddenDim + p];
                    }
                }

                for (int j = 0; j < posEncodingGrad.Length; j++)
                {
                    posEncodingGrad[j] = numOps.Add(posEncodingGrad[j], gradSkip[j]);
                }

                grad = new Tensor<T>(gradHidden, [numPoints, _hiddenDim]);
            }
            else
            {
                grad = _positionLayers[i].Backward(grad);
            }
        }

        for (int j = 0; j < posEncodingGrad.Length; j++)
        {
            posEncodingGrad[j] = numOps.Add(posEncodingGrad[j], grad.Data[j]);
        }

        if (_lastPositions == null || _lastDirections == null)
        {
            throw new InvalidOperationException("Cached inputs missing from forward pass.");
        }

        var posEncodedGradTensor = new Tensor<T>(posEncodingGrad, [numPoints, posDim]);
        var posGrad = PositionalEncodingBackward(_lastPositions, _positionEncodingLevels, posEncodedGradTensor);
        var normalizedDirections = NormalizeDirections(_lastDirections);
        var dirGrad = PositionalEncodingBackward(normalizedDirections, _directionEncodingLevels, gradDirEncodedTensor);

        var inputGrad = new T[numPoints * 6];
        for (int i = 0; i < numPoints; i++)
        {
            int baseIdx = i * 6;
            int posBase = i * 3;
            int dirBase = i * 3;
            for (int d = 0; d < 3; d++)
            {
                inputGrad[baseIdx + d] = posGrad.Data[posBase + d];
                inputGrad[baseIdx + 3 + d] = dirGrad.Data[dirBase + d];
            }
        }

        return new Tensor<T>(inputGrad, [numPoints, 6]);
    }

    /// <summary>
    /// Trains the model on input data.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Set training mode
        SetTrainingMode(true);

        // Forward pass
        var prediction = ForwardWithMemory(input);

        // Compute loss
        var flatPrediction = prediction.ToVector();
        var flatExpected = expectedOutput.ToVector();
        LastLoss = _lossFunction.CalculateLoss(flatPrediction, flatExpected);

        // Compute gradients
        var lossGradient = _lossFunction.CalculateDerivative(flatPrediction, flatExpected);
        Backpropagate(Tensor<T>.FromVector(lossGradient));

        // Update layer parameters
        foreach (var layer in Layers)
        {
            if (layer.SupportsTraining && layer.ParameterCount > 0)
            {
                layer.UpdateParameters(_learningRate);
            }
        }

        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates model parameters using gradient descent.
    /// </summary>
    public override void UpdateParameters(Vector<T> gradients)
    {
        if (gradients == null)
        {
            throw new ArgumentNullException(nameof(gradients));
        }

        // Apply gradient descent: params = params - learning_rate * gradients
        var currentParams = GetParameters();
        for (int i = 0; i < currentParams.Length; i++)
        {
            currentParams[i] = NumOps.Subtract(currentParams[i], NumOps.Multiply(_learningRate, gradients[i]));
        }

        SetParameters(currentParams);
    }

    /// <summary>
    /// Makes a prediction using the model.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        SetTrainingMode(false);
        return ForwardWithMemory(input);
    }

    #endregion

    #region Metadata and Serialization

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "PositionEncodingLevels", _positionEncodingLevels },
                { "DirectionEncodingLevels", _directionEncodingLevels },
                { "HiddenDim", _hiddenDim },
                { "NumLayers", _numLayers },
                { "ColorHiddenDim", _colorHiddenDim },
                { "ColorNumLayers", _colorNumLayers },
                { "UseHierarchicalSampling", _useHierarchicalSampling },
                { "RenderSamples", _renderSamples },
                { "HierarchicalSamples", _hierarchicalSamples },
                { "RenderNearBound", NumOps.ToDouble(_renderNearBound) },
                { "RenderFarBound", NumOps.ToDouble(_renderFarBound) },
                { "LearningRate", NumOps.ToDouble(_learningRate) },
                { "LayerCount", Layers.Count },
                { "TotalParameters", ParameterCount }
            },
            ModelData = Serialize()
        };
    }

    /// <summary>
    /// Serializes network-specific data.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_positionEncodingLevels);
        writer.Write(_directionEncodingLevels);
        writer.Write(_hiddenDim);
        writer.Write(_numLayers);
        writer.Write(_colorHiddenDim);
        writer.Write(_colorNumLayers);
        writer.Write(_useHierarchicalSampling);
        writer.Write(_renderSamples);
        writer.Write(_hierarchicalSamples);
        writer.Write(NumOps.ToDouble(_renderNearBound));
        writer.Write(NumOps.ToDouble(_renderFarBound));
        writer.Write(NumOps.ToDouble(_learningRate));
    }

    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int positionLevels = reader.ReadInt32();
        int directionLevels = reader.ReadInt32();
        int hiddenDim = reader.ReadInt32();
        int numLayers = reader.ReadInt32();
        int colorHiddenDim = reader.ReadInt32();
        int colorNumLayers = reader.ReadInt32();
        bool useHierarchical = reader.ReadBoolean();
        int renderSamples = reader.ReadInt32();
        int hierarchicalSamples = reader.ReadInt32();
        double renderNear = reader.ReadDouble();
        double renderFar = reader.ReadDouble();
        double learningRate = reader.ReadDouble();

        if (positionLevels != _positionEncodingLevels ||
            directionLevels != _directionEncodingLevels ||
            hiddenDim != _hiddenDim ||
            numLayers != _numLayers ||
            colorHiddenDim != _colorHiddenDim ||
            colorNumLayers != _colorNumLayers ||
            useHierarchical != _useHierarchicalSampling ||
            renderSamples != _renderSamples ||
            hierarchicalSamples != _hierarchicalSamples ||
            Math.Abs(renderNear - NumOps.ToDouble(_renderNearBound)) > 1e-8 ||
            Math.Abs(renderFar - NumOps.ToDouble(_renderFarBound)) > 1e-8 ||
            Math.Abs(learningRate - NumOps.ToDouble(_learningRate)) > 1e-8)
        {
            throw new InvalidOperationException("Serialized NeRF configuration does not match this instance.");
        }
    }

    /// <summary>
    /// Creates a new instance of this model for cloning.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new NeRF<T>(
            positionEncodingLevels: _positionEncodingLevels,
            directionEncodingLevels: _directionEncodingLevels,
            hiddenDim: _hiddenDim,
            numLayers: _numLayers,
            colorHiddenDim: _colorHiddenDim,
            colorNumLayers: _colorNumLayers,
            useHierarchicalSampling: _useHierarchicalSampling,
            renderSamples: _renderSamples,
            hierarchicalSamples: _hierarchicalSamples,
            renderNearBound: NumOps.ToDouble(_renderNearBound),
            renderFarBound: NumOps.ToDouble(_renderFarBound),
            learningRate: NumOps.ToDouble(_learningRate),
            lossFunction: _lossFunction);
    }

    #endregion
}
