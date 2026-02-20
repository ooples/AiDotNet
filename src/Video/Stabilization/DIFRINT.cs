using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Video.Options;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.Stabilization;

/// <summary>
/// DIFRINT: Deep Iterative Frame Interpolation for Full-frame Video Stabilization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DIFRINT stabilizes shaky video by generating smooth intermediate frames.
/// Unlike traditional stabilization that crops the frame, DIFRINT synthesizes full frames
/// without losing any content from the edges.
///
/// Key advantages:
/// - Full-frame stabilization (no cropping)
/// - Handles large camera motions
/// - Synthesizes missing content from warping
/// - Real-time performance possible
///
/// Example usage:
/// <code>
/// var model = new DIFRINT&lt;double&gt;(arch);
/// var stabilizedFrames = model.Stabilize(shakyFrames);
/// </code>
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Iterative refinement of stabilized frames
/// - Flow-based motion estimation
/// - Content synthesis for occluded regions
/// - Temporal consistency enforcement
/// </para>
/// <para>
/// <b>Reference:</b> "DIFRINT: A Framework for Full-Frame Video Stabilization"
/// https://arxiv.org/abs/2005.07055
/// </para>
/// </remarks>
public class DIFRINT<T> : VideoStabilizationBase<T>
{
    private readonly DIFRINTOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly string? _onnxModelPath;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly int _numFeatures;
    private readonly int _numIterations;
    private readonly int _imageHeight;
    private readonly int _imageWidth;

    #endregion

    #region Properties

    internal bool UseNativeMode => _useNativeMode;
    public override bool SupportsTraining => _useNativeMode;
    internal int NumFeatures => _numFeatures;
    internal int NumIterations => _numIterations;

    #endregion

    #region Constructors

    public DIFRINT(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numFeatures = 64,
        int numIterations = 3,
        DIFRINTOptions? options = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new DIFRINTOptions();
        Options = _options;
        _useNativeMode = true;
        _numFeatures = numFeatures;
        _numIterations = numIterations;
        _imageHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 480;
        _imageWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 640;

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    public DIFRINT(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        DIFRINTOptions? options = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new DIFRINTOptions();
        Options = _options;
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"DIFRINT ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _numFeatures = 64;
        _numIterations = 3;
        _imageHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 480;
        _imageWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 640;
        _lossFunction = new MeanSquaredErrorLoss<T>();

        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load ONNX model: {ex.Message}", ex); }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Stabilizes a sequence of video frames.
    /// </summary>
    public List<Tensor<T>> Stabilize(List<Tensor<T>> frames)
    {
        var stabilized = new List<Tensor<T>>();

        // Compute smooth camera path
        var smoothPath = ComputeSmoothPath(frames);

        for (int i = 0; i < frames.Count; i++)
        {
            var prevFrame = i > 0 ? frames[i - 1] : frames[i];
            var currFrame = frames[i];
            var nextFrame = i < frames.Count - 1 ? frames[i + 1] : frames[i];

            var stacked = StackFrames([prevFrame, currFrame, nextFrame]);
            var stabilizedFrame = _useNativeMode ? Forward(stacked) : PredictOnnx(stacked);

            stabilized.Add(stabilizedFrame);
        }

        return stabilized;
    }

    /// <summary>
    /// Stabilizes a single frame using neighboring frames.
    /// </summary>
    public Tensor<T> StabilizeFrame(Tensor<T> prevFrame, Tensor<T> currentFrame, Tensor<T> nextFrame)
    {
        var stacked = StackFrames([prevFrame, currentFrame, nextFrame]);
        return _useNativeMode ? Forward(stacked) : PredictOnnx(stacked);
    }

    /// <summary>
    /// Estimates camera motion between frames.
    /// </summary>
    public List<(double Dx, double Dy, double Rotation)> EstimateMotionPath(List<Tensor<T>> frames)
    {
        var motions = new List<(double, double, double)>();

        for (int i = 0; i < frames.Count - 1; i++)
        {
            var motion = EstimateMotionBetweenFrames(frames[i], frames[i + 1]);
            motions.Add(motion);
        }

        return motions;
    }

    #endregion

    #region Private Methods

    protected override Tensor<T> Forward(Tensor<T> input)
    {
        var result = input;
        foreach (var layer in Layers) result = layer.Forward(result);
        return result;
    }

    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++) inputData[i] = Convert.ToSingle(input.Data.Span[i]);

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_onnxSession.InputMetadata.Keys.First(), onnxInput) };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++) outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));

        return new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
    }

    private Tensor<T> StackFrames(List<Tensor<T>> frames)
    {
        var first = frames[0];
        int c = first.Rank == 4 ? first.Shape[1] : first.Shape[0];
        int h = first.Rank == 4 ? first.Shape[2] : first.Shape[1];
        int w = first.Rank == 4 ? first.Shape[3] : first.Shape[2];

        var stacked = new Tensor<T>([1, c * frames.Count, h, w]);
        for (int f = 0; f < frames.Count; f++)
            frames[f].Data.Span.Slice(0, c * h * w).CopyTo(stacked.Data.Span.Slice(f * c * h * w, c * h * w));
        return stacked;
    }

    private List<(double Dx, double Dy)> ComputeSmoothPath(List<Tensor<T>> frames)
    {
        var rawPath = new List<(double, double)>();
        double x = 0, y = 0;

        for (int i = 0; i < frames.Count - 1; i++)
        {
            var motion = EstimateMotionBetweenFrames(frames[i], frames[i + 1]);
            x += motion.Dx;
            y += motion.Dy;
            rawPath.Add((x, y));
        }

        // Apply Gaussian smoothing
        var smoothPath = new List<(double, double)>();
        int windowSize = 15;
        for (int i = 0; i < rawPath.Count; i++)
        {
            double sumX = 0, sumY = 0;
            int count = 0;
            for (int j = Math.Max(0, i - windowSize); j < Math.Min(rawPath.Count, i + windowSize); j++)
            {
                sumX += rawPath[j].Item1;
                sumY += rawPath[j].Item2;
                count++;
            }
            smoothPath.Add((sumX / count, sumY / count));
        }

        return smoothPath;
    }

    /// <summary>
    /// Estimates motion between two frames using block matching with subpixel refinement.
    /// Returns the global motion parameters (translation and rotation).
    /// </summary>
    private (double Dx, double Dy, double Rotation) EstimateMotionBetweenFrames(Tensor<T> frame1, Tensor<T> frame2)
    {
        int c = frame1.Rank == 4 ? frame1.Shape[1] : frame1.Shape[0];
        int h = frame1.Rank == 4 ? frame1.Shape[2] : frame1.Shape[1];
        int w = frame1.Rank == 4 ? frame1.Shape[3] : frame1.Shape[2];

        // Block matching parameters
        const int blockSize = 16;
        const int searchRange = 8;
        const int gridStep = 32; // Sample grid for efficiency

        var motionVectors = new List<(double dx, double dy, int x, int y)>();

        // Perform block matching at grid points
        for (int by = blockSize + searchRange; by < h - blockSize - searchRange; by += gridStep)
        {
            for (int bx = blockSize + searchRange; bx < w - blockSize - searchRange; bx += gridStep)
            {
                // Find best matching block in frame2
                double bestSad = double.MaxValue;
                int bestDx = 0, bestDy = 0;

                // Search within range
                for (int dy = -searchRange; dy <= searchRange; dy++)
                {
                    for (int dx = -searchRange; dx <= searchRange; dx++)
                    {
                        double sad = ComputeBlockSAD(frame1, frame2, bx, by, bx + dx, by + dy, blockSize, c, w);
                        if (sad < bestSad)
                        {
                            bestSad = sad;
                            bestDx = dx;
                            bestDy = dy;
                        }
                    }
                }

                // Subpixel refinement using parabolic fitting
                if (bestDx > -searchRange && bestDx < searchRange &&
                    bestDy > -searchRange && bestDy < searchRange)
                {
                    double sadLeft = ComputeBlockSAD(frame1, frame2, bx, by, bx + bestDx - 1, by + bestDy, blockSize, c, w);
                    double sadRight = ComputeBlockSAD(frame1, frame2, bx, by, bx + bestDx + 1, by + bestDy, blockSize, c, w);
                    double sadUp = ComputeBlockSAD(frame1, frame2, bx, by, bx + bestDx, by + bestDy - 1, blockSize, c, w);
                    double sadDown = ComputeBlockSAD(frame1, frame2, bx, by, bx + bestDx, by + bestDy + 1, blockSize, c, w);

                    // Parabolic subpixel refinement
                    double subDx = bestDx;
                    double subDy = bestDy;

                    double denom = 2 * (sadLeft + sadRight - 2 * bestSad);
                    if (Math.Abs(denom) > 1e-6)
                        subDx = bestDx - (sadRight - sadLeft) / denom;

                    denom = 2 * (sadUp + sadDown - 2 * bestSad);
                    if (Math.Abs(denom) > 1e-6)
                        subDy = bestDy - (sadDown - sadUp) / denom;

                    motionVectors.Add((subDx, subDy, bx, by));
                }
                else
                {
                    motionVectors.Add((bestDx, bestDy, bx, by));
                }
            }
        }

        if (motionVectors.Count == 0)
            return (0, 0, 0);

        // Robust estimation of global motion using RANSAC-like outlier rejection
        var sortedDx = motionVectors.Select(v => v.dx).OrderBy(x => x).ToList();
        var sortedDy = motionVectors.Select(v => v.dy).OrderBy(x => x).ToList();

        // Use median for robust translation estimation
        double medianDx = sortedDx[sortedDx.Count / 2];
        double medianDy = sortedDy[sortedDy.Count / 2];

        // Filter inliers (within 2 pixels of median)
        var inliers = motionVectors.Where(v =>
            Math.Abs(v.dx - medianDx) < 2 && Math.Abs(v.dy - medianDy) < 2).ToList();

        if (inliers.Count < 4)
        {
            // Fall back to median if too few inliers
            return (medianDx, medianDy, 0);
        }

        // Compute refined translation from inliers
        double globalDx = inliers.Average(v => v.dx);
        double globalDy = inliers.Average(v => v.dy);

        // Estimate rotation from motion field
        double rotation = EstimateRotation(inliers, w / 2.0, h / 2.0, globalDx, globalDy);

        return (globalDx, globalDy, rotation);
    }

    /// <summary>
    /// Computes Sum of Absolute Differences (SAD) between two blocks.
    /// </summary>
    private double ComputeBlockSAD(Tensor<T> frame1, Tensor<T> frame2,
        int x1, int y1, int x2, int y2, int blockSize, int channels, int width)
    {
        double sad = 0;
        int halfBlock = blockSize / 2;

        for (int dy = -halfBlock; dy < halfBlock; dy++)
        {
            for (int dx = -halfBlock; dx < halfBlock; dx++)
            {
                for (int ch = 0; ch < channels; ch++)
                {
                    int idx1 = ch * width * (frame1.Shape[frame1.Rank == 4 ? 2 : 1]) + (y1 + dy) * width + (x1 + dx);
                    int idx2 = ch * width * (frame2.Shape[frame2.Rank == 4 ? 2 : 1]) + (y2 + dy) * width + (x2 + dx);

                    if (idx1 >= 0 && idx1 < frame1.Data.Length && idx2 >= 0 && idx2 < frame2.Data.Length)
                    {
                        double v1 = Convert.ToDouble(frame1.Data.Span[idx1]);
                        double v2 = Convert.ToDouble(frame2.Data.Span[idx2]);
                        sad += Math.Abs(v2 - v1);
                    }
                }
            }
        }

        return sad;
    }

    /// <summary>
    /// Estimates rotation angle from motion vectors using least squares fitting.
    /// </summary>
    private double EstimateRotation(List<(double dx, double dy, int x, int y)> vectors,
        double centerX, double centerY, double globalDx, double globalDy)
    {
        // Estimate rotation from residual motion after translation removal
        double sumAngle = 0;
        int count = 0;

        foreach (var (dx, dy, x, y) in vectors)
        {
            // Position relative to center
            double rx = x - centerX;
            double ry = y - centerY;
            double r = Math.Sqrt(rx * rx + ry * ry);

            if (r > 10) // Only use points away from center
            {
                // Residual motion after translation
                double resDx = dx - globalDx;
                double resDy = dy - globalDy;

                // Expected tangential direction for rotation
                double tangX = -ry / r;
                double tangY = rx / r;

                // Project residual onto tangential direction
                double tangentialMotion = resDx * tangX + resDy * tangY;

                // Angle = arctan(tangential motion / radius)
                double angle = Math.Atan2(tangentialMotion, r);
                sumAngle += angle;
                count++;
            }
        }

        return count > 0 ? sumAngle / count : 0;
    }

    public override Tensor<T> Predict(Tensor<T> input) => Forward(input);

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Training is not supported in ONNX mode.");

        var prediction = Predict(input);
        LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

        var gradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
        var gradTensor = new Tensor<T>(prediction.Shape, gradient);

        for (int i = Layers.Count - 1; i >= 0; i--) gradTensor = Layers[i].Backward(gradTensor);
        _optimizer?.UpdateParameters(Layers);
    }

    #endregion

    #region Serialization

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
            Layers.AddRange(Architecture.Layers);
        else
        {
            int ch = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
            Layers.AddRange(LayerHelper<T>.CreateDefaultDIFRINTLayers(ch, _imageHeight, _imageWidth, _numFeatures, _numIterations));
        }
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Parameter updates are not supported in ONNX mode.");
        int offset = 0;
        foreach (var layer in Layers)
        {
            var p = layer.GetParameters();
            if (p.Length > 0 && offset + p.Length <= parameters.Length)
            {
                var slice = new Vector<T>(p.Length);
                for (int i = 0; i < p.Length; i++) slice[i] = parameters[offset + i];
                layer.SetParameters(slice);
                offset += p.Length;
            }
        }
    }

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        ModelType = ModelType.VideoStabilization,
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "DIFRINT" }, { "NumFeatures", _numFeatures }, { "NumIterations", _numIterations }
        },
        ModelData = _useNativeMode ? this.Serialize() : []
    };

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_numFeatures); writer.Write(_numIterations);
        writer.Write(_imageHeight); writer.Write(_imageWidth);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        for (int i = 0; i < 4; i++) _ = reader.ReadInt32();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new DIFRINT<T>(Architecture, _optimizer, _lossFunction, _numFeatures, _numIterations);

    #endregion

    #region Base Class Abstract Methods

    /// <inheritdoc/>
    public override Tensor<T> Stabilize(Tensor<T> unstableFrames)
    {
        return Forward(unstableFrames);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames)
    {
        return NormalizeFrames(rawFrames);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        return DenormalizeFrames(modelOutput);
    }

    #endregion

}
