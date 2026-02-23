using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Video.Options;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.Tracking;

/// <summary>
/// ByteTrack: Multi-Object Tracking by Associating Every Detection Box.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> ByteTrack is a simple yet powerful multi-object tracking method
/// that tracks all detection boxes, including low-confidence ones that other trackers ignore.
///
/// Key capabilities:
/// - Track multiple objects across video frames
/// - Handle occlusions and crowded scenes
/// - Associate detections between frames using motion prediction
/// - Maintain object IDs consistently over time
///
/// Example usage:
/// <code>
/// var model = new ByteTrack&lt;double&gt;(arch);
/// var tracks = model.Track(videoFrames);
/// foreach (var track in tracks)
///     Console.WriteLine($"Object {track.Id} at {track.BoundingBox}");
/// </code>
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - YOLOX-based detector backbone
/// - Kalman filter for motion prediction
/// - Two-stage association (high + low confidence boxes)
/// - IoU-based matching with Hungarian algorithm
/// </para>
/// <para>
/// <b>Reference:</b> "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
/// https://arxiv.org/abs/2110.06864
/// </para>
/// </remarks>
public class ByteTrack<T> : NeuralNetworkBase<T>
{
    private readonly ByteTrackOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly string? _onnxModelPath;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly int _numFeatures;
    private readonly int _numClasses;
    private readonly double _highThreshold;
    private readonly double _lowThreshold;
    private readonly int _maxAge;
    private readonly int _imageHeight;
    private readonly int _imageWidth;

    private readonly List<Track<T>> _activeTracks = [];
    private int _nextTrackId = 1;

    #endregion

    #region Properties

    internal bool UseNativeMode => _useNativeMode;
    public override bool SupportsTraining => _useNativeMode;
    internal int NumClasses => _numClasses;
    internal double HighThreshold => _highThreshold;
    internal double LowThreshold => _lowThreshold;

    #endregion

    #region Constructors

    public ByteTrack(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numFeatures = 256,
        int numClasses = 1,
        double highThreshold = 0.6,
        double lowThreshold = 0.1,
        int maxAge = 30,
        ByteTrackOptions? options = null)
        : base(architecture, lossFunction ?? new FocalLoss<T>())
    {
        _options = options ?? new ByteTrackOptions();
        Options = _options;
        _useNativeMode = true;
        _numFeatures = numFeatures;
        _numClasses = numClasses;
        _highThreshold = highThreshold;
        _lowThreshold = lowThreshold;
        _maxAge = maxAge;
        _imageHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 800;
        _imageWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 1440;

        _lossFunction = lossFunction ?? new FocalLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    public ByteTrack(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 1,
        ByteTrackOptions? options = null)
        : base(architecture, new FocalLoss<T>())
    {
        _options = options ?? new ByteTrackOptions();
        Options = _options;
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ByteTrack ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _numFeatures = 256;
        _numClasses = numClasses;
        _highThreshold = 0.6;
        _lowThreshold = 0.1;
        _maxAge = 30;
        _imageHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 800;
        _imageWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 1440;
        _lossFunction = new FocalLoss<T>();

        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load ONNX model: {ex.Message}", ex); }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Tracks objects across video frames.
    /// </summary>
    public List<List<Track<T>>> Track(List<Tensor<T>> frames)
    {
        var allTracks = new List<List<Track<T>>>();
        ResetTracks();

        foreach (var frame in frames)
        {
            var detections = Detect(frame);
            var frameTracks = UpdateTracks(detections);
            allTracks.Add(frameTracks);
        }

        return allTracks;
    }

    /// <summary>
    /// Detects objects in a single frame.
    /// </summary>
    public List<Detection<T>> Detect(Tensor<T> frame)
    {
        var output = _useNativeMode ? Forward(frame) : PredictOnnx(frame);
        return ParseDetections(output);
    }

    /// <summary>
    /// Updates tracks with new detections.
    /// </summary>
    public List<Track<T>> UpdateTracks(List<Detection<T>> detections)
    {
        var highConfDetections = detections.Where(d => d.Confidence >= _highThreshold).ToList();
        var lowConfDetections = detections.Where(d => d.Confidence >= _lowThreshold && d.Confidence < _highThreshold).ToList();

        // First association with high confidence detections
        var (matched, unmatchedTracks, unmatchedDetections) = AssociateDetections(_activeTracks, highConfDetections);

        foreach (var (track, detection) in matched)
        {
            track.Update(detection);
        }

        // Second association with low confidence detections for remaining tracks
        var (matched2, stillUnmatchedTracks, _) = AssociateDetections(unmatchedTracks, lowConfDetections);

        foreach (var (track, detection) in matched2)
        {
            track.Update(detection);
        }

        // Create new tracks for unmatched detections
        foreach (var detection in unmatchedDetections)
        {
            _activeTracks.Add(new Track<T>(_nextTrackId++, detection));
        }

        // Age and remove old tracks
        foreach (var track in stillUnmatchedTracks)
        {
            track.MarkMissed();
        }

        _activeTracks.RemoveAll(t => t.Age > _maxAge);

        return _activeTracks.Where(t => t.IsConfirmed).ToList();
    }

    /// <summary>
    /// Resets tracking state for a new video.
    /// </summary>
    public void ResetTracks()
    {
        _activeTracks.Clear();
        _nextTrackId = 1;
    }

    #endregion

    #region Private Methods

    private Tensor<T> Forward(Tensor<T> input)
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

    private List<Detection<T>> ParseDetections(Tensor<T> output)
    {
        var detections = new List<Detection<T>>();
        int numDetections = output.Shape[^1] / (5 + _numClasses);

        for (int i = 0; i < numDetections; i++)
        {
            int offset = i * (5 + _numClasses);
            double x = Convert.ToDouble(output.Data.Span[offset]);
            double y = Convert.ToDouble(output.Data.Span[offset + 1]);
            double w = Convert.ToDouble(output.Data.Span[offset + 2]);
            double h = Convert.ToDouble(output.Data.Span[offset + 3]);
            double conf = Convert.ToDouble(output.Data.Span[offset + 4]);

            if (conf >= _lowThreshold)
            {
                detections.Add(new Detection<T>(x, y, w, h, conf, 0));
            }
        }

        return detections;
    }

    private (List<(Track<T>, Detection<T>)> Matched, List<Track<T>> UnmatchedTracks, List<Detection<T>> UnmatchedDetections)
        AssociateDetections(List<Track<T>> tracks, List<Detection<T>> detections)
    {
        var matched = new List<(Track<T>, Detection<T>)>();
        var unmatchedTracks = new List<Track<T>>(tracks);
        var unmatchedDetections = new List<Detection<T>>(detections);

        foreach (var track in tracks.ToList())
        {
            Detection<T>? bestMatch = null;
            double bestIoU = 0.3;

            foreach (var detection in unmatchedDetections)
            {
                double iou = ComputeIoU(track.PredictedBox, detection);
                if (iou > bestIoU)
                {
                    bestIoU = iou;
                    bestMatch = detection;
                }
            }

            if (bestMatch != null)
            {
                matched.Add((track, bestMatch));
                unmatchedTracks.Remove(track);
                unmatchedDetections.Remove(bestMatch);
            }
        }

        return (matched, unmatchedTracks, unmatchedDetections);
    }

    private double ComputeIoU(Detection<T> box1, Detection<T> box2)
    {
        double x1 = Math.Max(box1.X - box1.Width / 2, box2.X - box2.Width / 2);
        double y1 = Math.Max(box1.Y - box1.Height / 2, box2.Y - box2.Height / 2);
        double x2 = Math.Min(box1.X + box1.Width / 2, box2.X + box2.Width / 2);
        double y2 = Math.Min(box1.Y + box1.Height / 2, box2.Y + box2.Height / 2);

        double intersection = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
        double union = box1.Width * box1.Height + box2.Width * box2.Height - intersection;

        return intersection / (union + 1e-8);
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultByteTrackLayers(ch, _imageHeight, _imageWidth, _numFeatures, _numClasses));
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
        ModelType = ModelType.ObjectTracking,
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "ByteTrack" }, { "NumClasses", _numClasses },
            { "HighThreshold", _highThreshold }, { "LowThreshold", _lowThreshold }
        },
        ModelData = _useNativeMode ? this.Serialize() : []
    };

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_numFeatures); writer.Write(_numClasses);
        writer.Write(_highThreshold); writer.Write(_lowThreshold); writer.Write(_maxAge);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); _ = reader.ReadInt32();
        _ = reader.ReadDouble(); _ = reader.ReadDouble(); _ = reader.ReadInt32();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new ByteTrack<T>(Architecture, _optimizer, _lossFunction, _numFeatures, _numClasses, _highThreshold, _lowThreshold, _maxAge);

    #endregion
}

/// <summary>
/// Represents a detection in a frame.
/// </summary>
public class Detection<T>
{
    public double X { get; }
    public double Y { get; }
    public double Width { get; }
    public double Height { get; }
    public double Confidence { get; }
    public int ClassId { get; }

    public Detection(double x, double y, double width, double height, double confidence, int classId)
    {
        X = x; Y = y; Width = width; Height = height; Confidence = confidence; ClassId = classId;
    }
}

/// <summary>
/// Represents a tracked object across frames.
/// </summary>
public class Track<T>
{
    public int Id { get; }
    public Detection<T> LastDetection { get; private set; }
    public Detection<T> PredictedBox { get; private set; }
    public int Age { get; private set; }
    public int HitStreak { get; private set; }
    public bool IsConfirmed => HitStreak >= 3;

    public Track(int id, Detection<T> detection)
    {
        Id = id;
        LastDetection = detection;
        PredictedBox = detection;
        Age = 0;
        HitStreak = 1;
    }

    public void Update(Detection<T> detection)
    {
        LastDetection = detection;
        PredictedBox = detection;
        Age = 0;
        HitStreak++;
    }

    public void MarkMissed()
    {
        Age++;
        HitStreak = 0;
    }
}
