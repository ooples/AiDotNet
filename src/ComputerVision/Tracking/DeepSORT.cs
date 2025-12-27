using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ComputerVision.Tracking;

/// <summary>
/// DeepSORT (Deep SORT) tracking with appearance features.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> DeepSORT extends SORT by adding deep appearance features
/// (Re-ID embeddings) to improve association accuracy, especially for occlusions and
/// ID switches. It uses a cascade matching strategy.</para>
///
/// <para>Key features:
/// - Deep appearance descriptor (Re-ID network)
/// - Cascade matching (prioritize recent tracks)
/// - Combined motion and appearance matching
/// - Gallery of appearance features per track
/// </para>
///
/// <para>Reference: Wojke et al., "Simple Online and Realtime Tracking with a Deep
/// Association Metric", ICIP 2017</para>
/// </remarks>
public class DeepSORT<T> : ObjectTrackerBase<T>
{
    private readonly List<DeepTrack<T>> _deepTracks;
    private readonly ReIDNetwork<T>? _reidNetwork;
    private readonly int _maxGallerySize;

    /// <inheritdoc/>
    public override string Name => "DeepSORT";

    /// <summary>
    /// Creates a new DeepSORT tracker.
    /// </summary>
    public DeepSORT(TrackingOptions<T> options) : base(options)
    {
        _deepTracks = new List<DeepTrack<T>>();
        _maxGallerySize = 100;

        if (options.UseAppearance)
        {
            _reidNetwork = new ReIDNetwork<T>();
        }
    }

    /// <inheritdoc/>
    public override TrackingResult<T> Update(List<Detection<T>> detections)
    {
        return Update(detections, null);
    }

    /// <inheritdoc/>
    public override TrackingResult<T> Update(List<Detection<T>> detections, Tensor<T>? image)
    {
        var startTime = DateTime.UtcNow;
        FrameCount++;

        // Filter detections by confidence
        var filteredDetections = detections
            .Where(d => NumOps.ToDouble(d.Confidence) >= NumOps.ToDouble(Options.ConfidenceThreshold))
            .ToList();

        // Extract appearance features if network available and image provided
        List<Tensor<T>> appearanceFeatures = new();
        if (_reidNetwork != null && image != null)
        {
            appearanceFeatures = ExtractAppearanceFeatures(image, filteredDetections);
        }

        // Predict new locations
        foreach (var track in _deepTracks)
        {
            track.Predict();
        }

        // Split tracks into confirmed and unconfirmed
        var confirmedTracks = _deepTracks.Where(t => t.State == TrackState.Confirmed).ToList();
        var unconfirmedTracks = _deepTracks.Where(t => t.State == TrackState.Tentative).ToList();

        // Cascade matching for confirmed tracks
        var (matchedConfirmed, unmatchedDetections) = CascadeMatching(
            confirmedTracks, filteredDetections, appearanceFeatures);

        // IoU matching for unconfirmed tracks
        var (matchedUnconfirmed, remainingDetections) = IoUMatching(
            unconfirmedTracks, unmatchedDetections, filteredDetections, appearanceFeatures);

        // Update matched tracks
        foreach (var (trackIdx, detIdx) in matchedConfirmed)
        {
            var feat = appearanceFeatures.Count > detIdx ? appearanceFeatures[detIdx] : null;
            confirmedTracks[trackIdx].Update(filteredDetections[detIdx], feat);
        }

        foreach (var (trackIdx, detIdx) in matchedUnconfirmed)
        {
            var feat = appearanceFeatures.Count > detIdx ? appearanceFeatures[detIdx] : null;
            unconfirmedTracks[trackIdx].Update(filteredDetections[detIdx], feat);
        }

        // Mark unmatched tracks
        var matchedConfirmedSet = matchedConfirmed.Select(m => m.trackIdx).ToHashSet();
        var matchedUnconfirmedSet = matchedUnconfirmed.Select(m => m.trackIdx).ToHashSet();

        for (int i = 0; i < confirmedTracks.Count; i++)
        {
            if (!matchedConfirmedSet.Contains(i))
            {
                confirmedTracks[i].MarkMissed();
            }
        }

        for (int i = 0; i < unconfirmedTracks.Count; i++)
        {
            if (!matchedUnconfirmedSet.Contains(i))
            {
                unconfirmedTracks[i].MarkMissed();
            }
        }

        // Create new tracks for remaining detections
        foreach (int detIdx in remainingDetections)
        {
            var feat = appearanceFeatures.Count > detIdx ? appearanceFeatures[detIdx] : null;
            var newTrack = new DeepTrack<T>(
                NextTrackId++,
                filteredDetections[detIdx],
                feat,
                _maxGallerySize,
                NumOps);
            _deepTracks.Add(newTrack);
        }

        // Remove deleted tracks
        _deepTracks.RemoveAll(t => t.State == TrackState.Deleted ||
                                   t.TimeSinceUpdate > Options.MaxAge);

        // Update base class tracks
        Tracks.Clear();
        foreach (var dt in _deepTracks)
        {
            if (dt.State == TrackState.Confirmed)
            {
                Tracks.Add(dt.ToTrack());
            }
        }

        return new TrackingResult<T>
        {
            Tracks = GetConfirmedTracks(),
            FrameNumber = FrameCount,
            TrackingTime = DateTime.UtcNow - startTime
        };
    }

    private List<Tensor<T>> ExtractAppearanceFeatures(Tensor<T> image, List<Detection<T>> detections)
    {
        var features = new List<Tensor<T>>();

        foreach (var det in detections)
        {
            var crop = CropImage(image, det.Box);
            var feat = _reidNetwork!.Extract(crop);
            features.Add(feat);
        }

        return features;
    }

    private Tensor<T> CropImage(Tensor<T> image, BoundingBox<T> box)
    {
        int channels = image.Shape[1];
        int imgH = image.Shape[2];
        int imgW = image.Shape[3];

        int x1 = Math.Max(0, (int)NumOps.ToDouble(box.X1));
        int y1 = Math.Max(0, (int)NumOps.ToDouble(box.Y1));
        int x2 = Math.Min(imgW, (int)NumOps.ToDouble(box.X2));
        int y2 = Math.Min(imgH, (int)NumOps.ToDouble(box.Y2));

        int cropW = Math.Max(1, x2 - x1);
        int cropH = Math.Max(1, y2 - y1);

        // Resize to 128x64 (standard Re-ID size)
        int targetH = 128;
        int targetW = 64;

        var crop = new Tensor<T>(new[] { 1, channels, targetH, targetW });

        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < targetH; h++)
            {
                for (int w = 0; w < targetW; w++)
                {
                    int srcH = y1 + (int)(h * (double)cropH / targetH);
                    int srcW = x1 + (int)(w * (double)cropW / targetW);

                    srcH = MathHelper.Clamp(srcH, 0, imgH - 1);
                    srcW = MathHelper.Clamp(srcW, 0, imgW - 1);

                    crop[0, c, h, w] = image[0, c, srcH, srcW];
                }
            }
        }

        return crop;
    }

    private (List<(int trackIdx, int detIdx)> matched, List<int> unmatched) CascadeMatching(
        List<DeepTrack<T>> tracks, List<Detection<T>> detections, List<Tensor<T>> features)
    {
        var matched = new List<(int trackIdx, int detIdx)>();
        var unmatchedDets = Enumerable.Range(0, detections.Count).ToList();

        // Cascade by time since update (prioritize recently seen tracks)
        for (int cascade = 0; cascade <= Options.MaxAge; cascade++)
        {
            var tracksAtCascade = new List<int>();
            for (int t = 0; t < tracks.Count; t++)
            {
                if (tracks[t].TimeSinceUpdate == cascade)
                {
                    tracksAtCascade.Add(t);
                }
            }

            if (tracksAtCascade.Count == 0 || unmatchedDets.Count == 0)
                continue;

            // Build cost matrix for this cascade level
            var costMatrix = BuildCombinedCostMatrix(tracks, detections, features, tracksAtCascade, unmatchedDets);

            // Solve assignment
            var assignment = HungarianAssignment(costMatrix);

            // Process assignments
            var newlyMatchedDets = new List<int>();
            for (int i = 0; i < assignment.Length; i++)
            {
                if (assignment[i] >= 0)
                {
                    int trackIdx = tracksAtCascade[i];
                    int detLocalIdx = assignment[i];
                    int detIdx = unmatchedDets[detLocalIdx];

                    if (costMatrix[i, detLocalIdx] < 0.7) // Threshold
                    {
                        matched.Add((trackIdx, detIdx));
                        newlyMatchedDets.Add(detIdx);
                    }
                }
            }

            unmatchedDets = unmatchedDets.Where(d => !newlyMatchedDets.Contains(d)).ToList();
        }

        return (matched, unmatchedDets);
    }

    private (List<(int trackIdx, int detIdx)> matched, List<int> unmatched) IoUMatching(
        List<DeepTrack<T>> tracks, List<int> detectionIndices,
        List<Detection<T>> allDetections, List<Tensor<T>> features)
    {
        var matched = new List<(int trackIdx, int detIdx)>();

        if (tracks.Count == 0 || detectionIndices.Count == 0)
        {
            return (matched, detectionIndices);
        }

        // Build IoU cost matrix
        var costMatrix = new double[tracks.Count, detectionIndices.Count];

        for (int t = 0; t < tracks.Count; t++)
        {
            var predictedBox = tracks[t].GetPredictedBox();

            for (int d = 0; d < detectionIndices.Count; d++)
            {
                int detIdx = detectionIndices[d];
                double iou = ComputeIoU(predictedBox, allDetections[detIdx].Box);
                costMatrix[t, d] = 1.0 - iou;
            }
        }

        var assignment = HungarianAssignment(costMatrix);

        var matchedDets = new HashSet<int>();
        for (int t = 0; t < assignment.Length; t++)
        {
            if (assignment[t] >= 0)
            {
                int localDetIdx = assignment[t];
                int detIdx = detectionIndices[localDetIdx];

                if (costMatrix[t, localDetIdx] < (1.0 - NumOps.ToDouble(Options.IouThreshold)))
                {
                    matched.Add((t, detIdx));
                    matchedDets.Add(detIdx);
                }
            }
        }

        var unmatched = detectionIndices.Where(d => !matchedDets.Contains(d)).ToList();
        return (matched, unmatched);
    }

    private double[,] BuildCombinedCostMatrix(
        List<DeepTrack<T>> tracks, List<Detection<T>> detections,
        List<Tensor<T>> features, List<int> trackIndices, List<int> detIndices)
    {
        var costMatrix = new double[trackIndices.Count, detIndices.Count];

        for (int i = 0; i < trackIndices.Count; i++)
        {
            var track = tracks[trackIndices[i]];
            var predictedBox = track.GetPredictedBox();

            for (int j = 0; j < detIndices.Count; j++)
            {
                int detIdx = detIndices[j];
                var detection = detections[detIdx];

                // IoU cost
                double iouCost = 1.0 - ComputeIoU(predictedBox, detection.Box);

                // Appearance cost
                double appearanceCost = 1.0;
                if (Options.UseAppearance && features.Count > detIdx && track.HasAppearanceFeatures())
                {
                    double similarity = track.GetMinAppearanceDistance(features[detIdx]);
                    appearanceCost = 1.0 - similarity;
                }

                // Combined cost
                if (Options.UseAppearance)
                {
                    costMatrix[i, j] = Options.AppearanceWeight * appearanceCost +
                                       (1 - Options.AppearanceWeight) * iouCost;
                }
                else
                {
                    costMatrix[i, j] = iouCost;
                }

                // Gate by Mahalanobis distance (simplified: use IoU threshold)
                if (iouCost > 0.9) // Very low IoU
                {
                    costMatrix[i, j] = 1e5; // Infinite cost
                }
            }
        }

        return costMatrix;
    }

    /// <inheritdoc/>
    public override void Reset()
    {
        base.Reset();
        _deepTracks.Clear();
    }
}

/// <summary>
/// Track with appearance feature gallery for DeepSORT.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class DeepTrack<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _trackId;
    private int _classId;
    private T _confidence;
    private readonly int _maxGallerySize;

    // Kalman state
    private double[] _state;
    private double[] _velocity;

    // Appearance gallery
    private readonly List<Tensor<T>> _gallery;

    public int TimeSinceUpdate { get; private set; }
    public int Hits { get; private set; }
    public int Age { get; private set; }
    public TrackState State { get; private set; }

    public DeepTrack(int trackId, Detection<T> detection, Tensor<T>? feature,
        int maxGallerySize, INumericOperations<T> numOps)
    {
        _trackId = trackId;
        _numOps = numOps;
        _classId = detection.ClassId;
        _confidence = detection.Confidence;
        _maxGallerySize = maxGallerySize;

        // Initialize state
        var (cx, cy, s, r) = BoxToState(detection.Box);
        _state = new double[] { cx, cy, s, r };
        _velocity = new double[] { 0, 0, 0, 0 };

        // Initialize gallery
        _gallery = new List<Tensor<T>>();
        if (feature != null)
        {
            _gallery.Add(feature);
        }

        TimeSinceUpdate = 0;
        Hits = 1;
        Age = 1;
        State = TrackState.Tentative;
    }

    public void Predict()
    {
        // Simple motion model
        _state[0] += _velocity[0];
        _state[1] += _velocity[1];
        _state[2] += _velocity[2];

        if (_state[2] <= 0)
            _state[2] = 1;

        Age++;
        TimeSinceUpdate++;
    }

    public void Update(Detection<T> detection, Tensor<T>? feature)
    {
        var (cx, cy, s, r) = BoxToState(detection.Box);

        // Update velocity
        _velocity[0] = 0.5 * _velocity[0] + 0.5 * (cx - _state[0]);
        _velocity[1] = 0.5 * _velocity[1] + 0.5 * (cy - _state[1]);
        _velocity[2] = 0.5 * _velocity[2] + 0.5 * (s - _state[2]);

        // Update state
        _state = new double[] { cx, cy, s, r };

        _classId = detection.ClassId;
        _confidence = detection.Confidence;
        TimeSinceUpdate = 0;
        Hits++;

        // Update gallery
        if (feature != null)
        {
            _gallery.Add(feature);
            if (_gallery.Count > _maxGallerySize)
            {
                _gallery.RemoveAt(0);
            }
        }

        // Confirm track
        if (State == TrackState.Tentative && Hits >= 3)
        {
            State = TrackState.Confirmed;
        }
    }

    public void MarkMissed()
    {
        if (State == TrackState.Tentative)
        {
            State = TrackState.Deleted;
        }
    }

    public BoundingBox<T> GetPredictedBox()
    {
        return StateToBox(_state[0], _state[1], _state[2], _state[3]);
    }

    public bool HasAppearanceFeatures()
    {
        return _gallery.Count > 0;
    }

    public double GetMinAppearanceDistance(Tensor<T> feature)
    {
        if (_gallery.Count == 0)
            return 0;

        double maxSimilarity = 0;

        foreach (var galleryFeat in _gallery)
        {
            double similarity = ComputeCosineSimilarity(galleryFeat, feature);
            maxSimilarity = Math.Max(maxSimilarity, similarity);
        }

        return maxSimilarity;
    }

    private double ComputeCosineSimilarity(Tensor<T> a, Tensor<T> b)
    {
        if (a.Length != b.Length)
            return 0;

        double dot = 0, normA = 0, normB = 0;

        for (int i = 0; i < a.Length; i++)
        {
            double va = _numOps.ToDouble(a[i]);
            double vb = _numOps.ToDouble(b[i]);
            dot += va * vb;
            normA += va * va;
            normB += vb * vb;
        }

        if (normA <= 0 || normB <= 0)
            return 0;

        return dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }

    public Track<T> ToTrack()
    {
        var box = GetPredictedBox();
        var track = new Track<T>(_trackId, box, _classId, _confidence)
        {
            VelocityX = _velocity[0],
            VelocityY = _velocity[1],
            TimeSinceUpdate = TimeSinceUpdate,
            Age = Age,
            Hits = Hits,
            State = State
        };

        if (_gallery.Count > 0)
        {
            track.AppearanceFeature = _gallery[^1];
        }

        return track;
    }

    private (double cx, double cy, double s, double r) BoxToState(BoundingBox<T> box)
    {
        double x1 = _numOps.ToDouble(box.X1);
        double y1 = _numOps.ToDouble(box.Y1);
        double x2 = _numOps.ToDouble(box.X2);
        double y2 = _numOps.ToDouble(box.Y2);

        double w = x2 - x1;
        double h = y2 - y1;
        double cx = x1 + w / 2;
        double cy = y1 + h / 2;
        double s = w * h;
        double r = h > 0 ? w / h : 1;

        return (cx, cy, s, r);
    }

    private BoundingBox<T> StateToBox(double cx, double cy, double s, double r)
    {
        double w = Math.Sqrt(Math.Max(1, s * r));
        double h = r > 0 ? w / r : w;

        double x1 = cx - w / 2;
        double y1 = cy - h / 2;
        double x2 = cx + w / 2;
        double y2 = cy + h / 2;

        return new BoundingBox<T>(
            _numOps.FromDouble(x1), _numOps.FromDouble(y1),
            _numOps.FromDouble(x2), _numOps.FromDouble(y2),
            BoundingBoxFormat.XYXY);
    }
}

/// <summary>
/// Simple Re-ID network for appearance feature extraction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class ReIDNetwork<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _featureDim;

    public ReIDNetwork(int featureDim = 128)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _featureDim = featureDim;
    }

    public Tensor<T> Extract(Tensor<T> crop)
    {
        // Simplified: average pooling + projection
        // In practice, this would be a CNN like ResNet-18

        int channels = crop.Shape[1];
        int height = crop.Shape[2];
        int width = crop.Shape[3];

        // Global average pooling
        var pooled = new double[channels];
        for (int c = 0; c < channels; c++)
        {
            double sum = 0;
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    sum += _numOps.ToDouble(crop[0, c, h, w]);
                }
            }
            pooled[c] = sum / (height * width);
        }

        // Simple projection to feature dimension
        var feature = new Tensor<T>(new[] { _featureDim });

        for (int f = 0; f < _featureDim; f++)
        {
            double val = 0;
            for (int c = 0; c < channels; c++)
            {
                // Simple mixing (in practice, learned projection)
                val += pooled[c] * Math.Sin((f + 1) * c * 0.1);
            }
            feature[f] = _numOps.FromDouble(Math.Tanh(val));
        }

        // L2 normalize
        double norm = 0;
        for (int i = 0; i < _featureDim; i++)
        {
            double v = _numOps.ToDouble(feature[i]);
            norm += v * v;
        }
        norm = Math.Sqrt(norm);

        if (norm > 0)
        {
            for (int i = 0; i < _featureDim; i++)
            {
                feature[i] = _numOps.FromDouble(_numOps.ToDouble(feature[i]) / norm);
            }
        }

        return feature;
    }
}
