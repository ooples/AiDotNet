using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
/// <para>This follows the official deep_sort tracker: an xyah Kalman filter, a matching cascade by
/// time since update with the nearest-neighbour cosine metric (budget 100, threshold
/// <see cref="TrackingOptions{T}.MaxCosineDistance"/>) gated at the chi-square 0.95 Mahalanobis
/// quantile, then IoU matching (max IoU distance 0.7) for unconfirmed tracks and tracks missed for one
/// frame, all with optimal assignment. Without appearance features the cascade uses the gated IoU
/// distance instead of the cosine metric.</para>
///
/// <para>Reference: Wojke et al., "Simple Online and Realtime Tracking with a Deep
/// Association Metric", ICIP 2017</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Detection)]
[ModelTask(ModelTask.Tracking)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Simple Online and Realtime Tracking with a Deep Association Metric",
    "https://arxiv.org/abs/1703.07402",
    Year = 2017,
    Authors = "Nicolai Wojke, Alex Bewley, Dietrich Paulus")]
public class DeepSORT<T> : ObjectTrackerBase<T>
{
    /// <summary>Cost given to gated (inadmissible) pairs, deep_sort's <c>INFTY_COST</c>.</summary>
    private const double InfinityCost = 1e5;

    /// <summary>Largest IoU distance accepted in the IoU round (<c>max_iou_distance</c>).</summary>
    private const double MaxIouDistance = 0.7;

    private readonly List<DeepTrack<T>> _deepTracks;
    private readonly ReIDNetwork<T>? _reidNetwork;
    private readonly Func<Tensor<T>, Tensor<T>>? _appearanceEncoder;
    private readonly XyahKalmanFilter<T> _kalman;
    private readonly Dictionary<int, List<Tensor<T>>> _samples;
    private readonly int _maxGallerySize;

    /// <inheritdoc/>
    public override string Name => "DeepSORT";

    /// <summary>
    /// Creates a new DeepSORT tracker. With <see cref="TrackingOptions{T}.UseAppearance"/> set, crops are
    /// described by the built-in fixed (untrained) descriptor; use the overload that takes an encoder to
    /// supply a trained re-identification network as in the paper.
    /// </summary>
    public DeepSORT(TrackingOptions<T> options) : this(options, null)
    {
    }

    /// <summary>
    /// Creates a new DeepSORT tracker whose appearance descriptors come from
    /// <paramref name="appearanceEncoder"/>, which maps a <c>[1, C, 128, 64]</c> crop to a feature
    /// vector (the paper uses a wide residual network trained for person re-identification).
    /// </summary>
    public DeepSORT(TrackingOptions<T> options, Func<Tensor<T>, Tensor<T>>? appearanceEncoder) : base(options)
    {
        _deepTracks = new List<DeepTrack<T>>();
        _kalman = new XyahKalmanFilter<T>();
        _samples = new Dictionary<int, List<Tensor<T>>>();
        _maxGallerySize = 100;
        _appearanceEncoder = appearanceEncoder;

        if (appearanceEncoder is null && options.UseAppearance)
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
        var engine = AiDotNetEngine.Current;

        // Filter detections by confidence
        var scores = TrackingAssociation<T>.Scores(detections);
        var keep = TrackingAssociation<T>.Where(
            engine.TensorGreaterEqual(scores, engine.TensorFullLike(scores, Options.ConfidenceThreshold)));
        var filteredDetections = keep.Select(i => detections[i]).ToList();

        // Extract appearance features if an encoder is available and an image is provided
        List<Tensor<T>> appearanceFeatures = new();
        if (image is not null && (_appearanceEncoder is not null || _reidNetwork is not null))
        {
            appearanceFeatures = ExtractAppearanceFeatures(image, filteredDetections);
        }

        foreach (var track in _deepTracks)
        {
            track.Predict();
        }

        var (matches, unmatchedTracks, unmatchedDetections) = Match(filteredDetections, appearanceFeatures);

        foreach (var (trackIdx, detIdx) in matches)
        {
            var feat = appearanceFeatures.Count > detIdx ? appearanceFeatures[detIdx] : null;
            _deepTracks[trackIdx].Update(filteredDetections[detIdx], feat);
        }

        foreach (int trackIdx in unmatchedTracks)
        {
            _deepTracks[trackIdx].MarkMissed();
        }

        foreach (int detIdx in unmatchedDetections)
        {
            var feat = appearanceFeatures.Count > detIdx ? appearanceFeatures[detIdx] : null;
            _deepTracks.Add(new DeepTrack<T>(NextTrackId++, filteredDetections[detIdx], feat, _kalman,
                Options.MinHits, Options.MaxAge));
        }

        _deepTracks.RemoveAll(t => t.State == TrackState.Deleted);
        UpdateGallery();

        // Report confirmed tracks updated in this or the previous frame (deep_sort_app's output rule).
        Tracks.Clear();
        foreach (var dt in _deepTracks)
        {
            if (dt.State == TrackState.Confirmed && dt.TimeSinceUpdate <= 1)
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

    /// <summary>
    /// <c>Tracker._match</c>: a matching cascade over confirmed tracks by time since update with the
    /// gated appearance metric, then an IoU round for unconfirmed tracks and confirmed tracks missed
    /// for exactly one frame.
    /// </summary>
    private (List<(int Track, int Det)> Matches, List<int> UnmatchedTracks, List<int> UnmatchedDetections)
        Match(List<Detection<T>> detections, List<Tensor<T>> features)
    {
        var confirmed = Enumerable.Range(0, _deepTracks.Count).Where(i => _deepTracks[i].State == TrackState.Confirmed).ToList();
        var unconfirmed = Enumerable.Range(0, _deepTracks.Count).Where(i => _deepTracks[i].State != TrackState.Confirmed).ToList();
        bool useAppearance = features.Count == detections.Count && detections.Count > 0;

        // Matching cascade (deep_sort linear_assignment.matching_cascade).
        var matchesA = new List<(int Track, int Det)>();
        var unmatchedDetections = Enumerable.Range(0, detections.Count).ToList();
        for (int level = 0; level < Options.MaxAge; level++)
        {
            if (unmatchedDetections.Count == 0)
            {
                break;
            }

            var tracksAtLevel = confirmed.Where(k => _deepTracks[k].TimeSinceUpdate == 1 + level).ToList();
            if (tracksAtLevel.Count == 0)
            {
                continue;
            }

            var (cost, maxDistance) = GatedMetric(tracksAtLevel, unmatchedDetections, detections, features, useAppearance);
            var (pairs, _, unmatchedCols) = TrackingAssociation<T>.MinCostMatching(cost, maxDistance);
            matchesA.AddRange(pairs.Select(p => (tracksAtLevel[p.Row], unmatchedDetections[p.Col])));
            unmatchedDetections = unmatchedCols.Select(c => unmatchedDetections[c]).ToList();
        }

        var matchedA = new HashSet<int>(matchesA.Select(m => m.Track));
        var unmatchedTracksA = confirmed.Where(k => !matchedA.Contains(k)).ToList();

        // IoU round.
        var iouCandidates = unconfirmed.Concat(unmatchedTracksA.Where(k => _deepTracks[k].TimeSinceUpdate == 1)).ToList();
        unmatchedTracksA = unmatchedTracksA.Where(k => _deepTracks[k].TimeSinceUpdate != 1).ToList();

        var matchesB = new List<(int Track, int Det)>();
        var unmatchedTracksB = iouCandidates;
        if (iouCandidates.Count > 0 && unmatchedDetections.Count > 0)
        {
            var (pairs, unmatchedRows, unmatchedCols) = TrackingAssociation<T>.MinCostMatching(
                IouCost(iouCandidates, unmatchedDetections, detections), NumOps.FromDouble(MaxIouDistance));
            matchesB = pairs.Select(p => (iouCandidates[p.Row], unmatchedDetections[p.Col])).ToList();
            unmatchedTracksB = unmatchedRows.Select(r => iouCandidates[r]).ToList();
            unmatchedDetections = unmatchedCols.Select(c => unmatchedDetections[c]).ToList();
        }

        var matches = matchesA.Concat(matchesB).ToList();
        var unmatchedTracks = unmatchedTracksA.Concat(unmatchedTracksB).Distinct().ToList();
        return (matches, unmatchedTracks, unmatchedDetections);
    }

    /// <summary>
    /// Cost of the cascade: with appearance features, <c>w * cosine + (1 - w) * Mahalanobis</c>
    /// (Eq. 5 with w = 1 - lambda; the official code uses w = 1); without them, the IoU distance.
    /// Pairs outside the chi-square 0.95 Mahalanobis gate, or beyond the cosine threshold, get
    /// <c>INFTY_COST</c>.
    /// </summary>
    private (Tensor<T> Cost, T MaxDistance) GatedMetric(List<int> trackIndices, List<int> detIndices,
        List<Detection<T>> detections, List<Tensor<T>> features, bool useAppearance)
    {
        var engine = AiDotNetEngine.Current;
        var detBoxes = TrackingAssociation<T>.BoxesToTensor(detIndices.Select(d => detections[d].Box).ToList());
        var measurements = TrackingAssociation<T>.BoxesToXyah(detBoxes);
        var maha = engine.TensorStack(
            trackIndices.Select(t => _deepTracks[t].GatingDistance(measurements)).ToArray(), 0);
        T chi2 = NumOps.FromDouble(XyahKalmanFilter<T>.GatingThreshold4Dof);
        var gate = engine.TensorGreaterThan(maha, chi2);

        Tensor<T> cost;
        T maxDistance;
        if (useAppearance)
        {
            var queries = engine.TensorStack(detIndices.Select(d => Flatten(features[d])).ToArray(), 0);
            var cosine = engine.TensorStack(
                trackIndices.Select(t => NearestNeighborCosineDistance(_deepTracks[t].TrackId, queries)).ToArray(), 0);
            T weight = Options.AppearanceWeight;
            T motionWeight = NumOps.Subtract(NumOps.One, weight);
            cost = engine.TensorAdd(engine.TensorMultiplyScalar(cosine, weight), engine.TensorMultiplyScalar(maha, motionWeight));
            gate = engine.TensorMax(gate, engine.TensorGreaterThan(cosine, Options.MaxCosineDistance));
            maxDistance = NumOps.Add(NumOps.Multiply(weight, Options.MaxCosineDistance), NumOps.Multiply(motionWeight, chi2));
        }
        else
        {
            var trackBoxes = TrackingAssociation<T>.BoxesToTensor(trackIndices.Select(t => _deepTracks[t].GetPredictedBox()).ToList());
            cost = TrackingAssociation<T>.IouDistance(trackBoxes, detBoxes);
            maxDistance = NumOps.FromDouble(MaxIouDistance);
        }

        return (engine.TensorWhere(gate, engine.TensorFullLike(cost, NumOps.FromDouble(InfinityCost)), cost), maxDistance);
    }

    /// <summary>deep_sort <c>iou_cost</c>: 1 - IoU, with <c>INFTY_COST</c> rows for tracks missed for more than one frame.</summary>
    private Tensor<T> IouCost(List<int> trackIndices, List<int> detIndices, List<Detection<T>> detections)
    {
        var engine = AiDotNetEngine.Current;
        var trackBoxes = TrackingAssociation<T>.BoxesToTensor(trackIndices.Select(t => _deepTracks[t].GetPredictedBox()).ToList());
        var detBoxes = TrackingAssociation<T>.BoxesToTensor(detIndices.Select(d => detections[d].Box).ToList());
        var cost = TrackingAssociation<T>.IouDistance(trackBoxes, detBoxes);

        T one = NumOps.One;
        T zero = NumOps.Zero;
        var stale = new Tensor<T>(trackIndices.Select(t => _deepTracks[t].TimeSinceUpdate > 1 ? one : zero).ToArray(),
            new[] { trackIndices.Count, 1 });
        return engine.TensorWhere(engine.TensorBroadcastTo(stale, cost.Shape.ToArray()),
            engine.TensorFullLike(cost, NumOps.FromDouble(InfinityCost)), cost);
    }

    /// <summary>
    /// <c>NearestNeighborDistanceMetric</c> ("cosine"): the smallest cosine distance between each query
    /// row and the target's stored samples; 1 for every query when the target has no samples yet.
    /// </summary>
    private Tensor<T> NearestNeighborCosineDistance(int trackId, Tensor<T> queries)
    {
        var engine = AiDotNetEngine.Current;
        if (!_samples.TryGetValue(trackId, out var gallery) || gallery.Count == 0)
        {
            var unseen = new Tensor<T>(new[] { queries.Shape[0] });
            engine.TensorFill(unseen, NumOps.One);
            return unseen;
        }

        var samples = NormalizeRows(engine.TensorStack(gallery.ToArray(), 0));
        var normalizedQueries = NormalizeRows(queries);
        var distances = engine.ScalarMinusTensor(NumOps.One,
            engine.TensorMatMul(samples, engine.TensorTranspose(normalizedQueries)));
        return engine.TensorAmin(distances, new[] { 0 }, keepDims: false);
    }

    private static Tensor<T> NormalizeRows(Tensor<T> rows)
    {
        var engine = AiDotNetEngine.Current;
        var norms = engine.TensorNorm(rows, 1, keepDims: true);
        return engine.TensorDivide(rows, engine.TensorBroadcastTo(norms, rows.Shape.ToArray()));
    }

    private static Tensor<T> Flatten(Tensor<T> feature)
    {
        return AiDotNetEngine.Current.Reshape(feature, new[] { feature.Length });
    }

    /// <summary>
    /// <c>Tracker.update</c>'s metric step: confirmed tracks hand their new features to their gallery
    /// (capped at the newest <c>nn_budget</c> = 100), and galleries of inactive targets are dropped.
    /// </summary>
    private void UpdateGallery()
    {
        var active = new HashSet<int>();
        foreach (var track in _deepTracks)
        {
            if (track.State != TrackState.Confirmed)
            {
                continue;
            }

            active.Add(track.TrackId);
            if (!_samples.TryGetValue(track.TrackId, out var gallery))
            {
                gallery = new List<Tensor<T>>();
                _samples[track.TrackId] = gallery;
            }

            gallery.AddRange(track.TakeFeatures().Select(Flatten));
            if (gallery.Count > _maxGallerySize)
            {
                gallery.RemoveRange(0, gallery.Count - _maxGallerySize);
            }
        }

        foreach (int id in _samples.Keys.Where(k => !active.Contains(k)).ToList())
        {
            _samples.Remove(id);
        }
    }

    private List<Tensor<T>> ExtractAppearanceFeatures(Tensor<T> image, List<Detection<T>> detections)
    {
        var features = new List<Tensor<T>>();
        foreach (var det in detections)
        {
            var crop = CropImage(image, det.Box);
            if (_appearanceEncoder is not null)
            {
                features.Add(_appearanceEncoder(crop));
            }
            else if (_reidNetwork is not null)
            {
                features.Add(_reidNetwork.Extract(crop));
            }
        }

        return features;
    }

    /// <summary>Nearest-neighbour crop of the box from image 0, resized to the 128 x 64 Re-ID input.</summary>
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

        const int targetH = 128;
        const int targetW = 64;

        var rows = Enumerable.Range(0, targetH)
            .Select(h => MathHelper.Clamp(y1 + (int)(h * (double)cropH / targetH), 0, imgH - 1)).ToArray();
        var cols = Enumerable.Range(0, targetW)
            .Select(w => MathHelper.Clamp(x1 + (int)(w * (double)cropW / targetW), 0, imgW - 1)).ToArray();

        var engine = AiDotNetEngine.Current;
        var first = engine.TensorSlice(image, new[] { 0, 0, 0, 0 }, new[] { 1, channels, imgH, imgW });
        var picked = engine.TensorIndexSelect(first, new Tensor<int>(rows, new[] { targetH }), 2);
        return engine.TensorIndexSelect(picked, new Tensor<int>(cols, new[] { targetW }), 3);
    }

    /// <inheritdoc/>
    public override void Reset()
    {
        base.Reset();
        _deepTracks.Clear();
        _samples.Clear();
    }
}

/// <summary>
/// deep_sort <c>Track</c>: an xyah Kalman state plus the appearance features observed since the last
/// gallery update.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal sealed class DeepTrack<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly XyahKalmanFilter<T> _kalman;
    private readonly int _nInit;
    private readonly int _maxAge;
    private readonly List<Tensor<T>> _features;
    private Tensor<T> _mean;
    private Tensor<T> _covariance;
    private int _classId;
    private T _confidence;
    private Tensor<T>? _lastFeature;

    public DeepTrack(int trackId, Detection<T> detection, Tensor<T>? feature, XyahKalmanFilter<T> kalman,
        int nInit, int maxAge)
    {
        TrackId = trackId;
        _kalman = kalman;
        _nInit = nInit;
        _maxAge = maxAge;
        _classId = detection.ClassId;
        _confidence = detection.Confidence;
        (_mean, _covariance) = kalman.Initiate(ToXyah(detection.Box));
        _features = new List<Tensor<T>>();
        if (feature is not null)
        {
            _features.Add(feature);
            _lastFeature = feature;
        }

        TimeSinceUpdate = 0;
        Hits = 1;
        Age = 1;
        State = TrackState.Tentative;
    }

    public int TrackId { get; }
    public int TimeSinceUpdate { get; private set; }
    public int Hits { get; private set; }
    public int Age { get; private set; }
    public TrackState State { get; private set; }

    public void Predict()
    {
        (_mean, _covariance) = _kalman.Predict(_mean, _covariance);
        Age++;
        TimeSinceUpdate++;
    }

    public void Update(Detection<T> detection, Tensor<T>? feature)
    {
        (_mean, _covariance) = _kalman.Update(_mean, _covariance, ToXyah(detection.Box));
        if (feature is not null)
        {
            _features.Add(feature);
            _lastFeature = feature;
        }

        _classId = detection.ClassId;
        _confidence = detection.Confidence;
        Hits++;
        TimeSinceUpdate = 0;
        if (State == TrackState.Tentative && Hits >= _nInit)
        {
            State = TrackState.Confirmed;
        }
    }

    public void MarkMissed()
    {
        if (State == TrackState.Tentative || TimeSinceUpdate > _maxAge)
        {
            State = TrackState.Deleted;
        }
    }

    /// <summary>Squared Mahalanobis distance to each xyah measurement row.</summary>
    public Tensor<T> GatingDistance(Tensor<T> measurements)
    {
        return _kalman.GatingDistance(_mean, _covariance, measurements);
    }

    /// <summary>Returns and clears the features gathered since the last gallery update.</summary>
    public List<Tensor<T>> TakeFeatures()
    {
        var taken = new List<Tensor<T>>(_features);
        _features.Clear();
        return taken;
    }

    public BoundingBox<T> GetPredictedBox()
    {
        T half = NumOps.FromDouble(0.5);
        T h = _mean[3, 0];
        T w = NumOps.Multiply(_mean[2, 0], h);
        T x1 = NumOps.Subtract(_mean[0, 0], NumOps.Multiply(w, half));
        T y1 = NumOps.Subtract(_mean[1, 0], NumOps.Multiply(h, half));
        return new BoundingBox<T>(x1, y1, NumOps.Add(x1, w), NumOps.Add(y1, h), BoundingBoxFormat.XYXY);
    }

    public Track<T> ToTrack()
    {
        return new Track<T>(TrackId, GetPredictedBox(), _classId, _confidence)
        {
            VelocityX = _mean[4, 0],
            VelocityY = _mean[5, 0],
            TimeSinceUpdate = TimeSinceUpdate,
            Age = Age,
            Hits = Hits,
            State = State,
            AppearanceFeature = _lastFeature
        };
    }

    private static Tensor<T> ToXyah(BoundingBox<T> box)
    {
        T half = NumOps.FromDouble(0.5);
        T w = NumOps.Subtract(box.X2, box.X1);
        T h = NumOps.Subtract(box.Y2, box.Y1);
        return new Tensor<T>(new[]
        {
            NumOps.Add(box.X1, NumOps.Multiply(w, half)),
            NumOps.Add(box.Y1, NumOps.Multiply(h, half)),
            NumOps.Divide(w, h),
            h
        }, new[] { 4 });
    }
}

/// <summary>
/// A fixed, untrained appearance descriptor: channel means projected onto sinusoids, squashed with
/// tanh and L2-normalised. It stands in when no trained re-identification encoder is supplied.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class ReIDNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _featureDim;

    public ReIDNetwork(int featureDim = 128)
    {
        _featureDim = featureDim;
    }

    public Tensor<T> Extract(Tensor<T> crop)
    {
        var engine = AiDotNetEngine.Current;
        int channels = crop.Shape[1];

        // Global average pooling of image 0: [C, 1]
        var first = engine.TensorSlice(crop, new[] { 0, 0, 0, 0 }, new[] { 1, channels, crop.Shape[2], crop.Shape[3] });
        var pooled = engine.Reshape(engine.ReduceMean(first, new[] { 2, 3 }, keepDims: false), new[] { channels, 1 });

        // projection[f, c] = sin((f + 1) * c * 0.1)
        var rowsIndex = engine.TensorArange<T>(1, _featureDim + 1);
        var colsIndex = engine.TensorArange<T>(0, channels);
        var projection = engine.TensorSin(engine.TensorMultiplyScalar(
            engine.TensorOuter(rowsIndex, colsIndex), NumOps.FromDouble(0.1)));

        var feature = engine.Reshape(engine.TensorTanh(engine.TensorMatMul(projection, pooled)), new[] { _featureDim });

        T norm = NumOps.Sqrt(engine.TensorSumOfSquares(feature));
        return NumOps.GreaterThan(norm, NumOps.Zero) ? engine.TensorDivideScalar(feature, norm) : feature;
    }
}
