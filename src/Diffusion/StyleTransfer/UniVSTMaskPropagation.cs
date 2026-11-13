using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Diffusion.StyleTransfer;

/// <summary>
/// UniVST's point-matching mask propagation (Song et al., arXiv:2410.20084, TPAMI 2025).
/// Propagates a first-frame mask across a video using DDIM-inversion feature maps.
/// </summary>
/// <remarks>
/// <para>
/// This is the component that lets UniVST drop the external tracking model other localized-editing
/// pipelines depend on: the diffusion model's own inversion features already encode correspondence,
/// so a nearest-neighbour vote over them propagates the region directly.
/// </para>
/// <para>
/// Each point in frame i is matched against points in the ANCHOR frames — the first frame plus the
/// previous <c>MaskAnchorHistory</c> frames — by cosine similarity in feature space. The k best
/// matches vote, and the majority label becomes the point's label. Pinning the first frame is what
/// keeps the mask from drifting; a purely local chain accumulates error frame over frame.
/// </para>
/// <para><b>For Beginners:</b> Given a mask drawn on the first frame, this figures out where that
/// same region sits in every later frame, by finding which points look most alike.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class UniVSTMaskPropagation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _neighbors;
    private readonly int _anchorHistory;
    private readonly double _downsampleRate;
    private readonly Random _random;

    /// <summary>
    /// The ambient tensor engine. Read per use rather than captured in the constructor, so the
    /// propagator follows a later engine switch instead of pinning whichever engine happened to be
    /// current when it was built.
    /// </summary>
    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>Gets k, the number of nearest neighbours that vote on each point.</summary>
    public int Neighbors => _neighbors;

    /// <summary>Gets how many preceding frames anchor the vote, in addition to the first frame.</summary>
    public int AnchorHistory => _anchorHistory;

    /// <summary>Gets the anchor-point downsampling rate.</summary>
    public double DownsampleRate => _downsampleRate;

    /// <summary>
    /// Creates a mask propagator.
    /// </summary>
    /// <param name="neighbors">k, the neighbours that vote. Paper: 10.</param>
    /// <param name="anchorHistory">Preceding frames used as anchors. Paper: 9.</param>
    /// <param name="downsampleRate">Anchor-point keep rate in (0, 1]. Paper: 0.5.</param>
    /// <param name="seed">
    /// Optional seed for the anchor downsampling. Supplied in tests so a propagation run is
    /// reproducible; left null in production so successive runs decorrelate their sampling.
    /// </param>
    public UniVSTMaskPropagation(
        int neighbors = 10, int anchorHistory = 9, double downsampleRate = 0.5, int? seed = null)
    {
        if (neighbors <= 0)
            throw new ArgumentOutOfRangeException(nameof(neighbors), neighbors, "neighbors must be positive.");
        if (anchorHistory < 0)
            throw new ArgumentOutOfRangeException(nameof(anchorHistory), anchorHistory, "anchorHistory cannot be negative.");
        if (downsampleRate is <= 0.0 or > 1.0)
            throw new ArgumentOutOfRangeException(nameof(downsampleRate), downsampleRate, "downsampleRate must be in (0, 1].");

        _neighbors = neighbors;
        _anchorHistory = anchorHistory;
        _downsampleRate = downsampleRate;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Propagates <paramref name="firstFrameMask"/> across every frame.
    /// </summary>
    /// <param name="features">
    /// Per-frame inversion feature maps, each shaped [height, width, channels]. All frames must
    /// share a shape.
    /// </param>
    /// <param name="firstFrameMask">
    /// The frame-0 mask, shaped [height, width]. Values above 0.5 are foreground.
    /// </param>
    /// <returns>One mask per frame, each [height, width], with 1 for foreground and 0 for background.</returns>
    public IReadOnlyList<Tensor<T>> Propagate(IReadOnlyList<Tensor<T>> features, Tensor<T> firstFrameMask)
    {
        if (features == null) throw new ArgumentNullException(nameof(features));
        if (firstFrameMask == null) throw new ArgumentNullException(nameof(firstFrameMask));
        if (features.Count == 0)
            throw new ArgumentException("At least one frame of features is required.", nameof(features));

        var shape = features[0].Shape;
        if (shape.Length != 3)
            throw new ArgumentException(
                $"Feature maps must be [height, width, channels]; got rank {shape.Length}.", nameof(features));

        int height = shape[0], width = shape[1], channels = shape[2];
        for (int i = 1; i < features.Count; i++)
        {
            var s = features[i].Shape;
            if (s.Length != 3 || s[0] != height || s[1] != width || s[2] != channels)
                throw new ArgumentException(
                    $"Frame {i} has shape [{string.Join(",", s)}] but frame 0 has [{height},{width},{channels}]; " +
                    "propagation compares points position-for-position and cannot mix resolutions.",
                    nameof(features));
        }

        if (firstFrameMask.Shape.Length != 2 || firstFrameMask.Shape[0] != height || firstFrameMask.Shape[1] != width)
            throw new ArgumentException(
                $"firstFrameMask must be [{height},{width}] to match the feature maps.", nameof(firstFrameMask));

        int points = height * width;

        // Row-normalized [points, channels] matrices, so a cosine similarity is a plain dot product
        // and the whole point-to-anchor comparison becomes one matmul per frame.
        var normalized = new Tensor<T>[features.Count];
        for (int f = 0; f < features.Count; f++) normalized[f] = NormalizeRows(features[f], points, channels);

        var labels = new bool[features.Count][];
        var firstForeground = Engine.TensorGreaterThan(firstFrameMask, NumOps.FromDouble(0.5)).Cast<int>().ToArray();
        labels[0] = Array.ConvertAll(firstForeground, v => v != 0);

        for (int f = 1; f < features.Count; f++)
        {
            var anchors = SelectAnchorFrames(f);
            labels[f] = PropagateOneFrame(normalized, labels, anchors, f, points, channels);
        }

        var masks = new List<Tensor<T>>(features.Count);
        for (int f = 0; f < features.Count; f++)
        {
            var indicator = new Vector<int>(Array.ConvertAll(labels[f], b => b ? 1 : 0));
            masks.Add(new Tensor<int>(new[] { height, width }, indicator).Cast<T>());
        }
        return masks;
    }

    /// <summary>
    /// Returns the anchor frame indices for frame <paramref name="frame"/>: the first frame plus up
    /// to <see cref="AnchorHistory"/> immediately preceding frames, without duplicates.
    /// </summary>
    public IReadOnlyList<int> SelectAnchorFrames(int frame)
    {
        if (frame <= 0) return Array.Empty<int>();

        var set = new List<int> { 0 };
        int from = Math.Max(1, frame - _anchorHistory);
        for (int f = from; f < frame; f++) set.Add(f);
        return set;
    }

    private bool[] PropagateOneFrame(
        Tensor<T>[] normalized, bool[][] labels, IReadOnlyList<int> anchorFrames,
        int frame, int points, int channels)
    {
        // Candidate anchor points, downsampled but keeping the source foreground/background
        // proportions. Uniform sampling would under-represent whichever region is smaller and tilt
        // every majority vote toward the larger one.
        var candidates = BuildCandidates(labels, anchorFrames, points);
        int anchorCount = candidates.Count;
        if (anchorCount == 0) return new bool[points];

        // Gather the candidates into one [anchors, channels] matrix (one index select over the anchor frames'
        // rows) so every point-to-anchor cosine similarity is a SINGLE matmul: [points, channels] x
        // [channels, anchors] -> [points, anchors].
        var frameSlots = new Dictionary<int, int>();
        var anchorFrameRows = new List<Tensor<T>>();
        var rowIndex = new int[anchorCount];
        var anchorIsForeground = new int[anchorCount];
        for (int a = 0; a < anchorCount; a++)
        {
            var (anchorFrame, anchorPoint, isForeground) = candidates[a];
            if (!frameSlots.TryGetValue(anchorFrame, out int slot))
            {
                slot = anchorFrameRows.Count;
                frameSlots[anchorFrame] = slot;
                anchorFrameRows.Add(normalized[anchorFrame]);
            }
            rowIndex[a] = (slot * points) + anchorPoint;
            anchorIsForeground[a] = isForeground ? 1 : 0;
        }
        var pool = anchorFrameRows.Count == 1 ? anchorFrameRows[0] : Engine.TensorConcatenate(anchorFrameRows.ToArray(), 0);
        var anchorRows = Engine.TensorIndexSelect(pool, new Tensor<int>(new[] { anchorCount }, new Vector<int>(rowIndex)), 0);

        // Nearest neighbour = MAXIMUM cosine similarity. The paper prints "arg min CosSim(...)",
        // which reads as minimising cosine DISTANCE; minimising similarity would select the least
        // similar point and invert the whole scheme.
        var similarity = Engine.TensorMatMul(normalized[frame], Engine.TensorTranspose(anchorRows));

        // k nearest anchors per point, ties at the k-th similarity resolved toward the earlier anchor (the
        // first-come order of a running top-k): everything above the k-th value is in, and the remaining slots
        // go to the first anchors equal to it. Counts stay in int so no element type can overflow them.
        int k = Math.Min(_neighbors, anchorCount);
        var grid = new[] { points, anchorCount };
        var topValues = Engine.TensorTopK(similarity, k, 1, out _);
        var kth = Engine.TensorBroadcastTo(Engine.TensorSelect(topValues, 1, k - 1).Reshape(new[] { points, 1 }), grid);
        var above = Engine.TensorGreaterThan(similarity, kth).Cast<int>();
        var tied = Engine.TensorEquals(similarity, kth).Cast<int>();
        var foreground = Engine.TensorBroadcastTo(
            new Tensor<int>(new[] { 1, anchorCount }, new Vector<int>(anchorIsForeground)), grid);

        var openSlots = Engine.ScalarMinusTensor(k + 1, Engine.ReduceSum(above, new[] { 1 }, true));
        var tieRank = Engine.TensorCumSum(tied, 1);
        var tiedTaken = Engine.TensorMultiply(tied, Engine.TensorGreaterThan(Engine.TensorBroadcastTo(openSlots, grid), tieRank));
        var chosen = Engine.TensorAdd(above, tiedTaken);
        var foregroundVotes = Engine.ReduceSum(Engine.TensorMultiply(chosen, foreground), new[] { 1 }, false);

        // Strict majority: a tie is left as background so the region cannot creep outward on
        // ambiguous points.
        var majority = Engine.TensorGreaterThan(Engine.TensorMultiplyScalar(foregroundVotes, 2), k).ToArray();
        return Array.ConvertAll(majority, v => v != 0);
    }

    private List<(int Frame, int Point, bool Foreground)> BuildCandidates(
        bool[][] labels, IReadOnlyList<int> anchorFrames, int points)
    {
        var foreground = new List<(int, int, bool)>();
        var background = new List<(int, int, bool)>();

        foreach (int af in anchorFrames)
        {
            var lab = labels[af];
            for (int p = 0; p < points; p++)
            {
                if (lab[p]) foreground.Add((af, p, true));
                else background.Add((af, p, false));
            }
        }

        var kept = new List<(int, int, bool)>();
        KeepProportion(foreground, kept);
        KeepProportion(background, kept);

        // Every anchor point being dropped would leave nothing to vote with; fall back to the full
        // set rather than silently returning an empty mask.
        if (kept.Count == 0)
        {
            kept.AddRange(foreground);
            kept.AddRange(background);
        }
        return kept;
    }

    private void KeepProportion(List<(int, int, bool)> source, List<(int, int, bool)> destination)
    {
        if (source.Count == 0) return;

        // Proportional: the keep count is a fraction OF THIS CLASS, so foreground and background
        // survive at the same rate and their ratio is preserved.
        int keep = (int)Math.Round(source.Count * _downsampleRate);
        if (keep < 1) keep = 1;
        if (keep >= source.Count) { destination.AddRange(source); return; }

        // Partial Fisher-Yates: draw `keep` distinct entries without shuffling the whole list.
        for (int i = 0; i < keep; i++)
        {
            int j = i + _random.Next(source.Count - i);
            (source[i], source[j]) = (source[j], source[i]);
            destination.Add(source[i]);
        }
    }

    /// <summary>
    /// Row-normalizes a [height, width, channels] feature map into a [points, channels] matrix, so a
    /// cosine similarity reduces to a dot product and the comparison can be done as a matmul.
    /// </summary>
    internal static Tensor<T> NormalizeRows(Tensor<T> features, int points, int channels)
    {
        var engine = AiDotNetEngine.Current;
        var rows = features.Reshape(new[] { points, channels });
        var grid = new[] { points, channels };
        var ones = Tensor<T>.CreateDefault(new[] { points, 1 }, NumOps.One);

        // Each row is first divided by its largest magnitude, so the squares summed below are at most 1 each:
        // squaring the raw features overflows Half (65504) once a row's sum of squares passes it. x / |x| is
        // unchanged by the pre-scaling.
        var peaks = engine.ReduceMax(engine.TensorAbs(rows), new[] { 1 }, true);
        var safePeaks = engine.TensorWhere(engine.TensorGreaterThan(peaks, NumOps.Zero), peaks, ones);
        var scaled = engine.TensorDivide(rows, engine.TensorBroadcastTo(safePeaks, grid));

        var norms = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(scaled), new[] { 1 }, true));
        // A zero-length feature vector has no direction; leaving it zero makes its similarity to
        // everything 0 rather than NaN, so it simply never wins a vote (its norm is replaced by 1).
        var safeNorms = engine.TensorWhere(engine.TensorGreaterThan(norms, NumOps.Zero), norms, ones);
        return engine.TensorDivide(scaled, engine.TensorBroadcastTo(safeNorms, grid));
    }
}
