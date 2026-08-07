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
        labels[0] = new bool[points];
        for (int p = 0; p < points; p++)
            labels[0][p] = Convert.ToDouble(NumOps.ToDouble(firstFrameMask[p])) > 0.5;

        for (int f = 1; f < features.Count; f++)
        {
            var anchors = SelectAnchorFrames(f);
            labels[f] = PropagateOneFrame(normalized, labels, anchors, f, points, channels);
        }

        var masks = new List<Tensor<T>>(features.Count);
        for (int f = 0; f < features.Count; f++)
        {
            var mask = new Tensor<T>(new[] { height, width });
            for (int p = 0; p < points; p++) mask[p] = labels[f][p] ? NumOps.One : NumOps.Zero;
            masks.Add(mask);
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

        // Gather the candidates into one [channels, anchors] matrix so every point-to-anchor cosine
        // similarity is a SINGLE matmul: [points, channels] x [channels, anchors] -> [points, anchors].
        // At paper scale (4096 points, 10 anchor frames) the element-wise form is ~168M dot products
        // per frame, which is exactly the kind of managed hot loop that has to be profiled out later.
        var anchorT = new Tensor<T>(new[] { channels, anchorCount });
        var anchorIsForeground = new bool[anchorCount];
        for (int a = 0; a < anchorCount; a++)
        {
            var (anchorFrame, anchorPoint, isForeground) = candidates[a];
            anchorIsForeground[a] = isForeground;
            var source = normalized[anchorFrame];
            int offset = anchorPoint * channels;
            for (int c = 0; c < channels; c++) anchorT[(c * anchorCount) + a] = source[offset + c];
        }

        // Nearest neighbour = MAXIMUM cosine similarity. The paper prints "arg min CosSim(...)",
        // which reads as minimising cosine DISTANCE; minimising similarity would select the least
        // similar point and invert the whole scheme.
        var similarity = Engine.TensorMatMul(normalized[frame], anchorT);

        var result = new bool[points];
        var bestSim = new double[_neighbors];
        var bestFg = new bool[_neighbors];

        for (int p = 0; p < points; p++)
        {
            int filled = 0;
            for (int i = 0; i < _neighbors; i++) { bestSim[i] = double.NegativeInfinity; bestFg[i] = false; }
            int row = p * anchorCount;

            // Top-k is a selection, not a matrix product, so it stays here rather than being forced
            // into the engine.
            for (int a = 0; a < anchorCount; a++)
            {
                double sim = NumOps.ToDouble(similarity[row + a]);

                if (filled < _neighbors)
                {
                    bestSim[filled] = sim; bestFg[filled] = anchorIsForeground[a]; filled++;
                    if (filled == _neighbors) SortDescending(bestSim, bestFg, filled);
                }
                else if (sim > bestSim[_neighbors - 1])
                {
                    bestSim[_neighbors - 1] = sim; bestFg[_neighbors - 1] = anchorIsForeground[a];
                    SortDescending(bestSim, bestFg, _neighbors);
                }
            }

            int fg = 0;
            for (int i = 0; i < filled; i++) if (bestFg[i]) fg++;
            // Strict majority: a tie is left as background so the region cannot creep outward on
            // ambiguous points.
            result[p] = filled > 0 && fg * 2 > filled;
        }

        return result;
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

    private static void SortDescending(double[] sim, bool[] fg, int count)
    {
        for (int i = 1; i < count; i++)
        {
            double s = sim[i];
            bool f = fg[i];
            int j = i - 1;
            while (j >= 0 && sim[j] < s) { sim[j + 1] = sim[j]; fg[j + 1] = fg[j]; j--; }
            sim[j + 1] = s; fg[j + 1] = f;
        }
    }

    /// <summary>
    /// Row-normalizes a [height, width, channels] feature map into a [points, channels] matrix, so a
    /// cosine similarity reduces to a dot product and the comparison can be done as a matmul.
    /// </summary>
    private static Tensor<T> NormalizeRows(Tensor<T> features, int points, int channels)
    {
        var result = new Tensor<T>(new[] { points, channels });
        for (int p = 0; p < points; p++)
        {
            int offset = p * channels;
            double norm = 0.0;
            for (int c = 0; c < channels; c++)
            {
                double v = NumOps.ToDouble(features[offset + c]);
                norm += v * v;
            }

            norm = Math.Sqrt(norm);
            // A zero-length feature vector has no direction; leaving it zero makes its similarity to
            // everything 0 rather than NaN, so it simply never wins a vote.
            if (norm <= 0.0) continue;

            for (int c = 0; c < channels; c++)
                result[offset + c] = NumOps.FromDouble(NumOps.ToDouble(features[offset + c]) / norm);
        }
        return result;
    }
}
