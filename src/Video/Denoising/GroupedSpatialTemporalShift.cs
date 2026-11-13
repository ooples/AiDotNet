using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Video.Denoising;

/// <summary>
/// The grouped spatial-temporal shift module: Shift-Net's alignment mechanism, which replaces optical
/// flow, deformable convolution and cross-frame attention with pure tensor shifting.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Li, Wang, Zhang, Wang, Loy, "A Simple Baseline for Video Restoration with Grouped Spatial-temporal
/// Shift" (CVPR 2023, arXiv:2206.10810).
/// </para>
/// <para><b>The scheme, in the paper's terms:</b></para>
/// <code>
///   temporal   f_i in R^(h x w x c)  is split EQUALLY along channels into
///              f_i^a, f_i^b in R^(h x w x c/2)
///              Forward (FTS) and Backward (BTS) temporal shift blocks alternate over
///              adjacent frame pairs, propagating the b-group between neighbours.
///
///   spatial    f_{i-1}^b is split along channels into M slices
///              f_{i-1,m}^b in R^(h x w x c/2M)
///              f'_{i-1,m}^b = Shift(f_{i-1,m}^b, dx_m, dy_m)
///              |dx_m| = kx*(s-1)+1, |dy_m| = ky*(s-1)+1,   s = 5
///              implementation: M = 25, dx_m, dy_m in {-9, -5, 0, 5, 9}
/// </code>
/// <para>
/// <b>Shift is FREE, which is the entire argument.</b> Moving a feature slice costs no multiply-adds,
/// so alignment becomes a memory operation and the model reaches the accuracy of attention-based
/// restorers at a fraction of the cost. Anything here that quietly became a convolution or a weighted
/// blend would defeat the point of the paper.
/// </para>
/// <para>
/// <b>Both halves matter and they are different.</b> The temporal half moves information BETWEEN
/// frames; the spatial half moves it WITHIN a frame across 25 displacements, which is what supplies the
/// large effective receptive field that stands in for explicit correspondence search. Implementing only
/// the temporal shift — the obvious reading of "temporal shift" — leaves the module unable to align
/// anything that moved sideways.
/// </para>
/// <para><b>For Beginners:</b> To clean up a video frame it helps to look at neighbouring frames, but
/// things move between them. Instead of computing where everything went, this simply slides copies of
/// the neighbour's features in 25 different directions and lets the following convolution pick whichever
/// slide happens to line up. Sliding is free; searching is not.</para>
/// </remarks>
public sealed class GroupedSpatialTemporalShift<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>The paper's base shift length s.</summary>
    public const int BaseShiftLength = 5;

    /// <summary>
    /// The displacement values the paper's implementation uses per axis: <c>{-9, -5, 0, 5, 9}</c>.
    /// </summary>
    /// <remarks>
    /// The formula <c>|d| = k*(s-1)+1</c> with <c>s = 5</c> generates magnitudes 1, 5, 9 for
    /// <c>k = 0, 1, 2</c>. The paper states its implementation uses the five values below, whose 5x5
    /// product gives exactly the stated <c>M = 25</c> slices; the ±1 magnitude the formula also permits
    /// is not among them. Recorded as the implementation's set rather than derived, because deriving it
    /// from the formula alone would not reproduce M = 25.
    /// </remarks>
    public static readonly int[] Displacements = { -9, -5, 0, 5, 9 };

    /// <summary>The paper's slice count M, the number of (dx, dy) displacement pairs.</summary>
    public static int SliceCount => Displacements.Length * Displacements.Length;

    /// <summary>
    /// The M displacement pairs, in row-major order over <see cref="Displacements"/>.
    /// </summary>
    public static IReadOnlyList<(int Dx, int Dy)> Offsets { get; } = BuildOffsets();

    private static (int Dx, int Dy)[] BuildOffsets()
    {
        var offsets = new (int, int)[SliceCount];
        int k = 0;
        foreach (int dy in Displacements)
        {
            foreach (int dx in Displacements) offsets[k++] = (dx, dy);
        }
        return offsets;
    }

    /// <summary>
    /// Splits a feature map EQUALLY along the channel axis into the <c>a</c> and <c>b</c> groups.
    /// </summary>
    /// <param name="features">A <c>[height, width, channels]</c> feature map with an even channel count.</param>
    /// <returns>
    /// <c>a</c> keeps the frame's own information; <c>b</c> is the group propagated to neighbours.
    /// </returns>
    /// <remarks>
    /// Equal halves, per <c>f_i^a, f_i^b in R^(h x w x c/2)</c>. An odd channel count cannot be split
    /// equally and is rejected rather than silently rounded, since an off-by-one here shifts every
    /// downstream slice boundary.
    /// </remarks>
    public (Tensor<T> A, Tensor<T> B) SplitGroups(Tensor<T> features)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));
        if (features.Shape.Length != 3)
            throw new ArgumentException(
                $"Expected [height, width, channels]; got rank {features.Shape.Length}.", nameof(features));

        int h = features.Shape[0], w = features.Shape[1], c = features.Shape[2];
        if (c % 2 != 0)
            throw new ArgumentException(
                $"The channel count must be even to split into equal a/b groups; got {c}.", nameof(features));

        int half = c / 2;
        var engine = AiDotNetEngine.Current;
        // Detached copies, as before: the groups are handed to callers, so they must not alias the input
        return (Detached(engine.TensorNarrow(features, 2, 0, half)),
            Detached(engine.TensorNarrow(features, 2, half, half)));
    }

    /// <summary>A fresh dense copy of <paramref name="view"/> that shares no storage with its source.</summary>
    private static Tensor<T> Detached(Tensor<T> view) => new Tensor<T>(view.ToArray(), view.Shape.ToArray());

    /// <summary>
    /// Shifts a feature map spatially by <paramref name="dx"/>, <paramref name="dy"/>.
    /// </summary>
    /// <remarks>
    /// Vacated positions are ZERO-filled, not clamped or wrapped. Clamping would smear border features
    /// inward and invent correspondences that do not exist; wrapping would align the left edge with the
    /// right. Zero says "nothing was shifted in from here", which is what the following convolution
    /// needs to learn to ignore.
    /// </remarks>
    public Tensor<T> Shift(Tensor<T> features, int dx, int dy)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));
        if (features.Shape.Length != 3)
            throw new ArgumentException(
                $"Expected [height, width, channels]; got rank {features.Shape.Length}.", nameof(features));

        if (dx == 0 && dy == 0)
        {
            // A zero shift still returns a new map, never the caller's tensor
            return Detached(features);
        }

        var engine = AiDotNetEngine.Current;
        return ShiftAxis(engine, ShiftAxis(engine, features, 0, dy), 1, dx);
    }

    /// <summary>
    /// Moves the content of <paramref name="features"/> by <paramref name="displacement"/> positions along
    /// <paramref name="axis"/> (output index = source index + displacement), filling vacated positions with zeros.
    /// </summary>
    private static Tensor<T> ShiftAxis(IEngine engine, Tensor<T> features, int axis, int displacement)
    {
        if (displacement == 0)
        {
            return features;
        }

        int extent = features.Shape[axis];
        var zeroShape = (int[])features._shape.Clone();
        zeroShape[axis] = Math.Min(Math.Abs(displacement), extent);
        var zeros = new Tensor<T>(zeroShape);
        if (Math.Abs(displacement) >= extent)
        {
            return zeros;
        }

        int kept = extent - Math.Abs(displacement);
        return displacement > 0
            ? engine.TensorConcatenate(new[] { zeros, engine.TensorNarrow(features, axis, 0, kept) }, axis)
            : engine.TensorConcatenate(new[] { engine.TensorNarrow(features, axis, -displacement, kept), zeros }, axis);
    }

    /// <summary>
    /// The grouped SPATIAL shift: splits a feature map into <see cref="SliceCount"/> channel slices and
    /// shifts each by its own displacement.
    /// </summary>
    /// <param name="features">The <c>b</c> group of a neighbouring frame.</param>
    /// <returns>
    /// A feature map of the same shape, with slice <c>m</c> displaced by <c>Offsets[m]</c>.
    /// </returns>
    /// <remarks>
    /// Each slice gets a DIFFERENT displacement — that is what "grouped" means. Shifting the whole map
    /// by one offset would align only content that happened to move that far; 25 slices covering
    /// {-9,-5,0,5,9}^2 give the following convolution a choice of alignments to select from, which is
    /// how the module substitutes for correspondence estimation.
    /// </remarks>
    public Tensor<T> GroupedSpatialShift(Tensor<T> features)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));
        if (features.Shape.Length != 3)
            throw new ArgumentException(
                $"Expected [height, width, channels]; got rank {features.Shape.Length}.", nameof(features));

        int h = features.Shape[0], w = features.Shape[1], c = features.Shape[2];
        int slices = SliceCount;
        if (c < slices)
            throw new ArgumentException(
                $"Need at least {slices} channels to form M = {slices} slices; got {c}. The paper splits " +
                $"the c/2 group into M slices of c/2M channels each.", nameof(features));

        int per = c / slices;              // channels per slice
        int assigned = per * slices;       // any remainder stays unshifted, see below
        var engine = AiDotNetEngine.Current;
        var parts = new List<Tensor<T>>(slices + 1);

        for (int m = 0; m < slices; m++)
        {
            var (dx, dy) = Offsets[m];
            parts.Add(Shift(engine.TensorNarrow(features, 2, m * per, per), dx, dy));
        }

        // Channels beyond M*per (when c is not a multiple of M) are copied through unshifted rather than
        // dropped. Leaving them zero would silently discard a slice of the neighbour's features.
        if (assigned < c)
        {
            parts.Add(engine.TensorNarrow(features, 2, assigned, c - assigned));
        }

        return engine.TensorConcatenate(parts.ToArray(), 2);
    }

    /// <summary>
    /// A forward temporal shift (FTS): frame <c>i</c> keeps its own <c>a</c> group and receives the
    /// spatially-shifted <c>b</c> group of frame <c>i - 1</c>.
    /// </summary>
    /// <param name="current">Frame i's features, <c>[height, width, channels]</c>.</param>
    /// <param name="previous">Frame i-1's features, same shape.</param>
    /// <remarks>
    /// Forward means information flows from PAST to present. The first frame of a sequence has no
    /// predecessor, so a caller should pass it as its own <paramref name="previous"/> — which reduces to
    /// self-alignment rather than injecting zeros.
    /// </remarks>
    public Tensor<T> ForwardTemporalShift(Tensor<T> current, Tensor<T> previous)
        => TemporalShift(current, previous);

    /// <summary>
    /// A backward temporal shift (BTS): frame <c>i</c> keeps its own <c>a</c> group and receives the
    /// spatially-shifted <c>b</c> group of frame <c>i + 1</c>.
    /// </summary>
    /// <remarks>
    /// FTS and BTS blocks ALTERNATE in the paper, so information propagates in both directions across
    /// the sequence. Using only one direction makes the first (or last) frames unrecoverable, since they
    /// would never receive anything.
    /// </remarks>
    public Tensor<T> BackwardTemporalShift(Tensor<T> current, Tensor<T> next)
        => TemporalShift(current, next);

    private Tensor<T> TemporalShift(Tensor<T> current, Tensor<T> neighbour)
    {
        if (current is null) throw new ArgumentNullException(nameof(current));
        if (neighbour is null) throw new ArgumentNullException(nameof(neighbour));
        for (int d = 0; d < 3; d++)
        {
            if (current.Shape[d] != neighbour.Shape[d])
                throw new ArgumentException(
                    "The current and neighbouring frames must have identical shapes.", nameof(neighbour));
        }

        var (a, _) = SplitGroups(current);
        var (_, neighbourB) = SplitGroups(neighbour);

        // The neighbour's b group is spatially shifted before being handed over; the frame's own a group
        // passes through untouched.
        var shiftedB = GroupedSpatialShift(neighbourB);

        return AiDotNetEngine.Current.TensorConcatenate(new[] { a, shiftedB }, 2);
    }

    /// <summary>
    /// The paper's loss: <c>L = (1/T) * sum_i ||H_i - O_i||_1</c>.
    /// </summary>
    /// <remarks>
    /// Plain L1 over frames — no perceptual or adversarial term. Shift-Net's claim is that a simple
    /// baseline suffices, so adding auxiliary losses would confound exactly the comparison it makes.
    /// </remarks>
    public static double Loss(Vector<T> predicted, Vector<T> target)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (predicted.Length != target.Length)
            throw new ArgumentException(
                $"Predicted ({predicted.Length}) and target ({target.Length}) must be the same length.",
                nameof(predicted));

        if (predicted.Length == 0)
        {
            return 0.0;
        }

        // In double: the result is a double, and a float difference such as 3e38 - (-3e38) overflows T
        var engine = AiDotNetEngine.Current;
        return engine.TensorMean(engine.TensorAbs(engine.TensorSubtract(
            TensorCopyHelper.CastLogical<T, double>(predicted.AsTensor()),
            TensorCopyHelper.CastLogical<T, double>(target.AsTensor()))));
    }
}
