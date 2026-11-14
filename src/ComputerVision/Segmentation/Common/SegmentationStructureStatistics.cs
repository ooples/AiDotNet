using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Per-structure statistics of a medical segmentation (pixel/voxel count and mean confidence of every
/// class present in a label map), computed with whole-tensor engine operations.
/// </summary>
/// <remarks>
/// Counts are taken in <c>int</c> and means are formed from values pre-scaled by <c>1 / sqrt(count)</c>, so
/// a narrow <typeparamref name="T"/> such as Half neither rounds a count above 2048 nor overflows a sum
/// above 65504 (a 512 x 512 slice holds 262144 pixels).
/// </remarks>
internal static class SegmentationStructureStatistics
{
    /// <summary>
    /// Summarizes every class that occupies at least one pixel/voxel of <paramref name="labels"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of the label map and probabilities.</typeparam>
    /// <param name="labels">Class index map ([H, W] or [D, H, W]) holding integral class ids in [0, numClasses).</param>
    /// <param name="probabilities">Class probabilities [numClasses, ...] with the same spatial size as
    /// <paramref name="labels"/>.</param>
    /// <param name="skipClassId">A class id to leave out (for example 0 for background), or null to keep all.</param>
    /// <param name="firstOccurrenceOrder">When true, structures are listed in the order their class first
    /// appears in a raster scan of <paramref name="labels"/>; otherwise by ascending class id.</param>
    /// <returns>One <see cref="SegmentedStructure"/> per present class: its area/volume in pixels/voxels and the
    /// mean probability the model gave that class over those pixels/voxels.</returns>
    public static List<SegmentedStructure> Summarize<T>(
        Tensor<T> labels, Tensor<T> probabilities, int? skipClassId = null, bool firstOccurrenceOrder = false)
    {
        var engine = AiDotNetEngine.Current;
        int numClasses = probabilities.Shape[0];
        int spatial = labels.Length;
        if (numClasses == 0 || spatial == 0)
        {
            return new List<SegmentedStructure>();
        }

        var classIds = engine.Reshape(labels, new[] { spatial });
        var classIndex = classIds.Cast<int>();

        // Pixel/voxel count of every class (exact, in int), and the probability each pixel/voxel gave its own class.
        int[] areas = engine.TensorBinCount(classIndex, numClasses).ToArray();
        // Gathered straight from the class-first storage at class * spatial + position, so no transposed copy of
        // the whole probability volume is materialized to read one value per pixel/voxel.
        var flatIndex = engine.TensorAdd(
            engine.TensorMultiplyScalar(classIndex, spatial), engine.TensorArange<int>(0, spatial, 1));
        var assignedProbability = engine.TensorTake(
            engine.Reshape(probabilities, new[] { numClasses * spatial }), flatIndex);

        IEnumerable<int> classOrder = Enumerable.Range(0, numClasses);
        if (firstOccurrenceOrder)
        {
            // First raster position of each class = spatial - max(spatial - position) over that class's pixels.
            var reversedPosition = engine.ScalarMinusTensor(spatial, engine.TensorArange<int>(0, spatial, 1));
            int[] maxReversed = engine.ScatterMax(reversedPosition, classIndex, out _, 0, numClasses).ToArray();
            classOrder = classOrder.OrderByDescending(c => areas[c] > 0 ? maxReversed[c] : int.MinValue);
        }

        // SegmentedStructure reports its statistics as double (shared by every medical model).
        return classOrder
            .Where(c => areas[c] > 0 && c != skipClassId)
            .Select(c => new SegmentedStructure
            {
                ClassId = c,
                Name = $"Class_{c}",
                VolumeOrArea = areas[c],
                MeanConfidence = ClassMeanAsDouble(assignedProbability, classIndex, c, areas[c]),
            })
            .ToList();
    }

    /// <summary>
    /// The number of non-zero entries of <paramref name="mask"/>, counted in <c>int</c>.
    /// </summary>
    /// <remarks>Summing a 0/1 mask in Half rounds every count above 2048 and overflows above 65504.</remarks>
    public static int CountNonZero<T>(Tensor<T> mask)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var nonZero = engine.ScalarMinusTensor(numOps.One, engine.TensorEquals(mask, numOps.Zero));
        return engine.TensorSum(nonZero.Cast<int>());
    }

    /// <summary>
    /// The mean of <paramref name="values"/> over the <paramref name="count"/> positions where
    /// <paramref name="mask"/> is non-zero, or zero when the mask is empty.
    /// </summary>
    /// <remarks>
    /// Positions outside the mask are selected away, not multiplied by zero, so a NaN or infinite value
    /// there cannot turn the mean into NaN. Each value is scaled by <c>1 / sqrt(count)</c> before the full
    /// engine sum (accumulated in float for Half), so the sum stays near <c>sqrt(count)</c> instead of
    /// <c>count</c> and cannot overflow a narrow <typeparamref name="T"/>.
    /// </remarks>
    public static T MaskedMean<T>(Tensor<T> values, Tensor<T> mask, int count)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        if (count <= 0) return numOps.Zero;

        var engine = AiDotNetEngine.Current;
        T scale = numOps.FromDouble(1.0 / Math.Sqrt(count));
        var selected = engine.TensorWhere(mask,
            engine.TensorMultiplyScalar(values, scale), engine.TensorFullLike(values, numOps.Zero));
        return numOps.Multiply(engine.TensorSum(selected), scale);
    }

    /// <summary>The mean of <paramref name="assigned"/> over the pixels labelled <paramref name="classId"/>.</summary>
    private static double ClassMeanAsDouble<T>(Tensor<T> assigned, Tensor<int> classIndex, int classId, int count)
    {
        var mask = AiDotNetEngine.Current.TensorEquals(classIndex, classId).Cast<T>();
        return MathHelper.GetNumericOperations<T>().ToDouble(MaskedMean(assigned, mask, count));
    }
}
