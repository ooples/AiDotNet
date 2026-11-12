using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Per-structure statistics of a medical segmentation (pixel/voxel count and mean confidence of every
/// class present in a label map), computed with whole-tensor engine operations.
/// </summary>
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

        // Pixel/voxel count of every class, and the probability each pixel/voxel gave its own class summed per class.
        var counts = engine.TensorBinCount(classIndex, numClasses);
        int[] areas = counts.ToArray();
        var probabilityByPixel = engine.TensorPermute(
            engine.Reshape(probabilities, new[] { numClasses, spatial }), new[] { 1, 0 });
        var assignedProbability = engine.TensorGatherClassValues(probabilityByPixel, classIds);
        var confidenceSums = engine.TensorScatterAdd(
            new Tensor<T>(new[] { numClasses }), classIndex, assignedProbability, 0);
        var meanConfidence = engine.TensorDivide(
            confidenceSums,
            engine.TensorClampMin(counts.Cast<T>(), MathHelper.GetNumericOperations<T>().One));

        // SegmentedStructure reports its statistics as double (shared by every medical model).
        double[] meanConfidenceValues = meanConfidence.Cast<double>().ToArray();

        IEnumerable<int> classOrder = Enumerable.Range(0, numClasses);
        if (firstOccurrenceOrder)
        {
            // First raster position of each class = spatial - max(spatial - position) over that class's pixels.
            var reversedPosition = engine.ScalarMinusTensor(spatial, engine.TensorArange<int>(0, spatial, 1));
            int[] maxReversed = engine.ScatterMax(reversedPosition, classIndex, out _, 0, numClasses).ToArray();
            classOrder = classOrder.OrderByDescending(c => areas[c] > 0 ? maxReversed[c] : int.MinValue);
        }

        return classOrder
            .Where(c => areas[c] > 0 && c != skipClassId)
            .Select(c => new SegmentedStructure
            {
                ClassId = c,
                Name = $"Class_{c}",
                VolumeOrArea = areas[c],
                MeanConfidence = meanConfidenceValues[c],
            })
            .ToList();
    }
}
