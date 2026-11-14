using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Assembles a slice-by-slice volume segmentation: each slice's <c>[H, W]</c> labels and <c>[numClasses, H, W]</c>
/// probabilities are copied into their depth slot of a <c>[D, H, W]</c> label volume and a
/// <c>[numClasses, D, H, W]</c> probability volume.
/// </summary>
/// <remarks>
/// The destination volumes are allocated once the first slice fixes the spatial size and class count, and every
/// slice is copied in as it arrives, so no list of per-slice tensors is held alongside the assembled volume. A
/// three-class float volume of 256 x 512 x 512 is 768 MiB, and concatenating retained slices needed that twice.
/// </remarks>
/// <typeparam name="T">The numeric type of the labels and probabilities.</typeparam>
internal sealed class SliceVolumeAssembler<T>
{
    private readonly int _depth;
    private Tensor<T>? _labels;
    private Tensor<T>? _probabilities;

    /// <summary>Creates an assembler for a volume of <paramref name="depth"/> slices.</summary>
    public SliceVolumeAssembler(int depth)
    {
        if (depth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(depth), depth, "A volume needs at least one slice.");
        }

        _depth = depth;
    }

    /// <summary>
    /// Copies slice <paramref name="depthIndex"/>'s labels <c>[H, W]</c> and probabilities <c>[numClasses, H, W]</c>
    /// into the volume.
    /// </summary>
    public void Add(int depthIndex, Tensor<T> sliceLabels, Tensor<T> sliceProbabilities)
    {
        if (sliceLabels is null) throw new ArgumentNullException(nameof(sliceLabels));
        if (sliceProbabilities is null) throw new ArgumentNullException(nameof(sliceProbabilities));
        if (depthIndex < 0 || depthIndex >= _depth)
        {
            throw new ArgumentOutOfRangeException(nameof(depthIndex), depthIndex, $"The volume has {_depth} slices.");
        }

        if (_labels is null || _probabilities is null)
        {
            _labels = new Tensor<T>(new[] { _depth, sliceLabels.Shape[0], sliceLabels.Shape[1] });
            _probabilities = new Tensor<T>(new[]
            {
                sliceProbabilities.Shape[0], _depth, sliceProbabilities.Shape[1], sliceProbabilities.Shape[2],
            });
        }

        _labels.SetSlice(0, depthIndex, sliceLabels);
        _probabilities.SetSlice(1, depthIndex, sliceProbabilities);
    }

    /// <summary>The assembled <c>[D, H, W]</c> label volume.</summary>
    public Tensor<T> Labels => _labels ?? throw new InvalidOperationException("No slice has been added.");

    /// <summary>The assembled <c>[numClasses, D, H, W]</c> probability volume.</summary>
    public Tensor<T> Probabilities => _probabilities ?? throw new InvalidOperationException("No slice has been added.");
}
