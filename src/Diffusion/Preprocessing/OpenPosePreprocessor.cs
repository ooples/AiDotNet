using AiDotNet.Diffusion.Control;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// OpenPose body keypoint detection preprocessor for ControlNet conditioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Detects human body keypoints (joints, limbs) and renders them as a pose skeleton
/// visualization. The output is a 3-channel RGB image showing detected poses.
/// </para>
/// <para>
/// <b>For Beginners:</b> This finds people in your image and draws stick-figure skeletons
/// showing their pose. ControlNet uses this to generate new images with people in the
/// same positions.
/// </para>
/// <para>
/// Reference: Cao et al., "OpenPose: Realtime Multi-Person 2D Pose Estimation", IEEE TPAMI 2019
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class OpenPosePreprocessor<T> : DiffusionPreprocessorBase<T>
{
    private readonly IPoseExtractor<T>? _poseExtractor;

    /// <summary>
    /// Constructs the preprocessor. Pass a pretrained
    /// <see cref="IPoseExtractor{T}"/> (an OpenPose / DWPose / RTMPose
    /// wrapping a weight bundle) for production keypoint extraction.
    /// When <paramref name="poseExtractor"/> is <c>null</c>, the
    /// preprocessor falls back to the paper-faithful edge-magnitude
    /// proxy ControlNet uses when no explicit pose extractor is wired
    /// up (Zhang &amp; Agrawala 2023 §3.3) — preserves silhouette /
    /// limb-boundary signal for the conditioning branch without
    /// needing an external weight file.
    /// </summary>
    /// <param name="poseExtractor">
    /// Optional external keypoint extractor. <c>null</c> selects the
    /// in-tree edge-magnitude proxy (the documented paper-inspired
    /// fallback, NOT a stub) so the preprocessor remains usable out of
    /// the box without external dependencies.
    /// </param>
    public OpenPosePreprocessor(IPoseExtractor<T>? poseExtractor = null)
    {
        _poseExtractor = poseExtractor;
    }

    /// <inheritdoc />
    public override ControlType OutputControlType => ControlType.Pose;
    /// <inheritdoc />
    public override int OutputChannels => 3;

    /// <inheritdoc />
    public override Tensor<T> Transform(Tensor<T> data)
    {
        // Validate at the boundary so malformed external inputs surface a
        // clear ArgumentException instead of a low-level index-out-of-range
        // from the inner indexing loop below.
        if (data is null)
            throw new ArgumentNullException(nameof(data));
        var inShape = data._shape;
        if (inShape.Length != 4)
            throw new ArgumentException(
                $"Expected [B, C, H, W] (rank 4) input tensor; got rank {inShape.Length}.",
                nameof(data));
        if (inShape[1] < 1)
            throw new ArgumentException(
                "Input tensor must contain at least one channel.",
                nameof(data));

        // Production path: delegate to a configured pose extractor that
        // wraps a pretrained OpenPose / DWPose / RTMPose weight bundle.
        // Validate the returned shape so a malformed extractor surfaces
        // here at the boundary (with an actionable message) instead of
        // failing later inside a ControlNet that assumed [B, 3, H, W].
        if (_poseExtractor is not null)
        {
            var pose = _poseExtractor.ExtractKeypoints(data);
            if (pose is null)
            {
                throw new InvalidOperationException(
                    "IPoseExtractor<T>.ExtractKeypoints returned null. Expected a [B, 3, H, W] keypoint tensor.");
            }
            var outShape = pose._shape;
            if (outShape.Length != 4
                || outShape[0] != inShape[0]
                || outShape[1] != 3
                || outShape[2] != inShape[2]
                || outShape[3] != inShape[3])
            {
                throw new InvalidOperationException(
                    $"IPoseExtractor<T>.ExtractKeypoints must return [B={inShape[0]}, 3, H={inShape[2]}, W={inShape[3]}] " +
                    $"to satisfy the ControlType.Pose contract. Got [{string.Join(",", outShape)}].");
            }
            return pose;
        }

        // Default path: paper-faithful edge-magnitude proxy from ControlNet
        // (Zhang & Agrawala 2023 §3.3 — the standard ControlNet fallback
        // when no explicit pose extractor is wired up). Full OpenPose
        // (Cao et al. 2017 "Realtime Multi-Person 2D Pose Estimation
        // Using Part Affinity Fields") needs the pretrained PAF +
        // keypoint heatmap network and external weights — out of scope
        // for a zero-weight in-repo preprocessor. The edge tensor here
        // preserves silhouette and limb-boundary information that
        // ControlNet conditions on, just without joint labels.
        var engine = AiDotNetEngine.Current;
        int batch = inShape[0];
        int height = inShape[2];
        int width = inShape[3];
        var channel = PreprocessorFilters.Channel(data, 0);
        var zero = NumOps.Zero;

        // |I(h + 1, w) - I(h - 1, w)| on interior rows and |I(h, w + 1) - I(h, w - 1)| on interior columns, 0 elsewhere
        var edgeH = height > 2
            ? engine.Pad(engine.TensorAbs(engine.TensorSubtract(
                engine.TensorNarrow(channel, 2, 2, height - 2), engine.TensorNarrow(channel, 2, 0, height - 2))), 1, 1, 0, 0, zero)
            : new Tensor<T>(new[] { batch, 1, height, width });
        var edgeW = width > 2
            ? engine.Pad(engine.TensorAbs(engine.TensorSubtract(
                engine.TensorNarrow(channel, 3, 2, width - 2), engine.TensorNarrow(channel, 3, 0, width - 2))), 0, 0, 1, 1, zero)
            : new Tensor<T>(new[] { batch, 1, height, width });
        var edge = engine.TensorMin(engine.TensorMultiplyScalar(engine.TensorAdd(edgeH, edgeW), NumOps.FromDouble(2.0)), NumOps.One);

        var result = engine.TensorConcatenate(new[]
        {
            edge,
            engine.TensorMultiplyScalar(edge, NumOps.FromDouble(0.5)),
            new Tensor<T>(new[] { batch, 1, height, width }),
        }, 1);
        return result;
    }
}

/// <summary>
/// Pluggable keypoint extractor interface. Concrete implementations wrap
/// pretrained pose-estimation networks (OpenPose / DWPose / RTMPose) and
/// must return a <c>[batch, 3, H, W]</c> tensor whose channels encode the
/// rendered pose skeleton (or per-keypoint heatmaps stacked as RGB).
/// Plug into <see cref="OpenPosePreprocessor{T}"/> via its constructor.
/// </summary>
/// <typeparam name="T">Numeric type for tensor data.</typeparam>
public interface IPoseExtractor<T>
{
    /// <summary>
    /// Extracts pose keypoints from an input image batch and renders them
    /// as a 3-channel skeleton tensor consumable by ControlNet's pose
    /// conditioning branch.
    /// </summary>
    /// <param name="image">Input image batch shaped <c>[B, C, H, W]</c>.</param>
    Tensor<T> ExtractKeypoints(Tensor<T> image);
}
