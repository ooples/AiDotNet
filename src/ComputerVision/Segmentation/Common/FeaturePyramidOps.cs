using AiDotNet.Interfaces;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Multi-scale feature helpers shared by the segmentation models whose papers read several backbone
/// resolutions: kMaX-DeepLab's pixel decoder and the FPN under CUPS's Cascade Mask R-CNN.
/// </summary>
internal static class FeaturePyramidOps
{
    /// <summary>
    /// Runs <c>layers[start..end)</c> on <paramref name="input"/> and taps the backbone stages: for every spatial
    /// resolution the run passes through, the last feature map produced at that resolution.
    /// </summary>
    /// <returns>The final output and the stage outputs, in the order the run reached them (finest first for a
    /// downsampling backbone).</returns>
    internal static (Tensor<T> Output, List<Tensor<T>> Stages) RunAndTapStages<T>(
        IReadOnlyList<ILayer<T>> layers, int start, int end, Tensor<T> input)
    {
        if (layers is null) throw new ArgumentNullException(nameof(layers));
        if (input is null) throw new ArgumentNullException(nameof(input));
        var stages = new List<Tensor<T>>();
        var x = input;
        for (int i = start; i < end; i++)
        {
            x = layers[i].Forward(x);
            if (x.Rank != 4) continue;
            if (stages.Count > 0 && stages[stages.Count - 1].Shape[2] == x.Shape[2] && stages[stages.Count - 1].Shape[3] == x.Shape[3])
                stages[stages.Count - 1] = x;
            else
                stages.Add(x);
        }

        return (x, stages);
    }

    /// <summary>
    /// The FPN top-down pathway (Lin et al. 2017): <c>P_coarsest = L(C_coarsest)</c>, then
    /// <c>P_i = L_i(C_i) + Upsample(P_{i+1})</c> (nearest), and every level is smoothed by its output convolution.
    /// </summary>
    /// <param name="stages">Backbone stages, finest first.</param>
    /// <param name="lateral">One lateral (1x1) convolution per stage.</param>
    /// <param name="output">One output (3x3) convolution per stage.</param>
    /// <returns>The pyramid levels, finest first.</returns>
    internal static List<Tensor<T>> TopDown<T>(
        IReadOnlyList<Tensor<T>> stages, IReadOnlyList<Func<Tensor<T>, Tensor<T>>> lateral,
        IReadOnlyList<Func<Tensor<T>, Tensor<T>>> output)
    {
        if (stages is null) throw new ArgumentNullException(nameof(stages));
        if (lateral is null) throw new ArgumentNullException(nameof(lateral));
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (stages.Count == 0 || lateral.Count != stages.Count || output.Count != stages.Count)
            throw new ArgumentException("Give one lateral and one output convolution per backbone stage.", nameof(stages));

        var engine = AiDotNetEngine.Current;
        var levels = new Tensor<T>[stages.Count];
        Tensor<T>? coarser = null;
        for (int i = stages.Count - 1; i >= 0; i--)
        {
            var merged = lateral[i](stages[i]);
            if (coarser is not null)
            {
                var size = new[] { merged.Shape[2], merged.Shape[3] };
                merged = engine.TensorAdd(merged, engine.Interpolate(coarser, size, InterpolateMode.Nearest, false));
            }

            coarser = merged;
            levels[i] = output[i](merged);
        }

        return levels.ToList();
    }
}
