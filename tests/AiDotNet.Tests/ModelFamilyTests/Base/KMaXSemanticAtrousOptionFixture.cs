using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ComputerVision.Segmentation.Panoptic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Option-sensitivity probe for <see cref="KMaXDeepLabOptions.SemanticAtrousRates"/> on deterministic multi-pixel
/// pyramid features (<c>stage5</c> at <c>5x5</c>), avoiding the <c>1x1</c> stride-32 collapse of the <c>32x32</c>
/// image clamp while executing the rebuilt <see cref="KMaXDeepLab{T}"/> instance's own semantic decoder layers.
/// </summary>
internal static class KMaXSemanticAtrousOptionFixture
{
    internal static bool Handles(string option)
        => option == nameof(KMaXDeepLabOptions.SemanticAtrousRates);

    internal static void PrepareOptions(KMaXDeepLabOptions options, string option)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        if (!Handles(option)) return;

        // On a 5x5 stage5 feature map, rates 1, 2 and 3 sample distinct in-bounds neighbor offsets (1, 2 and 3 pixels
        // apart), whereas the paper's [6, 12, 18] rates were sized for 640x640 (20x20 stage5) and all exceed 5.
        options.SemanticAtrousRates = [1, 2, 3];
    }

    internal static long Observe<T>(KMaXDeepLab<T> model)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));

        // Materialize lazy weights through KMaXDeepLab.ResolveLazyLayerShapes before running the semantic decoder.
        _ = model.GetParameters();
        model.SetTrainingMode(false);

        var ops = MathHelper.GetNumericOperations<T>();
        var channels = model.BackboneStageChannels;
        int c0 = channels.Count > 0 ? channels[0] : 256;
        int c1 = channels.Count > 1 ? channels[1] : c0;
        int c2 = channels.Count > 2 ? channels[2] : c1;
        int c3 = channels.Count > 3 ? channels[3] : c2;

        // Finest first (20x20, 10x10, 5x5, 5x5; ASPP reads stages[stages.Count - 1] = 5x5).
        var stages = new List<Tensor<T>>(4)
        {
            BuildStage<T>(c0, 20, 20, 0.13, 0.0),
            BuildStage<T>(c1, 10, 10, 0.19, 0.7),
            BuildStage<T>(c2, 5, 5, 0.29, 1.4),
            BuildStage<T>(c3, 5, 5, 0.37, 2.1),
        };

        using var noGrad = new NoGradScope<T>();
        var logits = model.RunSemanticDecoder(stages);
        return OptionSensitivityHarness.Fingerprint(
            Enumerable.Range(0, logits.Length).Select(i => ops.ToDouble(logits[i])));
    }

    private static Tensor<T> BuildStage<T>(int channels, int height, int width, double freq, double phase)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int spatial = height * width;
        int total = channels * spatial;
        var data = new T[total];
        for (int c = 0; c < channels; c++)
        {
            double cPhase = phase + 0.11 * (c % 17);
            int baseIdx = c * spatial;
            for (int r = 0; r < height; r++)
            {
                for (int col = 0; col < width; col++)
                {
                    double v = 0.35 * Math.Sin(freq * (r + 1) * (col + 2) + cPhase)
                             + 0.25 * Math.Cos(0.41 * (3 * r + col + 1) - 0.07 * (c % 13));
                    data[baseIdx + r * width + col] = ops.FromDouble(v);
                }
            }
        }

        return new Tensor<T>(new[] { 1, channels, height, width }, new Vector<T>(data));
    }
}
