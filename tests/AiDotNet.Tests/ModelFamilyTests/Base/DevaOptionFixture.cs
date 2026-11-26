using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.ComputerVision.Segmentation.Video;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

internal static class DevaOptionFixture
{
    internal static bool Handles(string option) => option is "MemoryTopK" or "MergeEvery"
        or "MaxMissedMerges" or "AssociationIoUThreshold" or "MaxMemoryFrames";

    internal static long Observe<T>(DEVA<T> model, string option)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var results = Run<T>(new DEVAOptions((DEVAOptions)model.GetOptions()), option);
        return OptionSensitivityHarness.Fingerprint(results.SelectMany(result =>
            result.ObjectIds.Select(id => (double)id).Prepend(result.ObjectIds.Length)
                .Concat(result.Masks.Select(ops.ToDouble)).Concat(result.Confidences)));
    }

    // DEVA explicitly supports supplied image layers. Identity logits make the proposals and keys
    // known, rather than letting an untrained decoder emit background throughout the video.
    // The temporal propagation, association, deletion and eviction are the real public DEVA task.
    internal static IReadOnlyList<VideoSegmentationResult<T>> Run<T>(DEVAOptions source, string option)
    {
        const int side = 8, pixels = side * side;
        var options = new DEVAOptions(source) { NumClasses = 2 };
        if (option != "MergeEvery") options.MergeEvery = option is "MemoryTopK" or "MaxMemoryFrames" ? 0 : 1;
        if (option != "MaxMissedMerges") options.MaxMissedMerges = option == "MergeEvery" ? 1 : 20;
        if (option != "MemoryTopK") options.MemoryTopK = 4096;
        if (option != "MaxMemoryFrames") options.MaxMemoryFrames = 20;
        var architecture = new NeuralNetworkArchitecture<T>(inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification, inputHeight: side,
            inputWidth: side, inputDepth: 2, outputSize: 2,
            layers: new List<ILayer<T>> { new ActivationLayer<T>((IActivationFunction<T>)new IdentityActivation<T>()) });
        using var model = new DEVA<T>(architecture, options: options);
        var ops = MathHelper.GetNumericOperations<T>();
        Tensor<T> Tensor(int[] shape, IEnumerable<double> values) =>
            new(shape, new Vector<T>(values.Select(ops.FromDouble).ToArray()));
        Tensor<T> Frame(Func<int, double> background, Func<int, double> foreground) =>
            Tensor(new[] { 1, 2, side, side }, Enumerable.Range(0, pixels).Select(background)
                .Concat(Enumerable.Range(0, pixels).Select(foreground)));
        Tensor<T> Mask(double value) => Tensor(new[] { side, side }, Enumerable.Repeat(value, pixels));
        Tensor<T> Annotation(double value) => Tensor(new[] { 1, side, side }, Enumerable.Repeat(value, pixels));
        var zero = Frame(_ => 0, _ => 0);
        var results = new List<VideoSegmentationResult<T>>();
        if (option == "MemoryTopK")
        {
            var initial = Frame(i => i * .03, _ => 0);
            var mask = Tensor(new[] { 1, side, side }, Enumerable.Range(0, pixels).Select(i => .1 + .8 * i / 63));
            model.InitializeTracking(initial, mask, new[] { 7 });
            results.Add(model.PropagateToFrame(Frame(_ => .2, _ => 0)));
        }
        else if (option == "MaxMemoryFrames")
        {
            model.InitializeTracking(zero, Annotation(.1), new[] { 7 });
            for (int step = 1; step <= 12; step++)
            {
                model.AddCorrection(7, Mask(.1 + .05 * step));
                results.Add(model.PropagateToFrame(zero));
            }
            results.Add(model.PropagateToFrame(zero));
        }
        else
        {
            model.InitializeTracking(zero, Annotation(1), new[] { 7 });
            // One foreground half-image has exactly IoU 1/2 with the propagated full annotation.
            // An all-background image instead supplies consecutive unmatched integrations.
            var next = option == "AssociationIoUThreshold"
                ? Frame(i => i % side < side / 2 ? -2 : 2, i => i % side < side / 2 ? 2 : -2)
                : Frame(_ => 2, _ => -2);
            int steps = option == "AssociationIoUThreshold" ? 1 : 12;
            for (int step = 0; step < steps; step++) results.Add(model.PropagateToFrame(next));
        }
        model.ResetTracking();
        return results;
    }
}
