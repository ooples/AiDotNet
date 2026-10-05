using System;
using System.Linq;
using AiDotNet.ComputerVision;
using AiDotNet.ComputerVision.Detection.ObjectDetection.DETR;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// "Look forward twice" (DINO Sec. 3.5, used by RT-DETR's decoder): layer i's training box is refined from
/// layer i-1's box without detaching it. So the loss on the LAST layer's boxes must reach the previous
/// layer's box head. A decoder that detached every reference would give the same values but a zero
/// gradient there, so no value-based check can see the difference.
/// </summary>
public class RTDETRLookForwardTwiceTests
{
    [Fact]
    public void LastLayerBoxLoss_ReachesThePreviousLayersBoxHead()
    {
        using var model = new RTDETR<double>(new ObjectDetectionOptions<double>
        {
            InputSize = new[] { 64, 64 },
            Size = ModelSize.Nano,
            NumClasses = 2
        });
        var rng = new Random(5);
        var input = new Tensor<double>(new[] { 1, 3, 64, 64 });
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();

        var decoder = model.GetDecoder();
        Assert.True(decoder.BoxHeads.Count >= 2);
        var previousBias = decoder.BoxHeads[decoder.BoxHeads.Count - 2].Layers.Last().Bias;
        var lastBias = decoder.BoxHeads[decoder.BoxHeads.Count - 1].Layers.Last().Bias;

        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new GradientTape<double>())
        {
            var pass = model.ForwardPass(input);
            var lastBoxes = pass.Boxes[pass.Boxes.Count - 1];
            var loss = TensorModelTrainer<double>.MeanSquaredError(lastBoxes, new Tensor<double>(lastBoxes._shape));
            gradients = tape.ComputeGradients(loss, new[] { previousBias, lastBias });
        }

        double Norm(Tensor<double> t) => Math.Sqrt(t.ToArray().Sum(v => v * v));
        Assert.True(gradients.ContainsKey(lastBias) && Norm(gradients[lastBias]) > 0, "the last layer's own box head gets no gradient");
        Assert.True(gradients.ContainsKey(previousBias) && Norm(gradients[previousBias]) > 0,
            "the last layer's box loss does not reach the previous layer's box head: its reference was detached");
    }
}
