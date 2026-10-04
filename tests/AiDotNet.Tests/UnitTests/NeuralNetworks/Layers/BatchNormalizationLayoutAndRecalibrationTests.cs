using System;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Review regressions for #2272's BatchNorm changes: a declared ChannelsLast layout must beat the NCHW heuristic, and a
/// recalibration pass must use batch statistics even when the layer is in evaluation mode.
/// </summary>
public class BatchNormalizationLayoutAndRecalibrationTests
{
    private static Tensor<double> Ramp(params int[] shape)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = ((i * 37) % 23) * 0.5 + (i % 4) * 3.0;
        return tensor;
    }

    [Fact]
    public void DeclaredChannelsLast_NormalizesTheTrailingAxis_EvenWhenAxisOneAlsoMatches()
    {
        // [2, 4, 3, 4]: axis 1 AND the trailing axis both equal the 4 features. Declared ChannelsLast, the features are
        // the trailing axis, so each trailing index must come out with zero mean over the other 24 values.
        var layer = new BatchNormalizationLayer<double>(4) { Layout = BatchNormDataLayout.ChannelsLast };
        layer.SetTrainingMode(true);
        var input = Ramp(2, 4, 3, 4);
        var output = layer.Forward(input);

        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
        for (int feature = 0; feature < 4; feature++)
        {
            double sum = 0;
            int count = 0;
            for (int i = feature; i < output.Length; i += 4) { sum += output[i]; count++; }
            Assert.True(Math.Abs(sum / count) < 1e-9,
                $"Trailing feature {feature} has mean {sum / count} after normalization; the layer normalized axis 1.");
        }
    }

    [Fact]
    public void Recalibration_UsesBatchStatistics_WhenTheLayerIsInEvaluationMode()
    {
        // A PredictCore that forces evaluation mode must not turn a recalibration pass into plain inference: with the
        // overwrite flag set, an eval-mode forward still replaces the running statistics with the batch's own.
        var layer = new BatchNormalizationLayer<double>(4);
        layer.SetTrainingMode(false);
        var input = Ramp(3, 4, 5, 5);
        layer.OverwriteRunningStatistics = true;
        try { layer.Forward(input); }
        finally { layer.OverwriteRunningStatistics = false; }

        var runningMean = layer.GetRunningMean();
        for (int channel = 0; channel < 4; channel++)
        {
            double sum = 0;
            int count = 0;
            for (int b = 0; b < 3; b++)
                for (int p = 0; p < 25; p++) { sum += input[(b * 4 + channel) * 25 + p]; count++; }
            Assert.Equal(sum / count, runningMean[channel], 9);
        }
    }
}
