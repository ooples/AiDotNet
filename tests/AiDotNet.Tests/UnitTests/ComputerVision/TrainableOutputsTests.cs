using AiDotNet.ComputerVision;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// A raw training step takes a detector's data outputs (the boxes a two-stage head received) out of its loss:
/// their positions carry the target's values, every other position the prediction's.
/// </summary>
public class TrainableOutputsTests
{
    private static Tensor<double> Filled(int[] shape, double start)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = start + i;
        return t;
    }

    [Fact]
    public void DataOutputsCarryTheTarget_AndTheRestCarryThePrediction()
    {
        // A two-stage layout: per-RoI heads beside per-image RPN maps, so everything is flattened into [1, total].
        var outputs = new List<Tensor<double>> { Filled(new[] { 3, 2 }, 1), Filled(new[] { 3, 4 }, 100), Filled(new[] { 1, 5 }, 200) };
        var prediction = CvTensorOps<double>.ConcatenateOutputs(outputs);
        var target = Filled(prediction.Shape.ToArray(), -50);

        var trainable = CvTensorOps<double>.ConcatenateTrainableOutputs(outputs, new[] { 1 }, target);

        Assert.Equal(prediction.Shape.ToArray(), trainable.Shape.ToArray());
        for (int i = 0; i < trainable.Length; i++)
        {
            bool data = i >= 6 && i < 18;
            Assert.Equal(data ? target[i] : prediction[i], trainable[i]);
        }
    }

    [Fact]
    public void WithoutAMatchingTarget_DataOutputsAreZero()
    {
        var outputs = new List<Tensor<double>> { Filled(new[] { 2, 2 }, 1), Filled(new[] { 2, 3 }, 10) };
        var trainable = CvTensorOps<double>.ConcatenateTrainableOutputs(outputs, new[] { 0 }, new Tensor<double>(new[] { 1, 1 }));
        var prediction = CvTensorOps<double>.ConcatenateOutputs(outputs);
        for (int row = 0; row < 2; row++)
        {
            for (int column = 0; column < 5; column++)
            {
                Assert.Equal(column < 2 ? 0.0 : prediction[row, column], trainable[row, column]);
            }
        }
    }

    [Fact]
    public void NoDataOutputs_IsThePrediction()
    {
        var outputs = new List<Tensor<double>> { Filled(new[] { 2, 2 }, 1), Filled(new[] { 2, 3 }, 10) };
        var prediction = CvTensorOps<double>.ConcatenateOutputs(outputs);
        var trainable = CvTensorOps<double>.ConcatenateTrainableOutputs(outputs, Array.Empty<int>(), null);
        Assert.Equal(prediction.ToArray(), trainable.ToArray());
    }
}
