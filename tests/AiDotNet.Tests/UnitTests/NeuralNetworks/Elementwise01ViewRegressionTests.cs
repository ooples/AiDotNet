#if NET5_0_OR_GREATER
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public partial class Elementwise01NumericalRegressionTests
{
    // Rows 2 and 3 of a [4, 2] tensor as a view: storage offset 4, not 0.
    private static Tensor<float> LastTwoRowsView(params float[] values)
    {
        var data = new Tensor<float>(values, new[] { 4, 2 });
        return AiDotNetEngine.Current.TensorNarrow(data, 0, 2, 2);
    }

    [Fact]
    public void DeepBeliefNormalization_ScalesTheRowsOfASlicedViewNotTheStartOfItsStorage()
    {
        // Tensor.Cast reads a view from the start of its storage, so the cast inside the min-max
        // scaling returned rows 0-1 (100..400) scaled by the min and range of rows 2-3.
        var view = LastTwoRowsView(100f, 200f, 300f, 400f, 2f, 4f, 6f, 10f);
        var model = NumericModel<float, DeepBeliefNetwork<float>>();

        var scaled = (Tensor<float>)Invoke(typeof(DeepBeliefNetwork<float>), model, "NormalizeBernoulliVisibleInput", view)!;

        // (x - 2) / (10 - 2) for x = 2, 4, 6, 10
        Assert.Equal(new[] { 0f, 0.25f, 0.5f, 1f }, scaled.ToArray());
    }

    [Fact]
    public void Gpt4VisionNormalization_ScalesTheRowsOfASlicedViewNotTheStartOfItsStorage()
    {
        var view = LastTwoRowsView(100f, 200f, 300f, 400f, 2f, 4f, 6f, 10f);
        var model = NumericModel<float, Gpt4VisionNeuralNetwork<float>>();

        var scaled = (Tensor<float>)Invoke(typeof(Gpt4VisionNeuralNetwork<float>), model, "NormalizeTensor", view)!;

        Assert.Equal(new[] { 0f, 0.25f, 0.5f, 1f }, scaled.ToArray());
    }

    [Theory]
    [InlineData(7)]
    [InlineData(50)]
    [InlineData(126)]
    public void GraFPrintCoordinates_AreTheDoubleQuotientRoundedOnceToFloat(int size)
    {
        // time = w / (W - 1) and frequency = h / (H - 1), each one double division rounded to float.
        // Multiplying by a float reciprocal is an ulp off for about one coordinate in five:
        // 5 / 6 came out as 0.8333334 where the quotient rounds to 0.8333333.
        var input = new Tensor<float>(new[] { 1, size, size });
        for (int i = 0; i < input.Length; i++) input.SetFlat(i, i % 17);

        var output = new GraFPrintCoordinateAugmentationLayer<float>().Forward(input);

        Assert.Equal(new[] { 3, size, size }, output.Shape.ToArray());
        for (int h = 0; h < size; h++)
        {
            for (int w = 0; w < size; w++)
            {
                Assert.Equal((float)(w / (double)(size - 1)), output[0, h, w]);
                Assert.Equal((float)(h / (double)(size - 1)), output[1, h, w]);
            }
        }
    }
}
#endif
