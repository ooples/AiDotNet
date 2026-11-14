#if NET5_0_OR_GREATER
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public partial class Elementwise01NumericalRegressionTests
{
    private static void CheckDecimalSoftmax<TModel>() where TModel : NeuralNetworkBase<decimal>
    {
        var model = NumericModel<decimal, TModel>();
        var result = Assert.IsType<List<decimal>>(Invoke(typeof(TModel), model, "Softmax", new List<decimal> { 0m, 1m }));
        Assert.InRange(result[0], 0.26894142136999m, 0.26894142137m);
        Assert.InRange(result[1], 0.73105857863m, 0.73105857863001m);
    }

    private static void CheckHalfSoftmax<TModel>() where TModel : NeuralNetworkBase<Half>
    {
        var model = NumericModel<Half, TModel>();
        var result = Assert.IsType<List<Half>>(Invoke(typeof(TModel), model, "Softmax",
            Enumerable.Repeat((Half)0, 70000).ToList()));
        Assert.All(result, value => Assert.Equal((Half)(1.0 / 70000), value));
    }

    private static void CheckNaNSoftmax<TModel>() where TModel : NeuralNetworkBase<double>
    {
        var model = NumericModel<double, TModel>();
        foreach (var input in new[] { new List<double> { double.NaN, 1 }, new List<double> { 1, double.NaN }, new List<double> { double.NaN } })
        {
            var result = Assert.IsType<List<double>>(Invoke(typeof(TModel), model, "Softmax", input));
            Assert.All(result, value => Assert.True(double.IsNaN(value)));
        }
    }

    [Fact]
    public void MultimodalSoftmax_DecimalFiniteInputsDoNotRequireInfinityConversion()
    {
        CheckDecimalSoftmax<Gpt4VisionNeuralNetwork<decimal>>();
        CheckDecimalSoftmax<ImageBindNeuralNetwork<decimal>>();
        CheckDecimalSoftmax<LLaVANeuralNetwork<decimal>>();
    }

    [Fact]
    public void MultimodalSoftmax_HalfLargeClassCountsPreserveUniformProbabilities()
    {
        CheckHalfSoftmax<Gpt4VisionNeuralNetwork<Half>>();
        CheckHalfSoftmax<ImageBindNeuralNetwork<Half>>();
        CheckHalfSoftmax<LLaVANeuralNetwork<Half>>();
        var clip = NumericModel<Half, ClipNeuralNetwork<Half>>();
        var probabilities = Assert.IsType<Vector<Half>>(Invoke(typeof(ClipNeuralNetwork<Half>), clip,
            "ScaledSoftmax", new Vector<Half>(70000)));
        Assert.All(probabilities.ToArray(), value => Assert.Equal((Half)(1.0 / 70000), value));
    }

    [Fact]
    public void MultimodalSoftmax_PreservesNaNPropagation()
    {
        CheckNaNSoftmax<Gpt4VisionNeuralNetwork<double>>();
        CheckNaNSoftmax<ImageBindNeuralNetwork<double>>();
        CheckNaNSoftmax<LLaVANeuralNetwork<double>>();
    }

    [Fact]
    public void VisionNormalization_DecimalAndFullScaleHalfPreserveRange()
    {
        var model = NumericModel<decimal, Gpt4VisionNeuralNetwork<decimal>>();
        var result = Assert.IsType<Tensor<decimal>>(Invoke(typeof(Gpt4VisionNeuralNetwork<decimal>), model,
            "NormalizeTensor", new Tensor<decimal>(new[] { -2m, 0m, 2m }, new[] { 1, 3 })));
        Assert.Equal(new[] { 0m, 0.5m, 1m }, result.ToArray());
        var halfModel = NumericModel<Half, Gpt4VisionNeuralNetwork<Half>>();
        var halfResult = Assert.IsType<Tensor<Half>>(Invoke(typeof(Gpt4VisionNeuralNetwork<Half>), halfModel,
            "NormalizeTensor", new Tensor<Half>(new[] { (Half)(-60000), (Half)0, (Half)60000 }, new[] { 1, 3 })));
        Assert.Equal(new[] { (Half)0, (Half)0.5, (Half)1 }, halfResult.ToArray());
    }
}
#endif
