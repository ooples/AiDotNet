using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class NormedConvBroadcastGradientTests
{
    [Theory]
    [InlineData(ConvolutionNormalization.None, false)]
    [InlineData(ConvolutionNormalization.Weight, false)]
    [InlineData(ConvolutionNormalization.Spectral, false)]
    [InlineData(ConvolutionNormalization.None, true)]
    [InlineData(ConvolutionNormalization.Weight, true)]
    [InlineData(ConvolutionNormalization.Spectral, true)]
    public void BroadcastNormalizationAndBiasMatchFiniteDifferences(
        ConvolutionNormalization normalization, bool transposed)
    {
        var engine = AiDotNetEngine.Current;
        var layer = new NormedConv1DLayer<double>(2, 3, 3, 1, 1, 1, 1, transposed, normalization, true);
        // Hold spectral power-iteration state fixed while differentiating the convolution.
        layer.SetTrainingMode(false);
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(new List<ILayer<double>> { layer });
        var input = new Tensor<double>(new[] { 2, 2, 5 }, new Vector<double>(
            Enumerable.Range(0, 20).Select(i => Math.Sin(i * .3 + .2)).ToArray()));
        Tensor<double> Loss()
        {
            var output = layer.Forward(input);
            return engine.ReduceMean(engine.TensorMultiply(output, output), new[] { 0, 1, 2 }, false);
        }

        Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new GradientTape<double>())
            gradients = tape.ComputeGradients(Loss(), sources: parameters);

        const double epsilon = 1e-6;
        using var noGrad = new NoGradScope<double>();
        foreach (var parameter in parameters)
        {
            var analytical = gradients[parameter].ToArray();
            for (int i = 0; i < parameter.Length; i++)
            {
                double original = parameter[i];
                double Evaluate(double value)
                {
                    parameter[i] = value;
                    engine.InvalidatePersistentTensor(parameter);
                    return Loss()[0];
                }
                double numerical;
                try { numerical = (Evaluate(original + epsilon) - Evaluate(original - epsilon)) / (2 * epsilon); }
                finally { parameter[i] = original; engine.InvalidatePersistentTensor(parameter); }
                Assert.True(Math.Abs(analytical[i] - numerical) <= 2e-5 * Math.Max(1, Math.Abs(numerical)),
                    $"{normalization}, transposed={transposed}, index={i}: gradient {analytical[i]} != {numerical}");
            }
        }
    }
}
