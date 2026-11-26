using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedTrainingSerial")]
public sealed class NormedConvNativeBiasTests
{
    private static void VerifyCpu<T>(int groups, bool transposed)
    {
        var prior = AiDotNetEngine.Current;
        try { AiDotNetEngine.Current = new AiDotNet.Tensors.Engines.CpuEngine(); Verify<T>(groups, transposed); }
        finally { AiDotNetEngine.Current = prior; }
    }

    private static void Verify<T>(int groups, bool transposed)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        using var layer = new NormedConv1DLayer<T>(2, 2, 3, transposed ? 2 : 1, 1, groups, 1,
            transposed, ConvolutionNormalization.None, true);
        var parameters = AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(new List<ILayer<T>> { layer });
        foreach (var parameter in parameters)
        {
            parameter.AsWritableSpan().Fill(ops.Zero);
            engine.InvalidatePersistentTensor(parameter);
        }
        var bias = parameters.Single(parameter => parameter.Rank == 1);
        bias[0] = ops.FromDouble(-.25);
        bias[1] = ops.FromDouble(.5);
        var input = new Tensor<T>(new[] { 2, 2, 5 });
        for (int i = 0; i < input.Length; i++) input[i] = ops.FromDouble(Math.Sin(i * .3));
        using var tape = new GradientTape<T>();
        var output = layer.Forward(input);
        int time = transposed ? 9 : 5;
        Assert.Equal(new[] { 2, 2, time }, output.Shape.ToArray());
        var seed = new Tensor<T>(output.Shape.ToArray());
        for (int i = 0; i < seed.Length; i++) seed[i] = ops.FromDouble((i % 7 - 3) * .125);
        var loss = engine.ReduceSum(engine.TensorMultiply(output, seed));
        var gradient = tape.ComputeGradients(loss, new[] { bias })[bias].ToArray();
        var actual = output.ToArray();
        var expectedBias = new double[2];
        for (int i = 0; i < output.Length; i++)
        {
            int channel = i / time % 2;
            Assert.Equal(channel == 0 ? -.25 : .5, ops.ToDouble(actual[i]), 6);
            expectedBias[channel] += ops.ToDouble(seed[i]);
        }
        for (int channel = 0; channel < 2; channel++) Assert.Equal(expectedBias[channel], ops.ToDouble(gradient[channel]), 6);
    }

    [Theory]
    [InlineData(1, false)] [InlineData(2, false)]
    [InlineData(1, true)] [InlineData(2, true)]
    public void Float_bias_values_and_batch_time_gradients_match_channel_reference(int groups, bool transposed)
        => VerifyCpu<float>(groups, transposed);

    [Theory]
    [InlineData(1, false)] [InlineData(2, false)]
    [InlineData(1, true)] [InlineData(2, true)]
    public void Double_bias_values_and_batch_time_gradients_match_channel_reference(int groups, bool transposed)
        => VerifyCpu<double>(groups, transposed);

    [SkippableTheory]
    [InlineData(1, false)] [InlineData(2, false)]
    [InlineData(1, true)] [InlineData(2, true)]
    public void Gpu_bias_values_and_batch_time_gradients_match_channel_reference(int groups, bool transposed)
    {
        using var gpu = new AiDotNet.Tensors.Engines.DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No direct GPU backend available.");
        var prior = AiDotNetEngine.Current;
        try { AiDotNetEngine.Current = gpu; Verify<float>(groups, transposed); }
        finally { AiDotNetEngine.Current = prior; }
    }
}
