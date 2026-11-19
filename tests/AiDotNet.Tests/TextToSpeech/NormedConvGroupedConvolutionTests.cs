using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedTrainingSerial")]
public sealed class NormedConvGroupedConvolutionTests
{
    private sealed record Sample(float[] Values, bool LearnedState);

    private static Sample[] Run(IEngine engine, bool grouped, bool compile)
    {
        string? previous = Environment.GetEnvironmentVariable("AIDOTNET_GROUPED_CONV2D");
        Environment.SetEnvironmentVariable("AIDOTNET_GROUPED_CONV2D", grouped ? "1" : "0");
        try
        {
            using var layer = new NormedConv1DLayer<float>(128, 256, 41, 2, 1, 16, 20, false, ConvolutionNormalization.Weight);
            layer.LoadTorchWeights(Enumerable.Range(0, 256 * 8 * 41).Select(i => .05 * Math.Sin(i * .07)).ToArray(),
                Enumerable.Range(0, 256).Select(i => .7 + .01 * (i % 9)).ToArray(), new double[256]);
            layer.SetTrainingMode(true);
            using var input = new Tensor<float>(Enumerable.Range(0, 2 * 128 * 65).Select(i => (float)(.1 * Math.Sin(i * .13))).ToArray(),
                new[] { 2, 128, 65 });
            var parameters = layer.GetTrainableParameters().ToArray();
            Tensor<float> Objective()
            {
                var output = layer.Forward(input);
                return engine.ReduceSum(engine.TensorMultiply(output, output), null);
            }
            if (!compile)
            {
                using var tape = new GradientTape<float>(); tape.BindEngineIfUnset(engine);
                var loss = Objective(); var gradients = tape.ComputeGradients(loss, parameters);
                return new[] { new Sample(loss.ToArray(), false) }
                    .Concat(parameters.Select(parameter => new Sample(gradients[parameter].ToArray(), false))).ToArray();
            }
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(parameters)) plan = scope.CompileTraining(parameters, Objective());
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: .0001);
                var values = new List<Sample>();
                for (int step = 0; step < 3; ++step)
                {
                    input[0] = .1f + step * .01f;
                    values.Add(new Sample(plan.Step().ToArray(), false));
                    values.AddRange(parameters.Select(parameter => new Sample(parameter.Grad?.ToArray()
                        ?? throw new InvalidOperationException("A normalized grouped-convolution parameter gradient was not published."), false)));
                    values.AddRange(parameters.Select(parameter => new Sample(parameter.ToArray(), true)));
                }
                return values.ToArray();
            }
        }
        finally { Environment.SetEnvironmentVariable("AIDOTNET_GROUPED_CONV2D", previous); }
    }

    private static void Compare(IEngine engine, bool compile, double tolerance)
    {
        var previous = AiDotNetEngine.Current; AiDotNetEngine.Current = engine;
        try
        {
            var expected = Run(engine, false, compile); var actual = Run(engine, true, compile);
            Assert.Equal(expected.Length, actual.Length);
            for (int tensor = 0; tensor < expected.Length; ++tensor)
            {
                Assert.Equal(expected[tensor].LearnedState, actual[tensor].LearnedState);
                var eValues = expected[tensor].Values; var aValues = actual[tensor].Values;
                Assert.Equal(eValues.Length, aValues.Length);
                double squaredError = 0, referenceNorm = 0, actualNorm = 0, dot = 0;
                for (int element = 0; element < eValues.Length; ++element)
                {
                    double e = eValues[element], a = aValues[element];
                    Assert.True(!double.IsNaN(a) && !double.IsInfinity(a));
                    if (!expected[tensor].LearnedState)
                        Assert.True(Math.Abs(e - a) <= tolerance * (1 + Math.Abs(e)),
                        $"Tensor {tensor}, element {element}: expected {e:R}, actual {a:R}.");
                    squaredError += (e - a) * (e - a); referenceNorm += e * e; actualNorm += a * a; dot += e * a;
                }
                if (expected[tensor].LearnedState)
                {
                    // The user-approved learned-state policy; raw outputs and gradients retain the strict checks above.
                    double relative = Math.Sqrt(squaredError / Math.Max(referenceNorm, 1e-30));
                    double cosine = referenceNorm == 0 || actualNorm == 0 ? (squaredError == 0 ? 1 : 0)
                        : dot / Math.Sqrt(referenceNorm * actualNorm);
                    Assert.True(relative <= (engine.SupportsGpu ? .003 : .001), $"Learned-state relative L2 {relative:R}.");
                    Assert.True(cosine >= (engine.SupportsGpu ? .995 : .999), $"Learned-state cosine {cosine:R}.");
                }
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Theory] [InlineData(false)] [InlineData(true)]
    public void Cpu_full_width_1x41_weightnorm_gradients_and_updates_match_split(bool compile)
        => Compare(new CpuEngine(), compile, 2e-5);

    [SkippableTheory] [InlineData(false)] [InlineData(true)]
    public void Cuda_full_width_1x41_weightnorm_gradients_and_updates_match_split(bool compile)
    {
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.IsGpuAvailable && engine.GetBackend() is CudaBackend, "CUDA unavailable.");
        Compare(engine, compile, 2e-4);
    }
}
