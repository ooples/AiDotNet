using System.Reflection;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class NormedConvSpectralStateTests
{
    [Fact]
    public void FloatPowerIterationAndSavedGradientMatchReference() => Check<float>(3e-5);

    [Fact]
    public void DoublePowerIterationAndSavedGradientMatchReference() => Check<double>(1e-9);

    [Fact]
    public void CompiledTrainingReplaysPowerIterationAndMatchesEagerUpdates()
    {
        var engine = AiDotNetEngine.Current;
        NormedConv1DLayer<double> Create() => new(2, 2, 3, 1, 1, 1, 1, false, ConvolutionNormalization.Spectral);
        var eager = Create();
        var compiled = Create();
        eager.SetTrainingMode(true);
        compiled.SetTrainingMode(true);
        Tensor<double> Field(NormedConv1DLayer<double> layer, string name) => Assert.IsType<Tensor<double>>(
            Assert.IsAssignableFrom<FieldInfo>(typeof(NormedConv1DLayer<double>).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)).GetValue(layer));
        foreach (string name in new[] { "_direction", "_u", "_v" })
        {
            var a = Field(eager, name);
            var b = Field(compiled, name);
            for (int i = 0; i < a.Length; i++) a[i] = b[i] = .1 + .03 * (i + 1);
            engine.InvalidatePersistentTensor(a);
            engine.InvalidatePersistentTensor(b);
        }
        var eagerDirection = Field(eager, "_direction");
        var compiledDirection = Field(compiled, "_direction");
        Tensor<double> Loss(NormedConv1DLayer<double> layer) => engine.ReduceSum(layer.Kernel(), null, false);
        CompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.EnableTraining(new[] { compiledDirection }))
            plan = scope.CompileTraining(new[] { compiledDirection }, Loss(compiled));
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: .01);
            for (int step = 0; step < 3; step++)
            {
                double expected;
                Dictionary<Tensor<double>, Tensor<double>> gradients;
                using (var tape = new GradientTape<double>())
                {
                    var loss = Loss(eager);
                    expected = loss[0];
                    gradients = tape.ComputeGradients(loss, sources: new[] { eagerDirection });
                }
                var gradient = gradients[eagerDirection].ToArray();
                for (int i = 0; i < eagerDirection.Length; i++) eagerDirection[i] -= .01 * gradient[i];
                engine.InvalidatePersistentTensor(eagerDirection);
                Assert.InRange(Math.Abs(plan.Step()[0] - expected), 0, 1e-8);
                foreach (string name in new[] { "_direction", "_u", "_v" })
                {
                    var a = Field(eager, name).ToArray();
                    var b = Field(compiled, name).ToArray();
                    for (int i = 0; i < a.Length; i++) Assert.InRange(Math.Abs(a[i] - b[i]), 0, 1e-8);
                }
            }
        }
    }

    private static void Check<T>(double tolerance)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        var layer = new NormedConv1DLayer<T>(2, 2, 3, 1, 1, 1, 1, false, ConvolutionNormalization.Spectral);
        layer.SetTrainingMode(true);
        Tensor<T> Field(string name) => Assert.IsType<Tensor<T>>(
            Assert.IsAssignableFrom<FieldInfo>(typeof(NormedConv1DLayer<T>).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)).GetValue(layer));
        var direction = Field("_direction");
        var u = Field("_u");
        var v = Field("_v");
        var weights = Enumerable.Range(0, 12).Select(i => .2 + .1 * Math.Sin(i * .7)).ToArray();
        var initialU = new[] { .6, .8 };
        for (int i = 0; i < weights.Length; i++) direction[i] = ops.FromDouble(weights[i]);
        for (int i = 0; i < initialU.Length; i++) u[i] = ops.FromDouble(initialU[i]);
        for (int i = 0; i < v.Length; i++) v[i] = ops.FromDouble(.1 * (i + 1));
        engine.InvalidatePersistentTensor(direction);
        engine.InvalidatePersistentTensor(u);
        engine.InvalidatePersistentTensor(v);

        double[] Normalize(double[] values)
        {
            double norm = Math.Max(Math.Sqrt(values.Sum(x => x * x)), 1e-12);
            return values.Select(x => x / norm).ToArray();
        }
        // PyTorch's parametrization updates the left vector from the OLD right vector, then the right from NEW left.
        // Deliberately use inconsistent initial u/v so a reversed iteration cannot pass from a converged initial state.
        var initialV = Enumerable.Range(0, 6).Select(i => .1 * (i + 1)).ToArray();
        var expectedU = Normalize(Enumerable.Range(0, 2)
            .Select(r => Enumerable.Range(0, 6).Sum(c => weights[r * 6 + c] * initialV[c])).ToArray());
        var expectedV = Normalize(Enumerable.Range(0, 6)
            .Select(c => weights[c] * expectedU[0] + weights[6 + c] * expectedU[1]).ToArray());
        double sigma = Enumerable.Range(0, 12).Sum(i => expectedU[i / 6] * weights[i] * expectedV[i % 6]);
        var upstreamValues = Enumerable.Range(0, 12).Select(i => .1 + .03 * i).ToArray();
        var upstream = new Tensor<T>(direction.Shape.ToArray(), new Vector<T>(upstreamValues.Select(ops.FromDouble)));
        using var tape = new GradientTape<T>();
        var kernel = layer.Kernel();
        for (int i = 0; i < 12; i++)
            Assert.InRange(Math.Abs(ops.ToDouble(kernel[i]) - weights[i] / sigma), 0, tolerance);
        for (int i = 0; i < u.Length; i++)
            Assert.InRange(Math.Abs(ops.ToDouble(u[i]) - expectedU[i]), 0, tolerance);
        for (int i = 0; i < v.Length; i++)
            Assert.InRange(Math.Abs(ops.ToDouble(v[i]) - expectedV[i]), 0, tolerance);

        var loss = engine.ReduceSum(engine.TensorMultiply(kernel, upstream), null, false);
        // GAN discriminators run more than once before backward. The first forward must retain its own u/v.
        using (new NoGradScope<T>()) _ = layer.Kernel();
        var gradients = tape.ComputeGradients(loss, sources: new[] { direction });
        double dot = Enumerable.Range(0, 12).Sum(i => upstreamValues[i] * weights[i]);
        for (int i = 0; i < 12; i++)
        {
            double expected = upstreamValues[i] / sigma - dot / (sigma * sigma) * expectedU[i / 6] * expectedV[i % 6];
            Assert.InRange(Math.Abs(ops.ToDouble(gradients[direction][i]) - expected), 0, tolerance);
        }
    }
}
