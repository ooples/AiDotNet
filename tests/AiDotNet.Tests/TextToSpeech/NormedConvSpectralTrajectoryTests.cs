using System.Reflection;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class NormedConvSpectralTrajectoryTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FloatTwoForwardsAndAdamWMatchAnalyticTrajectory(bool compiled) => Check<float>(compiled, 3e-5);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DoubleTwoForwardsAndAdamWMatchAnalyticTrajectory(bool compiled) => Check<double>(compiled, 1e-9);

    private static void Check<T>(bool compiled, double tolerance)
    {
        var previous = AiDotNetEngine.Current;
        var engine = new CpuEngine();
        AiDotNetEngine.Current = engine;
        try
        {
            var ops = MathHelper.GetNumericOperations<T>();
            var layer = new NormedConv1DLayer<T>(2, 3, 3, 1, 1, 1, 1, false, ConvolutionNormalization.Spectral);
            layer.SetTrainingMode(true);
            Tensor<T> Field(string name) => (Tensor<T>)typeof(NormedConv1DLayer<T>)
                .GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(layer)!;
            var direction = Field("_direction");
            var bias = Field("_bias");
            var u = Field("_u");
            var v = Field("_v");
            var state = new ReferenceState();
            void Set(Tensor<T> tensor, double[] values)
            {
                for (int i = 0; i < values.Length; i++) tensor[i] = ops.FromDouble(values[i]);
                engine.InvalidatePersistentTensor(tensor);
            }
            Set(direction, state.Weight);
            Set(bias, state.Bias);
            Set(u, state.U);
            Set(v, state.V);
            var realValues = Enumerable.Range(0, 10).Select(i => .4 * Math.Sin(.8 * i) - .15).ToArray();
            var fakeValues = Enumerable.Range(0, 10).Select(i => .3 * Math.Cos(.6 * i) + .07).ToArray();
            var real = new Tensor<T>(new[] { 1, 2, 5 }, new Vector<T>(realValues.Select(ops.FromDouble)));
            var fake = new Tensor<T>(new[] { 1, 2, 5 }, new Vector<T>(fakeValues.Select(ops.FromDouble)));
            var parameters = new[] { direction, bias };
            Tensor<T> Loss()
            {
                var realOutput = layer.Forward(real);
                var fakeOutput = layer.Forward(fake);
                var realError = engine.TensorAddScalar(realOutput, ops.FromDouble(-1));
                return engine.TensorAdd(
                    engine.ReduceMean(engine.TensorMultiply(realError, realError), null, false),
                    engine.ReduceMean(engine.TensorMultiply(fakeOutput, fakeOutput), null, false));
            }
            CompiledTrainingPlan<T>? plan = null;
            if (compiled)
            {
                using var scope = GraphMode.EnableTraining(parameters);
                plan = scope.CompileTraining(parameters, Loss());
                plan.ConfigureOptimizer(OptimizerType.AdamW, .0002, beta1: .8, beta2: .99, eps: 1e-8, weightDecay: .01);
            }
            using (plan)
            {
                var actualMoments = new Moments(21);
                for (int step = 1; step <= 3; step++)
                {
                    var expected = state.Step(realValues, fakeValues, step);
                    double actualLoss;
                    double[] actualDirectionGradient, actualBiasGradient;
                    if (plan is not null)
                    {
                        actualLoss = ops.ToDouble(plan.Step()[0]);
                        actualDirectionGradient = plan.Gradients[0].ToArray().Select(ops.ToDouble).ToArray();
                        actualBiasGradient = plan.Gradients[1].ToArray().Select(ops.ToDouble).ToArray();
                    }
                    else
                    {
                        using var tape = new GradientTape<T>();
                        var loss = Loss();
                        actualLoss = ops.ToDouble(loss[0]);
                        var gradients = tape.ComputeGradients(loss, sources: parameters);
                        actualDirectionGradient = gradients[direction].ToArray().Select(ops.ToDouble).ToArray();
                        actualBiasGradient = gradients[bias].ToArray().Select(ops.ToDouble).ToArray();
                        var values = direction.ToArray().Select(ops.ToDouble).Concat(bias.ToArray().Select(ops.ToDouble)).ToArray();
                        actualMoments.Update(values, actualDirectionGradient.Concat(actualBiasGradient).ToArray(), step);
                        Set(direction, values.Take(18).ToArray());
                        Set(bias, values.Skip(18).ToArray());
                    }
                    Close(new[] { expected.Loss }, new[] { actualLoss }, tolerance, $"step{step}/loss");
                    Close(expected.Gradient.Take(18).ToArray(), actualDirectionGradient, tolerance, $"step{step}/direction-gradient");
                    Close(expected.Gradient.Skip(18).ToArray(), actualBiasGradient, tolerance, $"step{step}/bias-gradient");
                    foreach (var (name, values) in new[] { ("_direction", state.Weight), ("_bias", state.Bias), ("_u", state.U), ("_v", state.V) })
                        Close(values, Field(name).ToArray().Select(ops.ToDouble).ToArray(), tolerance, $"step{step}/{name}");
                }
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    private static void Close(double[] expected, double[] actual, double tolerance, string phase)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(double.IsFinite(actual[i]) && Math.Abs(expected[i] - actual[i]) <= tolerance,
                $"{phase}[{i}]: expected {expected[i]:R}, actual {actual[i]:R}, tolerance {tolerance:R}");
    }

    // Independent scalar convolution and spectral VJP. Each forward snapshots its own power-iteration vectors.
    private sealed class ReferenceState
    {
        internal double[] Weight = Enumerable.Range(0, 18).Select(i => .2 * Math.Sin(.7 * i) + .12 * Math.Cos(.3 * i)).ToArray();
        internal double[] Bias = { .03, -.02, .04 };
        internal double[] U = Normalize(new[] { .4, -.7, .5 });
        internal double[] V = Normalize(new[] { -.2, .3, .1, -.6, .7, .4 });
        private readonly Moments _moments = new(21);

        internal (double Loss, double[] Gradient) Step(double[] real, double[] fake, int step)
        {
            var gradient = new double[21];
            double loss = Forward(real, 1, gradient) + Forward(fake, 0, gradient);
            var values = Weight.Concat(Bias).ToArray();
            _moments.Update(values, gradient, step);
            Weight = values.Take(18).ToArray();
            Bias = values.Skip(18).ToArray();
            return (loss, gradient);
        }

        private double Forward(double[] input, double target, double[] gradient)
        {
            V = Normalize(Enumerable.Range(0, 6).Select(c => Enumerable.Range(0, 3).Sum(r => U[r] * Weight[r * 6 + c])).ToArray());
            U = Normalize(Enumerable.Range(0, 3).Select(r => Enumerable.Range(0, 6).Sum(c => Weight[r * 6 + c] * V[c])).ToArray());
            double sigma = Enumerable.Range(0, 18).Sum(i => U[i / 6] * Weight[i] * V[i % 6]);
            var kernelGradient = new double[18];
            double loss = 0;
            for (int r = 0; r < 3; r++)
                for (int t = 0; t < 5; t++)
                {
                    double output = Bias[r];
                    for (int c = 0; c < 2; c++)
                        for (int k = 0; k < 3; k++)
                        {
                            int at = t + k - 1;
                            if (at >= 0 && at < 5) output += input[c * 5 + at] * Weight[r * 6 + c * 3 + k] / sigma;
                        }
                    double error = output - target;
                    loss += error * error / 15;
                    double upstream = 2 * error / 15;
                    gradient[18 + r] += upstream;
                    for (int c = 0; c < 2; c++)
                        for (int k = 0; k < 3; k++)
                        {
                            int at = t + k - 1;
                            if (at >= 0 && at < 5) kernelGradient[r * 6 + c * 3 + k] += upstream * input[c * 5 + at];
                        }
                }
            double dot = Enumerable.Range(0, 18).Sum(i => kernelGradient[i] * Weight[i]);
            for (int i = 0; i < 18; i++) gradient[i] += kernelGradient[i] / sigma - dot / (sigma * sigma) * U[i / 6] * V[i % 6];
            return loss;
        }
    }

    private static double[] Normalize(double[] value)
    {
        double norm = Math.Max(Math.Sqrt(value.Sum(x => x * x)), 1e-12);
        return value.Select(x => x / norm).ToArray();
    }

    private sealed class Moments
    {
        private readonly double[] _first, _second;
        internal Moments(int length) { _first = new double[length]; _second = new double[length]; }
        internal void Update(double[] values, double[] gradient, int step)
        {
            for (int i = 0; i < values.Length; i++)
            {
                _first[i] = .8 * _first[i] + .2 * gradient[i];
                _second[i] = .99 * _second[i] + .01 * gradient[i] * gradient[i];
                values[i] = values[i] * (1 - .0002 * .01) - .0002 * (_first[i] / (1 - Math.Pow(.8, step)))
                    / (Math.Sqrt(_second[i] / (1 - Math.Pow(.99, step))) + 1e-8);
            }
        }
    }
}
