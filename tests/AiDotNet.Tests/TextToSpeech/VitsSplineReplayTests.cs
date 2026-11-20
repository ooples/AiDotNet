using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.EndToEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("EngineCurrentGlobalState")]
public sealed class VitsSplineReplayTests
{
    private const double Bound = 5;
    private const double LogWeight = 0.17;

    // Independent scalar reference. Inversion uses bisection rather than the production quadratic solver.
    private static (double Value, double LogDet) Forward(double x, double[] p)
    {
        if (x < -Bound || x > Bound) return (x, 0);
        double[] Knots(int offset)
        {
            double maximum = p.Skip(offset).Take(3).Max();
            var probabilities = p.Skip(offset).Take(3).Select(value => Math.Exp(value - maximum)).ToArray();
            double total = probabilities.Sum();
            var knots = new double[4];
            knots[0] = -Bound;
            for (int i = 0; i < 3; i++) knots[i + 1] = knots[i] + 2 * Bound * (0.001 + 0.997 * probabilities[i] / total);
            return knots;
        }
        var xs = Knots(1);
        var ys = Knots(4);
        var derivatives = new[] { 1.0, 0.001 + Math.Log(1 + Math.Exp(p[7])), 0.001 + Math.Log(1 + Math.Exp(p[8])), 1.0 };
        int bin = 0;
        while (bin < 2 && x >= xs[bin + 1]) bin++;
        double width = xs[bin + 1] - xs[bin], height = ys[bin + 1] - ys[bin];
        double slope = height / width, theta = (x - xs[bin]) / width;
        double complement = 1 - theta, product = theta * complement;
        double denominator = slope + (derivatives[bin] + derivatives[bin + 1] - 2 * slope) * product;
        double y = ys[bin] + height * (slope * theta * theta + derivatives[bin] * product) / denominator;
        double derivative = slope * slope * (derivatives[bin + 1] * theta * theta + 2 * slope * product
            + derivatives[bin] * complement * complement) / (denominator * denominator);
        return (y, Math.Log(derivative));
    }

    private static double Objective(double[] point, bool inverse)
    {
        if (!inverse)
        {
            var result = Forward(point[0], point);
            return result.Value + LogWeight * result.LogDet;
        }
        if (point[0] < -Bound || point[0] > Bound) return point[0];
        double lo = -Bound, hi = Bound;
        for (int iteration = 0; iteration < 100; iteration++)
        {
            double mid = (lo + hi) / 2;
            if (Forward(mid, point).Value < point[0]) lo = mid; else hi = mid;
        }
        double root = (lo + hi) / 2;
        return root - LogWeight * Forward(root, point).LogDet;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Compiled_spline_replays_bin_and_tail_crossings_with_correct_input_and_parameter_gradients(bool inverse)
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            var engine = new CpuEngine();
            AiDotNetEngine.Current = engine;
            var x = new Tensor<double>(new[] { -4.3 }, new[] { 1 });
            var widths = new Tensor<double>(new[] { -0.3, 0.8, -0.6 }, new[] { 1, 3 });
            var heights = new Tensor<double>(new[] { 0.4, -0.2, 0.1 }, new[] { 1, 3 });
            var derivatives = new Tensor<double>(new[] { -0.3, 0.9 }, new[] { 1, 2 });
            var parameters = new[] { x, widths, heights, derivatives };
            CompiledTrainingPlan<double> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                var result = RationalQuadraticSpline.Apply(engine, x, widths, heights, derivatives, inverse, Bound);
                var loss = engine.TensorAdd(result.Outputs, engine.TensorMultiplyScalar(result.LogAbsDet, LogWeight));
                plan = scope.CompileTraining(parameters, engine.ReduceSum(loss, null));
            }
            using (plan)
            {
                var inputs = new[] { -7.0, -4.3, -0.2, 2.4, 7.0, -3.0 };
                for (int step = 0; step < inputs.Length; step++)
                {
                    var point = new[] { inputs[step], -0.3 + 0.1 * step, 0.8 - 0.2 * step, -0.6 + 0.15 * step,
                        0.4 - 0.1 * step, -0.2 + 0.3 * step, 0.1 - 0.05 * step, -0.3 + 0.2 * step, 0.9 - 0.1 * step };
                    int offset = 0;
                    foreach (var parameter in parameters)
                    {
                        for (int i = 0; i < parameter.Length; i++) parameter.SetFlat(i, point[offset++]);
                        parameter.IncrementVersion();
                    }
                    Assert.Equal(Objective(point, inverse), plan.Step().ToArray()[0], 9);
                    offset = 0;
                    for (int tensor = 0; tensor < parameters.Length; tensor++)
                    {
                        var actual = plan.Gradients[tensor].ToArray();
                        for (int index = 0; index < actual.Length; index++, offset++)
                        {
                            const double epsilon = 1e-5;
                            double saved = point[offset];
                            point[offset] = saved + epsilon;
                            double plus = Objective(point, inverse);
                            point[offset] = saved - epsilon;
                            double minus = Objective(point, inverse);
                            point[offset] = saved;
                            double expected = (plus - minus) / (2 * epsilon);
                            Assert.True(Math.Abs(actual[index] - expected) < 2e-5,
                                $"inverse={inverse}, step={step}, parameter={offset}: expected {expected:R}, got {actual[index]:R}");
                            Assert.Equal(saved, parameters[tensor].GetFlat(index));
                        }
                    }
                }
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}
