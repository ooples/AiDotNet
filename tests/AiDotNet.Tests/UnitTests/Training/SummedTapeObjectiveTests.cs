using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

public class SummedTapeObjectiveTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SeparateTapesPreserveSummedGradientsClippingAndOneUpdate(bool clip)
    {
        static (double[] Losses, double[] Weights, int Updates) Run(bool split, bool clip)
        {
            var engine = AiDotNetEngine.Current;
            var parameter = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 2.0 }));
            var diagnostic = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 5.0 }));
            var disconnected = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 9.0 }));
            var parameters = new[] { parameter, disconnected };
            Func<Tensor<double>> square = () => engine.TensorAdd(engine.TensorMultiply(parameter, parameter),
                engine.TensorMultiply(diagnostic, diagnostic));
            Func<Tensor<double>> linear = () => engine.TensorMultiplyScalar(parameter, 3.0);
            Tensor<double> Combined() => engine.TensorAdd(square(), linear());
            int updates = 0;
            int published = 0, regularized = 0, prepared = 0;
            var losses = new List<double>();
            var weights = new List<double>();
            var hooks = new TapeTrainingStepper<double>.EagerStepHooks
            {
                GradientSources = current => current.Concat(new[] { diagnostic }).ToArray(),
                OnTapeGradients = gradients => Assert.Equal(10.0, gradients[diagnostic][0], 12),
                OnGradients = gradients => { Assert.Single(gradients); published++; },
                RegularizeGradients = (current, gradients) =>
                {
                    Assert.Single(current);
                    gradients[parameter] = engine.TensorAdd(gradients[parameter],
                        engine.TensorMultiplyScalar(parameter, .2));
                    regularized++;
                },
                BeforeUpdate = () => prepared++,
                ModelGradientClip = clip ? 1.0 : 0,
            };
            for (int step = 0; step < 3; step++)
            {
                void Update(TapeStepContext<double> context)
                {
                    using var noGrad = new NoGradScope<double>();
                    updates++;
                    Assert.Single(context.Parameters);
                    Assert.Same(parameter, context.Parameters[0]);
                    Assert.Single(context.Gradients);
                    engine.TensorSubtractInPlace(parameter, engine.TensorMultiplyScalar(context.Gradients[parameter], .1));
                }
                using var arena = AiDotNet.Tensors.Helpers.TensorArena.Create();
                double loss = split
                    ? SummedTapeObjective<double>.Step(new[] { square, linear }, () => parameters, Update,
                        parameter, parameter, (_, _) => Combined(), (value, _) => value, hooks)
                    : TapeTrainingStepper<double>.EagerTapeStep(Combined, () => parameters, Update,
                        parameter, parameter, (_, _) => Combined(), (value, _) => value, hooks);
                losses.Add(loss);
                weights.Add(parameter[0]);
                Assert.Equal(5.0, diagnostic[0]);
                Assert.Equal(9.0, disconnected[0]);
            }
            Assert.Equal(3, published);
            Assert.Equal(3, regularized);
            Assert.Equal(3, prepared);
            return (losses.ToArray(), weights.ToArray(), updates);
        }
        var together = Run(false, clip);
        var separate = Run(true, clip);
        Assert.Equal(3, separate.Updates);
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(together.Losses[i], separate.Losses[i], 12);
            Assert.Equal(together.Weights[i], separate.Weights[i], 12);
        }
    }
}
