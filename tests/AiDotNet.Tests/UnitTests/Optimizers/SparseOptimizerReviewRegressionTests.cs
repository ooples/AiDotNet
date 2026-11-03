using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

[Collection("EngineCurrentGlobalState")]
public class SparseOptimizerReviewRegressionTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;
    public SparseOptimizerReviewRegressionTests() => AiDotNetEngine.Current = new CpuEngine();
    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> Make(string kind) => kind switch
    {
        "AMSGrad" => new AMSGradOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new AMSGradOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 0.01, EnableGradientClipping = false }),
        "AdaMax" => new AdaMaxOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new AdaMaxOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 0.01, EnableGradientClipping = false }),
        "Nadam" => new NadamOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new NadamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 0.01, EnableGradientClipping = false }),
        "Lion" => new LionOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new LionOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 0.01, WeightDecay = 0, EnableGradientClipping = false }),
        _ => throw new ArgumentException(nameof(kind)),
    };

    [Theory]
    [InlineData("AMSGrad")]
    [InlineData("AdaMax")]
    [InlineData("Nadam")]
    [InlineData("Lion")]
    public void ChangingSparseRows_PreservesUntouchedMomentumUpdates(string kind)
    {
        var engine = AiDotNetEngine.Current;
        var sparse = new Tensor<double>(new[] { 0.1, 0.2, 0.3 }, new[] { 3, 1 });
        var dense = new Tensor<double>(new[] { 0.1, 0.2, 0.3 }, new[] { 3, 1 });
        var sparseOptimizer = Make(kind); var denseOptimizer = Make(kind);
        double firstAfterStepOne = 0;
        for (int step = 0; step < 2; step++)
        {
            var ids = new Tensor<int>(new[] { step }, new[] { 1 });
            bool ran = false;
            using (var tape = new GradientTape<double>())
            {
                var rows = engine.TensorEmbeddingLookup<double, int>(sparse, ids);
                var loss = engine.ReduceSum(engine.TensorMultiplyScalar(rows, 0.5 + step * 0.25), null);
                tape.ComputeGradientsStreaming(loss, new[] { sparse }, (source, _) =>
                {
                    Assert.True(SparseEmbeddingOptimizerHelpers.HasSparseEmbeddingGrad(source));
                    sparseOptimizer.Step(new TapeStepContext<double>(new[] { source },
                        new Dictionary<Tensor<double>, Tensor<double>>(), 0.0));
                    ran = true;
                });
            }
            Assert.True(ran);
            var gradient = new Tensor<double>(new[] { 3, 1 });
            gradient[step] = 0.5 + step * 0.25;
            denseOptimizer.Step(new TapeStepContext<double>(new[] { dense },
                new Dictionary<Tensor<double>, Tensor<double>> { [dense] = gradient }, 0.0));
            for (int i = 0; i < 3; i++) Assert.Equal(dense[i], sparse[i], 10);
            if (step == 0) firstAfterStepOne = sparse[0];
        }
        Assert.NotEqual(firstAfterStepOne, sparse[0]);
    }

    [Fact]
    public void FtrlSparseStep_UsesBetaAndAlpha_ForAllNumericPaths()
    {
        CheckFtrl<double>(); CheckFtrl<float>(); CheckFtrl<decimal>();
    }

    private static void CheckFtrl<T>()
    {
        var engine = AiDotNetEngine.Current;
        var parameter = new Tensor<T>(new[] { 2, 1 });
        var optimizer = new FTRLOptimizer<T, Tensor<T>, Tensor<T>>(null,
            new FTRLOptimizerOptions<T, Tensor<T>, Tensor<T>>
            { Alpha = 0.005, Beta = 1, Lambda1 = 0, Lambda2 = 0, InitialLearningRate = 0.25, EnableGradientClipping = false });
        bool ran = false;
        using var tape = new GradientTape<T>();
        var rows = engine.TensorEmbeddingLookup<T, int>(parameter, new Tensor<int>(new[] { 0 }, new[] { 1 }));
        var loss = engine.ReduceSum(rows, null);
        tape.ComputeGradientsStreaming(loss, new[] { parameter }, (source, _) =>
        {
            Assert.True(SparseEmbeddingOptimizerHelpers.HasSparseEmbeddingGrad(source));
            optimizer.Step(new TapeStepContext<T>(new[] { source }, new Dictionary<Tensor<T>, Tensor<T>>(), default!));
            ran = true;
        });
        Assert.True(ran);
        Assert.InRange(Convert.ToDouble(parameter[0]), -0.00250001, -0.00249999);
        Assert.Equal(0.0, Convert.ToDouble(parameter[1]));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SparseSgd_PrefersCoalescedUnscaledRegularizedGradient(bool duplicateRows)
    {
        var engine = AiDotNetEngine.Current;
        var parameter = new Tensor<double>(new[] { 2.0, 3.0 }, new[] { 2, 1 });
        var optimizer = new StochasticGradientDescentOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new StochasticGradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>>
            { InitialLearningRate = 0.1, EnableGradientClipping = false });
        bool ran = false;
        using var tape = new GradientTape<double>();
        var ids = duplicateRows ? new[] { 0, 0 } : new[] { 0 };
        var rows = engine.TensorEmbeddingLookup<double, int>(parameter, new Tensor<int>(ids, new[] { ids.Length }));
        var loss = engine.ReduceSum(engine.TensorMultiplyScalar(rows, 3.0), null);
        tape.ComputeGradientsStreaming(loss, new[] { parameter }, (source, _) =>
        {
            var gradients = new Dictionary<Tensor<double>, Tensor<double>>();
            SparseEmbeddingOptimizerHelpers.MaterializeSparseGradients(new[] { source }, gradients, engine);
            double rawGradient = duplicateRows ? 6.0 : 3.0;
            Assert.Equal(rawGradient, gradients[source][0]); // Duplicate rows coalesce before preprocessing.
            gradients[source] = engine.TensorAdd(engine.TensorDivideScalar(gradients[source], rawGradient),
                engine.TensorMultiplyScalar(source, 0.5));
            var context = new TapeStepContext<double>(new[] { source }, gradients, 0.0);
            Assert.False(SparseEmbeddingOptimizerHelpers.HasUnprocessedSparseEmbeddingGrad(context, source));
            optimizer.Step(context);
            ran = true;
        });
        Assert.True(ran);
        Assert.Equal(1.8, parameter[0], 10);
        Assert.Equal(2.85, parameter[1], 10);
    }

    [Theory]
    [InlineData("AMSGradOptimizer")]
    [InlineData("AdaMaxOptimizer")]
    [InlineData("NadamOptimizer")]
    [InlineData("LionOptimizer")]
    [InlineData("FTRLOptimizer")]
    [InlineData("StochasticGradientDescentOptimizer")]
    [InlineData("AdagradOptimizer")]
    [InlineData("RootMeanSquarePropagationOptimizer")]
    [InlineData("ProximalGradientDescentOptimizer")]
    [InlineData("MomentumOptimizer")]
    [InlineData("MiniBatchGradientDescentOptimizer")]
    [InlineData("LARSOptimizer")]
    [InlineData("LAMBOptimizer")]
    [InlineData("GradientDescentOptimizer")]
    [InlineData("Adam8BitOptimizer")]
    [InlineData("AdaDeltaOptimizer")]
    [InlineData("AdamOptimizer")]
    [InlineData("AdamWOptimizer")]
    public void ProcessedSparseGradient_MatchesTheDenseControl(string kind)
    {
        IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> Create()
        {
            string name = "AiDotNet.Optimizers." + kind + "`3";
            var type = typeof(SparseOptimizerReviewRegressionTests).Assembly.GetType(name)
                ?? typeof(GradientBasedOptimizerBase<,,>).Assembly.GetType(name)!;
            type = type.MakeGenericType(typeof(double), typeof(Tensor<double>), typeof(Tensor<double>));
            var constructor = Array.Find(type.GetConstructors(), c => c.GetParameters().Length >= 2
                && c.GetParameters()[0].Name == "model" && c.GetParameters()[1].Name == "options")!;
            var parameters = constructor.GetParameters();
            var options = Activator.CreateInstance(parameters[1].ParameterType)!;
            void Set(string property, object value) => options.GetType().GetProperty(property)?.SetValue(options, value);
            Set("InitialLearningRate", 0.1); Set("EnableGradientClipping", false);
            Set("UseAdaptiveLearningRate", false); Set("BatchSize", 1); Set("WeightDecay", 0.0);
            var arguments = Array.ConvertAll(parameters, p => p.HasDefaultValue ? p.DefaultValue : null);
            arguments[1] = options;
            return (IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>)constructor.Invoke(arguments);
        }

        var sparse = new Tensor<double>(new[] { 2.0, 3.0 }, new[] { 2, 1 });
        var dense = new Tensor<double>(new[] { 2.0, 3.0 }, new[] { 2, 1 });
        var sparseOptimizer = Create(); var denseOptimizer = Create();
        var engine = AiDotNetEngine.Current;
        bool ran = false;
        using var tape = new GradientTape<double>();
        var rows = engine.TensorEmbeddingLookup<double, int>(sparse, new Tensor<int>(new[] { 0 }, new[] { 1 }));
        var loss = engine.ReduceSum(engine.TensorMultiplyScalar(rows, 3.0), null);
        tape.ComputeGradientsStreaming(loss, new[] { sparse }, (source, _) =>
        {
            var gradients = new Dictionary<Tensor<double>, Tensor<double>>();
            SparseEmbeddingOptimizerHelpers.MaterializeSparseGradients(new[] { source }, gradients, engine);
            gradients[source] = engine.TensorAdd(engine.TensorDivideScalar(gradients[source], 3.0),
                engine.TensorMultiplyScalar(source, 0.5));
            sparseOptimizer.Step(new TapeStepContext<double>(new[] { source }, gradients, 0.0));
            ran = true;
        });
        Assert.True(ran);
        denseOptimizer.Step(new TapeStepContext<double>(new[] { dense },
            new Dictionary<Tensor<double>, Tensor<double>>
            { [dense] = new Tensor<double>(new[] { 2.0, 1.5 }, new[] { 2, 1 }) }, 0.0));
        Assert.NotEqual(3.0, dense[1]); // The regularized untouched row is a positive control.
        for (int i = 0; i < 2; i++) Assert.Equal(dense[i], sparse[i], 10);
    }
}
