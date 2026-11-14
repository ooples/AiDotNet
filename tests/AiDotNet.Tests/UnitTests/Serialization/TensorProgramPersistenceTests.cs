using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.PhysicsInformed;
using AiDotNet.Serialization;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Serialization;

/// <summary>
/// The legacy-autodiff removal moved delegate persistence, compiled inference and PINN derivatives onto
/// AiDotNet.Tensors (TensorProgram / ProgramPackage / TensorFunc). These pin the behaviour that replaced it.
/// </summary>
[Collection("CompilationGlobalState")]
public class TensorProgramPersistenceTests
{
    private static Tensor<float> Random(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static void AssertClose(Tensor<float> expected, Tensor<float> actual, float tol = 1e-5f)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol, $"element {i}: {actual[i]:R} vs {expected[i]:R}");
    }

    // ── DelegateState ────────────────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Closure_OverEngineOps_SavesAsAProgram_WithItsCapturedWeight_AndReloadsAtAnyBatch()
    {
        var weight = Random(1, 4, 3);
        Func<Tensor<float>, Tensor<float>> f = x => AiDotNetEngine.Current.Tanh(AiDotNetEngine.Current.TensorMatMul(x, weight));

        string saved = DelegateState.Save(f, new[] { 4 });
        Assert.StartsWith(DelegateState.ProgramScheme, saved);

        var restored = DelegateState.Load<Func<Tensor<float>, Tensor<float>>>(saved, "TestLayer", "forward");
        foreach (var batch in new[] { 1, 5, 9 })
        {
            var x = Random(10 + batch, batch, 4);
            AssertClose(f(x), restored(x));
        }
    }

    [Fact]
    public void Closure_ThatReadsItsInputOnTheHost_IsNotSavedAsAProgram()
    {
        // Scales by a value read out of the input on the host: a program would bake in the first input's value.
        Func<Tensor<float>, Tensor<float>> f = x =>
            AiDotNetEngine.Current.TensorMultiplyScalar(x, x.ToArray()[0]);
        string saved = DelegateState.Save(f, new[] { 3 });
        Assert.False(saved.StartsWith(DelegateState.ProgramScheme, StringComparison.Ordinal));
    }

    [Fact]
    public void MethodReference_ToAnAiDotNetStaticMethod_RoundTrips()
    {
        Func<double, double> f = AiDotNet.Audio.Features.MelFilterbanks.HtkHzToMel;
        string saved = DelegateState.Save(f);
        Assert.StartsWith(DelegateState.MethodScheme, saved);
        var restored = DelegateState.Load<Func<double, double>>(saved, "TestLayer", "map");
        Assert.Equal(f(1000.0), restored(1000.0));
    }

    private static double DoubleIt(double x) => 2 * x;

    [Fact]
    public void MethodReference_OutsideAiDotNet_IsNeitherWrittenNorBound()
    {
        // A model file is data: binding any static method it names (here this test's own, or a BCL method a crafted file
        // could name) would let the file choose code to run. Only AiDotNet's and AiDotNet.Tensors' methods bind.
        Func<double, double> own = DoubleIt;
        Assert.Equal(string.Empty, DelegateState.Save(own));

        string crafted = DelegateState.MethodScheme + typeof(TensorProgramPersistenceTests).AssemblyQualifiedName
            + "|" + nameof(DoubleIt) + "|" + typeof(double).AssemblyQualifiedName;
        var error = Assert.Throws<InvalidOperationException>(
            () => DelegateState.Load<Func<double, double>>(crafted, "TestLayer", "map"));
        Assert.Contains("may only bind methods", error.Message);

        string bcl = DelegateState.MethodScheme + typeof(System.IO.Path).AssemblyQualifiedName
            + "|" + nameof(System.IO.Path.GetFullPath) + "|" + typeof(string).AssemblyQualifiedName;
        Assert.Throws<InvalidOperationException>(() => DelegateState.Load<Func<string, string>>(bcl, "TestLayer", "map"));
    }

    [Fact]
    public void LambdaLayer_BuiltFromAClosure_SurvivesADurableNetworkRoundTrip()
    {
        var scale = Random(2, 6);
        Func<Tensor<float>, Tensor<float>> forward = x => AiDotNetEngine.Current.TensorMultiply(x, scale);

        FeedForwardNeuralNetwork<float> Build() => new(new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: 5, outputSize: 2, layers: new List<ILayer<float>>
            {
                new InputLayer<float>(5),
                new DenseLayer<float>(6, activationFunction: new TanhActivation<float>()),
                new LambdaLayer<float>(new[] { 6 }, new[] { 6 }, forward, null, (IActivationFunction<float>)new IdentityActivation<float>()),
                new DenseLayer<float>(2, activationFunction: new IdentityActivation<float>()),
            }), lossFunction: new MeanSquaredErrorLoss<float>());

        var original = Build();
        var bytes = original.Serialize();
        var loaded = Build();
        loaded.Deserialize(bytes);

        var input = Random(3, 4, 5);
        AssertClose(original.Predict(input), loaded.Predict(input));
    }

    // ── Facade: compiled program embedded in a saved model ───────────────────────────────────────────────

    private static FeedForwardNeuralNetwork<float> Regressor() => new(new NeuralNetworkArchitecture<float>(
        inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
        inputSize: 4, outputSize: 1, layers: new List<ILayer<float>>
        {
            new InputLayer<float>(4),
            new DenseLayer<float>(8, activationFunction: new ReLUActivation<float>()),
            new DenseLayer<float>(1, activationFunction: new IdentityActivation<float>()),
        }), lossFunction: new MeanSquaredErrorLoss<float>());

    private static AiModelResult<float, Tensor<float>, Tensor<float>> Result(FeedForwardNeuralNetwork<float> network) =>
        new(new AiModelResultOptions<float, Tensor<float>, Tensor<float>>
        {
            OptimizationResult = new OptimizationResult<float, Tensor<float>, Tensor<float>> { BestSolution = network },
            JitCompilationConfig = JitCompilationConfig.Default,
        });

    [Fact]
    public void SavedModel_EmbedsItsCompiledProgram_AndTheLoadedModelPredictsIdentically()
    {
        var network = Regressor();
        var result = Result(network);
        var x = Random(4, 3, 4);
        var before = result.Predict(x);

        byte[] bytes;
        var restored = new AiModelResult<float, Tensor<float>, Tensor<float>>();
        using (ModelPersistenceGuard.InternalOperation())
        {
            bytes = result.Serialize();
            restored.Deserialize(bytes);
        }
        Assert.NotEmpty(result.CompiledProgram);
        Assert.Equal(result.CompiledProgram, restored.CompiledProgram);
        var embedded = AiDotNet.Tensors.Engines.Export.ProgramPackage.Load(new System.IO.MemoryStream(result.CompiledProgram));
        Assert.False(embedded.Program.ParametersBound);       // weights are not stored twice
        Assert.NotEmpty(embedded.Targets);                    // this device's compiled plan travels with the model

        AssertClose(before, restored.Predict(x));
        Assert.False(restored.CompiledPlansChanged); // same machine and shape: the embedded plan was used, not recompiled

        var other = Random(5, 7, 4); // another batch size and values: the program has a symbolic batch
        AssertClose(network.Predict(other), restored.Predict(other));
        Assert.True(restored.CompiledPlansChanged);  // a plan for the new shape was compiled from the program and added
    }

    [Fact]
    public void PersistCompiledProgram_False_KeepsTheModelFileFreeOfIt()
    {
        var result = new AiModelResult<float, Tensor<float>, Tensor<float>>(new AiModelResultOptions<float, Tensor<float>, Tensor<float>>
        {
            OptimizationResult = new OptimizationResult<float, Tensor<float>, Tensor<float>> { BestSolution = Regressor() },
            JitCompilationConfig = new JitCompilationConfig { PersistCompiledProgram = false },
        });
        result.Predict(Random(6, 2, 4));
        using (ModelPersistenceGuard.InternalOperation()) result.Serialize();
        Assert.Empty(result.CompiledProgram);
    }

    // ── PINN derivatives and input gradients ─────────────────────────────────────────────────────────────

    [Fact]
    public void NeuralNetworkDerivatives_MatchCentralDifferencesOfTheExactGradient()
    {
        var net = new FeedForwardNeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2, outputSize: 1, layers: new List<ILayer<double>>
            {
                new InputLayer<double>(2),
                new DenseLayer<double>(5, activationFunction: new TanhActivation<double>()),
                new DenseLayer<double>(1, activationFunction: new IdentityActivation<double>()),
            }), lossFunction: new MeanSquaredErrorLoss<double>());
        var point = new[] { 0.3, -0.7 };

        var d = NeuralNetworkDerivatives<double>.ComputeDerivatives(net, point, 1);
        double F(double[] p) => net.Predict(new Tensor<double>(p, new[] { 1, 2 }))[0];
        const double h = 1e-4;
        for (int i = 0; i < 2; i++)
        {
            var up = (double[])point.Clone(); up[i] += h;
            var down = (double[])point.Clone(); down[i] -= h;
            Assert.Equal((F(up) - F(down)) / (2 * h), d.FirstDerivatives![0, i], 6);

            // Second derivatives: central differences of the exact first derivatives.
            var gUp = NeuralNetworkDerivatives<double>.ComputeDerivatives(net, up, 1).FirstDerivatives!;
            var gDown = NeuralNetworkDerivatives<double>.ComputeDerivatives(net, down, 1).FirstDerivatives!;
            for (int j = 0; j < 2; j++)
                Assert.Equal((gUp[0, j] - gDown[0, j]) / (2 * h), d.SecondDerivatives![0, i, j], 5);
        }
    }

    [Fact]
    public void InputGradientHelper_GivesExactGradientsForAnEngineOpFunction()
    {
        var w = new Tensor<double>(new[] { 0.5, -1.0, 2.0, 0.25 }, new[] { 4, 1 });
        Func<Tensor<double>, Tensor<double>> f = x => AiDotNetEngine.Current.Tanh(AiDotNetEngine.Current.TensorMatMul(x, w));
        var helper = new InputGradientHelper<double>(f);
        var x0 = new Tensor<double>(new[] { 0.1, 0.2, -0.3, 0.4 }, new[] { 1, 4 });

        var g = helper.ComputeGradientTensor(x0, 0);
        Assert.True(helper.SupportsExactGradients);
        Assert.Equal("Backpropagation", helper.GradientMethod);
        double z = 0.1 * 0.5 + 0.2 * -1.0 + -0.3 * 2.0 + 0.4 * 0.25;
        double dt = 1 - Math.Tanh(z) * Math.Tanh(z);
        Assert.Equal(new[] { 0.5 * dt, -1.0 * dt, 2.0 * dt, 0.25 * dt }, g.ToArray(), new ToleranceComparer(1e-12));
    }

    [Fact]
    public void InputGradientHelper_NetworkGradients_AreNonZeroAndMatchFiniteDifferences()
    {
        // Predict runs under a no-grad scope, so a tape over it saw nothing and every gradient explainer
        // (saliency, integrated gradients, DeepLIFT, ...) got zeros. The helper must use the differentiable forward.
        var net = new FeedForwardNeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: 3, outputSize: 1, layers: new List<ILayer<double>>
            {
                new InputLayer<double>(3),
                new DenseLayer<double>(4, activationFunction: new TanhActivation<double>()),
                new DenseLayer<double>(1, activationFunction: new IdentityActivation<double>()),
            }), lossFunction: new MeanSquaredErrorLoss<double>());
        var helper = new InputGradientHelper<double>(net);
        var x = new Vector<double>(new[] { 0.2, -0.5, 0.9 });

        var g = helper.ComputeGradient(x, 0);
        double F(Vector<double> v) => net.Predict(new Tensor<double>(v.ToArray(), new[] { 1, 3 }))[0];
        const double h = 1e-5;
        for (int i = 0; i < 3; i++)
        {
            var up = x.Clone(); up[i] += h;
            var down = x.Clone(); down[i] -= h;
            Assert.Equal((F(up) - F(down)) / (2 * h), g[i], 6);
        }
        Assert.Contains(g.ToArray(), v => v != 0);
    }

    [Fact]
    public void InputGradientHelper_FallsBackToNumerical_WhenTheFunctionWorksOutsideTheEngine()
    {
        Func<Tensor<double>, Tensor<double>> f = x => new Tensor<double>(new[] { x.ToArray().Sum(v => v * v) }, new[] { 1, 1 });
        var helper = new InputGradientHelper<double>(f);
        var g = helper.ComputeGradientTensor(new Tensor<double>(new[] { 1.0, -2.0 }, new[] { 1, 2 }), 0);
        Assert.False(helper.SupportsExactGradients);
        Assert.Equal(2.0, g[0], 4);
        Assert.Equal(-4.0, g[1], 4);
    }

    private sealed class ToleranceComparer : IEqualityComparer<double>
    {
        private readonly double _tol;
        public ToleranceComparer(double tol) => _tol = tol;
        public bool Equals(double a, double b) => Math.Abs(a - b) <= _tol;
        public int GetHashCode(double v) => 0;
    }
}
