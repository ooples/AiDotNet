using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

[Collection("NonParallelIntegration")]
public class RecurrentConsumerDispatchTests
{
    private sealed class Engine : CpuEngine
    {
        internal bool Reference;
        internal int CellCalls, GateCalls, GlaCalls, XLstmCalls;
        public override (Tensor<T> Hidden, Tensor<T> Cell) TensorLstmCell<T>(Tensor<T> x, Tensor<T> h, Tensor<T> c,
            Tensor<T> wi, Tensor<T> wh, Tensor<T>? bi = null, Tensor<T>? bh = null)
        {
            CellCalls++;
            if (!Reference) return base.TensorLstmCell(x, h, c, wi, wh, bi, bh);
            bool single = x.Rank == 1;
            if (single) { x = Reshape(x, new[] { 1, x.Length }); h = Reshape(h, new[] { 1, h.Length }); c = Reshape(c, new[] { 1, c.Length }); }
            var gates = TensorAdd(TensorMatMulTransposed(x, wi), TensorMatMulTransposed(h, wh));
            if (bi is not null) gates = TensorAdd(gates, bi);
            if (bh is not null) gates = TensorAdd(gates, bh);
            var hidden = LstmCellFromGates(gates, c, out var cell);
            return single ? (Reshape(hidden, new[] { hidden.Length }), Reshape(cell, new[] { cell.Length })) : (hidden, cell);
        }
        // A compatibility-engine capability: this also compiles against the published tensor package.
        // Tensor tests independently verify the real fused kernel and its device backward.
#pragma warning disable CS0114 // Capability also compiles against packages without the new virtual method.
        public Tensor<T> LstmCellFromGates<T>(Tensor<T> gates, Tensor<T> old, out Tensor<T> cell)
        {
            GateCalls++;
            int width = old.Shape[old.Rank - 1];
            Tensor<T> Gate(int i) => TensorNarrow(gates, gates.Rank - 1, i * width, width);
            cell = TensorAdd(TensorMultiply(Sigmoid(Gate(1)), old), TensorMultiply(Sigmoid(Gate(0)), Tanh(Gate(2))));
            return TensorMultiply(Sigmoid(Gate(3)), Tanh(cell));
        }
#pragma warning restore CS0114
        public override Tensor<T> GlaScanForward<T>(Tensor<T> q, Tensor<T> k, Tensor<T> v, Tensor<T> gate, int heads)
        {
            GlaCalls++;
            if (!Reference) return base.GlaScanForward(q, k, v, gate, heads);
            Assert.Equal(q.Shape[2], heads); // MinLSTM's independent scalar heads.
            var state = new Tensor<T>(new[] { q.Shape[0], 1, heads }); var steps = new List<Tensor<T>>();
            for (int t = 0; t < q.Shape[1]; t++)
            {
                state = TensorAdd(TensorMultiply(TensorNarrow(gate, 1, t, 1), state),
                    TensorMultiply(TensorNarrow(k, 1, t, 1), TensorNarrow(v, 1, t, 1)));
                steps.Add(TensorMultiply(state, TensorNarrow(q, 1, t, 1)));
            }
            return TensorConcatenate(steps.ToArray(), 1);
        }
        public override Tensor<T> XLstmScanForward<T>(Tensor<T> q, Tensor<T> k, Tensor<T> v,
            Tensor<T> ig, Tensor<T> fg, Tensor<T> og, int heads)
        {
            XLstmCalls++;
            if (!Reference) return base.XLstmScanForward(q, k, v, ig, fg, og, heads);
            int b = q.Shape[0], steps = q.Shape[1], width = q.Shape[2], d = width / heads, hb = b * heads;
            var ops = MathHelper.GetNumericOperations<T>();
            var state = new Tensor<T>(new[] { hb, d, d }); var norm = new Tensor<T>(new[] { hb, d, 1 });
            var outputs = new List<Tensor<T>>();
            for (int t = 0; t < steps; t++)
            {
                Tensor<T> Col(Tensor<T> x) => Reshape(TensorNarrow(x, 1, t, 1), new[] { hb, d, 1 });
                Tensor<T> Gate(Tensor<T> x) => Reshape(TensorNarrow(x, 1, t, 1), new[] { hb, 1, 1 });
                var qc = Col(q); var kc = TensorMultiplyScalar(Col(k), ops.FromDouble(1 / Math.Sqrt(d))); var vc = Col(v);
                state = TensorAdd(TensorMultiply(state, Gate(fg)), TensorMultiply(BatchMatMul(vc, TensorPermute(kc, new[] { 0, 2, 1 })), Gate(ig)));
                norm = TensorAdd(TensorMultiply(norm, Gate(fg)), TensorMultiply(kc, Gate(ig)));
                var denom = TensorMax(TensorAbs(BatchMatMul(TensorPermute(norm, new[] { 0, 2, 1 }), qc)),
                    Tensor<T>.CreateDefault(new[] { hb, 1, 1 }, ops.One));
                var h = TensorDivide(TensorMultiply(BatchMatMul(state, qc), Gate(og)), TensorTile(denom, new[] { 1, d, 1 }));
                outputs.Add(Reshape(h, new[] { b, 1, width }));
            }
            return TensorConcatenate(outputs.ToArray(), 1);
        }
    }
    private static Tensor<double> Values(params int[] shape)
    {
        var t = new Tensor<double>(shape); for (int i = 0; i < t.Length; i++) t[i] = Math.Sin(i + .2) * .3; return t;
    }
    private static void Near(Tensor<double> a, Tensor<double> b, double tol = 1e-8)
    {
        Assert.Equal(a.Shape, b.Shape);
        for (int i = 0; i < a.Length; i++) Assert.InRange(Math.Abs(a[i] - b[i]), 0, tol);
    }
    private static (Tensor<double>, Tensor<double>[]) Evaluate(Func<Tensor<double>> forward, Tensor<double>[] sources, Engine e)
    {
        using var tape = new GradientTape<double>();
        var output = forward(); var loss = e.ReduceSum(e.TensorMultiply(output, output), null);
        var grads = tape.ComputeGradients(loss, sources);
        return (output, sources.Select(s => grads[s]).ToArray());
    }
    private static void Compare((Tensor<double> Output, Tensor<double>[] Grads) a,
        (Tensor<double> Output, Tensor<double>[] Grads) b, double tol = 1e-8)
    { Near(a.Output, b.Output, tol); for (int i = 0; i < a.Grads.Length; i++) Near(a.Grads[i], b.Grads[i], tol); }

    [Theory] [InlineData(false)] [InlineData(true)]
    public void MinimalAndExtendedLstm_DispatchAndMatchReferenceGradients(bool extended)
    {
        var prior = AiDotNetEngine.Current; var e = new Engine(); AiDotNetEngine.Current = e;
        try
        {
            LayerBase<double> layer = extended ? new ExtendedLSTMLayer<double>(4, 4, 2) : new MinLSTMLayer<double>(4, 4);
            var input = Values(2, 4, 4); var sources = new[] { input }.Concat(layer.GetTrainableParameters()).ToArray();
            var actual = Evaluate(() => layer.Forward(input), sources, e);
            Assert.Equal(1, extended ? e.XLstmCalls : e.GlaCalls);
            e.Reference = true; var expected = Evaluate(() => layer.Forward(input), sources, e);
            Compare(expected, actual, 1e-7);
        }
        finally { AiDotNetEngine.Current = prior; }
    }
    private sealed class CustomTanh : TanhActivation<double> { }
    [Fact]
    public void ConvLstm_PackedGatesMatchCustomActivationFallback()
    {
        var prior = AiDotNetEngine.Current; var e = new Engine(); AiDotNetEngine.Current = e;
        try
        {
            var layer = new ConvLSTMLayer<double>(new[] { 3, 3, 3, 2 }, 3, 2, 1, 1, (IActivationFunction<double>?)null);
            var reference = new ConvLSTMLayer<double>(new[] { 3, 3, 3, 2 }, 3, 2, 1, 1, (IActivationFunction<double>)new CustomTanh());
            reference.SetParameters(layer.GetParameters());
            var input = Values(2, 3, 3, 3, 2);
            var sources = new[] { input }.Concat(layer.GetTrainableParameters()).ToArray();
            var expectedSources = new[] { input }.Concat(reference.GetTrainableParameters()).ToArray();
            var actual = Evaluate(() => layer.Forward(input), sources, e);
            Assert.Equal(3, e.GateCalls);
            var expected = Evaluate(() => reference.Forward(input), expectedSources, e);
            Assert.Equal(3, e.GateCalls); // Custom activation must retain its own equation.
            Compare(expected, actual);
        }
        finally { AiDotNetEngine.Current = prior; }
    }
    [Theory] [InlineData(1)] [InlineData(2)]
    public void AttentionFeedbackCell_DispatchesAndPreservesEveryLeafGradient(int recurrentInputs)
    {
        var prior = AiDotNetEngine.Current; var e = new Engine(); AiDotNetEngine.Current = e;
        try
        {
            var packed = new Vector<double>(TapeLstmCell<double>.ParameterCount(3, 2, recurrentInputs));
            TapeLstmCell<double>.InitializeUniform(packed, 0, packed.Length, 2, new Random(9));
            var cell = new TapeLstmCell<double>(packed, 0, 3, 2, recurrentInputs);
            var input = Values(2, 3); var old = Values(2, 2); var h = Enumerable.Range(0, recurrentInputs).Select(_ => Values(2, 2)).ToArray();
            var sources = new[] { input, old }.Concat(h).Concat(cell.Leaves).ToArray();
            Tensor<double> Forward() { var (y, c) = cell.Step(input, h, old); return e.TensorConcatenate(new[] { y, c }, 1); }
            var actual = Evaluate(Forward, sources, e); Assert.Equal(1, e.CellCalls);
            e.Reference = true; Compare(Evaluate(Forward, sources, e), actual);
        }
        finally { AiDotNetEngine.Current = prior; }
    }
    [Fact]
    public void DeepArSingleStep_DispatchesAndPreservesStatesAndWeights()
    {
        var prior = AiDotNetEngine.Current; var e = new Engine(); AiDotNetEngine.Current = e;
        try
        {
            var cell = new DeepARLstmCellTape<double>(3, 2, 42);
            var x = Values(3, 2); var h = Values(2, 2); var c = Values(2, 2);
            var sources = new[] { x, h, c }.Concat(cell.GetTrainableParameters()).ToArray();
            Tensor<double> Forward() { var (y, state) = cell.Step(x, h, c); return e.TensorConcatenate(new[] { y, state }, 0); }
            var actual = Evaluate(Forward, sources, e); Assert.Equal(1, e.CellCalls);
            e.Reference = true; Compare(Evaluate(Forward, sources, e), actual);
        }
        finally { AiDotNetEngine.Current = prior; }
    }
    [Fact]
    public void NtmController_ChainedStepsPreserveInputAndGateWeightGradients()
    {
        var prior = AiDotNetEngine.Current; var e = new Engine(); AiDotNetEngine.Current = e;
        try
        {
            var controller = new LSTMNTMController<double, Matrix<double>, Vector<double>>(
                new NTMOptions<double, Matrix<double>, Vector<double>>(new AiDotNet.Regression.SimpleRegression<double>()) { ControllerHiddenSize = 3, MemoryWidth = 2, NumReadHeads = 1, RandomSeed = 42 });
            var x = Values(4); var read = new List<Tensor<double>>();
            controller.Forward(x, read); // Resolve input weights before collecting parameter leaves.
            var fields = new[] { "_weightsInput", "_weightsHidden", "_biases" };
            var sources = new[] { x }.Concat(fields.Select(name => (Tensor<double>)controller.GetType()
                .GetField(name, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(controller)!)).ToArray();
            Tensor<double> Forward()
            {
                controller.Reset(); var first = controller.Forward(x, read); var second = controller.Forward(x, read);
                return e.TensorConcatenate(new[] { first, second }, 0);
            }
            e.CellCalls = 0; var actual = Evaluate(Forward, sources, e); Assert.Equal(2, e.CellCalls);
            e.Reference = true; Compare(Evaluate(Forward, sources, e), actual);
        }
        finally { AiDotNetEngine.Current = prior; }
    }
    [Fact]
    public void DncController_UsesGateKernelAndPreservesTruncationBoundary()
    {
        var prior = AiDotNetEngine.Current; var e = new Engine(); AiDotNetEngine.Current = e;
        try
        {
            var model = new AiDotNet.NeuralNetworks.DifferentiableNeuralComputer<double>(
                new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
                    inputType: AiDotNet.Enums.InputType.OneDimensional, taskType: AiDotNet.Enums.NeuralNetworkTaskType.Regression,
                    inputSize: 2, outputSize: 1), memorySize: 3, memoryWordSize: 2, controllerSize: 2, readHeads: 1,
                    activationFunction: (IActivationFunction<double>?)null);
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            var method = model.GetType().GetMethod("ApplyControllerCell", flags)!;
            var stateField = model.GetType().GetField("_controllerCell", flags)!;
            var gates = Values(1, 8); var old = Values(1, 2); stateField.SetValue(model, old);
            using var tape = new GradientTape<double>();
            var h = (Tensor<double>)method.Invoke(model, new object[] { gates })!;
            Assert.Equal(1, e.GateCalls);
            var cell = (Tensor<double>)stateField.GetValue(model)!;
            var grads = tape.ComputeGradients(e.ReduceSum(e.TensorAdd(h, cell), null), new[] { gates, old });
            Assert.True(grads.ContainsKey(gates));
            Assert.True(!grads.TryGetValue(old, out var oldGrad) || oldGrad.ToArray().All(x => x == 0));
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    [Fact]
    public void GateCompatibilityBridge_UsesInstalledKernelAndPreservesBothAdjoints()
    {
        var e = new CpuEngine(); var gates = Values(2, 12); var old = Values(2, 3);
        var sources = new[] { gates, old };
        Tensor<double> actual; Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var h = LstmGateDispatch<double>.Forward(e, gates, old, out var c);
            actual = e.ReduceSum(e.TensorAdd(h, c), null);
            if (typeof(CpuEngine).GetMethods().Any(m => m.Name == "LstmCellFromGates")) Assert.True(tape.EntryCount < 12);
            grads = tape.ComputeGradients(actual, sources);
        }
        var reference = new Engine();
        using var referenceTape = new GradientTape<double>();
        var expectedH = reference.LstmCellFromGates(gates, old, out var expectedC);
        var expected = reference.ReduceSum(reference.TensorAdd(expectedH, expectedC), null);
        var expectedGrads = referenceTape.ComputeGradients(expected, sources);
        Near(expected, actual);
        foreach (var source in sources) Near(expectedGrads[source], grads[source]);
    }

}
