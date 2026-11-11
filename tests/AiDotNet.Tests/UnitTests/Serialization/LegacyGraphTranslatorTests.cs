using System.Globalization;
using System.Text.RegularExpressions;
using AiDotNet.Serialization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Serialization;

/// <summary>
/// Tests that graphs saved by the removed <c>GraphTrace</c> (the <c>graph:</c> scheme) still rebuild the function they
/// recorded.
/// </summary>
/// <remarks>
/// <para>
/// Every graph here is written by hand in exactly the text the legacy recorder emitted: post-order steps, the
/// <c>OperationType</c> name, parent ids, and the <c>OperationParams</c> the legacy <c>TensorOperations</c> method
/// recorded, with keys in ordinal order. Since the recorder refused captured constants, every operand is cut out of the
/// single input with Slice and Reshape, as a saved expression would have done.
/// </para>
/// <para>
/// The expected values are computed by <see cref="Legacy"/>, a plain loop-over-arrays transcription of the legacy
/// forward code, never by calling the engine op the translator maps to.
/// </para>
/// </remarks>
public class LegacyGraphTranslatorTests
{
    private const string Layer = "TestLayer";
    private const string Key = "expression";
    private const string Prefix = "Cannot rebuild TestLayer: constructor parameter 'expression' was saved as a legacy traced graph, but ";

    // ── Cases ───────────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>One legacy graph with its input and the legacy result.</summary>
    public sealed class Case
    {
        public Case(string name, string graph, int[] shape, Func<Nd, Nd> expected, bool batch = false, Func<int[], double[]>? data = null)
        {
            Name = name;
            Graph = graph;
            Shape = shape;
            Expected = expected;
            Batch = batch;
            Data = data ?? (s => Signed(s));
        }

        public string Name { get; }
        public string Graph { get; }
        public int[] Shape { get; }
        public Func<Nd, Nd> Expected { get; }

        /// <summary>Whether the graph is independent of axis 0's size, so it also replays at another batch size.</summary>
        public bool Batch { get; }

        public Func<int[], double[]> Data { get; }

        public Case Tolerance(double tolerance)
        {
            Tol = tolerance;
            return this;
        }

        /// <summary>
        /// Relative tolerance. Several engine kernels (pooling, convolution, scalar ops) accumulate in single precision
        /// even for double tensors, so the default allows float rounding; a wrong mapping is off by far more.
        /// </summary>
        public double Tol { get; private set; } = 1e-6;
    }

    /// <summary>Builds graph text in the legacy recorder's format.</summary>
    private sealed class G
    {
        private readonly List<string> _steps = new List<string> { "0=Input" };
        private int _next = 1;

        public int Op(string op, string args, string? parameters = null)
        {
            var id = _next++;
            _steps.Add(id + "=" + op + "(" + args + ")" + (parameters is null ? string.Empty : "{" + parameters + "}"));
            return id;
        }

        public int Op(string op, int arg, string? parameters = null) => Op(op, arg.ToString(CultureInfo.InvariantCulture), parameters);

        /// <summary>Elements [start, start + length) of the flat input, reshaped.</summary>
        public int Piece(int start, int length, params int[] shape)
        {
            var slice = Op("Slice", 0, "Axis=0,Length=" + length + ",Start=" + start + ",Step=1");
            return Op("Reshape", slice, "NewShape=" + Ints(shape));
        }

        public string Out(int output) => string.Join(";", _steps) + ";" + output;
    }

    private static string Ints(params int[] values) => "[" + string.Join("|", values) + "]";

    private static string Unary(string op, string? parameters = null)
        => "0=Input;1=" + op + "(0)" + (parameters is null ? string.Empty : "{" + parameters + "}") + ";1";

    private static Case U(string op, Func<double, double> f, string? parameters = null, bool positive = false)
        => new Case(op, Unary(op, parameters), new[] { 2, 3 }, x => Legacy.Map(x, f), batch: true,
            data: positive ? Positive : null);

    private static IEnumerable<Case> AllCases()
    {
        // ── elementwise ──
        yield return U("Exp", Math.Exp);
        yield return U("Log", Math.Log, positive: true);
        yield return U("Sqrt", Math.Sqrt, positive: true);
        yield return U("Tanh", Math.Tanh);
        yield return U("Sigmoid", Legacy.Sigmoid);
        yield return U("ReLU", v => v > 0 ? v : 0);
        yield return U("Negate", v => -v);
        yield return U("Abs", Math.Abs);
        yield return U("Square", v => v * v);
        yield return U("GELU", v => 0.5 * v * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (v + 0.044715 * v * v * v)))).Tolerance(1e-6);
        yield return U("Swish", v => v * Legacy.Sigmoid(v));
        yield return U("Mish", v => v * Math.Tanh(Legacy.Softplus(v)));
        yield return U("SoftPlus", Legacy.Softplus);
        yield return U("SELU", v => v >= 0 ? 1.0507009873554804934193349852946 * v
            : 1.0507009873554804934193349852946 * 1.6732632423543772848170429916717 * (Math.Exp(v) - 1));
        yield return U("HardSigmoid", v => Math.Max(0, Math.Min(1, v / 6.0 + 0.5)));
        yield return U("HardTanh", v => v < -1 ? -1 : v > 1 ? 1 : v);
        yield return U("SoftSign", v => v / (1 + Math.Abs(v)));
        yield return U("LiSHT", v => v * Math.Tanh(v));
        yield return U("BentIdentity", v => 0.5 * (Math.Sqrt(v * v + 1) - 1) + v);
        yield return U("Gaussian", v => Math.Exp(-v * v));
        yield return U("Power", v => Math.Pow(v, 3), "Exponent=3");
        yield return U("ELU", v => v > 0 ? v : 0.5 * (Math.Exp(v) - 1), "Alpha=0.5");
        yield return U("LeakyReLU", v => v > 0 ? v : 0.2 * v, "Alpha=0.2");
        yield return U("PReLU", v => v > 0 ? v : 0.25 * v, "Alpha=0.25");
        yield return U("CELU", v => Math.Max(0, v) + Math.Min(0, 0.7 * (Math.Exp(v / 0.7) - 1)), "alpha=0.7");
        yield return U("ScaledTanh", v => (1 - Math.Exp(-1.5 * v)) / (1 + Math.Exp(-1.5 * v)), "beta=1.5");
        yield return U("ThresholdedReLU", v => v > 0.3 ? v : 0, "Threshold=0.3");
        yield return U("ISRU", v => v / Math.Sqrt(1 + 2 * v * v), "Alpha=2");
        yield return U("Sign", v => v > 0 ? 1 : v < 0 ? -1 : 0, "SurrogateBeta=1");
        yield return U("SQRBF", v => Math.Exp(-0.5 * v * v), "Beta=0.5");
        // Outside training the legacy slope was the midpoint of [lower, upper], and it was recorded as Alpha.
        yield return U("RReLU", v => v >= 0 ? v : 0.229 * v, "Alpha=0.229,IsTraining=false,Lower=0.125,Upper=0.333");
        yield return U("SurrogateSpike", v => v > 0.2 ? 1 : 0, "SurrogateBeta=1,Threshold=0.2");
        yield return U("StraightThroughThreshold", v => v > 0.1 ? 1 : 0, "Threshold=0.1");
        yield return U("FakeQuantization", v => Legacy.FakeQuantize(v, 4, 0.1, 0, true), "NumBits=4,Scale=0.1,Symmetric=true,ZeroPoint=0");

        // ── binary arithmetic: operands are the two halves of a [B, 6] input ──
        foreach (var (op, f) in new (string, Func<double, double, double>)[]
                 {
                     ("Add", (p, q) => p + q), ("Subtract", (p, q) => p - q),
                     ("Multiply", (p, q) => p * q), ("Divide", (p, q) => p / q),
                 })
        {
            var graph = "0=Input;1=Slice(0){Axis=1,Length=3,Start=0,Step=1};2=Slice(0){Axis=1,Length=3,Start=3,Step=1};"
                        + "3=" + op + "(1,2);3";
            yield return new Case(op, graph, new[] { 2, 6 },
                x => Legacy.Zip(Legacy.Slice(x, 1, 0, 3), Legacy.Slice(x, 1, 3, 3), f), batch: true,
                data: op == "Divide" ? Positive : null);
        }

        // Broadcasting a reduced row back over the batch, as the legacy BroadcastAdd/BroadcastMultiply did.
        yield return new Case("Add (broadcast)", "0=Input;1=ReduceSum(0){Axes=[0],KeepDims=false};2=Add(0,1);2", new[] { 2, 3 },
            x => Legacy.Broadcast2(x, Legacy.SumAxis0(x), (p, q) => p + q), batch: true);
        yield return new Case("Multiply (broadcast)", "0=Input;1=ReduceSum(0){Axes=[0],KeepDims=false};2=Multiply(1,0);2", new[] { 2, 3 },
            x => Legacy.Broadcast2(x, Legacy.SumAxis0(x), (p, q) => p * q), batch: true);

        // ── matrix ──
        {
            var g = new G();
            var a = g.Piece(0, 6, 2, 3);
            var b = g.Piece(6, 6, 3, 2);
            yield return new Case("MatMul", g.Out(g.Op("MatMul", a + "," + b)), new[] { 12 },
                x => Legacy.MatMul(Legacy.Piece(x, 0, 6, 2, 3), Legacy.Piece(x, 6, 6, 3, 2)));
        }
        {
            var g = new G();
            var a = g.Piece(0, 12, 2, 2, 3);
            var b = g.Piece(12, 12, 2, 3, 2);
            yield return new Case("MatMul (batched)", g.Out(g.Op("MatMul", a + "," + b)), new[] { 24 },
                x => Legacy.BatchMatMul(Legacy.Piece(x, 0, 12, 2, 2, 3), Legacy.Piece(x, 12, 12, 2, 3, 2)));
        }
        yield return new Case("Transpose", Unary("Transpose"), new[] { 2, 3 }, x => Legacy.Permute(x, 1, 0), batch: true);

        // ── reductions ──
        yield return new Case("ReduceSum", Unary("ReduceSum", "Axes=[1],KeepDims=false"), new[] { 2, 3 },
            x => Legacy.Reduce(x, new[] { 1 }, false, Legacy.Sum), batch: true);
        yield return new Case("ReduceSum (all)", Unary("ReduceSum", "KeepDims=false"), new[] { 2, 3 },
            x => new Nd(new[] { 1 }, new[] { x.Data.Sum() }), batch: true);
        yield return new Case("Mean", Unary("Mean"), new[] { 2, 3 }, x => new Nd(new[] { 1 }, new[] { x.Data.Average() }), batch: true);
        yield return new Case("ReduceMean", Unary("ReduceMean", "Axes=[1],KeepDims=true"), new[] { 2, 3 },
            x => Legacy.Reduce(x, new[] { 1 }, true, v => v.Average()), batch: true);
        yield return new Case("ReduceMax", Unary("ReduceMax", "Axes=[0|1],KeepDims=false"), new[] { 2, 3 },
            x => Legacy.Reduce(x, new[] { 0, 1 }, false, v => v.Max()), batch: true);
        yield return new Case("ReduceLogVariance", Unary("ReduceLogVariance", "Axis=1,Epsilon=1E-08"), new[] { 2, 4 },
            x => Legacy.Reduce(x, new[] { 1 }, false, v => Math.Log(Legacy.Variance(v) + 1e-8)), batch: true);
        yield return new Case("Norm", Unary("Norm", "Axis=1,Epsilon=1E-12,KeepDims=false"), new[] { 2, 3 },
            x => Legacy.Reduce(x, new[] { 1 }, false, v => Math.Sqrt(v.Sum(t => t * t) + 1e-12)), batch: true);

        // ── shape ──
        yield return new Case("Reshape", Unary("Reshape", "NewShape=[3|2]"), new[] { 2, 3 }, x => new Nd(new[] { 3, 2 }, x.Data));
        yield return new Case("Permute", Unary("Permute", "Axes=[1|0|2]"), new[] { 2, 3, 2 }, x => Legacy.Permute(x, 1, 0, 2), batch: true);
        yield return new Case("Broadcast", Unary("Broadcast", "TargetShape=[3|4]"), new[] { 4 },
            x => new Nd(new[] { 3, 4 }, Enumerable.Range(0, 12).Select(i => x.Data[i % 4]).ToArray()));
        yield return new Case("Concat", "0=Input;1=Negate(0);2=Concat(0,1){Axis=1};2", new[] { 2, 3 },
            x => Legacy.Concat1(x, Legacy.Map(x, v => -v)), batch: true);
        yield return new Case("Pad", Unary("Pad", "Padding=[1|2]"), new[] { 2, 3 }, x => Legacy.Pad(x, 1, 2));
        yield return new Case("Crop", Unary("Crop", "Cropping=[1|0|0|1]"), new[] { 2, 2, 3, 3 }, x => Legacy.Crop(x, 1, 0, 0, 1), batch: true);
        yield return new Case("Slice", Unary("Slice", "Axis=1,Length=2,Start=1,Step=2"), new[] { 2, 5 },
            x => Legacy.Concat1(Legacy.Slice(x, 1, 1, 1), Legacy.Slice(x, 1, 3, 1)), batch: true);
        yield return new Case("Split", Unary("Split", "Axis=1,NumSplits=3,SplitIndex=1"), new[] { 2, 6 }, x => Legacy.Slice(x, 1, 2, 2), batch: true);
        yield return new Case("Upsample", Unary("Upsample", "Scale=2"), new[] { 2, 1, 2, 2 }, x => Legacy.Nearest(x, 1, 2, 2), batch: true);
        yield return new Case("Upsample3D", Unary("Upsample3D", "ScaleD=2,ScaleH=1,ScaleW=2"), new[] { 1, 1, 2, 2, 2 },
            x => Legacy.Nearest(x, 2, 1, 2), batch: true);
        yield return new Case("PixelShuffle", Unary("PixelShuffle", "UpscaleFactor=2"), new[] { 2, 4, 2, 2 }, x => Legacy.PixelShuffle(x, 2), batch: true);

        // ── softmax family (the legacy recorded the normalized axis) ──
        yield return new Case("Softmax", Unary("Softmax", "Axis=-1"), new[] { 2, 4 }, x => Legacy.AlongLast(x, Legacy.Softmax), batch: true);
        yield return new Case("LogSoftmax", Unary("LogSoftmax", "Axis=1"), new[] { 2, 4 },
            x => Legacy.AlongLast(x, r => Legacy.Softmax(r).Select(v => Math.Log(v)).ToArray()), batch: true);
        yield return new Case("Softmin", Unary("Softmin", "Axis=1"), new[] { 2, 4 },
            x => Legacy.AlongLast(x, r => Legacy.Softmax(r.Select(v => -v).ToArray())), batch: true);
        yield return new Case("LogSoftmin", Unary("LogSoftmin", "Axis=1"), new[] { 2, 4 },
            x => Legacy.AlongLast(x, r => Legacy.Softmax(r.Select(v => -v).ToArray()).Select(v => Math.Log(v)).ToArray()), batch: true);
        yield return new Case("SphericalSoftmax", Unary("SphericalSoftmax", "Axis=1"), new[] { 2, 4 },
            x => Legacy.AlongLast(x, Legacy.SphericalSoftmax), batch: true);
        yield return new Case("Sparsemax", Unary("Sparsemax", "Axis=1"), new[] { 2, 4 }, x => Legacy.AlongLast(x, Legacy.Sparsemax), batch: true);
        yield return new Case("TaylorSoftmax", Unary("TaylorSoftmax", "Axis=1,Order=2"), new[] { 2, 4 },
            x => Legacy.AlongLast(x, r => Legacy.TaylorSoftmax(r, 2)), batch: true);
        yield return new Case("TopKSoftmax", Unary("TopKSoftmax", "K=2"), new[] { 2, 4 }, x => Legacy.AlongLast(x, r => Legacy.TopKSoftmax(r, 2)), batch: true);
        yield return new Case("Maxout", Unary("Maxout", "NumPieces=2"), new[] { 2, 6 },
            x => Legacy.Reduce(new Nd(new[] { 2, 3, 2 }, x.Data), new[] { 2 }, false, v => v.Max()));
        {
            var g = new G();
            var input = g.Op("Slice", 0, "Axis=1,Length=3,Start=0,Step=1");
            var weights = g.Op("Slice", 0, "Axis=1,Length=3,Start=3,Step=1");
            yield return new Case("HierarchicalSoftmax", g.Out(g.Op("HierarchicalSoftmax", input + "," + weights, "NumClasses=4,TreeDepth=2")),
                new[] { 2, 6 }, x => Legacy.HierarchicalSoftmax(Legacy.Slice(x, 1, 0, 3), Legacy.Slice(x, 1, 3, 3), 4));
        }

        // ── pooling ──
        yield return new Case("MaxPool2D", Unary("MaxPool2D", "KernelSize=[2|2],Padding=[0|0],Stride=[2|2]"), new[] { 2, 1, 4, 4 },
            x => Legacy.Pool(x, 2, v => v.Max()), batch: true);
        yield return new Case("AvgPool2D", Unary("AvgPool2D", "KernelSize=[2|2],Padding=[0|0],Stride=[2|2]"), new[] { 2, 1, 4, 4 },
            x => Legacy.Pool(x, 2, v => v.Average()), batch: true);
        yield return new Case("MaxPool3D", Unary("MaxPool3D", "KernelSize=[2|2|2],Padding=[0|0|0],Stride=[2|2|2]"), new[] { 2, 1, 2, 4, 4 },
            x => Legacy.Pool(x, 2, v => v.Max()), batch: true);

        // ── convolutions ──
        {
            var g = new G();
            var input = g.Piece(0, 32, 1, 2, 4, 4);
            var kernel = g.Piece(32, 36, 2, 2, 3, 3);
            var bias = g.Piece(68, 2, 2);
            yield return new Case("Conv2D", g.Out(g.Op("Conv2D", input + "," + kernel + "," + bias, "Padding=[1|1],Stride=[1|1]")), new[] { 70 },
                x => Legacy.Conv(Legacy.Piece(x, 0, 32, 1, 2, 4, 4), Legacy.Piece(x, 32, 36, 2, 2, 3, 3), Legacy.Piece(x, 68, 2, 2), 1, 1, 1));
        }
        {
            var g = new G();
            var input = g.Piece(0, 25, 1, 1, 5, 5);
            var kernel = g.Piece(25, 4, 1, 1, 2, 2);
            yield return new Case("DilatedConv2D", g.Out(g.Op("DilatedConv2D", input + "," + kernel, "Dilation=[2|2],Padding=[0|0],Stride=[1|1]")),
                new[] { 29 }, x => Legacy.Conv(Legacy.Piece(x, 0, 25, 1, 1, 5, 5), Legacy.Piece(x, 25, 4, 1, 1, 2, 2), null, 1, 0, 2));
        }
        {
            var g = new G();
            var input = g.Piece(0, 27, 1, 1, 3, 3, 3);
            var kernel = g.Piece(27, 16, 2, 1, 2, 2, 2);
            var bias = g.Piece(43, 2, 2);
            yield return new Case("Convolution3D", g.Out(g.Op("Convolution3D", input + "," + kernel + "," + bias, "Padding=[0|0|0],Stride=[1|1|1]")),
                new[] { 45 }, x => Legacy.Conv(Legacy.Piece(x, 0, 27, 1, 1, 3, 3, 3), Legacy.Piece(x, 27, 16, 2, 1, 2, 2, 2), Legacy.Piece(x, 43, 2, 2), 1, 0, 1));
        }
        {
            var g = new G();
            var input = g.Piece(0, 8, 1, 2, 2, 2);
            var kernel = g.Piece(8, 18, 2, 1, 3, 3);
            var bias = g.Piece(26, 1, 1);
            yield return new Case("ConvTranspose2D",
                g.Out(g.Op("ConvTranspose2D", input + "," + kernel + "," + bias, "OutputPadding=[0|0],Padding=[0|0],Stride=[2|2]")), new[] { 27 },
                x => Legacy.ConvTranspose(Legacy.Piece(x, 0, 8, 1, 2, 2, 2), Legacy.Piece(x, 8, 18, 2, 1, 3, 3), Legacy.Piece(x, 26, 1, 1), 2));
        }
        {
            var g = new G();
            var input = g.Piece(0, 18, 1, 2, 3, 3);
            var kernel = g.Piece(18, 8, 2, 1, 2, 2);
            var bias = g.Piece(26, 2, 2);
            yield return new Case("DepthwiseConv2D", g.Out(g.Op("DepthwiseConv2D", input + "," + kernel + "," + bias, "Padding=[0|0],Stride=[1|1]")),
                new[] { 28 }, x => Legacy.Depthwise(Legacy.Piece(x, 0, 18, 1, 2, 3, 3), Legacy.Piece(x, 18, 8, 2, 1, 2, 2), Legacy.Piece(x, 26, 2, 2)));
        }
        {
            var g = new G();
            var input = g.Piece(0, 9, 1, 1, 3, 3);
            var weights = g.Piece(9, 16, 2, 2, 1, 1, 2, 2);
            var bias = g.Piece(25, 1, 1);
            yield return new Case("LocallyConnectedConv2D", g.Out(g.Op("LocallyConnectedConv2D", input + "," + weights + "," + bias, "Stride=[1|1]")),
                new[] { 26 }, x => Legacy.LocallyConnected(Legacy.Piece(x, 0, 9, 1, 1, 3, 3), Legacy.Piece(x, 9, 16, 2, 2, 1, 1, 2, 2), Legacy.Piece(x, 25, 1, 1)));
        }
        {
            // A zero offset field (o - o) makes the deformable convolution sample the regular grid.
            var g = new G();
            var input = g.Piece(0, 9, 1, 1, 3, 3);
            var kernel = g.Piece(9, 4, 1, 1, 2, 2);
            var raw = g.Piece(13, 32, 1, 8, 2, 2);
            var offset = g.Op("Subtract", raw + "," + raw);
            var bias = g.Piece(45, 1, 1);
            yield return new Case("DeformableConv2D",
                g.Out(g.Op("DeformableConv2D", input + "," + kernel + "," + offset + "," + bias, "Dilation=[1|1],Padding=[0|0],Stride=[1|1]")),
                new[] { 46 }, x => Legacy.Conv(Legacy.Piece(x, 0, 9, 1, 1, 3, 3), Legacy.Piece(x, 9, 4, 1, 1, 2, 2), Legacy.Piece(x, 45, 1, 1), 1, 0, 1));
        }

        // ── normalization: gamma and beta are reductions of the input, the only way they could reach a saved graph ──
        yield return new Case("LayerNorm",
            "0=Input;1=ReduceMean(0){Axes=[0],KeepDims=false};2=ReduceMax(0){Axes=[0],KeepDims=false};3=LayerNorm(0,1,2){Epsilon=1E-05};3",
            new[] { 2, 3 }, x => Legacy.LayerNorm(x, Legacy.Reduce(x, new[] { 0 }, false, v => v.Average()), Legacy.Reduce(x, new[] { 0 }, false, v => v.Max()), 1e-5),
            batch: true);
        yield return new Case("BatchNorm",
            "0=Input;1=ReduceMean(0){Axes=[0|2|3],KeepDims=false};2=ReduceMax(0){Axes=[0|2|3],KeepDims=false};3=BatchNorm(0,1,2){Epsilon=1E-05};3",
            new[] { 2, 2, 2, 2 }, x => Legacy.BatchNorm(x, Legacy.Reduce(x, new[] { 0, 2, 3 }, false, v => v.Average()),
                Legacy.Reduce(x, new[] { 0, 2, 3 }, false, v => v.Max()), 1e-5), batch: true);
        yield return new Case("GroupNormalization",
            "0=Input;1=ReduceMean(0){Axes=[0|2],KeepDims=false};2=ReduceMax(0){Axes=[0|2],KeepDims=false};3=GroupNormalization(0,1,2){Epsilon=1E-05,NumGroups=2};3",
            new[] { 2, 4, 3 }, x => Legacy.GroupNorm(x, 2, Legacy.Reduce(x, new[] { 0, 2 }, false, v => v.Average()),
                Legacy.Reduce(x, new[] { 0, 2 }, false, v => v.Max()), 1e-5), batch: true);
        yield return new Case("Squash", Unary("Squash", "Epsilon=1E-07"), new[] { 2, 3 }, x => Legacy.AlongLast(x, r => Legacy.Squash(r, 1e-7)), batch: true);

        // ── everything else ──
        yield return new Case("RBFKernel", "0=Input;1=Negate(0);2=ReduceMean(0){Axes=[1],KeepDims=false};3=Abs(2);4=RBFKernel(0,1,3);4",
            new[] { 2, 3 }, x => Legacy.Rbf(x, Legacy.Map(x, v => -v), Legacy.Map(Legacy.Reduce(x, new[] { 1 }, false, v => v.Average()), Math.Abs)));
        yield return new Case("AffineGrid", Unary("AffineGrid", "OutputSize=[3|4]"), new[] { 2, 2, 3 }, x => Legacy.AffineGrid(x, 3, 4), batch: true);
        {
            var g = new G();
            var input = g.Piece(0, 18, 1, 3, 3, 2);
            var grid = g.Piece(18, 8, 1, 2, 2, 2);
            yield return new Case("GridSample", g.Out(g.Op("GridSample", input + "," + grid)), new[] { 26 },
                x => Legacy.GridSample(Legacy.Piece(x, 0, 18, 1, 3, 3, 2), Legacy.Piece(x, 18, 8, 1, 2, 2, 2)),
                data: s => Signed(s, 0.9));
        }
        {
            var g = new G();
            var input = g.Piece(0, 6, 1, 3, 2);
            var adjacency = g.Piece(6, 9, 1, 3, 3);
            var weights = g.Piece(15, 4, 2, 2);
            var bias = g.Piece(19, 2, 2);
            yield return new Case("GraphConv", g.Out(g.Op("GraphConv", input + "," + adjacency + "," + weights + "," + bias)), new[] { 21 },
                x => Legacy.GraphConv(Legacy.Piece(x, 0, 6, 1, 3, 2), Legacy.Piece(x, 6, 9, 1, 3, 3), Legacy.Piece(x, 15, 4, 2, 2), Legacy.Piece(x, 19, 2, 2)));
        }
        {
            var g = new G();
            var table = g.Piece(0, 12, 4, 3);
            var indices = g.Piece(12, 4, 2, 2);
            yield return new Case("Embedding", g.Out(g.Op("Embedding", table + "," + indices)), new[] { 16 },
                x => Legacy.Embedding(Legacy.Piece(x, 0, 12, 4, 3), Legacy.Piece(x, 12, 4, 2, 2)),
                data: s => Signed(new[] { 12 }).Concat(new[] { 2.0, 0.0, 3.0, 1.0 }).ToArray());
        }
        {
            var g = new G();
            var input = g.Piece(0, 2, 1, 2);
            var hidden = g.Piece(2, 3, 1, 3);
            var cell = g.Piece(5, 3, 1, 3);
            var wIh = g.Piece(8, 24, 2, 12);
            var wHh = g.Piece(32, 36, 3, 12);
            var bias = g.Piece(68, 12, 12);
            var packet = g.Op("LSTMCell", string.Join(",", input, hidden, cell, wIh, wHh, bias),
                "gateOrder='ifgo,hiddenSize=3,stateLayout='hidden_then_cell,weightLayout='input_by_gates");
            var newCell = g.Op("Slice", packet, "Axis=1,Length=3,Start=3,Step=1");
            var newHidden = g.Op("Slice", packet, "Axis=1,Length=3,Start=0,Step=1");
            var both = g.Op("Concat", newHidden + "," + newCell, "Axis=1");
            yield return new Case("LSTMCell", g.Out(both), new[] { 80 },
                x => Legacy.Lstm(Legacy.Piece(x, 0, 2, 1, 2), Legacy.Piece(x, 2, 3, 1, 3), Legacy.Piece(x, 5, 3, 1, 3),
                    Legacy.Piece(x, 8, 24, 2, 12), Legacy.Piece(x, 32, 36, 3, 12), Legacy.Piece(x, 68, 12, 12)));
        }
        foreach (var format in new[] { "split", "interleaved" })
        {
            var g = new G();
            var a = g.Piece(0, 8, 2, 4);
            var b = g.Piece(8, 12, 4, 3);
            yield return new Case("ComplexMatMul (" + format + ")", g.Out(g.Op("ComplexMatMul", a + "," + b, "Format='" + format)), new[] { 20 },
                x => Legacy.ComplexMatMul(Legacy.Piece(x, 0, 8, 2, 4), Legacy.Piece(x, 8, 12, 4, 3), format == "interleaved"));
        }
        yield return new Case("ComplexMultiply",
            "0=Input;1=Slice(0){Axis=1,Length=4,Start=0,Step=1};2=Slice(0){Axis=1,Length=4,Start=4,Step=1};3=ComplexMultiply(1,2){Format='split};3",
            new[] { 2, 8 }, x => Legacy.ComplexMultiply(Legacy.Slice(x, 1, 0, 4), Legacy.Slice(x, 1, 4, 4)), batch: true);
        {
            var g = new G();
            var emissions = g.Piece(0, 6, 3, 2);
            var transitions = g.Piece(6, 4, 2, 2);
            var start = g.Piece(10, 2, 2);
            var end = g.Piece(12, 2, 2);
            yield return new Case("CRFForward", g.Out(g.Op("CRFForward", string.Join(",", emissions, transitions, start, end))), new[] { 14 },
                x => Legacy.Crf(Legacy.Piece(x, 0, 6, 3, 2), Legacy.Piece(x, 6, 4, 2, 2), Legacy.Piece(x, 10, 2, 2), Legacy.Piece(x, 12, 2, 2)));
        }
        {
            var g = new G();
            var input = g.Piece(0, 3, 3);
            var left = g.Piece(3, 2, 2);
            var right = g.Piece(5, 2, 2);
            yield return new Case("SoftSplit",
                g.Out(g.Op("SoftSplit", string.Join(",", input, left, right), "FeatureIndex=1,Temperature=0.5,Threshold=0.2")), new[] { 7 },
                x => Legacy.SoftSplit(Legacy.Piece(x, 0, 3, 3), Legacy.Piece(x, 3, 2, 2), Legacy.Piece(x, 5, 2, 2), 1, 0.2, 0.5));
        }
        {
            var g = new G();
            var input = g.Piece(0, 2, 2);
            var support = g.Piece(2, 6, 3, 2);
            var labels = g.Piece(8, 3, 3);
            yield return new Case("SoftKNN", g.Out(g.Op("SoftKNN", string.Join(",", input, support, labels), "Temperature=0.7")), new[] { 11 },
                x => Legacy.SoftKnn(Legacy.Piece(x, 0, 2, 2), Legacy.Piece(x, 2, 6, 3, 2), Legacy.Piece(x, 8, 3, 3), 0.7));
        }
        {
            var g = new G();
            var input = g.Piece(0, 16, 1, 16);
            var weights = g.Piece(16, 48, 3, 2, 8);
            var biases = g.Piece(64, 24, 3, 8);
            yield return new Case("OctonionMatMul", g.Out(g.Op("OctonionMatMul", string.Join(",", input, weights, biases))), new[] { 88 },
                x => Legacy.OctonionMatMul(Legacy.Piece(x, 0, 16, 1, 16), Legacy.Piece(x, 16, 48, 3, 2, 8), Legacy.Piece(x, 64, 24, 3, 8)));
        }

        // ── Poincaré ball: operands are the halves of a [B, 6] input with small entries ──
        foreach (var op in new[] { "MobiusAdd", "PoincareExpMap", "PoincareLogMap", "PoincareDistance" })
        {
            var graph = "0=Input;1=Slice(0){Axis=1,Length=3,Start=0,Step=1};2=Slice(0){Axis=1,Length=3,Start=3,Step=1};"
                        + "3=" + op + "(1,2){curvature=-1};3";
            yield return new Case(op, graph, new[] { 2, 6 }, x => Legacy.Hyperbolic(op, Legacy.Slice(x, 1, 0, 3), Legacy.Slice(x, 1, 3, 3), 1.0),
                batch: true, data: s => Signed(s, 0.4));
        }
        yield return new Case("Custom (PoincareProject)", Unary("Custom", "curvature=-1,epsilon=1E-05,op='PoincareProject"), new[] { 2, 3 },
            x => Legacy.AlongLast(x, r => Legacy.Project(r, 1.0, 1e-5)), batch: true,
            data: s => Enumerable.Range(0, s.Aggregate(1, (p, d) => p * d)).Select(i => i % 6 < 3 ? 0.9 - 0.1 * (i % 3) : 0.1 * (i % 3) - 0.15).ToArray());
        {
            var g = new G();
            var input = g.Piece(0, 6, 2, 3);
            var weights = g.Piece(6, 6, 2, 3);
            var biases = g.Piece(12, 6, 2, 3);
            yield return new Case("Custom (HyperbolicLinear)",
                g.Out(g.Op("Custom", string.Join(",", input, weights, biases), "curvature=-1,inputFeatures=3,op='HyperbolicLinear,outputFeatures=2")),
                new[] { 18 }, x => Legacy.HyperbolicLinear(Legacy.Piece(x, 0, 6, 2, 3), Legacy.Piece(x, 6, 6, 2, 3), Legacy.Piece(x, 12, 6, 2, 3), 1.0),
                data: s => Signed(s, 0.6));
        }
    }

    private static readonly Dictionary<string, Case> ByName = AllCases().ToDictionary(c => c.Name);

    public static IEnumerable<object[]> CaseNames => ByName.Keys.Select(n => new object[] { n });

    public static IEnumerable<object[]> BatchCaseNames => ByName.Values.Where(c => c.Batch).Select(c => new object[] { c.Name });

    // ── Per-op tests ────────────────────────────────────────────────────────────────────────────────────────

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Compile_replays_what_the_legacy_operation_computed(string name)
    {
        var c = ByName[name];
        var input = Input(c.Shape, c.Data);
        var actual = LegacyGraphTranslator.Compile<double>(c.Graph, Layer, Key)(input);
        AssertClose(c.Expected(ToNd(input)), actual, c.Tol, name);
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Translated_program_replays_what_the_legacy_operation_computed(string name)
    {
        var c = ByName[name];
        var input = Input(c.Shape, c.Data);
        var program = LegacyGraphTranslator.Translate<double>(c.Graph, c.Shape, Layer, Key);
        AssertClose(c.Expected(ToNd(input)), program.Run(input), c.Tol, name);
    }

    [Theory]
    [MemberData(nameof(BatchCaseNames))]
    public void Compile_and_the_translated_program_replay_at_another_batch_size(string name)
    {
        var c = ByName[name];
        var program = LegacyGraphTranslator.Translate<double>(c.Graph, c.Shape, Layer, Key);
        var compiled = LegacyGraphTranslator.Compile<double>(c.Graph, Layer, Key);

        var shape = (int[])c.Shape.Clone();
        shape[0] += 1;
        var input = Input(shape, c.Data);
        var expected = c.Expected(ToNd(input));

        AssertClose(expected, compiled(input), c.Tol, name + " (compiled)");
        AssertClose(expected, program.Run(input), c.Tol, name + " (program)");
    }

    [Fact]
    public void Every_mapped_operation_has_a_case()
    {
        var graphs = ByName.Values.Select(c => c.Graph).Concat(new[] { Gumbel(false), Gumbel(true) });
        var covered = new HashSet<string>(graphs.SelectMany(g =>
            Regex.Matches(g, @"=(\w+)\(", RegexOptions.None, TimeSpan.FromSeconds(1)).Cast<Match>().Select(m => m.Groups[1].Value)));
        var missing = LegacyGraphTranslator.MappedOperations.Where(op => !covered.Contains(op)).OrderBy(op => op).ToArray();
        Assert.True(missing.Length == 0, "No case for: " + string.Join(", ", missing));
    }

    private static string Gumbel(bool hard) => Unary("GumbelSoftmax", "Hard=" + (hard ? "true" : "false") + ",Temperature=0.5");

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GumbelSoftmax_replays_as_a_random_sample_of_the_simplex(bool hard)
    {
        // The legacy op drew fresh Gumbel noise on every call, so only its shape and invariants are reproducible.
        var graph = Gumbel(hard);
        var input = Input(new[] { 3, 4 }, s => Signed(s));
        foreach (var output in new[]
                 {
                     LegacyGraphTranslator.Compile<double>(graph, Layer, Key)(input),
                     LegacyGraphTranslator.Translate<double>(graph, new[] { 3, 4 }, Layer, Key).Run(input),
                 })
        {
            Assert.Equal(new[] { 3, 4 }, output.Shape.ToArray());
            for (int r = 0; r < 3; r++)
            {
                var row = Enumerable.Range(0, 4).Select(i => output.GetFlat(r * 4 + i)).ToArray();
                Assert.Equal(1.0, row.Sum(), 9);
                Assert.All(row, v => Assert.InRange(v, -1e-9, 1.0 + 1e-9));
                if (hard)
                {
                    // One-hot: one entry at 1, the rest at 0.
                    Assert.Equal(1, row.Count(v => Math.Abs(v - 1.0) < 1e-9));
                    Assert.Equal(3, row.Count(v => Math.Abs(v) < 1e-9));
                }
            }
        }
    }

    // ── Chains, formats and replays ─────────────────────────────────────────────────────────────────────────

    private const string Chain = "0=Input;1=Tanh(0);2=Square(1);3=Add(1,2);4=Softmax(3){Axis=-1};4";

    private static Nd ChainReference(Nd x)
    {
        var t = Legacy.Map(x, Math.Tanh);
        var sum = Legacy.Zip(t, Legacy.Map(t, v => v * v), (p, q) => p + q);
        return Legacy.AlongLast(sum, Legacy.Softmax);
    }

    [Fact]
    public void A_multi_node_chain_replays_through_compile_and_translate()
    {
        var input = Input(new[] { 2, 3 }, s => Signed(s));
        AssertClose(ChainReference(ToNd(input)), LegacyGraphTranslator.Compile<double>(Chain, Layer, Key)(input), 1e-10, "compile");
        AssertClose(ChainReference(ToNd(input)), LegacyGraphTranslator.Translate<double>(Chain, new[] { 2, 3 }, Layer, Key).Run(input), 1e-10, "program");
    }

    [Fact]
    public void Compile_replays_on_two_different_input_shapes()
    {
        var compiled = LegacyGraphTranslator.Compile<double>(Chain, Layer, Key);
        foreach (var shape in new[] { new[] { 2, 3 }, new[] { 4, 5, 2 } })
        {
            var input = Input(shape, s => Signed(s));
            AssertClose(ChainReference(ToNd(input)), compiled(input), 1e-10, string.Join("x", shape));
        }
    }

    [Fact]
    public void Translate_output_replays_at_another_batch_size()
    {
        // Traced at batch 2 with a lazy (-1) batch axis in the recorded shape, replayed at 5.
        var program = LegacyGraphTranslator.Translate<double>(Chain, new[] { -1, 3 }, Layer, Key);
        var input = Input(new[] { 5, 3 }, s => Signed(s));
        AssertClose(ChainReference(ToNd(input)), program.Run(input), 1e-10, "batch 5");
    }

    [Fact]
    public void Translate_works_at_float_precision()
    {
        var program = LegacyGraphTranslator.Translate<float>(Chain, new[] { 2, 3 }, Layer, Key);
        var data = Signed(new[] { 2, 3 });
        var input = new Tensor<float>(new[] { 2, 3 });
        for (int i = 0; i < data.Length; i++) input.SetFlat(i, (float)data[i]);
        var output = program.Run(input);
        var expected = ChainReference(new Nd(new[] { 2, 3 }, data));
        for (int i = 0; i < expected.Data.Length; i++) Assert.Equal(expected.Data[i], output.GetFlat(i), 5);
    }

    [Theory]
    [InlineData("graph:0=Input;1=Square(0);1")]
    [InlineData("0=Input;1=Square(0)>;1")]   // written before the separator fix: the '>' glued to the last step
    [InlineData("0=Input;1=Square(0);>;1")]  // the same marker as a part of its own
    public void Accepts_every_shape_the_legacy_recorder_wrote(string graph)
    {
        var input = Input(new[] { 3 }, s => Signed(s));
        AssertClose(Legacy.Map(ToNd(input), v => v * v), LegacyGraphTranslator.Compile<double>(graph, Layer, Key)(input), 1e-12, graph);
    }

    [Fact]
    public void Output_may_be_the_input_itself()
    {
        var input = Input(new[] { 3 }, s => Signed(s));
        AssertClose(ToNd(input), LegacyGraphTranslator.Compile<double>("0=Input;0", Layer, Key)(input), 0, "identity");
    }

    // ── Rejections ──────────────────────────────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData("", "no output node")]
    [InlineData("0=Input", "no output node")]
    [InlineData("0=Input;x", "no output node")]
    [InlineData("0=Input;1=Exp(0);-1", "no output node")]
    [InlineData("0=Input;1=Exp(0;1", "malformed")]
    [InlineData("0=Input;1=Exp(0)junk;1", "malformed")]
    [InlineData("0=Input;1=Exp(0){Alpha=1;1", "malformed")]
    [InlineData("0=Input;=Exp(0);1", "malformed")]
    [InlineData("0=Input;a=Exp(0);1", "malformed")]
    [InlineData("0=Input;1=Exp(x);1", "malformed")]
    [InlineData("0=Input;1=Exp(0,);1", "malformed")]
    [InlineData("0=Input;1=(0);1", "malformed")]
    [InlineData("0=Input;1=Exp(2);2=Exp(0);1", "before it is produced")]
    [InlineData("0=Input;1=Exp(1);1", "before it is produced")]
    [InlineData("0=Input;0=Exp(0);0", "recorded twice")]
    [InlineData("1=Exp(0);1", "before it is produced")]
    [InlineData("0=Input;1=Input;1", "more than one input")]
    [InlineData("1=Input;0", "never produced")]
    [InlineData("0=Input;1=Exp(0);5", "never produced")]
    [InlineData("0=Input;1=Power(0){Exponent=abc};1", "not a valid number")]
    [InlineData("0=Input;1=Softmax(0){Axis=1.5};1", "not a valid integer")]
    [InlineData("0=Input;1=Reshape(0){NewShape=[2|x]};1", "not a valid integer array")]
    [InlineData("0=Input;1=Reshape(0){NewShape=2};1", "not a valid integer array")]
    [InlineData("0=Input;1=ReduceSum(0){KeepDims=yes};1", "not a valid boolean")]
    [InlineData("0=Input;1=ComplexMultiply(0,0){Format=split};1", "not a valid string")]
    [InlineData("0=Input;1=Power(0);1", "does not record the parameter 'Exponent'")]
    [InlineData("0=Input;1=Exp(0){Foo=1};1", "does not take")]
    [InlineData("0=Input;1=Softmax(0){Axis=1,axis=0};1", "twice")]
    [InlineData("0=Input;1=Exp(0,0);1", "takes 1")]
    [InlineData("0=Input;1=Add(0);1", "takes 2")]
    [InlineData("0=Input;1=Frobnicate(0);1", "unknown operation 'Frobnicate'")]
    [InlineData("0=Input;1=Dropout(0);1", "never recorded")]
    [InlineData("0=Input;1=Activation(0);1", "cannot appear in a saved graph")]
    [InlineData("0=Input;1=Constant();1", "cannot appear in a saved graph")]
    [InlineData("0=Input;1=Custom(0){op='Teleport};1", "unknown custom operation")]
    [InlineData("0=Input;1=Custom(0,0){op='PoincareProject};1", "takes 1")]
    [InlineData("0=Input;1=ComplexMatMul(0,0){Format='polar};1", "neither 'split' nor 'interleaved'")]
    [InlineData("0=Input;1=LSTMCell(0,0,0,0,0,0){gateOrder='fgio};1", "only ever recorded 'ifgo'")]
    public void Malformed_or_unknown_graphs_are_rejected_when_compiled(string graph, string why)
    {
        var ex = Assert.Throws<InvalidOperationException>(() => LegacyGraphTranslator.Compile<double>(graph, Layer, Key));
        Assert.StartsWith(Prefix, ex.Message);
        Assert.Contains(why, ex.Message);

        var translated = Assert.Throws<InvalidOperationException>(() => LegacyGraphTranslator.Translate<double>(graph, new[] { 2, 3 }, Layer, Key));
        Assert.Equal(ex.Message, translated.Message);
    }

    [Fact]
    public void A_null_graph_is_rejected()
    {
        var ex = Assert.Throws<InvalidOperationException>(() => LegacyGraphTranslator.Compile<double>(null!, Layer, Key));
        Assert.StartsWith(Prefix, ex.Message);
    }

    [Fact]
    public void A_step_that_fails_on_its_input_names_the_step()
    {
        // Valid text, wrong input: Maxout needs a rank-2 input.
        var replay = LegacyGraphTranslator.Compile<double>(Unary("Maxout", "NumPieces=2"), Layer, Key);
        var ex = Assert.Throws<InvalidOperationException>(() => replay(Input(new[] { 4 }, s => Signed(s))));
        Assert.StartsWith(Prefix, ex.Message);
        Assert.Contains("1=Maxout(0){NumPieces=2}", ex.Message);

        var translate = Assert.Throws<InvalidOperationException>(() => LegacyGraphTranslator.Translate<double>(Unary("Maxout", "NumPieces=2"), new[] { 4 }, Layer, Key));
        Assert.StartsWith(Prefix, translate.Message);
    }

    // ── Helpers ─────────────────────────────────────────────────────────────────────────────────────────────

    private static double[] Signed(int[] shape) => Signed(shape, 1.5);

    /// <summary>Deterministic, irregular values in (-scale, scale), none exactly zero.</summary>
    private static double[] Signed(int[] shape, double scale)
    {
        var n = shape.Aggregate(1, (p, d) => p * d);
        var data = new double[n];
        for (int i = 0; i < n; i++)
        {
            var u = Math.Sin(i * 12.9898 + 78.233) * 43758.5453;
            u -= Math.Floor(u);
            data[i] = (u * 2 - 1) * scale;
            if (Math.Abs(data[i]) < 1e-3) data[i] = 0.05;
        }
        return data;
    }

    private static double[] Positive(int[] shape) => Signed(shape).Select(v => Math.Abs(v) + 0.1).ToArray();

    private static Tensor<double> Input(int[] shape, Func<int[], double[]> data)
    {
        var values = data(shape);
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < values.Length; i++) tensor.SetFlat(i, values[i]);
        return tensor;
    }

    private static Nd ToNd(Tensor<double> t)
    {
        var data = new double[t.Length];
        for (int i = 0; i < data.Length; i++) data[i] = t.GetFlat(i);
        return new Nd(t.Shape.ToArray(), data);
    }

    private static void AssertClose(Nd expected, Tensor<double> actual, double tolerance, string what)
    {
        Assert.True(expected.Shape.SequenceEqual(actual.Shape.ToArray()),
            $"{what}: shape [{string.Join(", ", actual.Shape.ToArray())}], expected [{string.Join(", ", expected.Shape)}]");
        for (int i = 0; i < expected.Data.Length; i++)
        {
            double e = expected.Data[i], a = actual.GetFlat(i);
            Assert.True(Math.Abs(e - a) <= tolerance * (1 + Math.Abs(e)), $"{what}: element {i} is {a}, expected {e}");
        }
    }

    // ── Reference: the legacy forward code over plain arrays ────────────────────────────────────────────────

    /// <summary>A plain row-major array with a shape.</summary>
    public sealed class Nd
    {
        public Nd(int[] shape, double[]? data = null)
        {
            Shape = shape;
            Data = data ?? new double[shape.Aggregate(1, (p, d) => p * d)];
        }

        public int[] Shape { get; }
        public double[] Data { get; }

        public int Offset(params int[] index)
        {
            int flat = 0;
            for (int d = 0; d < Shape.Length; d++) flat = flat * Shape[d] + index[d];
            return flat;
        }

        public double this[params int[] index]
        {
            get => Data[Offset(index)];
            set => Data[Offset(index)] = value;
        }
    }

    private static class Legacy
    {
        public static double Sigmoid(double v) => 1 / (1 + Math.Exp(-v));

        public static double Softplus(double v) => Math.Max(v, 0) + Math.Log(1 + Math.Exp(-Math.Abs(v)));

        public static double Sum(IEnumerable<double> v) => v.Sum();

        public static double Variance(IEnumerable<double> v)
        {
            var a = v.ToArray();
            var mean = a.Average();
            return a.Sum(t => (t - mean) * (t - mean)) / a.Length;
        }

        public static Nd Map(Nd x, Func<double, double> f) => new Nd(x.Shape, x.Data.Select(f).ToArray());

        public static Nd Zip(Nd a, Nd b, Func<double, double, double> f)
            => new Nd(a.Shape, a.Data.Select((v, i) => f(v, b.Data[i])).ToArray());

        public static Nd Piece(Nd x, int start, int length, params int[] shape)
            => new Nd(shape, x.Data.Skip(start).Take(length).ToArray());

        public static Nd SumAxis0(Nd x) => Reduce(x, new[] { 0 }, false, Sum);

        /// <summary>[B, F] op [F], the legacy broadcast of a row over the batch.</summary>
        public static Nd Broadcast2(Nd x, Nd row, Func<double, double, double> f)
        {
            var r = new Nd(x.Shape);
            for (int b = 0; b < x.Shape[0]; b++)
                for (int i = 0; i < x.Shape[1]; i++)
                    r[b, i] = f(x[b, i], row.Data[i]);
            return r;
        }

        /// <summary>Elements [start, start + length) of <paramref name="axis"/>.</summary>
        public static Nd Slice(Nd x, int axis, int start, int length)
        {
            var shape = (int[])x.Shape.Clone();
            shape[axis] = length;
            var r = new Nd(shape);
            ForEach(shape, idx =>
            {
                var src = (int[])idx.Clone();
                src[axis] += start;
                r[idx] = x[src];
            });
            return r;
        }

        public static Nd Concat1(Nd a, Nd b)
        {
            var shape = (int[])a.Shape.Clone();
            shape[1] += b.Shape[1];
            var r = new Nd(shape);
            ForEach(shape, idx =>
            {
                var src = (int[])idx.Clone();
                if (idx[1] < a.Shape[1]) r[idx] = a[src];
                else { src[1] -= a.Shape[1]; r[idx] = b[src]; }
            });
            return r;
        }

        public static void ForEach(int[] shape, Action<int[]> body)
        {
            var n = shape.Aggregate(1, (p, d) => p * d);
            var idx = new int[shape.Length];
            for (int flat = 0; flat < n; flat++)
            {
                int rem = flat;
                for (int d = shape.Length - 1; d >= 0; d--) { idx[d] = rem % shape[d]; rem /= shape[d]; }
                body(idx);
            }
        }

        public static Nd Permute(Nd x, params int[] axes)
        {
            var shape = axes.Select(a => x.Shape[a]).ToArray();
            var r = new Nd(shape);
            ForEach(x.Shape, idx => r[axes.Select(a => idx[a]).ToArray()] = x[idx]);
            return r;
        }

        /// <summary>Legacy reductions: kept dims (1 for reduced under keepDims), [1] when nothing remains.</summary>
        public static Nd Reduce(Nd x, int[] axes, bool keepDims, Func<IEnumerable<double>, double> f)
        {
            var outShape = new List<int>();
            for (int d = 0; d < x.Shape.Length; d++)
            {
                if (!axes.Contains(d)) outShape.Add(x.Shape[d]);
                else if (keepDims) outShape.Add(1);
            }
            if (outShape.Count == 0) outShape.Add(1);
            var groups = new Dictionary<int, List<double>>();
            var kept = Enumerable.Range(0, x.Shape.Length).Where(d => !axes.Contains(d)).ToArray();
            var keptShape = kept.Select(d => x.Shape[d]).ToArray();
            ForEach(x.Shape, idx =>
            {
                int key = 0;
                for (int k = 0; k < kept.Length; k++) key = key * keptShape[k] + idx[kept[k]];
                if (!groups.TryGetValue(key, out var list)) groups[key] = list = new List<double>();
                list.Add(x[idx]);
            });
            var r = new Nd(outShape.ToArray());
            for (int i = 0; i < r.Data.Length; i++) r.Data[i] = f(groups[i]);
            return r;
        }

        /// <summary>Applies <paramref name="f"/> to every row of the last axis.</summary>
        public static Nd AlongLast(Nd x, Func<double[], double[]> f)
        {
            int n = x.Shape[x.Shape.Length - 1];
            var r = new Nd(x.Shape);
            for (int row = 0; row < x.Data.Length / n; row++)
            {
                var o = f(x.Data.Skip(row * n).Take(n).ToArray());
                Array.Copy(o, 0, r.Data, row * n, n);
            }
            return r;
        }

        public static double[] Softmax(double[] r)
        {
            var max = r.Max();
            var e = r.Select(v => Math.Exp(v - max)).ToArray();
            var s = e.Sum();
            return e.Select(v => v / s).ToArray();
        }

        public static double[] SphericalSoftmax(double[] r)
        {
            var norm = Math.Max(Math.Sqrt(r.Sum(v => v * v)), 1e-12);
            return Softmax(r.Select(v => v / norm).ToArray());
        }

        public static double[] Sparsemax(double[] r)
        {
            var sorted = r.OrderByDescending(v => v).ToArray();
            double cum = 0, tau = 0;
            for (int i = 0; i < sorted.Length; i++)
            {
                cum += sorted[i];
                if (1 + (i + 1) * sorted[i] - cum > 0) tau = (cum - 1) / (i + 1);
            }
            return r.Select(v => Math.Max(v - tau, 0)).ToArray();
        }

        public static double[] TaylorSoftmax(double[] r, int order)
        {
            var t = r.Select(x =>
            {
                double sum = 1, power = 1, factorial = 1;
                for (int n = 1; n <= order; n++) { power *= x; factorial *= n; sum += power / factorial; }
                return sum > 0 ? sum : 1e-10;
            }).ToArray();
            var s = t.Sum();
            return t.Select(v => v / s).ToArray();
        }

        public static double[] TopKSoftmax(double[] r, int k)
        {
            var top = Enumerable.Range(0, r.Length).OrderByDescending(i => r[i]).Take(k).ToArray();
            var max = top.Max(i => r[i]);
            var sum = top.Sum(i => Math.Exp(r[i] - max));
            var o = new double[r.Length];
            foreach (var i in top) o[i] = Math.Exp(r[i] - max) / sum;
            return o;
        }

        public static double[] Squash(double[] r, double epsilon)
        {
            var sq = r.Sum(v => v * v);
            var norm = Math.Sqrt(sq + epsilon);
            return r.Select(v => sq / (1 + sq) * (v / norm)).ToArray();
        }

        public static double[] Project(double[] r, double c, double epsilon)
        {
            var maxRadius = 1 / Math.Sqrt(c) - epsilon;
            var norm = Math.Sqrt(r.Sum(v => v * v));
            var scale = norm >= maxRadius ? maxRadius / (norm + epsilon) : 1.0;
            return r.Select(v => v * scale).ToArray();
        }

        public static double FakeQuantize(double v, int bits, double scale, double zeroPoint, bool symmetric)
        {
            double qMin = symmetric ? -(1 << (bits - 1)) : 0, qMax = symmetric ? (1 << (bits - 1)) - 1 : (1 << bits) - 1;
            var shifted = Math.Round(v / scale) + zeroPoint;
            shifted = Math.Min(Math.Max(shifted, qMin), qMax);
            return (shifted - zeroPoint) * scale;
        }

        public static Nd HierarchicalSoftmax(Nd input, Nd weights, int classes)
        {
            int batch = input.Shape[0], dim = input.Shape[1], depth = weights.Shape[0];
            var r = new Nd(new[] { batch, classes });
            for (int b = 0; b < batch; b++)
            {
                var gate = new double[depth];
                for (int d = 0; d < depth; d++)
                {
                    double dot = 0;
                    for (int i = 0; i < dim; i++) dot += input[b, i] * weights[d, i];
                    gate[d] = 1 / (1 + Math.Exp(-dot));
                }
                for (int c = 0; c < classes; c++)
                {
                    double p = 1;
                    for (int d = 0; d < depth; d++)
                    {
                        p *= (c & (1 << (depth - d - 1))) != 0 ? gate[d] : 1 - gate[d];
                        if (p < 1e-10) { p = 1e-10; break; }
                    }
                    r[b, c] = p;
                }
            }
            return r;
        }

        public static Nd MatMul(Nd a, Nd b)
        {
            int m = a.Shape[0], k = a.Shape[1], n = b.Shape[1];
            var r = new Nd(new[] { m, n });
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double s = 0;
                    for (int t = 0; t < k; t++) s += a[i, t] * b[t, j];
                    r[i, j] = s;
                }
            return r;
        }

        public static Nd BatchMatMul(Nd a, Nd b)
        {
            int batch = a.Shape[0], m = a.Shape[1], k = a.Shape[2], n = b.Shape[2];
            var r = new Nd(new[] { batch, m, n });
            for (int q = 0; q < batch; q++)
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double s = 0;
                        for (int t = 0; t < k; t++) s += a[q, i, t] * b[q, t, j];
                        r[q, i, j] = s;
                    }
            return r;
        }

        public static Nd Pad(Nd x, params int[] padding)
        {
            var r = new Nd(x.Shape.Select((s, d) => s + 2 * padding[d]).ToArray());
            ForEach(x.Shape, idx => r[idx.Select((v, d) => v + padding[d]).ToArray()] = x[idx]);
            return r;
        }

        public static Nd Crop(Nd x, int top, int bottom, int left, int right)
        {
            int outH = x.Shape[2] - top - bottom, outW = x.Shape[3] - left - right;
            var r = new Nd(new[] { x.Shape[0], x.Shape[1], outH, outW });
            ForEach(r.Shape, i => r[i] = x[i[0], i[1], i[2] + top, i[3] + left]);
            return r;
        }

        /// <summary>Nearest-neighbour upsampling of the trailing two (or three) axes.</summary>
        public static Nd Nearest(Nd x, int sd, int sh, int sw)
        {
            var shape = (int[])x.Shape.Clone();
            int rank = shape.Length;
            shape[rank - 1] *= sw;
            shape[rank - 2] *= sh;
            if (rank == 5) shape[2] *= sd;
            var r = new Nd(shape);
            ForEach(shape, idx =>
            {
                var src = (int[])idx.Clone();
                src[rank - 1] /= sw;
                src[rank - 2] /= sh;
                if (rank == 5) src[2] /= sd;
                r[idx] = x[src];
            });
            return r;
        }

        public static Nd PixelShuffle(Nd x, int f)
        {
            int batch = x.Shape[0], c = x.Shape[1] / (f * f), h = x.Shape[2], w = x.Shape[3];
            var r = new Nd(new[] { batch, c, h * f, w * f });
            ForEach(r.Shape, i => r[i] = x[i[0], i[1] * f * f + (i[2] % f) * f + i[3] % f, i[2] / f, i[3] / f]);
            return r;
        }

        /// <summary>Non-overlapping pooling of the spatial axes (everything after batch and channel) with window k.</summary>
        public static Nd Pool(Nd x, int k, Func<IEnumerable<double>, double> f)
        {
            int spatial = x.Shape.Length - 2;
            var shape = x.Shape.Select((s, d) => d < 2 ? s : (s - k) / k + 1).ToArray();
            var r = new Nd(shape);
            ForEach(shape, o =>
            {
                var window = new List<double>();
                ForEach(Enumerable.Repeat(k, spatial).ToArray(), w =>
                    window.Add(x[o.Select((v, d) => d < 2 ? v : v * k + w[d - 2]).ToArray()]));
                r[o] = f(window);
            });
            return r;
        }

        /// <summary>Cross-correlation over the spatial axes, zero padding, as the engine Conv2D/Conv3D the legacy called.</summary>
        public static Nd Conv(Nd x, Nd kernel, Nd? bias, int stride, int padding, int dilation)
        {
            int spatial = x.Shape.Length - 2, inC = x.Shape[1], outC = kernel.Shape[0];
            var outShape = new int[x.Shape.Length];
            outShape[0] = x.Shape[0];
            outShape[1] = outC;
            for (int d = 0; d < spatial; d++)
                outShape[d + 2] = (x.Shape[d + 2] + 2 * padding - dilation * (kernel.Shape[d + 2] - 1) - 1) / stride + 1;
            var r = new Nd(outShape);
            var kSpatial = kernel.Shape.Skip(2).ToArray();
            ForEach(outShape, o =>
            {
                double s = bias is null ? 0 : bias.Data[o[1]];
                for (int ic = 0; ic < inC; ic++)
                {
                    ForEach(kSpatial, kk =>
                    {
                        var src = new int[x.Shape.Length];
                        src[0] = o[0];
                        src[1] = ic;
                        for (int d = 0; d < spatial; d++)
                        {
                            src[d + 2] = o[d + 2] * stride + kk[d] * dilation - padding;
                            if (src[d + 2] < 0 || src[d + 2] >= x.Shape[d + 2]) return;
                        }
                        s += x[src] * kernel[new[] { o[1], ic }.Concat(kk).ToArray()];
                    });
                }
                r[o] = s;
            });
            return r;
        }

        public static Nd ConvTranspose(Nd x, Nd kernel, Nd bias, int stride)
        {
            int batch = x.Shape[0], inC = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
            int outC = kernel.Shape[1], kh = kernel.Shape[2], kw = kernel.Shape[3];
            var r = new Nd(new[] { batch, outC, (h - 1) * stride + kh, (w - 1) * stride + kw });
            for (int b = 0; b < batch; b++)
                for (int ic = 0; ic < inC; ic++)
                    for (int i = 0; i < h; i++)
                        for (int j = 0; j < w; j++)
                            for (int oc = 0; oc < outC; oc++)
                                for (int p = 0; p < kh; p++)
                                    for (int q = 0; q < kw; q++)
                                        r[b, oc, i * stride + p, j * stride + q] += x[b, ic, i, j] * kernel[ic, oc, p, q];
            ForEach(r.Shape, o => r[o] += bias.Data[o[1]]);
            return r;
        }

        public static Nd Depthwise(Nd x, Nd kernel, Nd bias)
        {
            int batch = x.Shape[0], inC = x.Shape[1], mult = kernel.Shape[1], kh = kernel.Shape[2], kw = kernel.Shape[3];
            var r = new Nd(new[] { batch, inC * mult, x.Shape[2] - kh + 1, x.Shape[3] - kw + 1 });
            ForEach(r.Shape, o =>
            {
                int ic = o[1] / mult, m = o[1] % mult;
                double s = bias.Data[o[1]];
                for (int p = 0; p < kh; p++)
                    for (int q = 0; q < kw; q++)
                        s += x[o[0], ic, o[2] + p, o[3] + q] * kernel[ic, m, p, q];
                r[o] = s;
            });
            return r;
        }

        public static Nd LocallyConnected(Nd x, Nd weights, Nd bias)
        {
            int batch = x.Shape[0], inC = x.Shape[1], inH = x.Shape[2], inW = x.Shape[3];
            int outH = weights.Shape[0], outW = weights.Shape[1], outC = weights.Shape[2], kh = weights.Shape[4], kw = weights.Shape[5];
            var r = new Nd(new[] { batch, outC, outH, outW });
            ForEach(r.Shape, o =>
            {
                double s = 0;
                for (int ic = 0; ic < inC; ic++)
                    for (int p = 0; p < kh; p++)
                        for (int q = 0; q < kw; q++)
                        {
                            int ih = o[2] + p, iw = o[3] + q;
                            if (ih < inH && iw < inW) s += x[o[0], ic, ih, iw] * weights[o[2], o[3], o[1], ic, p, q];
                        }
                r[o] = s + bias.Data[o[1]];
            });
            return r;
        }

        public static Nd LayerNorm(Nd x, Nd gamma, Nd beta, double epsilon)
        {
            var r = new Nd(x.Shape);
            int f = x.Shape[1];
            for (int b = 0; b < x.Shape[0]; b++)
            {
                var row = Enumerable.Range(0, f).Select(i => x[b, i]).ToArray();
                double mean = row.Average(), std = Math.Sqrt(Variance(row) + epsilon);
                for (int i = 0; i < f; i++) r[b, i] = (x[b, i] - mean) / std * gamma.Data[i] + beta.Data[i];
            }
            return r;
        }

        public static Nd BatchNorm(Nd x, Nd gamma, Nd beta, double epsilon)
        {
            var r = new Nd(x.Shape);
            for (int c = 0; c < x.Shape[1]; c++)
            {
                var values = new List<double>();
                ForEach(x.Shape, i => { if (i[1] == c) values.Add(x[i]); });
                double mean = values.Average(), std = Math.Sqrt(Variance(values) + epsilon);
                ForEach(x.Shape, i => { if (i[1] == c) r[i] = (x[i] - mean) / std * gamma.Data[c] + beta.Data[c]; });
            }
            return r;
        }

        public static Nd GroupNorm(Nd x, int groups, Nd gamma, Nd beta, double epsilon)
        {
            var r = new Nd(x.Shape);
            int perGroup = x.Shape[1] / groups;
            for (int b = 0; b < x.Shape[0]; b++)
                for (int g = 0; g < groups; g++)
                {
                    var values = new List<double>();
                    ForEach(x.Shape, i => { if (i[0] == b && i[1] / perGroup == g) values.Add(x[i]); });
                    double mean = values.Average(), std = Math.Sqrt(Variance(values) + epsilon);
                    ForEach(x.Shape, i =>
                    {
                        if (i[0] == b && i[1] / perGroup == g) r[i] = (x[i] - mean) / std * gamma.Data[i[1]] + beta.Data[i[1]];
                    });
                }
            return r;
        }

        public static Nd Rbf(Nd input, Nd centers, Nd epsilons)
        {
            int batch = input.Shape[0], n = centers.Shape[0], f = input.Shape[1];
            var r = new Nd(new[] { batch, n });
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < n; c++)
                {
                    double sq = 0;
                    for (int i = 0; i < f; i++) sq += (input[b, i] - centers[c, i]) * (input[b, i] - centers[c, i]);
                    var distance = Math.Sqrt(sq);
                    r[b, c] = Math.Exp(-(epsilons.Data[c] * (distance * distance)));
                }
            return r;
        }

        public static Nd AffineGrid(Nd theta, int h, int w)
        {
            var r = new Nd(new[] { theta.Shape[0], h, w, 2 });
            for (int b = 0; b < theta.Shape[0]; b++)
                for (int i = 0; i < h; i++)
                    for (int j = 0; j < w; j++)
                    {
                        double x = (double)j / Math.Max(w - 1, 1) * 2.0 - 1.0, y = (double)i / Math.Max(h - 1, 1) * 2.0 - 1.0;
                        r[b, i, j, 0] = theta[b, 0, 0] * x + theta[b, 0, 1] * y + theta[b, 0, 2];
                        r[b, i, j, 1] = theta[b, 1, 0] * x + theta[b, 1, 1] * y + theta[b, 1, 2];
                    }
            return r;
        }

        public static Nd GridSample(Nd input, Nd grid)
        {
            int batch = input.Shape[0], inH = input.Shape[1], inW = input.Shape[2], channels = input.Shape[3];
            int outH = grid.Shape[1], outW = grid.Shape[2];
            var r = new Nd(new[] { batch, outH, outW, channels });
            for (int b = 0; b < batch; b++)
                for (int h = 0; h < outH; h++)
                    for (int w = 0; w < outW; w++)
                    {
                        double srcX = (grid[b, h, w, 0] + 1) * ((inW - 1) / 2.0), srcY = (grid[b, h, w, 1] + 1) * ((inH - 1) / 2.0);
                        int x0 = Math.Max(0, Math.Min((int)Math.Floor(srcX), inW - 1)), x1 = Math.Max(0, Math.Min(x0 + 1, inW - 1));
                        int y0 = Math.Max(0, Math.Min((int)Math.Floor(srcY), inH - 1)), y1 = Math.Max(0, Math.Min(y0 + 1, inH - 1));
                        double wx1 = Math.Max(0, srcX - x0), wx0 = Math.Max(0, 1 - (srcX - x0));
                        double wy1 = Math.Max(0, srcY - y0), wy0 = Math.Max(0, 1 - (srcY - y0));
                        for (int c = 0; c < channels; c++)
                            r[b, h, w, c] = input[b, y0, x0, c] * wx0 * wy0 + input[b, y0, x1, c] * wx1 * wy0
                                            + input[b, y1, x0, c] * wx0 * wy1 + input[b, y1, x1, c] * wx1 * wy1;
                    }
            return r;
        }

        public static Nd GraphConv(Nd input, Nd adjacency, Nd weights, Nd bias)
        {
            int batch = input.Shape[0], nodes = input.Shape[1], f = input.Shape[2], o = weights.Shape[1];
            var r = new Nd(new[] { batch, nodes, o });
            for (int b = 0; b < batch; b++)
                for (int i = 0; i < nodes; i++)
                    for (int k = 0; k < o; k++)
                    {
                        double s = 0;
                        for (int j = 0; j < nodes; j++)
                        {
                            double xw = 0;
                            for (int t = 0; t < f; t++) xw += input[b, j, t] * weights[t, k];
                            s += adjacency[b, i, j] * xw;
                        }
                        r[b, i, k] = s + bias.Data[k];
                    }
            return r;
        }

        public static Nd Embedding(Nd table, Nd indices)
        {
            int batch = indices.Shape[0], seq = indices.Shape.Length > 1 ? indices.Shape[1] : 1, dim = table.Shape[1];
            var r = new Nd(seq > 1 ? new[] { batch, seq, dim } : new[] { batch, dim });
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < seq; s++)
                {
                    var idx = (int)(seq > 1 ? indices[b, s] : indices.Data[b * indices.Data.Length / batch]);
                    for (int e = 0; e < dim; e++) r.Data[(b * seq + s) * dim + e] = table[idx, e];
                }
            return r;
        }

        public static Nd Lstm(Nd x, Nd h, Nd c, Nd wIh, Nd wHh, Nd bias)
        {
            int hidden = h.Shape[1];
            var gates = new double[4 * hidden];
            for (int g = 0; g < 4 * hidden; g++)
            {
                double s = bias.Data[g];
                for (int i = 0; i < x.Shape[1]; i++) s += x[0, i] * wIh[i, g];
                for (int i = 0; i < hidden; i++) s += h[0, i] * wHh[i, g];
                gates[g] = s;
            }
            var r = new Nd(new[] { 1, 2 * hidden });
            for (int j = 0; j < hidden; j++)
            {
                double ig = Sigmoid(gates[j]), fg = Sigmoid(gates[hidden + j]), gg = Math.Tanh(gates[2 * hidden + j]), og = Sigmoid(gates[3 * hidden + j]);
                double cell = fg * c[0, j] + ig * gg;
                r[0, j] = og * Math.Tanh(cell);
                r[0, hidden + j] = cell;
            }
            return r;
        }

        public static Nd ComplexMatMul(Nd a, Nd b, bool interleaved)
        {
            int m = a.Shape[0], k = a.Shape[1] / 2, n = b.Shape[1];
            var r = new Nd(new[] { m, 2 * n });
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double re = 0, im = 0;
                    for (int t = 0; t < k; t++)
                    {
                        double ar = interleaved ? a[i, 2 * t] : a[i, t], ai = interleaved ? a[i, 2 * t + 1] : a[i, k + t];
                        double br = interleaved ? b[2 * t, j] : b[t, j], bi = interleaved ? b[2 * t + 1, j] : b[k + t, j];
                        re += ar * br - ai * bi;
                        im += ar * bi + ai * br;
                    }
                    r[i, interleaved ? 2 * j : j] = re;
                    r[i, interleaved ? 2 * j + 1 : n + j] = im;
                }
            return r;
        }

        public static Nd ComplexMultiply(Nd a, Nd b)
        {
            int last = a.Shape.Length - 1, n = a.Shape[last] / 2;
            var r = new Nd(a.Shape);
            ForEach(a.Shape, idx =>
            {
                if (idx[last] >= n) return;
                var im = (int[])idx.Clone();
                im[last] += n;
                r[idx] = a[idx] * b[idx] - a[im] * b[im];
                r[im] = a[idx] * b[im] + a[im] * b[idx];
            });
            return r;
        }

        public static Nd Crf(Nd emissions, Nd transitions, Nd start, Nd end)
        {
            int seq = emissions.Shape[0], tags = emissions.Shape[1];
            var alpha = new double[tags];
            for (int j = 0; j < tags; j++) alpha[j] = start.Data[j] + emissions[0, j];
            for (int t = 1; t < seq; t++)
            {
                var next = new double[tags];
                for (int j = 0; j < tags; j++)
                {
                    var max = Enumerable.Range(0, tags).Max(i => alpha[i] + transitions[i, j]);
                    next[j] = emissions[t, j] + max + Math.Log(Enumerable.Range(0, tags).Sum(i => Math.Exp(alpha[i] + transitions[i, j] - max)));
                }
                alpha = next;
            }
            var maxFinal = Enumerable.Range(0, tags).Max(j => alpha[j] + end.Data[j]);
            var logZ = maxFinal + Math.Log(Enumerable.Range(0, tags).Sum(j => Math.Exp(alpha[j] + end.Data[j] - maxFinal)));
            return new Nd(new[] { 1 }, new[] { logZ });
        }

        public static Nd SoftSplit(Nd input, Nd left, Nd right, int feature, double threshold, double temperature)
        {
            var pLeft = 1 / (1 + Math.Exp(-((threshold - input.Data[feature]) / temperature)));
            return new Nd(left.Shape, left.Data.Select((v, i) => v * pLeft + right.Data[i] * (1 - pLeft)).ToArray());
        }

        public static Nd SoftKnn(Nd input, Nd support, Nd labels, double temperature)
        {
            int n = support.Shape[0], f = support.Shape[1];
            var scaled = Enumerable.Range(0, n).Select(i =>
                -Enumerable.Range(0, f).Sum(j => (input.Data[j] - support[i, j]) * (input.Data[j] - support[i, j])) / temperature).ToArray();
            var max = scaled.Max();
            var e = scaled.Select(v => Math.Exp(v - max)).ToArray();
            var sum = e.Sum();
            int outputs = Math.Max(labels.Data.Length / n, 1);
            var r = new Nd(new[] { outputs });
            for (int i = 0; i < n; i++)
                for (int j = 0; j < outputs; j++)
                    r.Data[j] += e[i] / sum * labels.Data[i * outputs + j];
            return r;
        }

        public static Nd OctonionMatMul(Nd input, Nd weights, Nd biases)
        {
            int batch = input.Shape[0], inF = weights.Shape[1], outF = weights.Shape[0];
            var r = new Nd(new[] { batch, outF * 8 });
            for (int b = 0; b < batch; b++)
                for (int o = 0; o < outF; o++)
                {
                    var acc = new double[8];
                    for (int i = 0; i < inF; i++)
                    {
                        var p = OctonionMultiply(
                            Enumerable.Range(0, 8).Select(k => input[b, i * 8 + k]).ToArray(),
                            Enumerable.Range(0, 8).Select(k => weights[o, i, k]).ToArray());
                        for (int k = 0; k < 8; k++) acc[k] += p[k];
                    }
                    for (int k = 0; k < 8; k++) r[b, o * 8 + k] = acc[k] + biases[o, k];
                }
            return r;
        }

        /// <summary>The legacy OctonionMultiplyComponents, term for term.</summary>
        private static double[] OctonionMultiply(double[] a, double[] b) => new[]
        {
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3] - a[4] * b[4] - a[5] * b[5] - a[6] * b[6] - a[7] * b[7],
            a[0] * b[1] + a[1] * b[0] - a[2] * b[3] - a[3] * b[2] - a[4] * b[5] + a[5] * b[4] + a[6] * b[7] + a[7] * b[6],
            a[0] * b[2] + a[2] * b[0] + a[1] * b[3] - a[3] * b[1] - a[4] * b[6] + a[6] * b[4] - a[5] * b[7] + a[7] * b[5],
            a[0] * b[3] + a[3] * b[0] - a[1] * b[2] - a[2] * b[1] + a[4] * b[7] - a[7] * b[4] + a[5] * b[6] + a[6] * b[5],
            a[0] * b[4] + a[4] * b[0] + a[1] * b[5] + a[5] * b[1] + a[2] * b[6] - a[6] * b[2] - a[3] * b[7] + a[7] * b[3],
            a[0] * b[5] + a[5] * b[0] + a[1] * b[4] - a[4] * b[1] + a[2] * b[7] - a[7] * b[2] + a[3] * b[6] - a[6] * b[3],
            a[0] * b[6] + a[6] * b[0] + a[1] * b[7] - a[7] * b[1] + a[2] * b[4] + a[4] * b[2] - a[3] * b[5] - a[5] * b[3],
            a[0] * b[7] + a[7] * b[0] + a[1] * b[6] + a[6] * b[1] - a[2] * b[5] - a[5] * b[2] - a[3] * b[4] + a[4] * b[3],
        };

        private static double Atanh(double x)
        {
            x = Math.Max(-0.999999, Math.Min(0.999999, x));
            return 0.5 * Math.Log((1.0 + x) / (1.0 - x));
        }

        private static double Dot(double[] a, double[] b) => a.Select((v, i) => v * b[i]).Sum();

        private static double[] Mobius(double[] x, double[] y, double c, double floor)
        {
            double dot = Dot(x, y), nx = Dot(x, x), ny = Dot(y, y);
            double numX = 1 + 2 * c * dot + c * ny, numY = 1 - c * nx;
            double den = Math.Max(1 + 2 * c * dot + c * c * nx * ny, floor);
            return x.Select((v, i) => (numX * v + numY * y[i]) / den).ToArray();
        }

        public static Nd Hyperbolic(string op, Nd x, Nd y, double c)
        {
            int batch = x.Shape[0], dim = x.Shape[1];
            double sqrtC = Math.Sqrt(c);
            var r = new Nd(op == "PoincareDistance" ? new[] { batch } : x.Shape);
            for (int b = 0; b < batch; b++)
            {
                var p = Enumerable.Range(0, dim).Select(i => x[b, i]).ToArray();
                var q = Enumerable.Range(0, dim).Select(i => y[b, i]).ToArray();
                double[] row;
                switch (op)
                {
                    case "MobiusAdd":
                        row = Mobius(p, q, c, 1e-10);
                        break;
                    case "PoincareExpMap":
                    {
                        double normV = Math.Sqrt(Dot(q, q)), lambda = 2.0 / Math.Max(1.0 - c * Dot(p, p), 1e-10);
                        if (normV < 1e-10) { row = p; break; }
                        double scale = Math.Tanh(sqrtC * lambda * normV / 2.0) / (sqrtC * normV);
                        row = Mobius(p, q.Select(v => v * scale).ToArray(), c, 1e-10);
                        break;
                    }
                    case "PoincareLogMap":
                    {
                        var diff = Mobius(p.Select(v => -v).ToArray(), q, c, 1e-10);
                        double normDiff = Math.Sqrt(Dot(diff, diff)), lambda = 2.0 / Math.Max(1.0 - c * Dot(p, p), 1e-10);
                        if (normDiff < 1e-10) { row = new double[dim]; break; }
                        double scale = 2.0 / (sqrtC * lambda) * Atanh(Math.Min(sqrtC * normDiff, 1.0 - 1e-10)) / normDiff;
                        row = diff.Select(v => v * scale).ToArray();
                        break;
                    }
                    default:
                    {
                        var diff = Mobius(p.Select(v => -v).ToArray(), q, c, 1e-10);
                        r.Data[b] = 2.0 / sqrtC * Atanh(Math.Min(sqrtC * Math.Sqrt(Dot(diff, diff)), 1.0 - 1e-10));
                        continue;
                    }
                }
                for (int i = 0; i < dim; i++) r[b, i] = row[i];
            }
            return r;
        }

        public static Nd HyperbolicLinear(Nd input, Nd weights, Nd biases, double c)
        {
            const double epsilon = 1e-5;
            int batch = input.Shape[0], f = input.Shape[1], outputs = weights.Shape[0];
            double sqrtC = Math.Sqrt(c);
            var r = new Nd(new[] { batch, outputs });
            for (int b = 0; b < batch; b++)
            {
                var x = Project(Enumerable.Range(0, f).Select(i => input[b, i]).ToArray(), c, epsilon);
                for (int o = 0; o < outputs; o++)
                {
                    var w = Enumerable.Range(0, f).Select(i => weights[o, i]).ToArray();
                    var normW = Math.Sqrt(Dot(w, w));
                    var weightScale = normW > epsilon ? Math.Tanh(sqrtC * normW / 2.0) / (sqrtC * normW) : 0.5;
                    var t = Mobius(x, w.Select(v => v * weightScale).ToArray(), c, epsilon);
                    var bias = Project(Enumerable.Range(0, f).Select(i => biases[o, i]).ToArray(), c, epsilon);
                    var final = Mobius(t, bias, c, epsilon);
                    r[b, o] = 2.0 / sqrtC * Atanh(Math.Min(sqrtC * Math.Sqrt(Dot(final, final)), 1.0 - epsilon));
                }
            }
            return r;
        }
    }
}
