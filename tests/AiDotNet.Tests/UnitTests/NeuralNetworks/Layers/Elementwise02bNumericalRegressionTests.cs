using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

public class Elementwise02bNumericalRegressionTests
{
    // Reference values below come from the official HiPPO transition formulas (Gu et al. 2020,
    // github.com/HazyResearch/hippo-code, transition()) evaluated with numpy, and from
    // scipy.linalg.expm for the ZOH Van Loan matrix. Generator: numpy 2.4.1 / scipy 1.18.1.

    [Theory]
    [InlineData("legs", new[]
    {
        -1.0, 0.0, 0.0, 0.0,
        -1.7320508075688772, -2.0, 0.0, 0.0,
        -2.23606797749979, -3.8729833462074175, -3.0, 0.0,
        -2.6457513110645907, -4.582575694955841, -5.916079783099616, -4.0,
    }, new[] { 1.0, 1.7320508075688772, 2.23606797749979, 2.6457513110645907 })]
    [InlineData("legt", new[]
    {
        -0.5, 0.8660254037844386, -1.118033988749895, 1.3228756555322954,
        -0.8660254037844386, -1.5, 1.9364916731037085, -2.29128784747792,
        -1.118033988749895, -1.9364916731037085, -2.5, 2.9580398915498085,
        -1.3228756555322954, -2.29128784747792, -2.9580398915498085, -3.5,
    }, new[] { 0.5, 0.8660254037844386, 1.118033988749895, 1.3228756555322954 })]
    [InlineData("lagt", new[]
    {
        -0.5, 0.0, 0.0, 0.0,
        -1.0, -0.5, 0.0, 0.0,
        -1.0, -1.0, -0.5, 0.0,
        -1.0, -1.0, -1.0, -0.5,
    }, new[] { 1.0, 1.0, 1.0, 1.0 })]
    public void HippoOperator_Order4_MatchesOfficialTransitionFormulas(string measure, double[] a, double[] b)
    {
        using var layer = new HippoMemoryCellLayer<double>(hiddenSize: 1, inputSize: 1, memoryOrder: 4,
            measure: measure, discretization: "bilinear", timeStep: 0.25, useGate: false);
        var method = typeof(HippoMemoryCellLayer<double>).GetMethod("CreateHippoOperator",
            BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new MissingMethodException("CreateHippoOperator");
        var result = method.Invoke(layer, new object[] { measure, 4 })
            ?? throw new InvalidOperationException("CreateHippoOperator returned null.");
        var (actualA, actualB) = ((Tensor<double>, Tensor<double>))result;
        Assert.Equal(new[] { 4, 4 }, actualA.Shape.ToArray());
        Assert.Equal(new[] { 1, 4 }, actualB.Shape.ToArray());
        AssertClose(a, actualA.ToArray(), 1e-12);
        AssertClose(b, actualB.ToArray(), 1e-12);
    }

    [Theory]
    [InlineData("legt", "forward",
        new[] { 0.875, 0.21650635094610965, -0.2795084971874737, -0.21650635094610965, 0.625, 0.4841229182759271, -0.2795084971874737, -0.4841229182759271, 0.3749999999999999 },
        new[] { 0.125, 0.21650635094610965, 0.2795084971874737 })]
    [InlineData("legt", "backward",
        new[] { 0.8802228412256268, 0.17368754616289575, -0.09965762573815218, -0.17368754616289572, 0.6239554317548746, 0.21576508892520427, -0.09965762573815218, -0.21576508892520427, 0.5682451253481894 },
        new[] { 0.11977715877437324, 0.17368754616289572, 0.09965762573815219 })]
    [InlineData("legt", "bilinear",
        new[] { 0.8828880045480386, 0.20481328480632546, -0.16271557766911487, -0.20481328480632544, 0.6009096077316658, 0.31706060366905525, -0.16271557766911485, -0.3170606036690552, 0.48266060261512217 },
        new[] { 0.11711199545196135, 0.20481328480632546, 0.16271557766911487 })]
    [InlineData("legt", "zoh",
        new[] { 0.8855314606701583, 0.20959652551394123, -0.15194717472579877, -0.2095965255139412, 0.5904524717252906, 0.3042785971326915, -0.15194717472579874, -0.3042785971326917, 0.4863814479098204 },
        new[] { 0.11446853932984165, 0.2095965255139412, 0.15194717472579883 })]
    [InlineData("lagt", "forward",
        new[] { 0.875, 0.0, 0.0, -0.25, 0.875, 0.0, -0.25, -0.25, 0.875 },
        new[] { 0.25, 0.25, 0.25 })]
    [InlineData("lagt", "backward",
        new[] { 0.8888888888888888, 0.0, 0.0, -0.19753086419753085, 0.8888888888888888, 0.0, -0.15363511659807955, -0.19753086419753085, 0.8888888888888888 },
        new[] { 0.2222222222222222, 0.1728395061728395, 0.13443072702331962 })]
    [InlineData("lagt", "bilinear",
        new[] { 0.8823529411764706, 0.0, 0.0, -0.22145328719723184, 0.8823529411764706, 0.0, -0.19539995929167517, -0.22145328719723184, 0.8823529411764706 },
        new[] { 0.23529411764705882, 0.20761245674740483, 0.18318746183594542 })]
    [InlineData("lagt", "zoh",
        new[] { 0.8824969025845955, 0.0, 0.0, -0.22062422564614886, 0.8824969025845955, 0.0, -0.19304619744038023, -0.22062422564614886, 0.8824969025845955 },
        new[] { 0.23500619483080917, 0.2062422564614885, 0.17985013841927197 })]
    public void HippoTransitions_Order3_MatchNumpyReference(string measure, string mode, double[] ad, double[] bd)
    {
        // The layer stores Ad^T and Bd^T (row-vector convention); compare against Ad and Bd.
        var precise = HippoTransitionMatrices<double>(measure, mode, order: 3, step: 0.25);
        AssertClose(ad, Transpose(precise.State, 3), 1e-12);
        AssertClose(bd, precise.Input, 1e-12);
        var single = HippoTransitionMatrices<float>(measure, mode, order: 3, step: 0.25);
        AssertClose(ad, Transpose(single.State, 3), 2e-6);
        AssertClose(bd, single.Input, 2e-6);
    }

    [Theory]
    [InlineData("forward", new[] { 0.55, 0.2598076211353315, 0.816357698556081 })]
    [InlineData("backward", new[] { 0.5, -0.06339745962155616, 0.21855018016122044 })]
    [InlineData("bilinear", new[] { 0.52, 0.03856406460551015, 0.32654995044041657 })]
    [InlineData("zoh", new[] { 0.5, -0.01786327949540828, 0.28108225192402875 })]
    public void HippoLegSUpdate_Order3_MatchesNumpyReference(string mode, double[] expected)
    {
        // c = [0.4, -0.3, 0.2], f = 0.7 at scaled time 2: dt = 1/2 (zoh interval log(3/2)). A swapped
        // triangle or a missing transpose in the LegS solve changes every entry past the first.
        AssertClose(expected, LegSUpdate<double>(mode), 1e-12);
        AssertClose(expected, LegSUpdate<float>(mode), 2e-6);
    }

    [Fact]
    public void HippoZoh_Order8_MatchesScipyExpmInDoubleAndFloat()
    {
        // ZOH with interval 0.5 on LegS order 8: the Van Loan matrix has 1-norm ~ 29, so the Pade-13
        // path scales and squares. Expected: scipy.linalg.expm of [[A dt, B dt], [0, 0]].
        double[] ad =
        {
            0.6065306597126336, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -0.41335603580156666, 0.36787944117144233, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -0.1136981165520909, -0.5606115547776246, 0.22313016014842987, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.12202417239796955, 0.11966906692358079, -0.5194014963577206, 0.1353352832366127, 0.0, 0.0, 0.0, 0.0,
            0.10228835971273191, 0.2326489129658125, 0.3379827410765774, -0.42266103098595087, 0.0820849986238987, 0.0, 0.0, 0.0,
            -0.0403419466133396, -0.020234882039330937, 0.1425607280125868, 0.45201146546433446, -0.3213603484930047, 0.04978706836786399, 0.0, 0.0,
            -0.08493231044794448, -0.15630252013613827, -0.1858510847146531, -0.03953873565869212, 0.47540861069811824, -0.23425856045404067, 0.030197383422318567, 0.0,
            -0.0021434669712627526, -0.03251526676257592, -0.11873385884150144, -0.24553983570983753, -0.2180488439753677, 0.44143795179062184, -0.16591953280571092, 0.018315638888734245,
        };
        double[] bd =
        {
            0.3934693402873666, 0.41335603580156677, 0.11369811655209087, -0.12202417239796956,
            -0.10228835971273198, 0.040341946613339355, 0.08493231044794469, 0.0021434669712626572,
        };

        var precise = HippoTransitionMatrices<double>("legs", "zoh", order: 8, step: 0.5);
        AssertClose(ad, Transpose(precise.State, 8), 1e-12);
        AssertClose(bd, precise.Input, 1e-12);
        // The same algorithm evaluated in float32 numpy is within 1.5e-7 of scipy; 2e-5 leaves room
        // for a different matmul accumulation order, and is ~100x tighter than an error that doubles
        // per extra squaring would allow.
        var single = HippoTransitionMatrices<float>("legs", "zoh", order: 8, step: 0.5);
        AssertClose(ad, Transpose(single.State, 8), 2e-5);
        AssertClose(bd, single.Input, 2e-5);
    }

    [Theory]
    [InlineData("legs", "forward")]
    [InlineData("legs", "backward")]
    [InlineData("legs", "bilinear")]
    [InlineData("legs", "zoh")]
    [InlineData("legt", "forward")]
    [InlineData("legt", "backward")]
    [InlineData("legt", "bilinear")]
    [InlineData("legt", "zoh")]
    [InlineData("lagt", "forward")]
    [InlineData("lagt", "backward")]
    [InlineData("lagt", "bilinear")]
    [InlineData("lagt", "zoh")]
    public void HippoSequence_Order3_InputGradientMatchesCentralDifference(string measure, string mode)
    {
        using var layer = new HippoMemoryCellLayer<double>(hiddenSize: 2, inputSize: 1, memoryOrder: 3,
            measure: measure, discretization: mode, timeStep: 0.25, useGate: false);
        double[] values = { 0.25, -0.5, 0.75, 0.1 };
        var input = new Tensor<double>((double[])values.Clone(), new[] { 4, 1 });

        Tensor<double>? gradient;
        using (var tape = new GradientTape<double>())
        {
            var loss = AiDotNetEngine.Current.ReduceSum(layer.Forward(input), new[] { 0, 1 }, false);
            var gradients = tape.ComputeGradients(loss, new[] { input });
            Assert.True(gradients.TryGetValue(input, out gradient));
        }

        Assert.NotNull(gradient);
        Assert.Equal(new[] { 4, 1 }, gradient.Shape.ToArray());
        const double h = 1e-6;
        for (int i = 0; i < values.Length; i++)
        {
            double numeric = (SumOutput(layer, values, i, h) - SumOutput(layer, values, i, -h)) / (2 * h);
            Assert.InRange(Math.Abs(gradient.GetFlat(i) - numeric), 0, 1e-6 * Math.Max(1.0, Math.Abs(numeric)));
        }
    }

    [Theory]
    [InlineData("legs", "backward")]
    [InlineData("legs", "bilinear")]
    [InlineData("legt", "backward")]
    [InlineData("legt", "bilinear")]
    [InlineData("lagt", "backward")]
    [InlineData("lagt", "bilinear")]
    public void HalfHippoSequence_Order3_MatchesDoubleOutputsAndGradients(string measure, string mode)
    {
        using var reference = new HippoMemoryCellLayer<double>(hiddenSize: 2, inputSize: 1, memoryOrder: 3,
            measure: measure, discretization: mode, timeStep: 0.25, useGate: false);
        using var layer = new HippoMemoryCellLayer<Half>(hiddenSize: 2, inputSize: 1, memoryOrder: 3,
            measure: measure, discretization: mode, timeStep: 0.25, useGate: false);
        var parameters = reference.GetParameters().ToArray();
        layer.SetParameters(new Vector<Half>(parameters.Select(p => (Half)p).ToArray()));
        // Compare against the double layer with the Half-rounded parameters, so only the arithmetic differs.
        reference.SetParameters(new Vector<double>(parameters.Select(p => (double)(Half)p).ToArray()));

        double[] values = { 0.25, 0.5, -0.25, 0.75 };
        var (expectedOutput, expectedGradient) = OutputAndInputGradient(reference,
            new Tensor<double>((double[])values.Clone(), new[] { 4, 1 }));
        var (actualOutput, actualGradient) = OutputAndInputGradient(layer,
            new Tensor<Half>(values.Select(v => (Half)v).ToArray(), new[] { 4, 1 }));

        for (int i = 0; i < expectedOutput.Length; i++)
        {
            double expected = expectedOutput[i];
            Assert.InRange(Math.Abs((double)actualOutput[i] - expected), 0, 0.03 * Math.Max(1.0, Math.Abs(expected)));
        }

        for (int i = 0; i < expectedGradient.Length; i++)
        {
            double expected = expectedGradient[i];
            Assert.InRange(Math.Abs((double)actualGradient[i] - expected), 0, 0.05 * Math.Max(1.0, Math.Abs(expected)));
        }
    }

    private static (T[] Output, T[] Gradient) OutputAndInputGradient<T>(HippoMemoryCellLayer<T> layer, Tensor<T> input)
    {
        using var tape = new GradientTape<T>();
        var output = layer.Forward(input);
        var loss = AiDotNetEngine.Current.ReduceSum(output, new[] { 0, 1 }, false);
        var gradients = tape.ComputeGradients(loss, new[] { input });
        Assert.True(gradients.TryGetValue(input, out var gradient));
        Assert.NotNull(gradient);
        return (output.ToArray(), gradient.ToArray());
    }

    private static double SumOutput(HippoMemoryCellLayer<double> layer, double[] values, int index, double delta)
    {
        var shifted = (double[])values.Clone();
        shifted[index] += delta;
        var output = layer.Forward(new Tensor<double>(shifted, new[] { shifted.Length, 1 }));
        return output.ToArray().Sum();
    }

    private static (double[] State, double[] Input) HippoTransitionMatrices<T>(
        string measure, string mode, int order, double step)
    {
        using var layer = new HippoMemoryCellLayer<T>(hiddenSize: 1, inputSize: 1, memoryOrder: order,
            measure: measure, discretization: mode, timeStep: step, useGate: false);
        var method = typeof(HippoMemoryCellLayer<T>).GetMethod("BuildDiscreteTransition",
            BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new MissingMethodException("BuildDiscreteTransition");
        var result = method.Invoke(layer, new object[] { step })
            ?? throw new InvalidOperationException("BuildDiscreteTransition returned null.");
        var transition = ((Tensor<T>, Tensor<T>))result;
        return (transition.Item1.Cast<double>().ToArray(), transition.Item2.Cast<double>().ToArray());
    }

    private static double[] LegSUpdate<T>(string mode)
    {
        using var layer = new HippoMemoryCellLayer<T>(hiddenSize: 1, inputSize: 1, memoryOrder: 3,
            measure: "legs", discretization: mode, useGate: false);
        var memory = new Tensor<double>(new[] { 0.4, -0.3, 0.2 }, new[] { 1, 1, 3 }).Cast<T>();
        var input = new Tensor<double>(new[] { 0.7 }, new[] { 1, 1 }).Cast<T>();
        var method = typeof(HippoMemoryCellLayer<T>).GetMethod("UpdateMemory",
            BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new MissingMethodException("UpdateMemory");
        var result = method.Invoke(layer, new object[] { memory, input, 2, 1 }) as Tensor<T>
            ?? throw new InvalidOperationException("UpdateMemory returned no tensor.");
        Assert.Equal(new[] { 1, 1, 3 }, result.Shape.ToArray());
        return result.Cast<double>().ToArray();
    }

    private static double[] Transpose(double[] matrix, int n)
    {
        var transposed = new double[matrix.Length];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                transposed[(j * n) + i] = matrix[(i * n) + j];
        return transposed;
    }

    private static void AssertClose(double[] expected, double[] actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance,
                $"Index {i}: expected {expected[i]:R}, actual {actual[i]:R}, tolerance {tolerance}.");
        }
    }

    [Theory]
    [InlineData("forward")]
    [InlineData("backward")]
    [InlineData("bilinear")]
    [InlineData("zoh")]
    public void HippoScalarTransitions_MatchClosedFormsInDoubleAndFloat(string mode)
    {
        double state = mode switch { "forward" => 0.75, "backward" => 0.8,
            "bilinear" => 7.0 / 9.0, _ => Math.Exp(-0.25) };
        double input = mode switch { "forward" => 0.25, "backward" => 0.2,
            "bilinear" => 2.0 / 9.0, _ => 1 - Math.Exp(-0.25) };
        var precise = HippoTransition<double>(mode);
        var single = HippoTransition<float>(mode);
        Assert.InRange(Math.Abs(precise.State - state), 0, 1e-12);
        Assert.InRange(Math.Abs(precise.Input - input), 0, 1e-12);
        Assert.InRange(Math.Abs(single.State - state), 0, 1e-5);
        Assert.InRange(Math.Abs(single.Input - input), 0, 1e-5);
    }

    [Theory]
    [InlineData("forward", 0.75, 0.25)]
    [InlineData("backward", 0.8, 0.2)]
    [InlineData("bilinear", 0.7777777777777778, 0.2222222222222222)]
    [InlineData("zoh", 0.7788007830714049, 0.22119921692859512)]
    public void HippoScalarTransitions_HalfRemainsSupported(string mode, double state, double input)
    {
        var actual = HippoTransition<Half>(mode);
        Assert.InRange(Math.Abs(actual.State - state), 0, 0.002);
        Assert.InRange(Math.Abs(actual.Input - input), 0, 0.002);
    }

    private static (double State, double Input) HippoTransition<T>(string mode)
    {
        using var layer = new HippoMemoryCellLayer<T>(hiddenSize: 1, inputSize: 1, memoryOrder: 1,
            measure: "legs", discretization: mode, timeStep: 0.25, useGate: false);
        var method = typeof(HippoMemoryCellLayer<T>).GetMethod("BuildDiscreteTransition",
            BindingFlags.Instance | BindingFlags.NonPublic)!;
        var transition = ((Tensor<T>, Tensor<T>))method.Invoke(layer, new object[] { 0.25 })!;
        return (transition.Item1.Cast<double>().GetFlat(0), transition.Item2.Cast<double>().GetFlat(0));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void HalfKosmosAttention_CausalMaskIsFiniteAndBlocksFutureValues(bool xPos)
    {
        var query = new Tensor<Half>(new[] { 2, 2 });
        var key = new Tensor<Half>(new[] { 2, 2 });
        var values = new Tensor<Half>(new Half[] { (Half)1, (Half)2, (Half)8, (Half)16 }, new[] { 2, 2 });
        var actual = MultiHeadAttentionMath<Half>.Attend(query, key, values, heads: 1, causal: true, xPos: xPos);
        Assert.Equal(new Half[] { (Half)1, (Half)2, (Half)4.5, (Half)9 }, actual.ToArray());
    }

    [Fact]
    public void StateCacheCompression_SubnormalFiniteRangePreservesTheEndpoints()
    {
        var cache = new SSMStateCache<float>(enableCompression: true, compressionBitWidth: 8);
        var state = new Tensor<float>(new[] { 0.0f, 1e-40f }, new[] { 2 });
        cache.CacheSSMState(0, state);
        Assert.Equal(state.ToArray(), cache.GetSSMState(0)!.ToArray());
    }
}
