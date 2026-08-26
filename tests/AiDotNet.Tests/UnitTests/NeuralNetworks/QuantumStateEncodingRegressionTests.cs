using System;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Regression tests for the two quantum state-encoding defects that made
/// <see cref="QuantumNeuralNetwork{T}"/> fail 10 of its 30 model-family invariants (issue class N1,
/// non-finite training).
/// </summary>
/// <remarks>
/// <para>
/// <b>Defect 1 — Sqrt of the raw feature.</b> <c>PrepareQuantumState</c> built amplitudes as
/// <c>Sqrt(x_i)</c>, which is only defined for a non-negative vector that already sums to one. A
/// single negative feature returns NaN, and that NaN propagates through every layer into the
/// Born-rule measurement. The generated fixture's own input carries 71 negative values out of 128,
/// so the model returned NaN for the very input its tests feed it.
/// </para>
/// <para>
/// <b>Defect 2 — normalising by the squared norm.</b> <c>QuantumLayer</c> divided the state by
/// <c>sum(|state|^2) + eps</c> rather than its square root, leaving a state of length
/// <c>1/||state||</c> instead of 1. Its own comment and its GPU path both specify the square root,
/// so the CPU and GPU paths disagreed on the state convention.
/// </para>
/// <para>
/// The encoding these pin is amplitude encoding as Qiskit's <c>initialize</c> and PennyLane's
/// <c>AmplitudeEmbedding</c> define it: normalise the feature vector to unit L2 norm and use it
/// directly as the amplitudes, so <c>sum |psi_i|^2 == 1</c>.
/// </para>
/// </remarks>
public class QuantumStateEncodingRegressionTests
{
    private static Tensor<float> Input(Func<int, float> f, int n = 128)
    {
        var t = new Tensor<float>(new[] { n });
        for (int i = 0; i < n; i++) t[i] = f(i);
        return t;
    }

    private static void AssertAllFinite(Tensor<float> t, string what)
    {
        for (int i = 0; i < t.Length; i++)
        {
            Assert.False(float.IsNaN(t[i]), $"{what}[{i}] is NaN");
            Assert.False(float.IsInfinity(t[i]), $"{what}[{i}] is Infinity");
        }
    }

    [Fact]
    public void NegativeFeatures_DoNotProduceNaN()
    {
        // The exact failure mode: Sqrt of a negative amplitude. Every element here is negative, so
        // the old encoding returned NaN for all of them.
        using var model = new QuantumNeuralNetwork<float>();
        using var input = Input(i => -0.5f - (0.001f * i));
        using var output = model.Predict(input);

        Assert.True(output.Length > 0);
        AssertAllFinite(output, "output");
    }

    [Fact]
    public void MixedSignFeatures_DoNotProduceNaN()
    {
        // Mirrors the generated fixture, which carries 71 negative values out of 128.
        using var model = new QuantumNeuralNetwork<float>();
        using var input = Input(i => (float)Math.Sin(i * 0.7));
        using var output = model.Predict(input);

        AssertAllFinite(output, "output");
    }

    [Fact]
    public void ZeroVector_DoesNotDivideByZero()
    {
        // A zero vector has no direction to encode. It must fall back to a finite state rather than
        // dividing by a zero norm, which would reintroduce the NaN the encoding exists to avoid.
        using var model = new QuantumNeuralNetwork<float>();
        using var input = Input(_ => 0f);
        using var output = model.Predict(input);

        AssertAllFinite(output, "output");
    }

    [Fact]
    public void Output_IsNotCrushedTowardZero_BySquaredNormNormalisation()
    {
        // Defect 2's signature. Dividing by the SQUARED norm shrinks the state by roughly the feature
        // count at EACH of the two quantum layers; at 128 features that drove Predict to ~4.6e-4 and
        // starved the gradients. A correctly normalised state keeps the readout on a usable scale.
        using var model = new QuantumNeuralNetwork<float>();
        using var input = Input(i => (float)Math.Sin(i * 0.7));
        using var output = model.Predict(input);

        double maxAbs = 0.0;
        for (int i = 0; i < output.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(output[i]));

        Assert.True(maxAbs > 1e-3,
            $"output collapsed to {maxAbs:G6}; the squared-norm normalisation bug produced ~4.6e-4 here");
    }

    [Fact]
    public void Training_KeepsParametersFinite()
    {
        // Integration-level: the NaN used to reach the parameters through training, which is what
        // ForwardPass_ShouldBeFinite_AfterTraining and Training_ShouldChangeParameters observed.
        using var model = new QuantumNeuralNetwork<float>();
        using var input = Input(i => (float)Math.Sin(i * 0.7));
        using var target = new Tensor<float>(new[] { 1 });
        target[0] = 0.25f;

        for (int step = 0; step < 5; step++) model.Train(input, target);

        var parameters = model.GetParameters();
        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.False(float.IsNaN(parameters[i]), $"parameter[{i}] is NaN after training");
            Assert.False(float.IsInfinity(parameters[i]), $"parameter[{i}] is Infinity after training");
        }

        using var output = model.Predict(input);
        AssertAllFinite(output, "post-training output");
    }

    [Fact]
    public void QuantumLayer_LargeFiniteInput_NormalizesWithoutOverflow()
    {
        using var layer = new QuantumLayer<float>(inputSize: 4, outputSize: 4, numQubits: 2);
        using var input = Input(i => (i & 1) == 0 ? float.MaxValue : -float.MaxValue, n: 4);
        using var output = layer.Forward(input);

        AssertAllFinite(output, "quantum-layer output");
        Assert.Equal(1.0f, output.ToArray().Sum(), 4);
    }

    /// <summary>
    /// Pins the GPU prescale against the CPU for the rows that sit at the edges of float: an
    /// all-zero row and a row of the smallest positive normal.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>The regression.</b> The GPU path used to floor the per-row maximum with
    /// <c>Clamp(max, float.Epsilon, float.MaxValue)</c>. That reads as the safe choice and is the
    /// opposite of one: <c>CL_FP_DENORM</c> is <i>optional</i> in OpenCL, so a conforming device
    /// may flush subnormals in arithmetic and turn the floor straight back into 0. The chain is
    /// then <c>Sqrt(0) = 0</c>, <c>Reciprocal(0) = +Inf</c>, and <c>0 * Inf = NaN</c> -- so on such
    /// a device <i>every all-zero row</i> came back NaN, not merely a contrived subnormal one. The
    /// fix mirrors the CPU's <c>TensorWhere(max &gt; 0, max, 1)</c>: a normal-valued sentinel
    /// cannot degrade on any device.
    /// </para>
    /// <para>
    /// <b>Why the smallest normal and not the smallest subnormal.</b> A subnormal row cannot be a
    /// hard CPU-parity assertion, because flushing it is behaviour OpenCL explicitly permits -- the
    /// measured device here returns 0 from the max reduction for such a row while preserving the
    /// value through <c>Abs</c>. The smallest positive normal is the strongest input every
    /// IEEE-754 device must handle exactly, and it still covers what the prescale is for: its
    /// reciprocal (~8.5e37) is within one power of two of overflowing, which is why the scale is
    /// applied as two <c>1/sqrt(max)</c> multiplies rather than one <c>1/max</c>.
    /// </para>
    /// <para>
    /// The subnormal row is still asserted, just on the invariant that actually holds everywhere:
    /// it must be finite, and it must either match the CPU or flush to zero -- with which of the
    /// two required decided by measuring the device rather than by accepting either.
    /// </para>
    /// </remarks>
    [SkippableFact]
    [Trait("Category", "GPU")]
    public void QuantumLayer_GpuPrescaling_MatchesCpuForZeroAndSmallestNormalRows()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "Requires DirectGpu execution.");

        var savedEngine = AiDotNetEngine.Current;
        float[] cpuValues;
        float[] gpuValues;
        bool devicePreservesSubnormals;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            using var cpuLayer = new QuantumLayer<float>(inputSize: 4, outputSize: 4, numQubits: 2);
            using var cpuInput = CreateEdgeMagnitudeRows();
            using var cpuOutput = cpuLayer.Forward(cpuInput);
            var parameters = cpuLayer.GetParameters();
            cpuValues = cpuOutput.ToArray();

            AiDotNetEngine.Current = gpu;
            using var gpuLayer = new QuantumLayer<float>(inputSize: 4, outputSize: 4, numQubits: 2);
            gpuLayer.SetParameters(parameters);
            var backend = gpu.GetBackend();
            Assert.NotNull(backend);
            devicePreservesSubnormals = PreservesSubnormalsInArithmetic(backend);
            using var gpuHostInput = CreateEdgeMagnitudeRows();
            using var gpuInput = GpuTensorHelper.UploadToGpu(backend, gpuHostInput, GpuTensorRole.Input);
            using var gpuOutput = gpuLayer.ForwardGpu(gpuInput);
            gpuValues = gpuOutput.ToArray();
        }
        finally
        {
            AiDotNetEngine.Current = savedEngine;
        }

        Assert.Equal(cpuValues.Length, gpuValues.Length);

        // Finiteness is device-independent: flushing a subnormal is allowed, producing NaN is not.
        for (int i = 0; i < gpuValues.Length; i++)
        {
            Assert.True(!float.IsNaN(gpuValues[i]) && !float.IsInfinity(gpuValues[i]),
                $"GPU output[{i}] is non-finite: {gpuValues[i]}.");
        }

        // Row 0 (all zeros) and row 2 (smallest normal) must match the CPU on every device.
        foreach (int row in new[] { ZeroRow, SmallestNormalRow })
        {
            for (int i = row * RowWidth; i < (row + 1) * RowWidth; i++)
            {
                Assert.True(Math.Abs(cpuValues[i] - gpuValues[i]) <= 1e-4f,
                    $"CPU/GPU output mismatch at {i}: CPU={cpuValues[i]:G9}, GPU={gpuValues[i]:G9}.");
            }
        }

        Assert.Equal(0.0f, RowOf(gpuValues, ZeroRow).Sum());
        Assert.Equal(1.0f, RowOf(gpuValues, SmallestNormalRow).Sum(), 4);
        Assert.Equal(1.0f, RowOf(cpuValues, SmallestSubnormalRow).Sum(), 4);

        // Row 1 (smallest subnormal) is asserted against what this device was measured to do.
        var subnormal = RowOf(gpuValues, SmallestSubnormalRow).ToArray();
        if (devicePreservesSubnormals)
        {
            for (int i = 0; i < subnormal.Length; i++)
            {
                int flat = (SmallestSubnormalRow * RowWidth) + i;
                Assert.True(Math.Abs(cpuValues[flat] - subnormal[i]) <= 1e-4f,
                    $"Device preserves subnormals, so output[{flat}] must match the CPU: "
                    + $"CPU={cpuValues[flat]:G9}, GPU={subnormal[i]:G9}.");
            }
        }
        else
        {
            Assert.All(subnormal, value => Assert.Equal(0.0f, value));
        }
    }

    private const int RowWidth = 4;
    private const int ZeroRow = 0;
    private const int SmallestSubnormalRow = 1;
    private const int SmallestNormalRow = 2;

    private static IEnumerable<float> RowOf(float[] values, int row)
        => values.Skip(row * RowWidth).Take(RowWidth);

    /// <summary>
    /// Measures whether this device keeps subnormals through the max reduction the prescale uses.
    /// <c>Abs</c> alone does not answer it -- that is a sign-bit mask and survives flush-to-zero
    /// hardware unchanged; the comparison inside the reduction is where a device drops them.
    /// </summary>
    private static bool PreservesSubnormalsInArithmetic(IDirectGpuBackend backend)
    {
        using var source = backend.AllocateBuffer(RowWidth);
        backend.Fill(source, float.Epsilon, RowWidth);
        using var reduced = backend.AllocateBuffer(1);
        backend.MaxAxis(source, reduced, 1, RowWidth);
        var host = new float[1];
        backend.DownloadBuffer(reduced, host);
        return host[0] > 0f;
    }

    private static Tensor<float> CreateEdgeMagnitudeRows()
    {
        var input = new Tensor<float>([3, RowWidth]);
        for (int i = RowWidth; i < 2 * RowWidth; i++)
            input[i] = (i & 1) == 0 ? float.Epsilon : -float.Epsilon;
        for (int i = 2 * RowWidth; i < 3 * RowWidth; i++)
            input[i] = (i & 1) == 0 ? MinNormal : -MinNormal;
        return input;
    }

    /// <summary>Smallest positive <b>normal</b> float; every IEEE-754 device represents it exactly.</summary>
    private const float MinNormal = 1.17549435E-38f;

    [Fact]
    public void NonFiniteFeatures_AreRejectedBeforeQuantumNormalization()
    {
        using var model = new QuantumNeuralNetwork<float>();
        foreach (float nonFinite in new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity })
        {
            using var input = Input(i => i == 7 ? nonFinite : 0.25f);
            var error = Assert.Throws<ArgumentException>(() => model.Predict(input));
            Assert.Contains("index 7", error.Message, StringComparison.Ordinal);
        }
    }
}
