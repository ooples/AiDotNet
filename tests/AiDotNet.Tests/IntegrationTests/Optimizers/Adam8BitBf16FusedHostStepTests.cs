using System;
using System.Collections.Generic;
using AiDotNet.MixedPrecision;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// #2447: the eager BF16-moment step of <see cref="Adam8BitOptimizer{T, TInput, TOutput}"/> built about ten
/// parameter-sized tensors per parameter (two widened moments, the scaled gradient, the squared gradient, both bias
/// corrected moments, the square root, the denominator, the quotient and the update). At ParaformerLarge's ~220M
/// parameters that is several GB of transient storage per step on top of the model, its gradients and the tape, and the
/// nightly heavy lane ran out of its 11 GB heap inside that step. The step now runs as one pass over the host arrays.
/// These tests pin the two halves of that: it is the same update, and it no longer allocates per parameter.
/// "The same update" is checked twice. Against a scalar statement of the recurrence the fused pass must agree bit
/// for bit at every length, which is what holds its vectorized blocks and its tail to one result. Against the engine
/// operations the step used before, it must agree to within rounding: an engine kernel is free to round a scalar
/// division differently from an IEEE divide, so the last bit of an update is not owed there.
/// </summary>
[Collection("FusedTrainingSerial")]
public class Adam8BitBf16FusedHostStepTests
{
    private const double LearningRate = 1e-2;
    private const double Beta1 = 0.9;
    private const double Beta2 = 0.999;
    private const double Epsilon = 1e-8;

    internal static Adam8BitOptimizer<float, Tensor<float>, Tensor<float>> NewOptimizer(double weightDecay, bool decoupled)
        => new(null, new Adam8BitOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            UseBFloat16MomentStorage = true,
            UseAMSGrad = false,
            // The reference below is the update rule on the raw gradient; the default norm clip is not part of it.
            EnableGradientClipping = false,
            InitialLearningRate = LearningRate,
            Beta1 = Beta1,
            Beta2 = Beta2,
            Epsilon = Epsilon,
            WeightDecay = weightDecay,
            UseDecoupledWeightDecay = decoupled,
        });

    /// <summary>
    /// How far a parameter may sit from the engine-operation recurrence: 1% of the largest update one step can make
    /// (the learning rate). A last-bit difference in an update is ~1e-9 here; the bound has to admit the rare case
    /// where that bit moves a moment across a BF16 rounding boundary (a 2^-8 relative change in that moment).
    /// </summary>
    internal const float EngineTolerance = (float)(LearningRate * 1e-2);

    /// <summary>
    /// 1 - beta^step as the optimizer forms it for a float model: from the float beta it stepped with, widened.
    /// </summary>
    private static float BiasCorrection(double beta, int step) => (float)(1.0 - Math.Pow((float)beta, step));

    // BitConverter.SingleToInt32Bits is not available on net471.
    private static int Bits(float value) => BitConverter.ToInt32(BitConverter.GetBytes(value), 0);

    internal static float[] RandomValues(int length, int seed, double scale)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var values = new float[length];
        for (int i = 0; i < length; i++) values[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return values;
    }

    /// <summary>
    /// The recurrence the step ran before it was fused, written with the same engine operations in the same order and
    /// keeping its moments as BF16 between steps.
    /// </summary>
    private sealed class TensorOpReference
    {
        private readonly CpuEngine _engine = new();
        private readonly ushort[] _m;
        private readonly ushort[] _v;
        private readonly double _weightDecay;
        private readonly bool _decoupled;
        private int _step;

        public TensorOpReference(int length, double weightDecay, bool decoupled)
        {
            _m = new ushort[length];
            _v = new ushort[length];
            _weightDecay = weightDecay;
            _decoupled = decoupled;
        }

        public void Step(Tensor<float> param, Tensor<float> grad)
        {
            _step++;
            float beta1 = (float)Beta1, beta2 = (float)Beta2;
            float bc1 = BiasCorrection(Beta1, _step);
            float bc2 = BiasCorrection(Beta2, _step);
            if (_weightDecay > 0.0)
            {
                if (_decoupled) _engine.TensorMultiplyScalarInPlace(param, (float)(1.0 - LearningRate * _weightDecay));
                else grad = _engine.TensorAdd(grad, _engine.TensorMultiplyScalar(param, (float)_weightDecay));
            }

            var m = new Tensor<float>(param.Shape.ToArray());
            var v = new Tensor<float>(param.Shape.ToArray());
            for (int i = 0; i < m.Length; i++)
            {
                m[i] = BitConverterHelper.Bf16BitsToFloat(_m[i]);
                v[i] = BitConverterHelper.Bf16BitsToFloat(_v[i]);
            }

            var gradScaled = _engine.TensorMultiplyScalar(grad, 1f - beta1);
            _engine.TensorMultiplyScalarInPlace(m, beta1);
            _engine.TensorAddInPlace(m, gradScaled);
            var gradSq = _engine.TensorMultiply(grad, grad);
            _engine.TensorMultiplyScalarInPlace(gradSq, 1f - beta2);
            _engine.TensorMultiplyScalarInPlace(v, beta2);
            _engine.TensorAddInPlace(v, gradSq);
            for (int i = 0; i < m.Length; i++)
            {
                _m[i] = BitConverterHelper.FloatToBf16Bits(m[i]);
                _v[i] = BitConverterHelper.FloatToBf16Bits(v[i]);
            }

            var mHat = _engine.TensorDivideScalar(m, bc1);
            var vHat = _engine.TensorDivideScalar(v, bc2);
            var denom = _engine.TensorAddScalar(_engine.TensorSqrt(vHat), (float)Epsilon);
            var update = _engine.TensorMultiplyScalar(_engine.TensorDivide(mHat, denom), (float)LearningRate);
            _engine.TensorSubtractInPlace(param, update);
        }
    }

    /// <summary>Adam with BF16 moments, one element at a time, as the recurrence is written.</summary>
    internal sealed class ScalarReference
    {
        private readonly ushort[] _m;
        private readonly ushort[] _v;
        private readonly double _weightDecay;
        private readonly bool _decoupled;
        private int _step;

        public ScalarReference(float[] initial, double weightDecay, bool decoupled)
        {
            Parameters = (float[])initial.Clone();
            _m = new ushort[initial.Length];
            _v = new ushort[initial.Length];
            _weightDecay = weightDecay;
            _decoupled = decoupled;
        }

        public float[] Parameters { get; }

        public void Step(float[] gradient)
        {
            _step++;
            float beta1 = (float)Beta1, beta2 = (float)Beta2, epsilon = (float)Epsilon, lr = (float)LearningRate;
            float bc1 = BiasCorrection(Beta1, _step);
            float bc2 = BiasCorrection(Beta2, _step);
            for (int i = 0; i < Parameters.Length; i++)
            {
                float p = Parameters[i];
                float g = gradient[i];
                if (_weightDecay > 0.0)
                {
                    if (_decoupled) p *= (float)(1.0 - LearningRate * _weightDecay);
                    else g += p * (float)_weightDecay;
                }

                float m = BitConverterHelper.Bf16BitsToFloat(_m[i]) * beta1 + g * (1f - beta1);
                float v = BitConverterHelper.Bf16BitsToFloat(_v[i]) * beta2 + g * g * (1f - beta2);
                _m[i] = BitConverterHelper.FloatToBf16Bits(m);
                _v[i] = BitConverterHelper.FloatToBf16Bits(v);
                float mHat = m / bc1;
                float denom = (float)Math.Sqrt(v / bc2) + epsilon;
                Parameters[i] = p - mHat / denom * lr;
            }
        }
    }

    [Theory]
    // Lengths below one SIMD block, across several, and not a multiple of the block (so the scalar tail runs too);
    // each with no decay, decoupled decay and coupled decay.
    [InlineData(1, 0.0, true)]
    [InlineData(7, 0.0, true)]
    [InlineData(64, 0.0, true)]
    [InlineData(1003, 0.0, true)]
    [InlineData(1003, 0.05, true)]
    [InlineData(1003, 0.05, false)]
    [InlineData(20011, 0.05, true)]
    [InlineData(20011, 0.05, false)]
    public void FusedStep_MatchesTheAdamRecurrence_AcrossSteps(int length, double weightDecay, bool decoupled)
    {
        var optimizer = NewOptimizer(weightDecay, decoupled);
        var initial = RandomValues(length, seed: 2447, scale: 1.0);
        var param = new Tensor<float>(new[] { length });
        var engineExpected = new Tensor<float>(new[] { length });
        for (int i = 0; i < length; i++) { param[i] = initial[i]; engineExpected[i] = initial[i]; }
        var reference = new TensorOpReference(length, weightDecay, decoupled);
        var scalar = new ScalarReference(initial, weightDecay, decoupled);

        for (int step = 0; step < 6; step++)
        {
            // Gradients spanning several orders of magnitude, so v's rounding to BF16 matters to later steps.
            var g = RandomValues(length, seed: 100 + step, scale: Math.Pow(10, step % 3 - 2));
            var grad = new Tensor<float>(new[] { length });
            var gradCopy = new Tensor<float>(new[] { length });
            for (int i = 0; i < length; i++) { grad[i] = g[i]; gradCopy[i] = g[i]; }

            optimizer.Step(new TapeStepContext<float>(
                parameters: new[] { param },
                gradients: new Dictionary<Tensor<float>, Tensor<float>> { [param] = grad },
                loss: 0f));
            reference.Step(engineExpected, gradCopy);
            scalar.Step(g);

            for (int i = 0; i < length; i++)
            {
                Assert.True(g[i] == grad[i], $"step {step + 1}: the step modified the tape's gradient at {i}.");
                // Element-wise IEEE arithmetic in a fixed order: vector block or tail, the pass owes the same bits.
                Assert.True(Bits(scalar.Parameters[i]) == Bits(param[i]),
                    $"step {step + 1}, element {i}: the scalar recurrence gives {scalar.Parameters[i]:R}, the step {param[i]:R}.");
                Assert.True(Math.Abs(engineExpected[i] - param[i]) <= EngineTolerance,
                    $"step {step + 1}, element {i}: the engine recurrence gives {engineExpected[i]:R}, the step {param[i]:R}.");
            }
        }
    }

    /// <summary>
    /// A double parameter takes the same fused pass in double arithmetic, with the moments still rounded through float
    /// into their BF16 slots. Bit for bit against the recurrence written out per element.
    /// </summary>
    [Theory]
    [InlineData(1, 0.0, true)]
    [InlineData(1003, 0.0, true)]
    [InlineData(1003, 0.05, true)]
    [InlineData(1003, 0.05, false)]
    public void FusedStep_OnADoubleParameter_MatchesTheAdamRecurrence_AcrossSteps(int length, double weightDecay, bool decoupled)
    {
        var optimizer = new Adam8BitOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new Adam8BitOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                UseBFloat16MomentStorage = true,
                UseAMSGrad = false,
                EnableGradientClipping = false,
                InitialLearningRate = LearningRate,
                Beta1 = Beta1,
                Beta2 = Beta2,
                Epsilon = Epsilon,
                WeightDecay = weightDecay,
                UseDecoupledWeightDecay = decoupled,
            });
        var initial = RandomValues(length, seed: 2447, scale: 1.0);
        var param = new Tensor<double>(new[] { length });
        var expected = new double[length];
        for (int i = 0; i < length; i++) { param[i] = initial[i]; expected[i] = initial[i]; }
        var m = new ushort[length];
        var v = new ushort[length];

        for (int step = 1; step <= 6; step++)
        {
            var g = RandomValues(length, seed: 100 + step, scale: Math.Pow(10, step % 3 - 2));
            var grad = new Tensor<double>(new[] { length });
            for (int i = 0; i < length; i++) grad[i] = g[i];

            optimizer.Step(new TapeStepContext<double>(
                parameters: new[] { param },
                gradients: new Dictionary<Tensor<double>, Tensor<double>> { [param] = grad },
                loss: 0.0));

            double bc1 = 1.0 - Math.Pow(Beta1, step), bc2 = 1.0 - Math.Pow(Beta2, step);
            for (int i = 0; i < length; i++)
            {
                double p = expected[i];
                double gi = g[i];
                if (weightDecay > 0.0)
                {
                    if (decoupled) p *= 1.0 - LearningRate * weightDecay;
                    else gi += p * weightDecay;
                }

                double mi = BitConverterHelper.Bf16BitsToFloat(m[i]) * Beta1 + gi * (1.0 - Beta1);
                double vi = BitConverterHelper.Bf16BitsToFloat(v[i]) * Beta2 + gi * gi * (1.0 - Beta2);
                m[i] = BitConverterHelper.FloatToBf16Bits((float)mi);
                v[i] = BitConverterHelper.FloatToBf16Bits((float)vi);
                expected[i] = p - mi / bc1 / (Math.Sqrt(vi / bc2) + Epsilon) * LearningRate;

                Assert.True((double)g[i] == grad[i], $"step {step}: the step modified the tape's gradient at {i}.");
                Assert.True(BitConverter.DoubleToInt64Bits(expected[i]) == BitConverter.DoubleToInt64Bits(param[i]),
                    $"step {step}, element {i}: the recurrence gives {expected[i]:R}, the step {param[i]:R}.");
            }
        }
    }

#if NET5_0_OR_GREATER
    // GC.GetTotalAllocatedBytes, the process-wide allocation counter this reads, does not exist on net471.
    [Fact]
    public void FusedStep_AllocatesNoParameterSizedTemporaries()
    {
        const int length = 4 * 1024 * 1024;                      // 16 MB of float32
        const long parameterBytes = (long)length * sizeof(float);
        var optimizer = NewOptimizer(weightDecay: 0.05, decoupled: false);
        var param = new Tensor<float>(new[] { length });
        var grad = new Tensor<float>(new[] { length });
        var values = RandomValues(length, seed: 7, scale: 1.0);
        for (int i = 0; i < length; i++) { param[i] = values[i]; grad[i] = values[length - 1 - i]; }
        var context = new TapeStepContext<float>(
            parameters: new[] { param },
            gradients: new Dictionary<Tensor<float>, Tensor<float>> { [param] = grad },
            loss: 0f);

        // The first step allocates the resident BF16 moments (2 x 2 bytes per element); measure a later one.
        optimizer.Step(context);
        long before = GC.GetTotalAllocatedBytes(precise: true);
        optimizer.Step(context);
        long allocated = GC.GetTotalAllocatedBytes(precise: true) - before;

        // The tensor-op recurrence built twelve parameter-sized tensors here (192 MB), counting its operations. The fused pass owes
        // none, so even a quarter of ONE is far more than its bookkeeping needs.
        Assert.True(allocated < parameterBytes / 4,
            $"A BF16 Adam step over a {parameterBytes / (1024 * 1024)} MB parameter allocated {allocated / (1024.0 * 1024.0):F1} MB.");
    }
#endif
}

/// <summary>
/// #2447: the fused host pass is for host parameters only. Under a GPU engine the BF16 step stays on the device (the
/// backend's BF16 Adam kernel, with the moments in device buffers), and the host pass declines so that an engine
/// without that kernel runs the recurrence through its own tensor operations rather than on the CPU. This drives the
/// step under a GPU engine and then moves the same optimizer to a CPU engine, where the fused pass has to continue
/// from the moments the device produced.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class Adam8BitBf16GpuEngineStepTests
{
    [SkippableTheory]
    [InlineData(0.0, true)]
    [InlineData(0.05, true)]
    [InlineData(0.05, false)]
    public void UnderAGpuEngine_StepsFollowTheRecurrence_AndAHostStepContinuesFromTheDeviceMoments(
        double weightDecay, bool decoupled)
    {
        // A missing backend is reported through IsGpuAvailable; a constructor exception is a real failure.
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.SupportsGpu || !gpu.IsGpuAvailable, "GPU engine unavailable.");
        const int length = 1003;
        const int deviceSteps = 3;
        const int steps = 5;
        var saved = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = gpu;
            var initial = Adam8BitBf16FusedHostStepTests.RandomValues(length, seed: 2447, scale: 1.0);
            var parameter = gpu.TensorMultiplyScalar(new Tensor<float>((float[])initial.Clone(), new[] { length }), 1f);
            var optimizer = Adam8BitBf16FusedHostStepTests.NewOptimizer(weightDecay, decoupled);
            var scalar = new Adam8BitBf16FusedHostStepTests.ScalarReference(initial, weightDecay, decoupled);

            for (int step = 0; step < steps; step++)
            {
                if (step == deviceSteps) AiDotNetEngine.Current = new CpuEngine();
                var g = Adam8BitBf16FusedHostStepTests.RandomValues(length, seed: 100 + step, scale: 0.1);
                var gradient = new Tensor<float>((float[])g.Clone(), new[] { length });
                if (step < deviceSteps) gradient = gpu.TensorMultiplyScalar(gradient, 1f);
                optimizer.Step(new TapeStepContext<float>(
                    new[] { parameter },
                    new Dictionary<Tensor<float>, Tensor<float>> { [parameter] = gradient },
                    0f));
                scalar.Step(g);

                // A device kernel may round a moment's last BF16 bit differently, which moves that element's
                // update by under 1% of a step; a step that read stale host values or restarted its moments is
                // off by a whole update (the learning rate) or more.
                float tolerance = (step + 1) * 2 * Adam8BitBf16FusedHostStepTests.EngineTolerance;
                var actual = parameter.ToArray();
                for (int i = 0; i < length; i++)
                {
                    Assert.True(Math.Abs(actual[i] - scalar.Parameters[i]) <= tolerance,
                        $"step {step + 1} ({(step < deviceSteps ? "GPU engine" : "CPU engine")}), element {i}: "
                        + $"the recurrence gives {scalar.Parameters[i]:R}, the step {actual[i]:R}.");
                }
            }
        }
        finally
        {
            AiDotNetEngine.Current = saved;
        }
    }
}
