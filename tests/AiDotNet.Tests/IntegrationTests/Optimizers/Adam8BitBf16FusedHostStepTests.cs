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

    private static Adam8BitOptimizer<float, Tensor<float>, Tensor<float>> NewOptimizer(double weightDecay, bool decoupled)
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
    private const float EngineTolerance = (float)(LearningRate * 1e-2);

    /// <summary>
    /// 1 - beta^step as the optimizer forms it for a float model: from the float beta it stepped with, widened.
    /// </summary>
    private static float BiasCorrection(double beta, int step) => (float)(1.0 - Math.Pow((float)beta, step));

    private static float[] RandomValues(int length, int seed, double scale)
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
    private sealed class ScalarReference
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
                Assert.True(BitConverter.SingleToInt32Bits(scalar.Parameters[i]) == BitConverter.SingleToInt32Bits(param[i]),
                    $"step {step + 1}, element {i}: the scalar recurrence gives {scalar.Parameters[i]:R}, the step {param[i]:R}.");
                Assert.True(Math.Abs(engineExpected[i] - param[i]) <= EngineTolerance,
                    $"step {step + 1}, element {i}: the engine recurrence gives {engineExpected[i]:R}, the step {param[i]:R}.");
            }
        }
    }

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

        // The tensor-op recurrence allocated about twelve parameter-sized tensors here (192 MB). The fused pass owes
        // none, so even a quarter of ONE is far more than its bookkeeping needs.
        Assert.True(allocated < parameterBytes / 4,
            $"A BF16 Adam step over a {parameterBytes / (1024 * 1024)} MB parameter allocated {allocated / (1024.0 * 1024.0):F1} MB.");
    }
}
