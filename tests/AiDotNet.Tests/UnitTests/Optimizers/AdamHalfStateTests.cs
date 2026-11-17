using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using System.Reflection;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

[Collection("EngineCurrentGlobalState")]
public sealed class AdamHalfStateTests
{
    private static AdamOptimizer<Half, Tensor<Half>, Tensor<Half>> TapeAdam(bool amsgrad = false, double rate = 0.01)
        => new(null, new AdamOptimizerOptions<Half, Tensor<Half>, Tensor<Half>>
        {
            InitialLearningRate = rate, UseAdaptiveLearningRate = false,
            UseAMSGrad = amsgrad, Epsilon = 1e-8, EnableGradientClipping = false, MaxGradientNorm = 0,
        });

    private static Tensor<Half> HalfTensor(params double[] values)
        => new(values.Select(value => (Half)value).ToArray(), new[] { values.Length });

    private static void TapeStep(AdamOptimizer<Half, Tensor<Half>, Tensor<Half>> optimizer,
        Tensor<Half>[] parameters, params (Tensor<Half> Parameter, Tensor<Half> Gradient)[] gradients)
        => optimizer.Step(new TapeStepContext<Half>(parameters,
            gradients.ToDictionary(item => item.Parameter, item => item.Gradient), (Half)0));

    private sealed class TapeOracle
    {
        private readonly float[] _master;
        private readonly float[] _m;
        private readonly float[] _v;
        private readonly float[] _maximum;
        internal long Age;
        internal TapeOracle(Tensor<Half> parameter)
        {
            _master = parameter.ToArray().Select(value => (float)value).ToArray();
            _m = new float[_master.Length]; _v = new float[_master.Length]; _maximum = new float[_master.Length];
        }
        internal void Rebase(Tensor<Half> parameter)
        {
            for (int i = 0; i < _master.Length; i++) _master[i] = (float)parameter.GetFlat(i);
        }
        internal Half[] Update(Tensor<Half> gradient, bool amsgrad = false, double rate = 0.01)
        {
            Age++;
            var result = new Half[_master.Length];
            for (int i = 0; i < result.Length; i++)
            {
                float g = (float)gradient.GetFlat(i);
                _m[i] = 0.9f * _m[i] + 0.1f * g;
                _v[i] = 0.999f * _v[i] + 0.001f * (g * g);
                _maximum[i] = Math.Max(_maximum[i], _v[i]);
                float mHat = _m[i] / (float)(1 - Math.Pow(0.9, Age));
                float vHat = (amsgrad ? _maximum[i] : _v[i]) / (float)(1 - Math.Pow(0.999, Age));
                _master[i] -= (float)(Half)rate * (mHat / ((float)Math.Sqrt(vHat) + 1e-8f));
                result[i] = (Half)_master[i];
            }
            return result;
        }
        internal Half[] UpdateSparse(int[] rows, Tensor<Half> gradient, int width, double rate = 0.01)
        {
            Age++;
            for (int position = 0; position < rows.Length; position++)
                for (int column = 0; column < width; column++)
                {
                    int i = rows[position] * width + column;
                    float g = (float)gradient[position, column];
                    _m[i] = 0.9f * _m[i] + 0.1f * g;
                    _v[i] = 0.999f * _v[i] + 0.001f * (g * g);
                    float mHat = _m[i] / (float)(1 - Math.Pow(0.9, Age));
                    float vHat = _v[i] / (float)(1 - Math.Pow(0.999, Age));
                    _master[i] -= (float)(Half)rate * (mHat / ((float)Math.Sqrt(vHat) + 1e-8f));
                }
            return _master.Select(value => (Half)value).ToArray();
        }
    }

    private static long TapeAge(AdamOptimizer<Half, Tensor<Half>, Tensor<Half>> optimizer, int index)
    {
        var snapshot = typeof(AdamOptimizer<Half, Tensor<Half>, Tensor<Half>>)
            .GetProperty("HalfTapeState", BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(optimizer);
        Assert.NotNull(snapshot);
        var slots = Assert.IsAssignableFrom<System.Collections.IDictionary>(snapshot.GetType().GetProperty("Slots")?.GetValue(snapshot));
        return slots.Contains(index) ? Assert.IsType<long>(slots[index]?.GetType().GetProperty("Age")?.GetValue(slots[index])) : 0;
    }

    [Fact]
    public void Tape_DefaultEpsilon_ZeroGradientsKeepParametersFiniteAndUnchanged()
        => OnCpu(() =>
        {
            var optimizer = TapeAdam();
            var parameter = HalfTensor(0.125, -0.25, 0.375);
            var expected = parameter.ToArray();
            for (int step = 0; step < 3; step++)
                TapeStep(optimizer, new[] { parameter }, (parameter, HalfTensor(0, 0, 0)));
            Assert.Equal(expected, parameter.ToArray());
            Assert.Equal(3, TapeAge(optimizer, 0));
        });

    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void Tape_LogicalShapesAndStridedViewsFollowIndependentFp32Recurrence(int layout)
        => OnCpu(() =>
        {
            int[] shape = layout == 0 ? Array.Empty<int>() : layout == 3 ? new[] { 2, 1, 2 } : new[] { 2, 2 };
            int length = layout == 0 ? 1 : 4;
            var data = Enumerable.Range(0, length).Select(i => (Half)(0.125 * (i + 1))).ToArray();
            var parameter = new Tensor<Half>(data, shape);
            if (layout == 4)
            {
                parameter = parameter.Transpose();
                Assert.False(parameter.IsContiguous);
            }
            var optimizer = TapeAdam(amsgrad: true); var oracle = new TapeOracle(parameter);
            for (int step = 0; step < 8; step++)
            {
                var gradient = new Tensor<Half>(Enumerable.Range(0, length)
                    .Select(i => (Half)((step % 2 == 0 ? 1 : -1) * (i + 1) * 1e-4)).ToArray(), shape);
                var expected = oracle.Update(gradient, amsgrad: true);
                TapeStep(optimizer, new[] { parameter }, (parameter, gradient));
                Assert.Equal(shape, parameter.Shape.ToArray()); Assert.Equal(expected, parameter.ToArray());
            }
            Assert.Equal(8, TapeAge(optimizer, 0));
        });

    [Fact]
    public void Tape_OffsetParameterAndOffsetStridedGradientUseLogicalValues()
        => OnCpu(() =>
        {
            var storage = new Tensor<Half>(new[] { (Half)9, (Half)10, (Half)0.125, (Half)(-0.25), (Half)0.375, (Half)(-0.5) }, new[] { 3, 2 });
            var parameter = storage.Slice(0, 1, 3);
            var optimizer = TapeAdam(); var oracle = new TapeOracle(parameter);
            for (int step = 0; step < 4; step++)
            {
                var gradientStorage = new Tensor<Half>(new[] { (Half)7, (Half)8,
                    (Half)((step + 1) * 1e-4), (Half)(-2e-4), (Half)3e-4, (Half)(-4e-5) }, new[] { 3, 2 });
                var gradient = gradientStorage.Slice(0, 1, 3).Transpose();
                Assert.False(gradient.IsContiguous);
                var expected = oracle.Update(gradient);
                TapeStep(optimizer, new[] { parameter }, (parameter, gradient));
                Assert.Equal(expected, parameter.ToArray());
                Assert.Equal((Half)9, storage.GetFlat(0)); Assert.Equal((Half)10, storage.GetFlat(1));
                Assert.Equal((Half)7, gradientStorage.GetFlat(0)); Assert.Equal((Half)8, gradientStorage.GetFlat(1));
            }
        });

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Tape_TinyGradientsAndFrozenParameters_FollowIndependentFp32Recurrences(bool amsgrad)
        => OnCpu(() =>
        {
            var optimizer = TapeAdam(amsgrad);
            var first = HalfTensor(0.125, -0.25, 0.375);
            var second = HalfTensor(-0.125, 0.5, -0.375);
            var firstOracle = new TapeOracle(first); var secondOracle = new TapeOracle(second);
            for (int step = 0; step < 16; step++)
            {
                var gradient = new Tensor<Half>(Gradient(step).ToArray(), new[] { 3 });
                var parameters = step % 2 == 0 ? new[] { first, second } : new[] { second, first };
                var frozen = step % 3 == 0 ? second : first;
                var active = step % 3 == 0 ? first : second;
                var oracle = step % 3 == 0 ? firstOracle : secondOracle;
                var beforeFrozen = frozen.ToArray();
                var expected = oracle.Update(gradient, amsgrad);
                TapeStep(optimizer, parameters, (active, gradient));
                Assert.Equal(expected, active.ToArray());
                Assert.Equal(beforeFrozen, frozen.ToArray());
                Assert.Equal(firstOracle.Age, TapeAge(optimizer, step % 2 == 0 ? 0 : 1));
                Assert.Equal(secondOracle.Age, TapeAge(optimizer, step % 2 == 0 ? 1 : 0));
            }
        });

    [Fact]
    public void Tape_PersistentMasterAccumulatesUpdatesBelowHalfUlp()
        => OnCpu(() =>
        {
            const double rate = 1e-5;
            var optimizer = TapeAdam(rate: rate);
            var parameter = HalfTensor(1);
            var oracle = new TapeOracle(parameter);
            for (int step = 0; step < 128; step++)
            {
                var gradient = HalfTensor(1e-4);
                var expected = oracle.Update(gradient, rate: rate);
                TapeStep(optimizer, new[] { parameter }, (parameter, gradient));
                Assert.Equal(expected, parameter.ToArray());
                if (step == 0) Assert.Equal((Half)1, parameter[0]);
            }
            Assert.True(parameter[0] < (Half)1);
        });

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Tape_CheckpointCopiesContinueIndependentFp32StateAndAges(bool amsgrad)
        => OnCpu(() =>
        {
            var original = TapeAdam(amsgrad);
            var first = HalfTensor(0.125, -0.25, 0.375);
            var second = HalfTensor(-0.125, 0.5, -0.375);
            for (int step = 0; step < 5; step++)
                TapeStep(original, new[] { first, second }, (first, new Tensor<Half>(Gradient(step).ToArray(), new[] { 3 })));
            var bytes = original.Serialize();
            var restored = TapeAdam(amsgrad); restored.Deserialize(bytes);
            var copy = TapeAdam(amsgrad); copy.Deserialize(bytes);
            var restoredFirst = new Tensor<Half>(first.ToArray(), new[] { 3 });
            var copiedFirst = new Tensor<Half>(first.ToArray(), new[] { 3 });
            var restoredSecond = new Tensor<Half>(second.ToArray(), new[] { 3 });
            var copiedSecond = new Tensor<Half>(second.ToArray(), new[] { 3 });
            for (int step = 5; step < 12; step++)
            {
                var g = new Tensor<Half>(Gradient(step).ToArray(), new[] { 3 });
                TapeStep(original, new[] { first, second }, (first, g), (second, g));
                TapeStep(restored, new[] { restoredFirst, restoredSecond }, (restoredFirst, g), (restoredSecond, g));
                TapeStep(copy, new[] { copiedFirst, copiedSecond }, (copiedFirst, g), (copiedSecond, g));
                Assert.Equal(first.ToArray(), restoredFirst.ToArray()); Assert.Equal(first.ToArray(), copiedFirst.ToArray());
                Assert.Equal(second.ToArray(), restoredSecond.ToArray()); Assert.Equal(second.ToArray(), copiedSecond.ToArray());
                Assert.Equal(step + 1, TapeAge(restored, 0)); Assert.Equal(step - 4, TapeAge(restored, 1));
            }
            TapeStep(original, new[] { first, second }, (first, HalfTensor(0.5, -0.25, 0.125)));
            var next = HalfTensor(2e-5, -4e-5, -8e-5);
            TapeStep(restored, new[] { restoredFirst, restoredSecond }, (restoredFirst, next));
            TapeStep(copy, new[] { copiedFirst, copiedSecond }, (copiedFirst, next));
            Assert.Equal(restoredFirst.ToArray(), copiedFirst.ToArray());
        });

    [Fact]
    public void Tape_EmptyEnrollmentCheckpointIsNotMistakenForLegacyState()
        => OnCpu(() =>
        {
            var original = TapeAdam(); var parameter = HalfTensor(0.125);
            for (int step = 0; step < 3; step++) TapeStep(original, new[] { parameter });
            var restored = TapeAdam(); restored.Deserialize(original.Serialize());
            var oracle = new TapeOracle(parameter); var gradient = HalfTensor(1e-4);
            var expected = oracle.Update(gradient);
            TapeStep(restored, new[] { parameter }, (parameter, gradient));
            Assert.Equal(expected, parameter.ToArray()); Assert.Equal(1, TapeAge(restored, 0));
        });

    [Fact]
    public void Tape_AnomalySkipDoesNotAdvanceMasterMomentsOrAge()
        => OnCpu(() =>
        {
            var optimizer = TapeAdam(); var parameter = HalfTensor(0.125);
            TapeStep(optimizer, new[] { parameter }, (parameter, HalfTensor(1e-4)));
            var before = parameter.ToArray(); var checkpoint = optimizer.Serialize();
            TapeStep(optimizer, new[] { parameter }, (parameter, HalfTensor(double.NaN)));
            Assert.Equal(before, parameter.ToArray()); Assert.Equal(checkpoint, optimizer.Serialize());
            Assert.Equal(1, TapeAge(optimizer, 0));
        });

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Tape_ClippingFeedsPromotedMomentsBeforeUpdate(bool byValue)
        => OnCpu(() =>
        {
            var optimizer = TapeAdam();
            var options = (AdamOptimizerOptions<Half, Tensor<Half>, Tensor<Half>>)optimizer.GetOptions();
            options.EnableGradientClipping = true;
            options.GradientClippingMethod = byValue ? GradientClippingMethod.ByValue : GradientClippingMethod.ByNorm;
            options.MaxGradientValue = 0.5; options.MaxGradientNorm = 1;
            var parameter = HalfTensor(0.125, -0.25); var oracle = new TapeOracle(parameter);
            var first = byValue ? HalfTensor(0.5, -0.5) : HalfTensor((double)(Half)(3f * (float)(Half)(1.0 / (5 + 1e-6))),
                (double)(Half)(-4f * (float)(Half)(1.0 / (5 + 1e-6))));
            var expected = oracle.Update(first);
            TapeStep(optimizer, new[] { parameter }, (parameter, HalfTensor(3, -4)));
            Assert.Equal(expected, parameter.ToArray());
            var small = HalfTensor(1e-4, -2e-4); expected = oracle.Update(small);
            TapeStep(optimizer, new[] { parameter }, (parameter, small));
            Assert.Equal(expected, parameter.ToArray());
        });

    [Fact]
    public void Tape_ExternalPublishedValueReplacementPreservesMoments_AndResetStartsFresh()
        => OnCpu(() =>
        {
            var optimizer = TapeAdam(); var parameter = HalfTensor(0.125); var oracle = new TapeOracle(parameter);
            var gradient = HalfTensor(1e-4);
            oracle.Update(gradient); TapeStep(optimizer, new[] { parameter }, (parameter, gradient));
            parameter[0] = (Half)0.5; oracle.Rebase(parameter);
            var next = HalfTensor(-2e-4); var expected = oracle.Update(next);
            TapeStep(optimizer, new[] { parameter }, (parameter, next));
            Assert.Equal(expected, parameter.ToArray()); Assert.Equal(2, TapeAge(optimizer, 0));
            optimizer.Reset(); oracle = new TapeOracle(parameter); expected = oracle.Update(gradient);
            TapeStep(optimizer, new[] { parameter }, (parameter, gradient));
            Assert.Equal(expected, parameter.ToArray()); Assert.Equal(1, TapeAge(optimizer, 0));
        });

    [Fact]
    public void Tape_RestoredShapeMismatchRefusesEntireBatchBeforeUpdatingEarlierParameter()
        => OnCpu(() =>
        {
            var original = TapeAdam(); var first = HalfTensor(0.125, -0.25); var second = HalfTensor(0.5, -0.375);
            var gradient = HalfTensor(1e-4, -2e-4);
            TapeStep(original, new[] { first, second }, (first, gradient), (second, gradient));
            var restored = TapeAdam(); restored.Deserialize(original.Serialize());
            var wrongShape = new Tensor<Half>(second.ToArray(), new[] { 1, 2 });
            var before = first.ToArray(); var checkpoint = restored.Serialize();
            Assert.Throws<InvalidOperationException>(() =>
                TapeStep(restored, new[] { first, wrongShape }, (first, gradient), (wrongShape, gradient)));
            Assert.Equal(before, first.ToArray()); Assert.Equal(second.ToArray(), wrongShape.ToArray());
            Assert.Equal(checkpoint, restored.Serialize());
        });

    [Fact]
    public void Tape_LateSameLengthWrongGradientShapeRefusesEntireBatchBeforeMutation()
        => OnCpu(() =>
        {
            var optimizer = TapeAdam(); var first = HalfTensor(0.125, -0.25); var second = HalfTensor(0.5, -0.375);
            var gradient = HalfTensor(1e-4, -2e-4);
            TapeStep(optimizer, new[] { first, second }, (first, gradient), (second, gradient));
            var checkpoint = optimizer.Serialize(); var firstBefore = first.ToArray(); var secondBefore = second.ToArray();
            var wrongGradient = new Tensor<Half>(gradient.ToArray(), new[] { 1, 2 });
            Assert.Throws<ArgumentException>(() =>
                TapeStep(optimizer, new[] { first, second }, (first, gradient), (second, wrongGradient)));
            Assert.Equal(firstBefore, first.ToArray()); Assert.Equal(secondBefore, second.ToArray());
            Assert.Equal(checkpoint, optimizer.Serialize());
        });

    [Fact]
    public void Tape_CheckpointBindsCanonicalOrderAfterLiveParameterReordering()
        => OnCpu(() =>
        {
            var original = TapeAdam(); var first = HalfTensor(0.125); var second = HalfTensor(-0.25);
            var firstOracle = new TapeOracle(first); var secondOracle = new TapeOracle(second);
            var firstGradient = HalfTensor(1e-4); var secondGradient = HalfTensor(-2e-4);
            firstOracle.Update(firstGradient);
            TapeStep(original, new[] { first, second }, (first, firstGradient));
            secondOracle.Update(secondGradient);
            TapeStep(original, new[] { second, first }, (second, secondGradient));
            var restored = TapeAdam(); restored.Deserialize(original.Serialize());
            var newFirst = new Tensor<Half>(first.ToArray(), new[] { 1 });
            var newSecond = new Tensor<Half>(second.ToArray(), new[] { 1 });
            var expectedFirst = firstOracle.Update(secondGradient); var expectedSecond = secondOracle.Update(firstGradient);
            TapeStep(restored, new[] { newSecond, newFirst }, (newFirst, secondGradient), (newSecond, firstGradient));
            Assert.Equal(expectedFirst, newFirst.ToArray()); Assert.Equal(expectedSecond, newSecond.ToArray());
            Assert.Equal(2, TapeAge(restored, 0)); Assert.Equal(2, TapeAge(restored, 1));
        });

    private static void SparseTapeStep(AdamOptimizer<Half, Tensor<Half>, Tensor<Half>> optimizer,
        Tensor<Half> table, int[] rows, Tensor<Half> weights)
    {
        var engine = AiDotNetEngine.Current;
        using var tape = new GradientTape<Half>();
        var selected = engine.TensorEmbeddingLookup<Half, int>(table, new Tensor<int>(rows, new[] { rows.Length }));
        var loss = engine.ReduceSum(engine.TensorMultiply(selected, weights), null, false);
        bool sawSparse = false;
        tape.ComputeGradientsStreaming(loss, new[] { table }, (source, _) =>
        {
            sawSparse = SparseEmbeddingOptimizerHelpers.HasSparseEmbeddingGrad(source);
            // Deliberately keep the raw sparse-only form alive for the optimizer's selected-row path.
            TapeStep(optimizer, new[] { source });
        });
        Assert.True(sawSparse);
    }

    [Fact]
    public void Tape_RawSparseOddRowsKeepMissingMomentsFrozen_AndResumeCheckpoint()
        => OnCpu(() =>
        {
            var optimizer = TapeAdam();
            ((AdamOptimizerOptions<Half, Tensor<Half>, Tensor<Half>>)optimizer.GetOptions()).AnomalyGuardMode = AdamAnomalyGuardMode.Never;
            var table = new Tensor<Half>(Enumerable.Range(0, 10).Select(i => (Half)(0.125 + i * 0.01)).ToArray(), new[] { 5, 2 });
            var oracle = new TapeOracle(table);
            int[][] rows = { new[] { 1, 3 }, new[] { 3 }, new[] { 1, 3 } };
            for (int step = 0; step < rows.Length; step++)
            {
                var values = step == 0 ? new[] { (Half)1e-4, (Half)(-2e-4), (Half)3e-4, (Half)(-1e-4) }
                    : step == 1 ? new[] { (Half)0, (Half)0 } : new[] { (Half)(-2e-4), (Half)1e-4, (Half)0, (Half)0 };
                var gradient = new Tensor<Half>(values, new[] { rows[step].Length, 2 });
                var expected = oracle.UpdateSparse(rows[step], gradient, 2);
                var before = table.ToArray();
                SparseTapeStep(optimizer, table, rows[step], gradient);
                Assert.Equal(expected, table.ToArray());
                // A touched zero-gradient row still evolves its prior moment; a missing row does not.
                if (step == 1)
                {
                    Assert.Equal(before[2], table.GetFlat(2)); Assert.Equal(before[3], table.GetFlat(3));
                    Assert.NotEqual(before[6], table.GetFlat(6));
                }
                Assert.Equal(step + 1, TapeAge(optimizer, 0));
                if (step == 0)
                {
                    var restored = TapeAdam(); restored.Deserialize(optimizer.Serialize());
                    optimizer = restored;
                    table = new Tensor<Half>(table.ToArray(), new[] { 5, 2 });
                }
            }
        });

    [Fact]
    public void Tape_DuplicateSparseRowsUseSummedDenseGradientContract()
        => OnCpu(() =>
        {
            var optimizer = TapeAdam();
            ((AdamOptimizerOptions<Half, Tensor<Half>, Tensor<Half>>)optimizer.GetOptions()).AnomalyGuardMode = AdamAnomalyGuardMode.Never;
            var table = new Tensor<Half>(new[] { (Half)0.125, (Half)0.25, (Half)0.375 }, new[] { 3, 1 });
            var oracle = new TapeOracle(table);
            var first = new Tensor<Half>(new[] { (Half)1e-4 }, new[] { 1, 1 });
            oracle.UpdateSparse(new[] { 1 }, first, 1); SparseTapeStep(optimizer, table, new[] { 1 }, first);
            var repeated = new Tensor<Half>(new[] { (Half)1e-4, (Half)2e-4 }, new[] { 2, 1 });
            var combined = (Half)((float)repeated[0, 0] + (float)repeated[1, 0]);
            var dense = new Tensor<Half>(new[] { (Half)0, (Half)0, combined }, new[] { 3, 1 });
            var expected = oracle.Update(dense);
            SparseTapeStep(optimizer, table, new[] { 2, 2 }, repeated);
            Assert.Equal(expected, table.ToArray()); Assert.Equal(2, TapeAge(optimizer, 0));
        });

    [Fact]
    public void Tape_UnpromotedStateIsRefusedRatherThanRestartedOrRecheckpointed()
        => OnCpu(() =>
        {
            var optimizer = TapeAdam(); var parameter = HalfTensor(0.125);
            typeof(AdamOptimizer<Half, Tensor<Half>, Tensor<Half>>).GetField("_tapeStep", BindingFlags.NonPublic | BindingFlags.Instance)
                ?.SetValue(optimizer, 3);
            Assert.Throws<NotSupportedException>(() => TapeStep(optimizer, new[] { parameter }, (parameter, HalfTensor(1e-4))));
            Assert.Throws<NotSupportedException>(() => optimizer.Serialize());
            Assert.Equal((Half)0.125, parameter[0]);
        });

    [Fact]
    public void Tape_PublicDeserializeOfMalformedSlotRollsBackExistingState()
        => OnCpu(() =>
        {
            var optimizer = TapeAdam(); var parameter = HalfTensor(0.125);
            TapeStep(optimizer, new[] { parameter }, (parameter, HalfTensor(1e-4)));
            var before = optimizer.Serialize(); var parameterBefore = parameter.ToArray();
            var property = typeof(AdamOptimizer<Half, Tensor<Half>, Tensor<Half>>)
                .GetProperty("HalfTapeState", BindingFlags.NonPublic | BindingFlags.Instance)
                ?? throw new InvalidOperationException("Missing Half tape checkpoint declaration.");
            var malformedSource = TapeAdam(); malformedSource.Deserialize(before);
            var snapshot = property.GetValue(malformedSource) ?? throw new InvalidOperationException("Missing snapshot.");
            var slots = Assert.IsAssignableFrom<System.Collections.IDictionary>(snapshot.GetType().GetProperty("Slots")?.GetValue(snapshot));
            var slot = slots[0] ?? throw new InvalidOperationException("Missing first slot.");
            slot.GetType().GetProperty("M")?.SetValue(slot, Array.Empty<float>());
            // Keep the normal envelope/options/tape metadata valid, corrupting only the promoted moment payload.
            // Exercise the public restore, whose commit must roll back after that slot is rejected.
            var malformed = malformedSource.Serialize();
            Assert.Throws<InvalidOperationException>(() => optimizer.Deserialize(malformed));
            Assert.Equal(before, optimizer.Serialize()); Assert.Equal(parameterBefore, parameter.ToArray());
        });

    private static AdamOptimizer<Half, Matrix<Half>, Vector<Half>> Adam(bool amsgrad = false)
        => new(null, new AdamOptimizerOptions<Half, Matrix<Half>, Vector<Half>>
        {
            InitialLearningRate = 0.01, UseAdaptiveLearningRate = false,
            UseAMSGrad = amsgrad, Epsilon = 1e-8,
        });

    private static Vector<Half> Values(params double[] values)
        => new(values.Select(value => (Half)value).ToArray());

    private static Vector<Half> Gradient(int step)
        => (step % 4) switch
        {
            0 => Values(1e-4, -2e-4, 5e-5),
            1 => Values(-3e-5, 1e-5, 0),
            2 => Values(0, 0, 0),
            _ => Values(2e-5, -4e-5, -8e-5),
        };

    private static void OnCpu(Action action)
    {
        var previous = AiDotNetEngine.Current;
        try { AiDotNetEngine.Current = new CpuEngine(); action(); }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Fact]
    public void DefaultEpsilon_ZeroGradientsLeaveFiniteParametersUnchanged()
        => OnCpu(() =>
        {
            var optimizer = Adam();
            var original = Values(0.125, -0.25, 0.375);
            var current = original;
            for (int step = 0; step < 3; step++)
            {
                current = optimizer.UpdateParameters(current, Values(0, 0, 0));
                Assert.Equal(original.ToArray(), current.ToArray());
            }
        });

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TinyGradients_RetainFp32MomentsAcrossSteps(bool amsgrad)
        => OnCpu(() =>
        {
            var optimizer = Adam(amsgrad);
            var actual = Values(0.125, -0.25, 0.375);
            var expected = actual.ToArray();
            var master = expected.Select(value => (float)value).ToArray();
            var m = new float[3]; var v = new float[3]; var maximum = new float[3];
            for (int step = 1; step <= 16; step++)
            {
                var gradient = Gradient(step - 1);
                // Independent scalar FP32 reference, quantizing only the updated parameters.
                for (int i = 0; i < expected.Length; i++)
                {
                    float g = (float)gradient[i];
                    m[i] = 0.9f * m[i] + 0.1f * g;
                    v[i] = 0.999f * v[i] + 0.001f * (g * g);
                    maximum[i] = Math.Max(maximum[i], v[i]);
                    float mHat = m[i] / (float)(1 - Math.Pow(0.9, step));
                    float vHat = (amsgrad ? maximum[i] : v[i]) / (float)(1 - Math.Pow(0.999, step));
                    float update = (float)(Half)0.01 * (mHat / ((float)Math.Sqrt(vHat) + 1e-8f));
                    master[i] -= update;
                    expected[i] = (Half)master[i];
                }
                actual = optimizer.UpdateParameters(actual, gradient);
                Assert.Equal(expected, actual.ToArray());
            }
        });

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CheckpointCopies_ContinueIndependentMomentState(bool amsgrad)
        => OnCpu(() =>
        {
            var original = Adam(amsgrad);
            var parameters = Values(0.125, -0.25, 0.375);
            for (int step = 0; step < 5; step++)
                parameters = original.UpdateParameters(parameters, Gradient(step));
            var bytes = original.Serialize();
            var restored = Adam(amsgrad); restored.Deserialize(bytes);
            var copy = Adam(amsgrad); copy.Deserialize(bytes);
            var expected = parameters; var resumed = parameters; var copied = parameters;
            for (int step = 5; step < 12; step++)
            {
                expected = original.UpdateParameters(expected, Gradient(step));
                resumed = restored.UpdateParameters(resumed, Gradient(step));
                copied = copy.UpdateParameters(copied, Gradient(step));
                Assert.Equal(expected.ToArray(), resumed.ToArray());
                Assert.Equal(expected.ToArray(), copied.ToArray());
            }
            // Advancing one copy must not mutate another copy's saved moments.
            original.UpdateParameters(expected, Values(0.5, -0.25, 0.125));
            Assert.Equal(restored.UpdateParameters(resumed, Gradient(12)).ToArray(),
                copy.UpdateParameters(copied, Gradient(12)).ToArray());
        });

    [Fact]
    public void ResetAndShapeChange_ClearMoments_AndReverseSurvivesCheckpoint()
        => OnCpu(() =>
        {
            var optimizer = Adam();
            var initial = Values(0.125, -0.25, 0.375);
            var first = optimizer.UpdateParameters(initial, Gradient(0));
            var copy = Adam(); copy.Deserialize(optimizer.Serialize());
            Assert.Equal(initial.ToArray(), copy.ReverseUpdate(first, Gradient(0)).ToArray());
            // ReverseUpdate returns weights; like the existing API it does not roll back optimizer time.
            Assert.Equal(optimizer.UpdateParameters(first, Gradient(1)).ToArray(),
                copy.UpdateParameters(first, Gradient(1)).ToArray());
            optimizer.Reset();
            Assert.Equal(Adam().UpdateParameters(initial, Gradient(0)).ToArray(),
                optimizer.UpdateParameters(initial, Gradient(0)).ToArray());
            var smaller = Values(0.125, -0.25);
            Assert.Equal(Adam().UpdateParameters(smaller, Values(1e-4, -2e-4)).ToArray(),
                optimizer.UpdateParameters(smaller, Values(1e-4, -2e-4)).ToArray());
        });

    [Fact]
    public void MatrixUpdates_UseTheSamePersistentFp32StateAsVectorUpdates()
        => OnCpu(() =>
        {
            var vectorOptimizer = Adam(); var matrixOptimizer = Adam();
            var vector = Values(0.125, -0.25, 0.375, -0.125);
            var matrix = new Matrix<Half>(new[,] { { vector[0], vector[1] }, { vector[2], vector[3] } });
            var gradient = Values(1e-4, -2e-4, 5e-5, -3e-5);
            var matrixGradient = new Matrix<Half>(new[,] { { gradient[0], gradient[1] }, { gradient[2], gradient[3] } });
            for (int step = 0; step < 5; step++)
            {
                vector = vectorOptimizer.UpdateParameters(vector, gradient);
                matrix = matrixOptimizer.UpdateParameters(matrix, matrixGradient);
                Assert.Equal(vector.ToArray(), matrix.ToRowVector().ToArray());
            }
        });

    [Fact]
    public void SubHalfUlpUpdates_AccumulateThroughCheckpointMatrixAndReverseCalls()
        => OnCpu(() =>
        {
            var optimizer = Adam(); var matrixOptimizer = Adam(); var restored = Adam();
            var parameters = Values(1000);
            var matrix = new Matrix<Half>(new[,] { { (Half)1000 } });
            var gradient = Values(1e-4);
            var matrixGradient = new Matrix<Half>(new[,] { { gradient[0] } });
            var resumed = parameters;
            float master = 1000, m = 0, v = 0, g = (float)gradient[0];
            for (int step = 1; step <= 64; step++)
            {
                m = 0.9f * m + 0.1f * g;
                v = 0.999f * v + 0.001f * (g * g);
                float mHat = m / (float)(1 - Math.Pow(0.9, step));
                float vHat = v / (float)(1 - Math.Pow(0.999, step));
                master -= (float)(Half)0.01 * (mHat / ((float)Math.Sqrt(vHat) + 1e-8f));
                var previous = parameters;
                parameters = optimizer.UpdateParameters(parameters, gradient);
                matrix = matrixOptimizer.UpdateParameters(matrix, matrixGradient);
                Assert.Equal((Half)master, parameters[0]);
                Assert.Equal(parameters[0], matrix[0, 0]);
                // Reverse returns the old published weights, while the optimizer retains the
                // current step's hidden FP32 residual and time for the next forward update.
                Assert.Equal(previous.ToArray(), optimizer.ReverseUpdate(parameters, gradient).ToArray());
                if (step == 17)
                {
                    restored.Deserialize(optimizer.Serialize());
                    resumed = parameters;
                    Assert.Equal(previous.ToArray(), restored.ReverseUpdate(resumed, gradient).ToArray());
                }
                else if (step > 17)
                {
                    resumed = restored.UpdateParameters(resumed, gradient);
                    Assert.Equal(parameters.ToArray(), resumed.ToArray());
                }
            }
            Assert.NotEqual((Half)1000, parameters[0]);
        });

    [Fact]
    public void CallerWeightReplacement_RebindsTheMaster_AndRetainsMoments()
        => OnCpu(() =>
        {
            var optimizer = Adam();
            var parameters = Values(1000);
            float m = 0, v = 0;
            for (int step = 1; step <= 6; step++)
            {
                var gradient = Values(step == 6 ? -4e-5 : 1e-4);
                float g = (float)gradient[0];
                m = 0.9f * m + 0.1f * g;
                v = 0.999f * v + 0.001f * (g * g);
                if (step == 6)
                    parameters = Values(10);
                parameters = optimizer.UpdateParameters(parameters, gradient);
                if (step == 6)
                {
                    float mHat = m / (float)(1 - Math.Pow(0.9, step));
                    float vHat = v / (float)(1 - Math.Pow(0.999, step));
                    float expected = 10 - (float)(Half)0.01 * (mHat / ((float)Math.Sqrt(vHat) + 1e-8f));
                    Assert.Equal((Half)expected, parameters[0]);
                }
            }
        });

    [Theory]
    [InlineData("legacy")]
    [InlineData("master")]
    [InlineData("second-moment")]
    public void UnsupportedOrIncompleteState_IsRejectedWithoutMutation(string state)
        => OnCpu(() =>
        {
            var optimizer = Adam();
            var parameters = Values(0.125, -0.25, 0.375);
            const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
            var type = optimizer.GetType();
            if (state == "legacy")
            {
                // The materialized fields of an older Half checkpoint: only T moments and time.
                type.GetField("_m", flags)!.SetValue(optimizer, Values(1e-4, -2e-4, 5e-5));
                type.GetField("_v", flags)!.SetValue(optimizer, Values(1e-4, 2e-4, 5e-5));
                type.GetField("_t", flags)!.SetValue(optimizer, 3);
            }
            else
            {
                parameters = optimizer.UpdateParameters(parameters, Gradient(0));
                type.GetField(state == "master" ? "_halfMaster" : "_halfV", flags)!.SetValue(optimizer, null);
            }
            var restored = Adam();
            restored.Deserialize(optimizer.Serialize());
            var before = restored.Serialize();
            var weights = parameters.ToArray();
            if (state == "legacy")
            {
                Assert.Throws<NotSupportedException>(() => restored.UpdateParameters(parameters, Gradient(1)));
                Assert.Throws<NotSupportedException>(() => restored.ReverseUpdate(parameters, Gradient(1)));
            }
            else
            {
                Assert.Throws<InvalidOperationException>(() => restored.UpdateParameters(parameters, Gradient(1)));
                Assert.Throws<InvalidOperationException>(() => restored.ReverseUpdate(parameters, Gradient(1)));
            }
            Assert.Equal(before, restored.Serialize());
            Assert.Equal(weights, parameters.ToArray());
        });

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void GpuUpdates_CommitNativeState_ForImmediateCheckpointAndCpuResume(bool amsgrad)
    {
        var previous = AiDotNetEngine.Current;
        using var gpu = new DirectGpuTensorEngine();
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(gpu.IsGpuAvailable, "The required GPU backend failed to initialize.");
        Skip.If(!gpu.IsGpuAvailable, "No DirectGpuTensorEngine is available.");
        try
        {
            var optimizer = Adam(amsgrad);
            var parameters = Values(1000, 1000, 1000);
            AiDotNetEngine.Current = new CpuEngine();
            var reference = Adam(amsgrad);
            var expected = parameters;
            AiDotNetEngine.Current = gpu;
            Vector<Half> prior = parameters;
            for (int step = 0; step < 19; step++)
            {
                prior = parameters;
                long launchesBefore = GpuLaunchProbe.TotalEver;
                parameters = optimizer.UpdateParameters(parameters, Gradient(step));
                Assert.True(GpuLaunchProbe.TotalEver > launchesBefore,
                    "The FP32 optimizer state update executed no GPU work.");
                AiDotNetEngine.Current = new CpuEngine();
                expected = reference.UpdateParameters(expected, Gradient(step));
                Assert.Equal(expected.ToArray(), parameters.ToArray());
                AiDotNetEngine.Current = gpu;
            }
            // Serialize immediately after a device update; the native arrays must already be authoritative.
            var checkpoint = optimizer.Serialize();
            AiDotNetEngine.Current = new CpuEngine();
            var restored = Adam(amsgrad); restored.Deserialize(checkpoint);
            if (!amsgrad)
                Assert.Equal(prior.ToArray(), restored.ReverseUpdate(parameters, Gradient(18)).ToArray());
            else
                Assert.Throws<NotSupportedException>(() => restored.ReverseUpdate(parameters, Gradient(18)));
            for (int step = 19; step < 64; step++)
            {
                parameters = restored.UpdateParameters(parameters, Gradient(step));
                expected = reference.UpdateParameters(expected, Gradient(step));
                Assert.Equal(expected.ToArray(), parameters.ToArray());
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}
