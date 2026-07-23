using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Regression suite for AiDotNet#1238 — <c>Adam8BitOptimizer.Step(TapeStepContext)</c>
/// was bypassing the byte[] quantized state entirely and storing m and v as
/// full-precision <see cref="Tensor{T}"/> per parameter, defeating the
/// memory-saving purpose of the "8Bit" name. PR #1229's foundation-scale
/// Sundial training pivot to Adam8Bit was ineffective because the tape Step
/// path used the same memory budget as <see cref="AdamOptimizer{T,TInput,TOutput}"/>.
///
/// <para>
/// Fix: tape Step now allocates byte-backed <see cref="Vector{T}"/> over
/// <c>byte</c> for m and v per parameter (plus a per-block
/// <see cref="Vector{T}"/> over <c>double</c> for scales — the codebase's
/// span-backed wrapper that AllocateTapeState uses for all optimizer
/// state). It dequantizes into transient tensors for the math, runs the
/// Adam recurrences, then re-quantizes the updates back to the byte
/// buffers. Resident state for a parameter of N elements drops from
/// <c>2 × N × sizeof(T)</c> bytes to <c>2 × (N + numBlocks × 8)</c> bytes,
/// roughly the 8× reduction the class name promised.
/// </para>
/// </summary>
public class Adam8BitTapeStepIssue1238Tests
{
    /// <summary>
    /// Walks the <c>Adam8BitOptimizer</c>'s tape state via the
    /// <c>GetTapeStateSnapshotForTests</c> internal accessor (visible to
    /// this assembly via <c>InternalsVisibleTo</c>) and confirms that:
    /// <list type="number">
    /// <item>The state dictionary is keyed by tensor reference (per-parameter).</item>
    /// <item>Each entry's m and v storage is the span-optimized
    ///   <see cref="Vector{T}"/> over <c>byte</c> (not a raw <c>byte[]</c>
    ///   nor a full-precision <see cref="Tensor{T}"/>) — the whole point of
    ///   #1238 plus the project's "no raw arrays" convention.</item>
    /// <item>Vector sizes equal the parameter length; scale Vectors size to
    ///   <c>ceil(length / BlockSize)</c>.</item>
    /// </list>
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task Step_AllocatesByteQuantizedTapeState_NotFullPrecisionTensors()
    {
        await Task.Yield();

        var options = new Adam8BitOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            BlockSize = 64,
            CompressBothMoments = true,
            InitialLearningRate = 0.01,
        };
        var optimizer = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        // A 256-element parameter gets ceil(256/64) = 4 blocks under the
        // configured BlockSize.
        var param = new Tensor<double>(new[] { 16, 16 });
        var grad = new Tensor<double>(new[] { 16, 16 });
        for (int i = 0; i < grad.Length; i++) grad[i] = 0.1;

        var ctx = new TapeStepContext<double>(
            parameters: new[] { param },
            gradients: new Dictionary<Tensor<double>, Tensor<double>> { [param] = grad },
            loss: 0.0);

        optimizer.Step(ctx);

        // Use the internal test snapshot accessor instead of reflection.
        // GetTapeStateSnapshotForTests is internal and visible to this
        // assembly via InternalsVisibleTo. Returns a copy of structural
        // state (lengths, presence flags, block counts) — we don't
        // touch private field names so future refactors that preserve
        // the public contract won't break this test.
        var snapshot = optimizer.GetTapeStateSnapshotForTests();
        Assert.Single(snapshot);

        var info = snapshot[param];
        Assert.Equal(256, info.Length);
        Assert.Equal(4, info.NumBlocks);

        Assert.True(info.HasMQuantized);
        Assert.Equal(256, info.MQuantizedLength);
        Assert.Equal(256, info.VQuantizedLength);

        Assert.True(info.HasMScales);
        Assert.Equal(4, info.MScalesLength);
        Assert.Equal(4, info.VScalesLength);
    }

    /// <summary>
    /// CompressBothMoments=false keeps the first moment as a Tensor (matching
    /// the legacy <see cref="Adam8BitOptimizer{T,TInput,TOutput}.UpdateSolution"/>
    /// contract) and only quantizes v. Verify that contract holds in the tape
    /// path too.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task Step_CompressBothMomentsFalse_KeepsMAsFullPrecisionTensor()
    {
        await Task.Yield();

        var options = new Adam8BitOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            BlockSize = 64,
            CompressBothMoments = false,
            InitialLearningRate = 0.01,
        };
        var optimizer = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        var param = new Tensor<double>(new[] { 8, 8 });
        var grad = new Tensor<double>(new[] { 8, 8 });
        for (int i = 0; i < grad.Length; i++) grad[i] = 0.1;

        var ctx = new TapeStepContext<double>(
            parameters: new[] { param },
            gradients: new Dictionary<Tensor<double>, Tensor<double>> { [param] = grad },
            loss: 0.0);

        optimizer.Step(ctx);

        // Use the internal test snapshot accessor instead of reflection
        // (see Step_AllocatesByteQuantizedTapeState_NotFullPrecisionTensors
        // for rationale).
        var snapshot = optimizer.GetTapeStateSnapshotForTests();
        var info = snapshot[param];

        Assert.False(info.HasMQuantized);
        Assert.False(info.HasMScales);
        Assert.True(info.HasMFullPrecision);
        // v stays quantized regardless.
        Assert.Equal(64, info.VQuantizedLength);
    }

    /// <summary>
    /// Convergence parity: on a well-conditioned quadratic, the tape Step
    /// should drive a single parameter tensor to zero from a non-zero start
    /// within a small budget of iterations — the classic Adam smoke test.
    /// Confirms the byte-quantized state isn't silently breaking the
    /// recurrences.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task Step_ConvergesQuadratic_FromNonZeroStart()
    {
        await Task.Yield();

        // f(x) = sum(x_i^2), gradient = 2x. Adam should drive x toward 0.
        // Pin every option that materially affects optimizer behavior so
        // the test's intended trajectory is fully specified by the test
        // inputs rather than framework defaults — covers both the
        // quantization knobs (UseStochasticRounding, QuantizationPercentile)
        // and the core Adam dynamics (Beta1, Beta2, Epsilon). If a future
        // PR shifts any default this test would silently start measuring
        // something different; pinning here makes drift visible at PR
        // review time.
        var options = new Adam8BitOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            BlockSize = 16,
            CompressBothMoments = true,
            InitialLearningRate = 0.5,
            UseStochasticRounding = false,
            QuantizationPercentile = 99.9,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
        };
        var optimizer = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        var param = new Tensor<double>(new[] { 16 });
        for (int i = 0; i < param.Length; i++) param[i] = 5.0; // start far from minimum

        for (int step = 0; step < 200; step++)
        {
            var grad = new Tensor<double>(new[] { param.Length });
            for (int i = 0; i < param.Length; i++) grad[i] = 2.0 * param[i]; // d/dx (x^2) = 2x
            var ctx = new TapeStepContext<double>(
                parameters: new[] { param },
                gradients: new Dictionary<Tensor<double>, Tensor<double>> { [param] = grad },
                loss: 0.0);
            optimizer.Step(ctx);
        }

        // Should be close to 0. Tolerance is intentionally loose because
        // 8-bit quantization adds noise to the moments, but 200 iterations
        // is more than enough to converge well below 0.5 from a start of 5.
        for (int i = 0; i < param.Length; i++)
        {
            Assert.True(Math.Abs(param[i]) < 0.5,
                $"param[{i}] = {param[i]} should be near 0 after 200 Adam8Bit steps from a quadratic-bowl start of 5.");
        }
    }

    /// <summary>
    /// Multi-parameter test: each tensor gets its own per-tensor block scales,
    /// not a global scale. This was the structural change the issue called
    /// out — the original Step was per-tensor at the tensor level (allocating
    /// a Tensor pair per param), but the byte[] state has to be PER-TENSOR
    /// AND per-block to give the 8× saving in practice.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task Step_MultipleParameters_GetIndependentTapeStates()
    {
        await Task.Yield();

        var options = new Adam8BitOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            BlockSize = 32,
            CompressBothMoments = true,
            InitialLearningRate = 0.01,
        };
        var optimizer = new Adam8BitOptimizer<double, Matrix<double>, Vector<double>>(null, options);

        var paramA = new Tensor<double>(new[] { 64 });   // 64 elements → 2 blocks
        var paramB = new Tensor<double>(new[] { 128 });  // 128 elements → 4 blocks
        var gradA = new Tensor<double>(new[] { 64 });
        var gradB = new Tensor<double>(new[] { 128 });
        for (int i = 0; i < gradA.Length; i++) gradA[i] = 0.1;
        for (int i = 0; i < gradB.Length; i++) gradB[i] = 0.2;

        var ctx = new TapeStepContext<double>(
            parameters: new[] { paramA, paramB },
            gradients: new Dictionary<Tensor<double>, Tensor<double>>
            {
                [paramA] = gradA,
                [paramB] = gradB,
            },
            loss: 0.0);

        optimizer.Step(ctx);

        // Use the internal test snapshot accessor instead of reflection.
        var snapshot = optimizer.GetTapeStateSnapshotForTests();
        Assert.Equal(2, snapshot.Count);

        var infoA = snapshot[paramA];
        var infoB = snapshot[paramB];

        Assert.Equal(64, infoA.Length);
        Assert.Equal(2, infoA.NumBlocks);
        Assert.Equal(128, infoB.Length);
        Assert.Equal(4, infoB.NumBlocks);

        // Each parameter gets its own quantization buffer; the snapshot
        // accessor exposes lengths but not the buffer references, so we
        // assert structural distinctness by lengths (non-aliasing of
        // buffers themselves is verified at the impl level by
        // AllocateTapeState always allocating a fresh Vector<byte>).
        Assert.NotEqual(infoA.VQuantizedLength, infoB.VQuantizedLength);
    }
}
