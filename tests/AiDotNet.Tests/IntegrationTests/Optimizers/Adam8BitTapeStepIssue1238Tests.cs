using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
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
/// Fix: tape Step now allocates byte[] m and v per parameter (plus per-block
/// double[] scales), dequantizes into transient tensors for the math, runs
/// the Adam recurrences, then re-quantizes the updates back to the byte
/// buffers. Resident state for a parameter of N elements drops from
/// <c>2 × N × sizeof(T)</c> bytes to <c>2 × (N + numBlocks × 8)</c> bytes,
/// roughly the 8× reduction the class name promised.
/// </para>
/// </summary>
public class Adam8BitTapeStepIssue1238Tests
{
    /// <summary>
    /// Walks the <c>Adam8BitOptimizer</c>'s private tape state via reflection
    /// and confirms that:
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

        var statesField = typeof(Adam8BitOptimizer<double, Matrix<double>, Vector<double>>)
            .GetField("_tapeStates", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(statesField);
        var states = (System.Collections.IDictionary)statesField!.GetValue(optimizer)!;
        Assert.Equal(1, states.Count);

        var stateObj = states[param]!;
        var stateType = stateObj.GetType();

        var lengthProp = stateType.GetField("Length", BindingFlags.Public | BindingFlags.Instance)!;
        var numBlocksProp = stateType.GetField("NumBlocks", BindingFlags.Public | BindingFlags.Instance)!;
        var mQuantized = (Vector<byte>?)stateType.GetField("MQuantized", BindingFlags.Public | BindingFlags.Instance)!.GetValue(stateObj);
        var vQuantized = (Vector<byte>)stateType.GetField("VQuantized", BindingFlags.Public | BindingFlags.Instance)!.GetValue(stateObj)!;
        var mScales = (Vector<double>?)stateType.GetField("MScales", BindingFlags.Public | BindingFlags.Instance)!.GetValue(stateObj);
        var vScales = (Vector<double>)stateType.GetField("VScales", BindingFlags.Public | BindingFlags.Instance)!.GetValue(stateObj)!;

        Assert.Equal(256, (int)lengthProp.GetValue(stateObj)!);
        Assert.Equal(4, (int)numBlocksProp.GetValue(stateObj)!);

        Assert.NotNull(mQuantized);
        Assert.Equal(256, mQuantized!.Length);
        Assert.Equal(256, vQuantized.Length);

        Assert.NotNull(mScales);
        Assert.Equal(4, mScales!.Length);
        Assert.Equal(4, vScales.Length);
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

        var statesField = typeof(Adam8BitOptimizer<double, Matrix<double>, Vector<double>>)
            .GetField("_tapeStates", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var states = (System.Collections.IDictionary)statesField.GetValue(optimizer)!;
        var stateObj = states[param]!;
        var stateType = stateObj.GetType();

        var mQuantized = (Vector<byte>?)stateType.GetField("MQuantized", BindingFlags.Public | BindingFlags.Instance)!.GetValue(stateObj);
        var mScales = (Vector<double>?)stateType.GetField("MScales", BindingFlags.Public | BindingFlags.Instance)!.GetValue(stateObj);
        var mFullPrecision = stateType.GetField("MFullPrecision", BindingFlags.Public | BindingFlags.Instance)!.GetValue(stateObj);
        var vQuantized = (Vector<byte>)stateType.GetField("VQuantized", BindingFlags.Public | BindingFlags.Instance)!.GetValue(stateObj)!;

        Assert.Null(mQuantized);
        Assert.Null(mScales);
        Assert.NotNull(mFullPrecision);
        // v stays quantized regardless.
        Assert.Equal(64, vQuantized.Length);
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
        var options = new Adam8BitOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            BlockSize = 16,
            CompressBothMoments = true,
            InitialLearningRate = 0.5,
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

        var statesField = typeof(Adam8BitOptimizer<double, Matrix<double>, Vector<double>>)
            .GetField("_tapeStates", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var states = (System.Collections.IDictionary)statesField.GetValue(optimizer)!;
        Assert.Equal(2, states.Count);

        var stateA = states[paramA]!;
        var stateB = states[paramB]!;
        var stateType = stateA.GetType();
        var lengthField = stateType.GetField("Length", BindingFlags.Public | BindingFlags.Instance)!;
        var numBlocksField = stateType.GetField("NumBlocks", BindingFlags.Public | BindingFlags.Instance)!;
        var vQuantizedField = stateType.GetField("VQuantized", BindingFlags.Public | BindingFlags.Instance)!;

        Assert.Equal(64, (int)lengthField.GetValue(stateA)!);
        Assert.Equal(2, (int)numBlocksField.GetValue(stateA)!);
        Assert.Equal(128, (int)lengthField.GetValue(stateB)!);
        Assert.Equal(4, (int)numBlocksField.GetValue(stateB)!);

        // The byte buffers must be distinct objects (no aliasing across params).
        Assert.NotSame(vQuantizedField.GetValue(stateA), vQuantizedField.GetValue(stateB));
    }
}
