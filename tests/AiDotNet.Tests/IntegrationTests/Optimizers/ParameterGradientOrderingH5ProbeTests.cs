using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// H5 probe: verifies that <c>NeuralNetworkBase{T}.GetParameters</c> and
/// the trainable-tensor walk used by <c>NeuralNetworkBase{T}.ComputeGradients</c>
/// (via <c>GetParameterChunks</c> / <c>CollectParameters</c>) produce flat
/// vectors of the SAME length AND, critically, the SAME per-index
/// correspondence.
///
/// <para>
/// Background: <c>NeuralNetworkBase{T}.GetParameters</c> walks the
/// top-level <c>Layers</c> list and concatenates each layer's
/// <c>GetParameters()</c> result. <c>NeuralNetworkBase{T}.ComputeGradients</c>
/// builds its flat gradient vector by iterating
/// <c>GetParameterChunks</c>, which calls
/// <c>TapeTrainingStep{T}.CollectTrainableLayers</c> (recursive
/// pre-order descent into trainable leaves) and yields each layer's
/// <c>GetTrainableParameters()</c> tensors in registration order. The
/// Adam optimizer's <c>UpdateSolution</c> reads <c>parameters =
/// GetParameters()</c>, applies the gradient flat vector elementwise,
/// and writes back via <c>SetParameters</c>. If these two walks
/// disagree on per-index correspondence, gradients are silently applied
/// to the wrong parameters — training appears to converge but plateaus
/// at random-update fidelity.
/// </para>
///
/// <para>
/// PR #1358 fixed the four mechanical issues (H1 double-1/N averaging,
/// H2 missing SetTrainingMode in Optimize, H3 buggy default L2
/// regularization, H4 default loss MSE) but left a 7-10% top-1 vs
/// per-sample 56% top-1 residual on the canary fixture. This probe
/// verifies whether the parameter/gradient flat-vector traversal order
/// is the underlying cause.
/// </para>
/// </summary>
[Collection("NonParallelIntegration")]
public class ParameterGradientOrderingH5ProbeTests
{
    private readonly ITestOutputHelper _output;

    public ParameterGradientOrderingH5ProbeTests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// Direct test for length equality between
    /// <c>GetParameters()</c> and the gradient flat vector produced by
    /// <c>ComputeGradients</c>. Length equality is a necessary
    /// condition for per-index correspondence — if the lengths
    /// disagree, Adam's vector ops would throw <c>ArgumentException</c>
    /// before any update is applied (this is the path the #1245
    /// regression test covers).
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task GetParameters_And_ComputeGradients_LengthsMustMatch()
    {
        await Task.Yield();

        var (model, x, y) = BuildCanaryFixture();
        var gradient = model.ComputeGradients(x, y, new CategoricalCrossEntropyLoss<float>());
        var parameters = model.GetParameters();

        _output.WriteLine($"GetParameters length:   {parameters.Length}");
        _output.WriteLine($"ComputeGradients length: {gradient.Length}");
        _output.WriteLine($"Difference:             {parameters.Length - gradient.Length}");

        Assert.Equal(parameters.Length, gradient.Length);
    }

    /// <summary>
    /// Per-layer traversal-order probe. Walks the network's
    /// <c>Layers</c> list and the recursive trainable-leaf walk used by
    /// <c>ComputeGradients</c> in parallel, logging the per-layer
    /// parameter spans for each path. Any divergence in count or order
    /// is the H5 bug.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task GetParameters_And_ComputeGradients_PerLayerSpansMustAgree()
    {
        await Task.Yield();

        var (model, x, y) = BuildCanaryFixture();

        // Force lazy-init: ComputeGradients runs a forward pass which
        // materializes every layer's parameter tensors. Calling it
        // before GetParameters mirrors the optimizer's actual call
        // sequence (gradient first, then param read).
        _ = model.ComputeGradients(x, y, new CategoricalCrossEntropyLoss<float>());

        // === Path A: GetParameters (top-level Layers walk) ===
        // For each top-level layer, log the count + offset of params
        // contributed to the flat vector.
        int flatOffsetA = 0;
        var flatA = model.GetParameters();
        var layersList = model.Layers;
        _output.WriteLine("=== Path A: GetParameters() flat-walk ===");
        for (int i = 0; i < layersList.Count; i++)
        {
            var layer = layersList[i];
            var layerParams = layer.GetParameters();
            _output.WriteLine($"  Layer[{i}] {layer.GetType().Name}: " +
                $"count={layerParams.Length}, offset={flatOffsetA}, total={flatOffsetA + layerParams.Length}");
            flatOffsetA += layerParams.Length;
        }
        _output.WriteLine($"  TOTAL Path A: {flatOffsetA} (flatA.Length = {flatA.Length})");

        // === Path B: GetParameterChunks (recursive trainable walk) ===
        int flatOffsetB = 0;
        int chunkIdx = 0;
        _output.WriteLine("=== Path B: GetParameterChunks() recursive walk ===");
        foreach (var chunk in model.GetParameterChunks())
        {
            _output.WriteLine($"  Chunk[{chunkIdx}]: count={chunk.Length}, offset={flatOffsetB}");
            flatOffsetB += chunk.Length;
            chunkIdx++;
        }
        _output.WriteLine($"  TOTAL Path B: {flatOffsetB}");

        Assert.Equal(flatOffsetA, flatOffsetB);
    }

    /// <summary>
    /// The deepest verification: this test directly probes the
    /// per-index correspondence of GetParameters and GetParameterChunks
    /// using a round-trip identity. We write a deterministic, distinct
    /// pattern into every parameter via SetParameters (path A), then
    /// read the chunks in path-B order and verify the values appear at
    /// the path-B offsets that the optimizer would write to during an
    /// update.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task GetParameters_And_GetParameterChunks_PerIndexCorrespondence()
    {
        await Task.Yield();

        var (model, x, y) = BuildCanaryFixture();

        // Materialize lazy weights so GetParameters length is stable.
        _ = model.ComputeGradients(x, y, new CategoricalCrossEntropyLoss<float>());

        // Build a deterministic pattern: parameters[i] = i + 1
        // (avoiding 0 so we can tell missing-vs-present).
        var template = model.GetParameters();
        var pattern = new Vector<float>(template.Length);
        for (int i = 0; i < pattern.Length; i++) pattern[i] = i + 1;
        model.SetParameters(pattern);

        // Now read parameters back via path A (flat) and path B
        // (chunks). If they correspond per-index, then chunk[k][j]
        // sitting at flat-offset offsetB[k]+j MUST equal the value
        // pattern[offsetB[k]+j] = offsetB[k]+j+1.
        var flatBack = model.GetParameters();
        Assert.Equal(template.Length, flatBack.Length);

        // Verify the pattern round-tripped through SetParameters/GetParameters
        bool flatRoundtrip = true;
        int firstMismatchFlat = -1;
        for (int i = 0; i < flatBack.Length; i++)
        {
            if (Math.Abs(flatBack[i] - (i + 1)) > 1e-3f)
            {
                flatRoundtrip = false;
                if (firstMismatchFlat < 0) firstMismatchFlat = i;
            }
        }
        _output.WriteLine($"Flat round-trip (SetParameters→GetParameters): " +
            $"{(flatRoundtrip ? "OK" : $"FAILED at index {firstMismatchFlat}")}");

        // Now read chunks and verify offsetB[k]+j corresponds to
        // pattern[offsetB[k]+j] = offsetB[k]+j+1.
        int offsetB = 0;
        int chunkIdx = 0;
        int firstMismatchChunk = -1;
        bool chunkCorrespondence = true;
        foreach (var chunk in model.GetParameterChunks())
        {
            for (int j = 0; j < chunk.Length; j++)
            {
                float expected = offsetB + j + 1;
                float actual = chunk[j];
                if (Math.Abs(actual - expected) > 1e-3f)
                {
                    chunkCorrespondence = false;
                    if (firstMismatchChunk < 0)
                    {
                        firstMismatchChunk = offsetB + j;
                        _output.WriteLine($"  Chunk[{chunkIdx}][{j}] at flat-offset {offsetB + j}: " +
                            $"expected {expected}, got {actual} (delta = {actual - expected})");
                    }
                }
            }
            offsetB += chunk.Length;
            chunkIdx++;
        }
        _output.WriteLine($"Chunk per-index correspondence with flat pattern: " +
            $"{(chunkCorrespondence ? "OK" : $"FAILED, first mismatch at flat-offset {firstMismatchChunk}")}");

        Assert.True(flatRoundtrip, "SetParameters→GetParameters round-trip failed");
        Assert.True(chunkCorrespondence,
            $"Path A (GetParameters) and Path B (GetParameterChunks) disagree on per-index correspondence. " +
            $"First mismatch at flat-offset {firstMismatchChunk}. This is the H5 bug — gradients are " +
            $"applied to the wrong parameters because the optimizer reads GetParameters() order but " +
            $"writes gradient values produced in GetParameterChunks() order.");
    }

    /// <summary>
    /// Adam-step probe: this is the realistic end-to-end check. We
    /// snapshot parameters before and after a single Adam-style update
    /// step, where the gradient flat vector is built from
    /// ComputeGradients (path B order) and the parameter update is
    /// applied to the GetParameters() vector (path A order). Then we
    /// reconstruct per-tensor deltas via GetParameterChunks (path B
    /// order) and verify each chunk's delta is proportional to the
    /// gradient that produced it (Adam scaling factor for t=1).
    ///
    /// <para>
    /// If H5 is the bug, the deltas observed in the chunks won't match
    /// the gradient that "should" have driven them — they'll match
    /// some OTHER gradient (a different chunk's gradient, swapped by
    /// the index mis-correspondence).
    /// </para>
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task AdamStep_GradientToParameter_Correspondence()
    {
        await Task.Yield();

        var (model, x, y) = BuildCanaryFixture();
        var loss = new CategoricalCrossEntropyLoss<float>();

        // Materialize and capture starting parameters in BOTH orders.
        var gradient = model.ComputeGradients(x, y, loss);
        var paramsBefore = model.GetParameters();

        // Capture chunk values (path B order) before any update.
        var chunksBefore = new List<float[]>();
        foreach (var chunk in model.GetParameterChunks())
        {
            var copy = new float[chunk.Length];
            for (int j = 0; j < chunk.Length; j++) copy[j] = chunk[j];
            chunksBefore.Add(copy);
        }

        // Apply a simple SGD step (parameters -= lr * gradient).
        // This bypasses Adam's momentum machinery so the parameter
        // delta is exactly -lr * gradient at every index — making
        // it trivial to detect a swapped correspondence.
        float lr = 0.01f;
        var paramsAfter = new Vector<float>(paramsBefore.Length);
        for (int i = 0; i < paramsBefore.Length; i++)
        {
            paramsAfter[i] = paramsBefore[i] - lr * gradient[i];
        }
        model.SetParameters(paramsAfter);

        // Now read chunks again (path B order) and compute the delta
        // each chunk index saw. If H5 is correct (= no swap), then
        // delta[k][j] at flat-offset offsetB[k]+j should equal
        // -lr * gradient[offsetB[k]+j] (exact match, since the
        // gradient flat vector was BUILT in path B order).
        int offsetB = 0;
        int chunkIdx = 0;
        double maxDeltaError = 0.0;
        int worstChunk = -1;
        int worstJ = -1;
        double worstExpected = 0;
        double worstActual = 0;

        foreach (var chunk in model.GetParameterChunks())
        {
            for (int j = 0; j < chunk.Length; j++)
            {
                float actualDelta = chunk[j] - chunksBefore[chunkIdx][j];
                float expectedDelta = -lr * gradient[offsetB + j];
                double err = Math.Abs(actualDelta - expectedDelta);
                if (err > maxDeltaError)
                {
                    maxDeltaError = err;
                    worstChunk = chunkIdx;
                    worstJ = j;
                    worstExpected = expectedDelta;
                    worstActual = actualDelta;
                }
            }
            offsetB += chunk.Length;
            chunkIdx++;
        }

        _output.WriteLine($"Max delta error: {maxDeltaError:E6}");
        if (worstChunk >= 0)
        {
            _output.WriteLine($"  Worst at chunk[{worstChunk}][{worstJ}]: " +
                $"expected delta={worstExpected:E6}, actual delta={worstActual:E6}");
        }

        // Tolerance is loose because FromMemory/CopyTo paths can
        // introduce small fp rounding when chunks are not the actual
        // backing storage. Pre-fix the error should be very large
        // (some chunks see a gradient meant for OTHER chunks).
        // Post-fix the error should be near 0 (within rounding).
        Assert.True(maxDeltaError < 1e-3,
            $"Gradient-to-parameter correspondence violated. " +
            $"At chunk[{worstChunk}][{worstJ}]: expected delta {worstExpected:E6}, " +
            $"got {worstActual:E6}. This means the gradient computed for one " +
            $"parameter was applied to a DIFFERENT parameter — the H5 bug.");
    }

    // ---- Fixture builder ----
    //
    // Uses the same dimensions as BuildAsyncResidualModeCollapseTests so
    // any divergence observed here can be cross-referenced with the
    // 5-arm diagnostic's residual gap.
    private const int SampleCount = 16;
    private const int CtxLen = 16;
    private const int VocabSize = 16;
    private const int DModel = 32;
    private const int Heads = 2;
    private const int FfDim = 64;
    private const int NumLayers = 2;
    private const int Seed = 1351;

    private static (Transformer<float>, Tensor<float>, Tensor<float>) BuildCanaryFixture()
    {
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: NumLayers,
            numDecoderLayers: 0,
            numHeads: Heads,
            modelDimension: DModel,
            feedForwardDimension: FfDim,
            inputSize: CtxLen,
            outputSize: VocabSize,
            maxSequenceLength: CtxLen,
            vocabularySize: VocabSize);

        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        var rng = new Random(Seed);
        var x = new Tensor<float>([SampleCount, CtxLen]);
        var y = new Tensor<float>([SampleCount, VocabSize]);
        for (int i = 0; i < SampleCount; i++)
        {
            int label = -1;
            for (int s = 0; s < CtxLen; s++)
            {
                int tok = rng.Next(VocabSize);
                x[i, s] = tok;
                if (s == CtxLen - 1) label = tok;
            }
            y[i, label] = 1.0f;
        }
        return (model, x, y);
    }
}
