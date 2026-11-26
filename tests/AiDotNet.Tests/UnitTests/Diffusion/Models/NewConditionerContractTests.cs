using AiDotNet.Diffusion.Conditioning;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Enums;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tokenization;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// Contract tests for Phase 1 conditioning infrastructure: text conditioners and noise predictors.
/// Tests pass an explicit (small-vocab) tokenizer — the conditioner ctors now require one,
/// matching the PyTorch convention where model construction and tokenizer loading are
/// separate concerns.
/// </summary>
// Shares one xUnit collection with FastGenContractTests so the two classes' foundation-scale FP32
// round-trip tests run sequentially (one multi-GB model resident at a time → no 16 GB CI OOM).
[Collection("FoundationScaleSerial")]
public class NewConditionerContractTests : DiffusionUnitTestBase
{
    #region Text Conditioner Constructor Tests

    [Fact(Timeout = 120000)]
    public async Task SigLIPTextConditioner_Construct_CreatesValidConditioner()
    {
        var conditioner = new SigLIPTextConditioner<double>(ClipTokenizerFactory.CreateSimple());

        Assert.NotNull(conditioner);
        Assert.True(conditioner.ProducesPooledOutput);
        Assert.True(conditioner.EmbeddingDimension > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SigLIP2TextConditioner_Construct_CreatesValidConditioner()
    {
        var conditioner = new SigLIP2TextConditioner<double>(ClipTokenizerFactory.CreateSimple());

        Assert.NotNull(conditioner);
        Assert.True(conditioner.ProducesPooledOutput);
        Assert.True(conditioner.EmbeddingDimension > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task DistilledT5TextConditioner_Construct_CreatesValidConditioner()
    {
        var tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.FlanT5);
        var conditioner = new DistilledT5TextConditioner<double>(tokenizer);

        Assert.NotNull(conditioner);
        Assert.True(conditioner.EmbeddingDimension > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task GemmaTextConditioner_Construct_CreatesValidConditioner()
    {
        var tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.LLaMA);
        var conditioner = new GemmaTextConditioner<double>(tokenizer);

        Assert.NotNull(conditioner);
        Assert.True(conditioner.EmbeddingDimension > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task Qwen2TextConditioner_Construct_CreatesValidConditioner()
    {
        var tokenizer = LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.Qwen);
        var conditioner = new Qwen2TextConditioner<double>(tokenizer);

        Assert.NotNull(conditioner);
        Assert.True(conditioner.EmbeddingDimension > 0);
    }

    #endregion

    #region Noise Predictor Constructor Tests

    [Fact(Timeout = 120000)]
    public async Task MMDiTXNoisePredictor_DefaultConstructor_CreatesValidPredictor()
    {
        var predictor = new MMDiTXNoisePredictor<double>();

        Assert.NotNull(predictor);
        Assert.False(predictor.WeightsMaterialized);
        Assert.True(predictor.ParameterCount > 0);
        Assert.False(predictor.WeightsMaterialized);
    }

    [Fact(Timeout = 120000)]
    public async Task FluxDoubleStreamPredictor_DefaultConstructor_CreatesValidPredictor()
    {
        var predictor = new FluxDoubleStreamPredictor<double>();

        Assert.NotNull(predictor);
        Assert.False(predictor.WeightsMaterialized);
        Assert.True(predictor.ParameterCount > 0);
        Assert.False(predictor.WeightsMaterialized);
    }

    [Fact(Timeout = 120000)]
    public async Task SiTPredictor_DefaultConstructor_CreatesValidPredictor()
    {
        var predictor = new SiTPredictor<double>();

        Assert.NotNull(predictor);
        Assert.False(predictor.WeightsMaterialized);
        Assert.True(predictor.ParameterCount > 0);
        Assert.False(predictor.WeightsMaterialized);
    }

    [Fact(Timeout = 120000)]
    public async Task EMMDiTPredictor_DefaultConstructor_CreatesValidPredictor()
    {
        var predictor = new EMMDiTPredictor<double>();

        Assert.NotNull(predictor);
        Assert.False(predictor.WeightsMaterialized);
        Assert.True(predictor.ParameterCount > 0);
        Assert.False(predictor.WeightsMaterialized);
    }

    #endregion

    #region Parameterizable Contract Tests

    // FP32 (production-canonical) so a materialized foundation-scale predictor fits the 16 GB CI runner;
    // serialized via the collection so only one is resident at a time. The allocation-free streaming
    // chunk API (#1624) yields the resident weight tensors by reference — a flat GetParameters() instead
    // builds a List<T> that exceeds the max array element count ("Array dimensions exceeded supported
    // range") at >2.1B parameters.
    //
    // HeavyTimeout (#1715): MMDiT-X defaults to SD3.5-Medium (hidden 2048 / 24 joint blocks ≈ 2.4 B
    // params / ~9.6 GB fp32). On the 16 GB runner the resident weight set exceeds the streaming engage
    // threshold's half-available-RAM headroom, so GetParameterChunks pages weights to a disk backing
    // file (bounded resident set + lossless write-back — verified memory-safe and round-trip-correct).
    // The OOM is fixed, but a ~9.6 GB disk round-trip's runtime sits near the per-test budget, so this
    // belongs in the nightly HeavyTimeout lane alongside the other foundation-scale predictors. Below
    // the half-RAM headroom (e.g. a roomier runner) it stays fully resident and runs fast.
    [Trait("Category", "HeavyTimeout")]
    [Fact(Timeout = 600000)]
    public async Task MMDiTXNoisePredictor_GetSetParameters_RoundTrips()
    {
        await Task.Yield();
        var predictor = new MMDiTXNoisePredictor<float>();
        AssertParameterChunksRoundTrip(predictor.GetParameterChunks, predictor.SetParameterChunks);
    }

    // HeavyTimeout (#1715): foundation-scale FLUX double-stream predictor (~billions of params). The
    // OOM is fixed — the chunked param IO now streams to disk with a bounded resident set + lossless
    // write-back (Tensors weight-streaming + the GetParameterChunks engagement) — but round-tripping
    // tens of GB across disk inherently exceeds the per-test budget on the 16 GB runner. Excluded from
    // the default gate and tracked in the nightly HeavyTimeout lane; graduates back when streaming IO
    // is fast enough. (Memory-safety verified; this is the orthogonal inherent-runtime bucket.)
    [Trait("Category", "HeavyTimeout")]
    [Fact(Timeout = 600000)]
    public async Task FluxDoubleStreamPredictor_GetSetParameters_RoundTrips()
    {
        await Task.Yield();
        var predictor = new FluxDoubleStreamPredictor<float>();
        AssertParameterChunksRoundTrip(predictor.GetParameterChunks, predictor.SetParameterChunks);
    }

    // HeavyTimeout (#1715): SiTPredictor<double> at paper scale (790 M parameters, 6.3 GB fp64). This
    // test exercises the FLAT GetParameters()/SetParameters() contract (one contiguous Vector<double>
    // over every parameter), which is all-resident by definition. The weights themselves stream (they
    // exceed half the heap), but two 6.3 GB flat vectors cannot both live under the nightly's 11.2 GB
    // heap cap, which is what OOMed here (#2447). So the first vector is reduced to an exact,
    // position-sensitive fingerprint and released before the read-back -- the same values are compared,
    // without holding two copies of a foundation model.
    [Trait("Category", "HeavyTimeout")]
    [Fact(Timeout = 120000)]
    public async Task SiTPredictor_GetSetParameters_RoundTrips()
    {
        var predictor = new SiTPredictor<double>();

        // What this proves at paper scale: a flat read, a flat write and a second flat read fit the lane's
        // memory, and the second read is bit-for-bit what was written. The first value is changed before the
        // fingerprint is taken and the vector written back, so a SetParameters that did nothing leaves the
        // original first value in place and the fingerprints differ. Only the one vector is ever held.
        var (length, expected) = ReadChangeFingerprintAndWriteBack(predictor);
        Assert.True(length > 0);

        var retrieved = predictor.GetParameters();
        Assert.Equal(length, retrieved.Length);
        Assert.Equal(expected, Fingerprint(retrieved));
    }

    // The first flat vector lives only inside this frame, so it is unreachable by the time the caller
    // reads the parameters back. Not inlined: an inlined body would keep the vector in the caller's frame.
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static (int Length, (double Sum, double Weighted, long Bits) Fingerprint) ReadChangeFingerprintAndWriteBack(
        SiTPredictor<double> predictor)
    {
        var parameters = predictor.GetParameters();
        Assert.True(parameters.Length > 0, "A paper-scale SiT predictor has parameters to round-trip.");
        // Differs from the stored value whatever it is, so the write is observable.
        parameters[0] = parameters[0] + 0.5;
        var fingerprint = Fingerprint(parameters);
        predictor.SetParameters(parameters);
        return (parameters.Length, fingerprint);
    }

    // Exact for a bitwise-identical vector (same values, same order, same summation order) and
    // sensitive to a value moving position, which a plain sum would not be.
    private static (double Sum, double Weighted, long Bits) Fingerprint(Vector<double> values)
    {
        double sum = 0, weighted = 0;
        long bits = 0;
        for (int i = 0; i < values.Length; i++)
        {
            double v = values[i];
            sum += v;
            weighted += v * ((i % 1009) + 1);
            bits = unchecked(bits * 31 + BitConverter.DoubleToInt64Bits(v));
        }
        return (sum, weighted, bits);
    }

    #endregion
}
