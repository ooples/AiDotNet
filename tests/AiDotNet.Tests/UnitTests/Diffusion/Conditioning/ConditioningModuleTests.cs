using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.Conditioning;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tokenization;
using AiDotNet.Training;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.Diffusion.Conditioning;

/// <summary>
/// Tests for text conditioning modules: CLIP, T5, Dual, and Triple.
/// Each conditioner ctor now requires an explicit <c>ITokenizer</c> (PyTorch-style:
/// model construction and tokenizer loading are separate concerns). Tests pass
/// the existing small-vocab factory tokenizers, which produce real (minimal-vocab)
/// BPE / SentencePiece output without any network I/O.
/// </summary>
public class ConditioningModuleTests
{
    private static CLIPTextConditioner<double> NewClip(CLIPVariant variant = CLIPVariant.ViTL14) =>
        new CLIPTextConditioner<double>(ClipTokenizerFactory.CreateSimple(), variant);

    private static T5TextConditioner<double> NewT5(T5Variant variant = T5Variant.Base) =>
        new T5TextConditioner<double>(
            LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.FlanT5),
            variant);

    #region CLIP Text Conditioner Tests

    [Fact(Timeout = 120000)]
    public async Task CLIPConditioner_DefaultVariant_Creates768DimEmbedding()
    {
        var clip = NewClip();

        Assert.Equal(768, clip.EmbeddingDimension);
        Assert.Equal(77, clip.MaxSequenceLength);
        Assert.True(clip.ProducesPooledOutput);
        Assert.Equal(ConditioningType.Text, clip.ConditioningType);
    }

    [Theory]
    [InlineData(CLIPVariant.ViTL14, 768)]
    [InlineData(CLIPVariant.ViTH14, 1024)]
    [InlineData(CLIPVariant.ViTBigG14, 1280)]
    public void CLIPConditioner_Variants_HaveCorrectDimensions(CLIPVariant variant, int expectedDim)
    {
        var clip = NewClip(variant);

        Assert.Equal(expectedDim, clip.EmbeddingDimension);
    }

    [Fact(Timeout = 120000)]
    public async Task CLIPConditioner_Tokenize_ReturnsCorrectShape()
    {
        var clip = NewClip();

        var tokens = clip.Tokenize("a cat sitting on a mat");

        Assert.Equal(2, tokens.Shape.Length);
        Assert.Equal(1, tokens.Shape[0]);
        Assert.Equal(77, tokens.Shape[1]);
    }

    [Fact(Timeout = 120000)]
    public async Task CLIPConditioner_TokenizeBatch_ReturnsCorrectShape()
    {
        var clip = NewClip();

        var tokens = clip.TokenizeBatch(new[] { "a cat", "a dog", "a bird" });

        Assert.Equal(2, tokens.Shape.Length);
        Assert.Equal(3, tokens.Shape[0]);
        Assert.Equal(77, tokens.Shape[1]);
    }

    #endregion

    #region T5 Text Conditioner Tests

    [Fact(Timeout = 120000)]
    public async Task T5Conditioner_DefaultVariant_HasCorrectDimensions()
    {
        var t5 = NewT5();

        Assert.Equal(768, t5.EmbeddingDimension);
        Assert.Equal(512, t5.MaxSequenceLength);
        Assert.False(t5.ProducesPooledOutput);
        Assert.Equal(ConditioningType.Text, t5.ConditioningType);
    }

    [Theory]
    [InlineData(T5Variant.Small, 512)]
    [InlineData(T5Variant.Base, 768)]
    [InlineData(T5Variant.Large, 1024)]
    [InlineData(T5Variant.XL, 2048)]
    [InlineData(T5Variant.XXL, 4096)]
    public void T5Conditioner_Variants_HaveCorrectDimensions(T5Variant variant, int expectedDim)
    {
        var t5 = NewT5(variant);

        Assert.Equal(expectedDim, t5.EmbeddingDimension);
    }

    [Fact(Timeout = 120000)]
    public async Task T5Conditioner_Tokenize_ReturnsCorrectShape()
    {
        var t5 = NewT5();

        var tokens = t5.Tokenize("a cat sitting on a mat");

        Assert.Equal(2, tokens.Shape.Length);
        Assert.Equal(1, tokens.Shape[0]);
        Assert.Equal(512, tokens.Shape[1]);
    }

    #endregion

    #region Dual Text Conditioner Tests

    [Fact(Timeout = 120000)]
    public async Task DualConditioner_FromEncoders_HasCorrectProperties()
    {
        var dual = new DualTextConditioner<double>(
            clipEncoder: NewClip(),
            t5Encoder: NewT5());

        Assert.True(dual.EmbeddingDimension > 0);
        Assert.True(dual.MaxSequenceLength > 0);
        Assert.Equal(ConditioningType.MultiModal, dual.ConditioningType);
    }

    #endregion

    #region Triple Text Conditioner Tests

    [Fact(Timeout = 120000)]
    public async Task TripleConditioner_FromEncoders_HasCorrectProperties()
    {
        var triple = new TripleTextConditioner<double>(
            clipLEncoder: NewClip(CLIPVariant.ViTL14),
            clipGEncoder: NewClip(CLIPVariant.ViTBigG14),
            t5Encoder: NewT5(T5Variant.XXL));

        Assert.True(triple.EmbeddingDimension > 0);
        Assert.Equal(ConditioningType.MultiModal, triple.ConditioningType);
    }

    [Fact(Timeout = 120000)]
    public async Task TripleConditioner_CustomVariants_DimensionsMatchSelection()
    {
        var triple = new TripleTextConditioner<double>(
            clipLEncoder: NewClip(CLIPVariant.ViTL14),
            clipGEncoder: NewClip(CLIPVariant.ViTH14),
            t5Encoder: NewT5(T5Variant.XL));

        Assert.Equal(768, triple.CLIPLEmbeddingDimension);
        Assert.Equal(1024, triple.CLIPGEmbeddingDimension);
        Assert.Equal(2048, triple.T5EmbeddingDimension);
        Assert.Equal(1792, triple.CombinedPooledDimension);
    }

    #endregion

    #region ResearchPaper Attribute Validation

    /// <summary>
    /// Reflection-based check that every conditioner's <c>[ResearchPaper]</c>
    /// attribute survives instance construction (the attribute's ctor
    /// validates URL format — title non-empty + url https://-prefixed —
    /// and throws on malformed values). Catches paper-URL rot at test time
    /// before review feedback notices a typo'd arXiv link.
    /// </summary>
    [Fact(Timeout = 30000)]
    public async Task AllConditioners_ResearchPaperUrls_AreWellFormed()
    {
        var conditionerTypes = new[]
        {
            typeof(CLIPTextConditioner<double>),
            typeof(SigLIPTextConditioner<double>),
            typeof(SigLIP2TextConditioner<double>),
            typeof(T5TextConditioner<double>),
            typeof(DistilledT5TextConditioner<double>),
            typeof(GemmaTextConditioner<double>),
            typeof(Qwen2TextConditioner<double>),
            typeof(ChatGLM3TextConditioner<double>),
        };

        foreach (var t in conditionerTypes)
        {
            var attr = (ResearchPaperAttribute?)Attribute.GetCustomAttribute(
                t, typeof(ResearchPaperAttribute));
            Assert.NotNull(attr);
            Assert.False(string.IsNullOrWhiteSpace(attr!.Title),
                $"{t.Name}.ResearchPaper.Title must be non-empty.");
            Assert.True(attr.Url.StartsWith("https://", StringComparison.OrdinalIgnoreCase),
                $"{t.Name}.ResearchPaper.Url ('{attr.Url}') must be a valid https:// link.");
            Assert.True(attr.Year > 1900,
                $"{t.Name}.ResearchPaper.Year ({attr.Year}) is implausible.");
            Assert.False(string.IsNullOrWhiteSpace(attr.Authors),
                $"{t.Name}.ResearchPaper.Authors must be non-empty.");
        }
    }

    #endregion

    #region FromPretrained Tokenizer Wiring (network — opt-in)

    /// <summary>
    /// Verifies that <see cref="CLIPTextConditioner{T}.FromPretrained"/> wires
    /// a real HuggingFace tokenizer (network I/O on first call, cached
    /// afterwards). Skipped by default so CI doesn't depend on hub
    /// availability — opt in by setting <c>AIDOTNET_RUN_NETWORK_TESTS=1</c>.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task CLIPConditioner_FromPretrained_LoadsRealHuggingFaceTokenizer()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_NETWORK_TESTS") != "1")
        {
            // Skip on default CI: this test downloads from HuggingFace Hub.
            return;
        }

        var clip = CLIPTextConditioner<double>.FromPretrained(CLIPVariant.ViTL14);
        Assert.NotNull(clip);
        Assert.Equal(77, clip.MaxSequenceLength);
        Assert.Equal(768, clip.EmbeddingDimension);
        // VocabSize comes from the loaded tokenizer; the canonical CLIP
        // BPE vocab is 49408. Allow any positive value because the actual
        // loader implementation may report a slightly different count
        // depending on which special tokens are included in the count.
        Assert.True(clip.VocabSize > 0);
    }

    #endregion

    #region End-to-End Forward-Pass Smoke Tests
    // These exercise the full Tokenize → EncodeText → GetPooledEmbedding path
    // so a build that compiles but breaks runtime shape contracts (e.g.
    // PreLNTransformerBlock's residual TensorAdd, EmbeddingLayer →
    // T5RelativeBiasAttention shape mesh, RoPE engagement inside MHA / GQA)
    // gets caught here rather than at first user-pipeline run.

    [Fact(Timeout = 120000)]
    public async Task CLIPConditioner_EncodeText_ProducesFiniteEmbeddings()
    {
        var clip = NewClip();
        var tokens = clip.Tokenize("a cat sitting on a couch");
        var embeddings = clip.EncodeText(tokens);

        Assert.Equal(3, embeddings.Shape.Length); // [B, S, D]
        Assert.Equal(1, embeddings.Shape[0]);
        Assert.Equal(77, embeddings.Shape[1]);
        Assert.Equal(768, embeddings.Shape[2]);

        var span = embeddings.AsSpan();
        for (int i = 0; i < Math.Min(200, span.Length); i++)
        {
            Assert.False(double.IsNaN(span[i]), $"CLIP embed[{i}] is NaN");
            Assert.False(double.IsInfinity(span[i]), $"CLIP embed[{i}] is Inf");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task CLIPConditioner_GetPooledEmbedding_ApplliesTextProjectionPostPool()
    {
        // Verify that CLIP's text_projection is applied to the EOS-pooled
        // embedding (rank-2 [B, D]) and NOT to every sequence position
        // (which would have been a paper-fidelity violation per Radford 2021 §3.1).
        var clip = NewClip();
        var tokens = clip.Tokenize("a starry night");
        var seqEmbeddings = clip.EncodeText(tokens);
        var pooled = clip.GetPooledEmbedding(seqEmbeddings);

        Assert.Equal(2, pooled.Shape.Length); // [B, D] — pooled is rank-2
        Assert.Equal(1, pooled.Shape[0]);
        Assert.Equal(768, pooled.Shape[1]);

        var span = pooled.AsSpan();
        for (int i = 0; i < span.Length; i++)
            Assert.False(double.IsNaN(span[i]), $"CLIP pooled[{i}] is NaN");
    }

    [Fact(Timeout = 300000)]
    public async Task T5Conditioner_EncodeText_ProducesFiniteEmbeddings()
    {
        // T5 exercises the T5RelativeBiasAttentionLayer + PreLNTransformerBlock
        // path. Uses T5-Small so per-iter cost stays inside the test timeout.
        var t5 = NewT5(T5Variant.Small);
        var tokens = t5.Tokenize("a serene mountain lake");
        var embeddings = t5.EncodeText(tokens);

        Assert.Equal(3, embeddings.Shape.Length);
        Assert.Equal(1, embeddings.Shape[0]);
        Assert.Equal(512, embeddings.Shape[1]);
        Assert.Equal(512, embeddings.Shape[2]); // Small hidden=512

        var span = embeddings.AsSpan();
        for (int i = 0; i < Math.Min(200, span.Length); i++)
            Assert.False(double.IsNaN(span[i]), $"T5 embed[{i}] is NaN");
    }

    #endregion

    #region Gradient-Flow Isolation Tests
    // These bisect where in the T5 layer stack the gradient stops flowing
    // back to the EmbeddingLayer. Each test substitutes a minimal custom
    // layer list via Architecture.Layers (which TextConditioningBase
    // honours over CreateDefaultLayers), trains for one step, and checks
    // whether the EmbeddingLayer's first parameters changed.
    //
    // Result interpretation:
    //   * EmbeddingOnly_Train passes  -> EmbeddingLayer + framework training
    //     plumbing work; bug is in downstream layers.
    //   * EmbeddingPlusRMSNorm passes -> RMSNorm propagates gradients.
    //   * EmbeddingPlusScale passes   -> ConstantScale propagates gradients.
    //   * Add layers one at a time until the first failure pinpoints the
    //     gradient blocker.

    private static T5TextConditioner<double> NewT5WithLayers(IEnumerable<AiDotNet.Interfaces.ILayer<double>> customLayers)
    {
        var arch = new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Custom,
            complexity: NetworkComplexity.Simple,
            inputSize: 1);
        foreach (var layer in customLayers) arch.Layers.Add(layer);
        return new T5TextConditioner<double>(
            tokenizer: LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.FlanT5),
            variant: T5Variant.Small,
            architecture: arch);
    }

    [Fact(Timeout = 120000)]
    public async Task GradientFlowIsolation_EmbeddingOnly_TrainsEmbedding()
    {
        TapeTrainingStep<double>.InvalidateCache();
        var emb = new AiDotNet.NeuralNetworks.Layers.EmbeddingLayer<double>(
            vocabularySize: 32128, embeddingDimension: 64);
        emb.InputMode = AiDotNet.NeuralNetworks.Layers.EmbeddingInputMode.Indices;

        var t5 = NewT5WithLayers(new[] { (AiDotNet.Interfaces.ILayer<double>)emb });
        var tokens = t5.Tokenize("test");

        // EncodeText runs RunLayerStack → InitializeLayers (lazy-init).
        // Layers now contains the supplied EmbeddingLayer, and
        // _embeddingTensor has real allocated values (since EmbeddingLayer
        // resolves from input on first forward).
        var initial = t5.EncodeText(tokens);
        Assert.True(initial.Shape.Length == 3, $"expected rank-3 output, got rank {initial.Shape.Length}");

        var rng = new Random(42);
        var target = new Tensor<double>(initial._shape);
        for (int i = 0; i < target.Length; i++) target[i] = rng.NextDouble() - 0.5;

        var paramsBefore = t5.GetParameters();
        Assert.True(paramsBefore.Length > 0,
            $"EmbeddingLayer must report a non-empty parameter list after lazy init. " +
            $"paramsBefore.Length = {paramsBefore.Length}. Layers.Count = {t5.Layers.Count}.");

        int n = Math.Min(64, paramsBefore.Length);
        var beforeSample = new double[n];
        for (int i = 0; i < n; i++) beforeSample[i] = paramsBefore[i];

        t5.Train(tokens, target);

        var paramsAfter = t5.GetParameters();
        Assert.Equal(paramsBefore.Length, paramsAfter.Length);
        int changed = 0;
        for (int i = 0; i < n; i++)
            if (Math.Abs(beforeSample[i] - paramsAfter[i]) > 1e-12) changed++;
        Assert.True(changed > 0,
            $"EmbeddingLayer alone failed to train ({changed}/{n} params changed). " +
            "Bug is upstream of every diffusion-specific layer.");
    }

    #endregion

    #region Training-Cycle Tests (Forward + Backward + Parameter Update)
    // These exercise the full tape-based training pipeline through the new
    // layers (RMSNorm, T5RelativeBiasAttention, PreLNTransformerBlock,
    // ConstantScale, plus EmbeddingLayer in Indices mode). If parameters
    // don't change after a training step, gradient flow is broken
    // somewhere in the composed Engine-op graph.

    [Fact(Timeout = 600000)]
    public async Task T5Conditioner_Training_ChangesParameters()
    {
        // TapeTrainingStep<T> caches the CollectParameters result keyed on
        // (firstLayer-ref, count, fingerprint). Clear it so this test sees
        // a clean walk regardless of what an earlier test in the suite ran.
        TapeTrainingStep<double>.InvalidateCache();

        // T5-Small exercises RMSNorm + T5RelativeBiasAttention +
        // PreLNTransformerBlock + ConstantScale + EmbeddingLayer-with-
        // Indices-mode through the full tape-based training loop. If
        // gradients flow correctly through every primitive, a single
        // Train() step must perturb the parameters at the EARLY end of
        // the parameter vector (EmbeddingLayer) — that only happens when
        // the gradient signal makes it all the way back through every
        // downstream layer's autodiff plumbing.
        var t5 = NewT5(T5Variant.Small);
        var tokens = t5.Tokenize("a serene mountain lake");

        var initialEmbeddings = t5.EncodeText(tokens);
        Assert.Equal(3, initialEmbeddings.Shape.Length);

        var rng = new Random(42);
        var target = new Tensor<double>(initialEmbeddings._shape);
        for (int i = 0; i < target.Length; i++)
            target[i] = (rng.NextDouble() - 0.5) * 0.1;

        var paramsBefore = t5.GetParameters();
        Assert.True(paramsBefore.Length > 0, "T5-Small should have trainable parameters.");
        int sampleSize = Math.Min(64, paramsBefore.Length);
        var beforeSample = new double[sampleSize];
        for (int i = 0; i < sampleSize; i++) beforeSample[i] = paramsBefore[i];

        t5.Train(tokens, target);

        var paramsAfter = t5.GetParameters();
        Assert.Equal(paramsBefore.Length, paramsAfter.Length);

        int changedCount = 0;
        for (int i = 0; i < sampleSize; i++)
        {
            if (Math.Abs(beforeSample[i] - paramsAfter[i]) > 1e-12)
                changedCount++;
        }
        Assert.True(changedCount > 0,
            $"After one training step, 0/{sampleSize} sampled EmbeddingLayer " +
            "parameters changed. Either gradient flow is broken through one " +
            "of the new primitives, or the suite-level static-state leak in " +
            "TapeTrainingStep<T> is masking the gradient on a non-isolated " +
            "run. The isolated test run passes — fix the state leak rather " +
            "than weakening this assertion.");
    }

    [Fact(Timeout = 600000)]
    public async Task CLIPConditioner_Training_ChangesParameters()
    {
        TapeTrainingStep<double>.InvalidateCache();
        // CLIP exercises the post-LN TransformerEncoderLayer path
        // (different shape contract than T5's pre-LN block) plus the
        // separate text_projection DenseLayer applied post-pool.
        var clip = NewClip();
        var tokens = clip.Tokenize("a beautiful sunset");

        var initialEmbeddings = clip.EncodeText(tokens);
        Assert.Equal(3, initialEmbeddings.Shape.Length);

        var rng = new Random(42);
        var target = new Tensor<double>(initialEmbeddings._shape);
        for (int i = 0; i < target.Length; i++)
            target[i] = (rng.NextDouble() - 0.5) * 0.1;

        var paramsBefore = clip.GetParameters();
        int sampleSize = Math.Min(64, paramsBefore.Length);
        var beforeSample = new double[sampleSize];
        for (int i = 0; i < sampleSize; i++) beforeSample[i] = paramsBefore[i];

        clip.Train(tokens, target);

        var paramsAfter = clip.GetParameters();
        int changedCount = 0;
        for (int i = 0; i < sampleSize; i++)
        {
            if (Math.Abs(beforeSample[i] - paramsAfter[i]) > 1e-12)
                changedCount++;
        }
        Assert.True(changedCount > 0,
            $"After one training step, 0/{sampleSize} sampled CLIP parameters changed.");
    }

    #endregion

    #region IConditioningModule Interface Compliance Tests

    [Fact(Timeout = 120000)]
    public async Task AllConditioners_ImplementIConditioningModule()
    {
        var clip = NewClip();
        var t5 = NewT5();
        var conditioners = new IConditioningModule<double>[]
        {
            clip,
            t5,
            new DualTextConditioner<double>(NewClip(), NewT5()),
            new TripleTextConditioner<double>(NewClip(CLIPVariant.ViTL14), NewClip(CLIPVariant.ViTBigG14), NewT5(T5Variant.XXL)),
        };

        foreach (var conditioner in conditioners)
        {
            Assert.True(conditioner.EmbeddingDimension > 0,
                $"{conditioner.GetType().Name} should have positive EmbeddingDimension");
            Assert.True(conditioner.MaxSequenceLength > 0,
                $"{conditioner.GetType().Name} should have positive MaxSequenceLength");

            var tokens = conditioner.Tokenize("test");
            Assert.NotNull(tokens);

            var batchTokens = conditioner.TokenizeBatch(new[] { "test1", "test2" });
            Assert.NotNull(batchTokens);
            Assert.Equal(2, batchTokens.Shape[0]);
        }
    }

    #endregion
}
