using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tokenization.HuggingFace;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// CLIP text encoder conditioning module (Radford et al., ICML 2021).
/// Primary text conditioner for Stable Diffusion 1.x / 2.x and one of two
/// encoders in SDXL / SD3 / FLUX.1.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Learning Transferable Visual Models From Natural Language Supervision",
    "https://arxiv.org/abs/2103.00020",
    Year = 2021,
    Authors = "Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever")]
public class CLIPTextConditioner<T> : TextConditioningBase<T>
{
    private readonly CLIPVariant _variant;

    /// <summary>
    /// CLIP text_projection: a separate learnable hidden→embedding linear
    /// projection applied ONLY to the EOS-pooled output (Radford 2021 §3.1),
    /// NOT to every sequence position. Kept outside the layer stack so the
    /// per-token attention path runs over hidden-dim representations and the
    /// pooled output gets projected to the shared image-text space.
    /// </summary>
    private readonly DenseLayer<T> _textProjection;

    public override bool ProducesPooledOutput => true;

    /// <summary>
    /// Constructs a CLIP text conditioner with an explicit paper-canonical
    /// tokenizer. PyTorch-style: model construction and tokenizer loading
    /// are separate concerns — no silent test-vocab default. Use
    /// <see cref="FromPretrained"/> for the production convenience path
    /// that loads the canonical HuggingFace CLIP tokenizer.
    /// </summary>
    /// <param name="tokenizer">The paper-canonical CLIP tokenizer (byte-level BPE).</param>
    /// <param name="variant">CLIP variant (selects hidden size / num layers / num heads).</param>
    /// <param name="architecture">Optional architecture override; pass user-supplied
    /// <see cref="NeuralNetworkArchitecture{T}.Layers"/> to bypass the default factory.</param>
    public CLIPTextConditioner(
        ITokenizer tokenizer,
        CLIPVariant variant = CLIPVariant.ViTL14,
        NeuralNetworkArchitecture<T>? architecture = null)
        : base(
            architecture: architecture ?? BuildDefaultArchitecture(variant),
            tokenizer: tokenizer,
            maxSequenceLength: 77,
            embeddingDimension: GetEmbeddingDim(variant))
    {
        Guard.NotNull(tokenizer);
        _variant = variant;
        _textProjection = new DenseLayer<T>(
            outputSize: GetEmbeddingDim(variant),
            activationFunction: new IdentityActivation<T>());
    }

    /// <summary>
    /// Loads a paper-canonical CLIP text conditioner with its real pretrained
    /// HuggingFace tokenizer. Network I/O happens here (the tokenizer is
    /// downloaded and cached on first call to <see cref="AutoTokenizer.FromPretrained(string, string?)"/>),
    /// so construction is explicit about its cost rather than hiding it
    /// inside a default constructor.
    /// </summary>
    /// <param name="variant">CLIP variant.</param>
    /// <param name="huggingFaceModelName">HuggingFace model ID (default: <c>openai/clip-vit-large-patch14</c>).</param>
    /// <param name="cacheDir">Optional cache directory for downloaded tokenizer files.</param>
    public static CLIPTextConditioner<T> FromPretrained(
        CLIPVariant variant = CLIPVariant.ViTL14,
        string huggingFaceModelName = "openai/clip-vit-large-patch14",
        string? cacheDir = null)
    {
        var tokenizer = AutoTokenizer.FromPretrained(huggingFaceModelName, cacheDir);
        return new CLIPTextConditioner<T>(tokenizer, variant);
    }

    protected override IEnumerable<ILayer<T>> CreateDefaultLayers() =>
        LayerHelper<T>.CreateDefaultCLIPTextLayers(
            vocabSize: VocabSize,
            maxSeqLen: MaxSequenceLength,
            hiddenSize: GetHiddenSize(_variant),
            numLayers: GetNumLayers(_variant),
            numHeads: GetNumHeads(_variant));

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new CLIPTextConditioner<T>(Tokenizer, _variant, Architecture);

    /// <summary>
    /// CLIP pools by extracting the embedding at the EOS token position
    /// (Radford 2021 §3.1) and then applying <see cref="_textProjection"/>
    /// to map hidden-dim → embedding-dim. Diffusion pipelines pad to fixed
    /// length so the canonical EOS placement is the last sequence position.
    /// </summary>
    public override Tensor<T> GetPooledEmbedding(Tensor<T> sequenceEmbeddings)
    {
        int rank = sequenceEmbeddings.Shape.Length;
        if (rank != 3)
            throw new ArgumentException(
                $"GetPooledEmbedding expects rank-3 [B, S, D]; got rank {rank}.");
        int batch = sequenceEmbeddings.Shape[0];
        int seqLen = sequenceEmbeddings.Shape[1];
        int dim = sequenceEmbeddings.Shape[2];

        // Gather the EOS-position embedding into a [B, hiddenSize] tensor.
        var eosPooled = new Vector<T>(batch * dim);
        for (int b = 0; b < batch; b++)
        {
            int eosPos = seqLen - 1;
            for (int d = 0; d < dim; d++)
                eosPooled[b * dim + d] = sequenceEmbeddings[b * seqLen * dim + eosPos * dim + d];
        }
        var eosTensor = new Tensor<T>(new[] { batch, dim }, eosPooled);

        // Project to the shared image-text embedding space.
        return _textProjection.Forward(eosTensor);
    }

    /// <summary>
    /// CLIP parameter count = layer-stack params + the post-pool projection.
    /// </summary>
    public override long ParameterCount
    {
        get
        {
            long basePc = 0;
            foreach (var layer in Layers) basePc += layer.ParameterCount;
            return basePc + _textProjection.ParameterCount;
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var basePart = base.GetParameters();
        return Vector<T>.Concatenate(basePart, _textProjection.GetParameters());
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int idx = 0;
        foreach (var layer in Layers)
        {
            int count = (int)layer.ParameterCount;
            if (count == 0) continue;
            layer.UpdateParameters(parameters.Slice(idx, count));
            idx += count;
        }
        int projCount = (int)_textProjection.ParameterCount;
        if (projCount > 0)
            _textProjection.UpdateParameters(parameters.Slice(idx, projCount));
    }

    /// <summary>
    /// PyTorch-style lazy architecture: token-ID inputs are rank-2
    /// <c>[batch, seqLen]</c>. We use <see cref="InputType.TwoDimensional"/>
    /// with <c>inputSize=1</c> (inferred to <c>[1, 1]</c>) so the architecture
    /// validator's "InputSize &gt; 0 for OneDimensional" gate is satisfied
    /// AND <see cref="NeuralNetworkBase{T}"/>'s auto-batch-promote /
    /// squeeze logic (which resolves an unbatched rank from <c>InputSize</c>)
    /// does NOT strip the rank-3 layer-stack output back to rank-2 — the
    /// expectedUnbatchedRank becomes 3, so our rank-2 token input never
    /// triggers promotion.
    /// </summary>
    private static NeuralNetworkArchitecture<T> BuildDefaultArchitecture(CLIPVariant variant) =>
        new NeuralNetworkArchitecture<T>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Custom,
            complexity: NetworkComplexity.Deep,
            inputSize: 1);

    private static int GetEmbeddingDim(CLIPVariant variant) => variant switch
    {
        CLIPVariant.ViTL14 => 768,
        CLIPVariant.ViTH14 => 1024,
        CLIPVariant.ViTBigG14 => 1280,
        _ => 768,
    };
    private static int GetHiddenSize(CLIPVariant variant) => GetEmbeddingDim(variant);
    private static int GetNumLayers(CLIPVariant variant) => variant switch
    {
        CLIPVariant.ViTL14 => 12,
        CLIPVariant.ViTH14 => 24,
        CLIPVariant.ViTBigG14 => 32,
        _ => 12,
    };
    private static int GetNumHeads(CLIPVariant variant) => variant switch
    {
        CLIPVariant.ViTL14 => 12,
        CLIPVariant.ViTH14 => 16,
        CLIPVariant.ViTBigG14 => 20,
        _ => 12,
    };
}
