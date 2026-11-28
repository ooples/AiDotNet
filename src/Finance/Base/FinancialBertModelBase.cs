using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.HuggingFace;
using AiDotNet.Tokenization.Interfaces;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Finance.Base;

/// <summary>
/// Base class of the BERT-based financial language models (FinBERT, FinancialBERT, SEC-BERT, FinBERT-tone).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// All four are BERT-base encoders (Devlin et al. 2019) pretrained or fine-tuned on financial text, so they share one
/// implementation: BERT's embeddings, a stack of Post-LN transformer blocks, the [CLS] pooler, and a linear
/// classification head; WordPiece tokenization; and loading a Hugging Face checkpoint. The models differ only in their
/// defaults and the paper they cite.
/// </para>
/// <para><b>For Beginners:</b> BERT reads a whole sentence at once, in both directions, and summarizes it in the hidden
/// state of a special first token, [CLS]. A small linear layer turns that summary into class scores.
/// </para>
/// </remarks>
public abstract partial class FinancialBertModelBase<T> : FinancialNLPModelBase<T>
{
    /// <summary>
    /// Initializes a BERT model built from native layers, which can be trained.
    /// </summary>
    /// <param name="architecture">The neural network architecture; custom layers replace the encoder.</param>
    /// <param name="maxSequenceLength">Longest token sequence the model reads.</param>
    /// <param name="vocabularySize">Token vocabulary size.</param>
    /// <param name="hiddenDimension">Hidden width.</param>
    /// <param name="numClasses">Number of classes the head predicts.</param>
    /// <param name="lossFunction">The loss; cross-entropy on logits when null.</param>
    protected FinancialBertModelBase(
        NeuralNetworkArchitecture<T> architecture,
        int maxSequenceLength,
        int vocabularySize,
        int hiddenDimension,
        int numClasses,
        ILossFunction<T>? lossFunction)
        : base(architecture, maxSequenceLength, vocabularySize, hiddenDimension, numClasses,
            lossFunction ?? new CrossEntropyWithLogitsLoss<T>())
    {
    }

    /// <summary>
    /// Initializes a BERT model that runs an exported ONNX graph.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model.</param>
    /// <param name="maxSequenceLength">Longest token sequence the model reads.</param>
    /// <param name="vocabularySize">Token vocabulary size.</param>
    /// <param name="hiddenDimension">Hidden width.</param>
    /// <param name="numClasses">Number of classes the graph predicts.</param>
    protected FinancialBertModelBase(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int maxSequenceLength,
        int vocabularySize,
        int hiddenDimension,
        int numClasses)
        : base(architecture, onnxModelPath, maxSequenceLength, vocabularySize, hiddenDimension, numClasses)
    {
    }

    /// <summary>The BERT settings: normalization placement, embeddings, attention dropout and checkpoint.</summary>
    protected abstract FinancialBertEncoderOptions<T> BertOptions { get; }

    /// <inheritdoc/>
    protected sealed override FinancialLanguageModelOptions<T> LanguageModelOptions => BertOptions;

    /// <inheritdoc/>
    /// <remarks>BERT summarizes a sequence in its first position, the [CLS] token.</remarks>
    protected sealed override SequenceTokenSliceLayer<T>.Position SequencePoolingPosition
        => SequenceTokenSliceLayer<T>.Position.First;

    /// <inheritdoc/>
    /// <remarks>
    /// The tokenizer stored with the checkpoint when one is loaded, so ids match the checkpoint's vocabulary;
    /// otherwise WordPiece with the bert-base-uncased vocabulary.
    /// </remarks>
    protected override ITokenizer CreateDefaultTokenizer()
    {
        var options = BertOptions;
        if (options.UsePretrainedWeights && !string.IsNullOrWhiteSpace(options.PretrainedModelPath))
            return HuggingFaceTokenizerLoader.LoadFromDirectory(CheckpointDirectory(options.PretrainedModelPath));
        return WordPieceTokenizer.BertBaseUncased();
    }

    /// <summary>
    /// Builds the BERT encoder and head and, when configured, loads the pretrained checkpoint into them.
    /// </summary>
    /// <param name="numAttentionHeads">Attention heads per block.</param>
    /// <param name="intermediateDimension">Feed-forward width.</param>
    /// <param name="numLayers">Number of transformer blocks.</param>
    /// <param name="numClasses">Number of classes the head predicts.</param>
    /// <param name="hiddenDropoutRate">Dropout on the embeddings, every sublayer and the head.</param>
    protected void InitializeBertLayers(
        int numAttentionHeads, int intermediateDimension, int numLayers, int numClasses, double hiddenDropoutRate)
    {
        if (!UseNativeMode)
            return;
        var options = BertOptions;
        InitializeEncoderAndHead(
            () => LayerHelper<T>.CreateFinancialBertEncoderLayers(
                VocabularySize, options.MaxPositionEmbeddings, options.TypeVocabSize, HiddenDimension,
                numAttentionHeads, intermediateDimension, numLayers, hiddenDropoutRate, options.AttentionDropoutRate,
                options.NormPlacement, pool: IsSequenceLevelTask),
            () => LayerHelper<T>.CreateFinancialClassificationHeadLayers(numClasses, hiddenDropoutRate));
        if (options.UsePretrainedWeights)
            LoadPretrainedCheckpointCore(options.PretrainedModelPath);
    }

    /// <summary>The folder holding a checkpoint, given the folder or a file inside it.</summary>
    private static string CheckpointDirectory(string path)
        => System.IO.Directory.Exists(path) ? path : System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(path)) ?? path;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Exported BERT graphs take <c>input_ids</c> and usually <c>attention_mask</c> and <c>token_type_ids</c>; the
    /// optional inputs are supplied only when the graph declares them. The ids are unpadded (see
    /// <see cref="FinancialNLPModelBase{T}.Tokenize"/>), so every position is attended to and every token is in segment
    /// 0.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
            throw new InvalidOperationException("ONNX session not initialized.");

        var shape = input.Rank == 1 ? new[] { 1, input.Shape[0] } : input.Shape.ToArray();
        var ids = Engine.Reshape(input.Contiguous(), new[] { input.Length }).Cast<long>().ToArray();
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", new OnnxTensors.DenseTensor<long>(ids, shape))
        };
        var declared = OnnxSession.InputMetadata;
        if (declared.ContainsKey("attention_mask"))
        {
            var mask = new long[ids.Length];
            for (int i = 0; i < mask.Length; i++) mask[i] = 1L;
            inputs.Add(NamedOnnxValue.CreateFromTensor("attention_mask", new OnnxTensors.DenseTensor<long>(mask, shape)));
        }

        if (declared.ContainsKey("token_type_ids"))
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor("token_type_ids",
                new OnnxTensors.DenseTensor<long>(new long[ids.Length], shape)));
        }

        using var results = OnnxSession.Run(inputs);
        var output = results.First().AsTensor<float>();
        var data = new T[output.Length];
        NumOps.FromFloatSpan(output.ToDenseTensor().Buffer.Span.Slice(0, (int)output.Length), data);
        return new Tensor<T>(output.Dimensions.ToArray(), new Vector<T>(data));
    }

    /// <inheritdoc/>
    /// <remarks>Token ids are [sequence] or [batch, sequence].</remarks>
    protected override void ValidateInputShape(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 1 && input.Rank != 2)
            throw new ArgumentException(
                $"Token ids must be [sequence] or [batch, sequence]; got rank {input.Rank}.", nameof(input));
    }
}
