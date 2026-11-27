using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Finance.Base;

/// <summary>
/// Base class of the decoder-only financial language models (FinGPT, FinMA, InvestLM, BloombergGPT) used as
/// classifiers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A causal decoder lets each position read only the positions before it, so the last position is the only one that
/// has read the whole input. Classification therefore reads the last position's hidden state into a linear head, as
/// sequence classification on a GPT-style model does.
/// </para>
/// <para><b>For Beginners:</b> These models read text left to right. Once they reach the last word they have seen
/// everything, so that word's summary is what the classifier uses.
/// </para>
/// </remarks>
public abstract class FinancialDecoderModelBase<T> : FinancialNLPModelBase<T>
{
    /// <summary>
    /// Initializes a decoder built from native layers, which can be trained.
    /// </summary>
    /// <param name="architecture">The neural network architecture; custom layers replace the decoder.</param>
    /// <param name="maxSequenceLength">Longest token sequence the model reads.</param>
    /// <param name="vocabularySize">Token vocabulary size.</param>
    /// <param name="hiddenDimension">Hidden width.</param>
    /// <param name="numClasses">Number of classes the head predicts.</param>
    /// <param name="lossFunction">The loss; cross-entropy on logits when null.</param>
    protected FinancialDecoderModelBase(
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
    /// Initializes a decoder that runs an exported ONNX graph.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model.</param>
    /// <param name="maxSequenceLength">Longest token sequence the model reads.</param>
    /// <param name="vocabularySize">Token vocabulary size.</param>
    /// <param name="hiddenDimension">Hidden width.</param>
    /// <param name="numClasses">Number of classes the graph predicts.</param>
    protected FinancialDecoderModelBase(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int maxSequenceLength,
        int vocabularySize,
        int hiddenDimension,
        int numClasses)
        : base(architecture, onnxModelPath, maxSequenceLength, vocabularySize, hiddenDimension, numClasses)
    {
    }

    /// <inheritdoc/>
    /// <remarks>Under the causal mask only the last position has read every token.</remarks>
    protected sealed override SequenceTokenSliceLayer<T>.Position SequencePoolingPosition
        => SequenceTokenSliceLayer<T>.Position.Last;

    /// <summary>The name of the tokenizer the released model uses, for the error raised when none is set.</summary>
    protected abstract string ReleasedTokenizerDescription { get; }

    /// <inheritdoc/>
    /// <remarks>
    /// The released tokenizers are vocabulary files distributed with the checkpoints, so none can be built here;
    /// set <see cref="FinancialLanguageModelOptions{T}.Tokenizer"/>, for example from
    /// <c>AutoTokenizer.FromPretrained</c>.
    /// </remarks>
    /// <exception cref="InvalidOperationException">Always: no tokenizer was configured.</exception>
    protected override ITokenizer CreateDefaultTokenizer()
        => throw new InvalidOperationException(
            $"{GetType().Name} has no built-in tokenizer: it uses {ReleasedTokenizerDescription}. Set the options' "
            + "Tokenizer (for example AutoTokenizer.FromPretrained with the model's tokenizer files) to tokenize text.");

    /// <summary>
    /// Builds the decoder and the classification head.
    /// </summary>
    /// <param name="decoder">Builds the decoder; its argument is whether to end with the last position's state.</param>
    /// <param name="numClasses">Number of classes the head predicts.</param>
    /// <param name="dropoutRate">Dropout before the head's linear layer.</param>
    protected void InitializeDecoderLayers(Func<bool, IEnumerable<ILayer<T>>> decoder, int numClasses, double dropoutRate)
    {
        if (decoder is null) throw new ArgumentNullException(nameof(decoder));
        if (!UseNativeMode)
            return;
        InitializeEncoderAndHead(
            () => decoder(IsSequenceLevelTask),
            () => LayerHelper<T>.CreateFinancialClassificationHeadLayers(numClasses, dropoutRate));
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
