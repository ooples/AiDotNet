using AiDotNet.Agentic.Models.Local;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.Agentic.SelfImproving;

/// <summary>
/// Tokenizes a text <see cref="FineTuningDataset"/> into tensor supervised-fine-tuning data so a tensor model
/// (e.g. <c>MambaLanguageModel</c>) can be fine-tuned directly. This is the "tokenizer in front of a tensor
/// model" step the string-only <see cref="FineTuningDataConverter"/> leaves to the model pipeline.
/// </summary>
/// <remarks>
/// <para>
/// Each example's "prompt + completion" text is tokenized to a fixed-length window. The input is the one-hot
/// encoding of tokens <c>[0..L-1]</c> and the target is the one-hot of the next tokens <c>[1..L]</c> — standard
/// next-token (causal LM) supervision. Both are shape <c>[1, L, vocab]</c>, so a language model's per-position
/// logits and the targets flatten to equal-length vectors for the cross-entropy loss. Each example's reward is
/// carried as its sample weight.
/// </para>
/// <para>
/// Unlike a <c>string</c>-typed fine-tune (where the SFT loss step cannot turn a string into a numeric vector),
/// tensor inputs/outputs convert cleanly, so this is the path that actually trains a tensor model end-to-end.
/// </para>
/// <para><b>For Beginners:</b> Turns "good run" text into the number grids a neural language model learns from:
/// for every position it records "given the words so far, the next word should be this one."
/// </para>
/// </remarks>
public static class TextTensorDatasetConverter
{
    /// <summary>
    /// Converts a reward-filtered dataset to tensor next-token supervised data.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensors.</typeparam>
    /// <param name="dataset">The reward-filtered dataset.</param>
    /// <param name="tokenizer">The tokenizer used to encode each example's text.</param>
    /// <param name="vocabSize">The model's vocabulary size (one-hot width).</param>
    /// <param name="sequenceLength">The fixed token window length L per example.</param>
    /// <returns>Supervised tensor data with one-hot input/target tensors of shape [1, L, vocab].</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="dataset"/> or <paramref name="tokenizer"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="vocabSize"/> or <paramref name="sequenceLength"/> is not positive.</exception>
    public static FineTuningData<T, Tensor<T>, Tensor<T>> ToTensorData<T>(
        FineTuningDataset dataset,
        IGenerationTokenizer tokenizer,
        int vocabSize,
        int sequenceLength)
    {
        Guard.NotNull(dataset);
        Guard.NotNull(tokenizer);
        Guard.Positive(vocabSize);
        Guard.Positive(sequenceLength);

        var one = (T)Convert.ChangeType(1.0, typeof(T));
        var count = dataset.Examples.Count;
        var inputs = new Tensor<T>[count];
        var outputs = new Tensor<T>[count];
        var weights = new double[count];

        for (var e = 0; e < count; e++)
        {
            var example = dataset.Examples[e];
            var text = string.IsNullOrEmpty(example.Completion)
                ? example.Prompt
                : example.Prompt + " " + example.Completion;
            var tokens = tokenizer.Encode(text);

            var input = new Tensor<T>(new[] { 1, sequenceLength, vocabSize });
            var target = new Tensor<T>(new[] { 1, sequenceLength, vocabSize });
            for (var t = 0; t < sequenceLength; t++)
            {
                var inputToken = ClampToken(t < tokens.Count ? tokens[t] : 0, vocabSize);
                var targetToken = ClampToken(t + 1 < tokens.Count ? tokens[t + 1] : 0, vocabSize);
                input[new[] { 0, t, inputToken }] = one;
                target[new[] { 0, t, targetToken }] = one;
            }

            inputs[e] = input;
            outputs[e] = target;
            weights[e] = example.Reward;
        }

        return new FineTuningData<T, Tensor<T>, Tensor<T>>
        {
            Inputs = inputs,
            Outputs = outputs,
            SampleWeights = weights,
        };
    }

    private static int ClampToken(int token, int vocabSize)
    {
        if (token < 0)
        {
            return 0;
        }

        return token >= vocabSize ? vocabSize - 1 : token;
    }
}
