using System;
using System.Collections.Generic;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Configuration;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.Models.Results;

/// <summary>
/// Partial class providing text-in prediction and KV-cached causal language-model decoding on <see cref="AiModelResult{T, TInput, TOutput}"/>.
/// </summary>
public partial class AiModelResult<T, TInput, TOutput>
{
    /// <summary>
    /// Predicts directly from raw text using the fitted text vectorizer captured during training.
    /// </summary>
    /// <param name="documents">The raw text documents to classify or score.</param>
    /// <returns>The model's predictions for the documents.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="documents"/> is null.</exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the model was not trained on text (no vectorizer was configured), or the model's input type is
    /// not a feature <see cref="Matrix{T}"/>.
    /// </exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> When you train on text with <c>ConfigureTextVectorizer(...)</c>, the result
    /// remembers how to turn words into numbers. This method applies that same conversion to new sentences and runs
    /// the model — so you can go straight from a <see cref="string"/> to a prediction without vectorizing by hand.
    /// </para>
    /// <example>
    /// <code>
    /// var texts = new[] { "I really enjoyed this!", "Total waste of time.", "Would recommend." };
/// var labels = new Vector&lt;double&gt;(new double[] { 1.0, 0.0, 1.0 });
///
/// var vectorizer = new CountVectorizer&lt;double&gt;();
    /// var result = await new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
    ///     .ConfigureModel(new LogisticRegression&lt;double&gt;())
    ///     .ConfigureTextVectorizer(vectorizer)
    ///     .ConfigureDataLoader(DataLoaders.FromTextDocuments(texts, labels, vectorizer))
    ///     .BuildAsync();
    ///
    /// var prediction = result.PredictText(new[] { "I really enjoyed this!" });
    /// </code>
    /// </example>
    /// </remarks>
    public TOutput PredictText(IEnumerable<string> documents)
    {
        if (documents is null)
        {
            throw new ArgumentNullException(nameof(documents));
        }

        if (TextVectorizer is null)
        {
            throw new InvalidOperationException(
                "This model was not trained on text. Call ConfigureTextVectorizer(...) before BuildAsync(), " +
                "or use Predict(...) with already-vectorized features.");
        }

        Matrix<T> features = TextVectorizer.Transform(documents);

        // The model's input type is the generic TInput; for text models it is a feature Matrix<T>.
        // Box through object so the type test is valid on the unconstrained TInput.
        if ((object)features is not TInput typedFeatures)
        {
            throw new InvalidOperationException(
                "PredictText requires a model whose input type is Matrix<T> (text features).");
        }

        return Predict(typedFeatures);
    }

    /// <summary>
    /// Generates text autoregressively from <paramref name="prompt"/> using the configured or auto-discovered
    /// tokenizer and KV-cached causal decoding when supported by the underlying model.
    /// </summary>
    /// <param name="prompt">Input text prompt. Never receives any target or ground-truth continuation.</param>
    /// <param name="maxNewTokens">Maximum number of new subword tokens to generate.</param>
    /// <param name="temperature">Sampling temperature; <c>&lt;= 0</c> selects deterministic greedy argmax decoding.</param>
    /// <param name="seed">Optional RNG seed when <paramref name="temperature"/> &gt; 0.</param>
    /// <param name="stopSequences">Optional text stop sequences that terminate generation early once emitted.</param>
    /// <returns>The newly generated continuation decoded to text (excluding the input prompt).</returns>
    public string GenerateText(
        string prompt,
        int maxNewTokens = 64,
        double temperature = 0.0,
        int? seed = null,
        IReadOnlyList<string>? stopSequences = null)
    {
        if (prompt is null)
            throw new ArgumentNullException(nameof(prompt));
        if (maxNewTokens <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxNewTokens), "maxNewTokens must be positive.");

        var tokenizer = RequireConfiguredTokenizer(nameof(GenerateText));
        var promptIds = EncodeCausalPrompt(tokenizer, prompt);
        if (promptIds.Count == 0)
            throw new ArgumentException("Prompt produced zero token IDs after tokenization.", nameof(prompt));

        int? eosId = TryResolveEosTokenId(tokenizer);
        bool greedy = temperature <= 0.0;
        var rng = greedy
            ? null
            : (seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom());

        var generatedIds = new List<int>(maxNewTokens);
        string latestDecoded = string.Empty;

        RunCausalDecodeLoop(promptIds, maxNewTokens, (step, logits) =>
        {
            int nextId = greedy
                ? ArgMaxLogits(logits)
                : SampleTemperatureLogits(logits, temperature, rng!);

            if (eosId.HasValue && nextId == eosId.Value)
                return (-1, false);

            generatedIds.Add(nextId);
            latestDecoded = tokenizer.Decode(generatedIds, skipSpecialTokens: true);

            if (stopSequences is { Count: > 0 })
            {
                for (int i = 0; i < stopSequences.Count; i++)
                {
                    var stop = stopSequences[i];
                    if (!string.IsNullOrEmpty(stop))
                    {
                        int stopIdx = latestDecoded.IndexOf(stop, StringComparison.Ordinal);
                        if (stopIdx >= 0)
                        {
                            latestDecoded = latestDecoded.Substring(0, stopIdx);
                            return (nextId, false);
                        }
                    }
                }
            }

            return (nextId, true);
        });

        return latestDecoded;
    }

    /// <summary>
    /// Predicts the single next whole word following <paramref name="prompt"/> using full-context, KV-cached
    /// greedy subword completion, stopping automatically at the first whitespace or punctuation boundary.
    /// </summary>
    /// <param name="prompt">The untruncated context passage preceding the word to predict.</param>
    /// <param name="maxSubwordTokens">Maximum subword steps to advance before stopping (default 8).</param>
    /// <returns>The predicted next word (trimmed of leading/trailing whitespace and outer punctuation).</returns>
    public string PredictNextWord(string prompt, int maxSubwordTokens = 8) =>
        PredictNextWordWithFirstToken(prompt, maxSubwordTokens).Word;

    internal (string Word, int FirstTokenId) PredictNextWordWithFirstToken(string prompt, int maxSubwordTokens = 8)
    {
        if (prompt is null)
            throw new ArgumentNullException(nameof(prompt));
        if (maxSubwordTokens <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxSubwordTokens), "maxSubwordTokens must be positive.");

        var tokenizer = RequireConfiguredTokenizer(nameof(PredictNextWord));
        string cleanPrompt = prompt.TrimEnd();
        if (cleanPrompt.Length == 0)
            return (string.Empty, -1);

        var promptIds = EncodeCausalPrompt(tokenizer, cleanPrompt);
        if (promptIds.Count == 0)
            return (string.Empty, -1);

        var specialIds = ResolveAllSpecialTokenIds(tokenizer);
        var wordTokenIds = new List<int>(maxSubwordTokens);
        bool seenWordChar = false;
        int firstPredId = -1;

        RunCausalDecodeLoop(promptIds, maxSubwordTokens, (step, logits) =>
        {
            int nextId = ArgMaxLogits(logits);
            if (step == 0)
                firstPredId = nextId;

            if (specialIds.Contains(nextId))
                return (-1, false);

            string piece = tokenizer.Decode(new List<int> { nextId }, skipSpecialTokens: true);
            if (seenWordChar)
            {
                if (piece.Length == 0 || char.IsWhiteSpace(piece[0]))
                    return (-1, false);

                // Stop before trailing punctuation (e.g. ".", ",", "\"", "!") while allowing word-internal
                // apostrophe contractions such as "'s", "'t", "'re", "'ve", "'ll", "'d", "'m".
                if (!char.IsLetterOrDigit(piece[0]))
                {
                    bool isContraction = (piece[0] == '\'' || piece[0] == '\u2019')
                        && piece.Length > 1
                        && char.IsLetter(piece[1]);
                    if (!isContraction)
                        return (-1, false);
                }
            }

            wordTokenIds.Add(nextId);
            string accumulated = tokenizer.Decode(wordTokenIds, skipSpecialTokens: true).TrimStart();
            if (!seenWordChar)
            {
                for (int i = 0; i < accumulated.Length; i++)
                {
                    if (char.IsLetterOrDigit(accumulated[i]))
                    {
                        seenWordChar = true;
                        break;
                    }
                }
            }

            // If a single subword token already contained trailing whitespace after word chars, stop immediately.
            if (seenWordChar)
            {
                for (int i = 0; i < accumulated.Length; i++)
                {
                    if (char.IsWhiteSpace(accumulated[i]))
                        return (-1, false);
                }
            }

            return (nextId, true);
        });

        if (wordTokenIds.Count == 0)
            return (string.Empty, firstPredId);

        string rawDecoded = tokenizer.Decode(wordTokenIds, skipSpecialTokens: true).TrimStart();
        return (ExtractFirstCleanWord(rawDecoded), firstPredId);
    }

    internal AiDotNet.Tokenization.Interfaces.ITokenizer RequireConfiguredTokenizer(string callerName)
    {
        if (Tokenizer != null)
            return Tokenizer;

        throw new InvalidOperationException(
            $"{callerName} requires a configured tokenizer. Call AiModelBuilder.ConfigureTokenizer(...) " +
            "or load a checkpoint with companion tokenizer files via ConfigureModel(PretrainedSource).");
    }

    internal List<int> EncodeCausalPrompt(AiDotNet.Tokenization.Interfaces.ITokenizer tokenizer, string text)
    {
        EncodingOptions opts;
        if (TokenizationConfig != null)
        {
            opts = TokenizationConfig.ToEncodingOptions();
            opts.Padding = false;
            opts.ReturnAttentionMask = false;
            opts.ReturnTokenTypeIds = false;
        }
        else
        {
            // Preserve BOS policy on causal BPE tokenizers while avoiding BERT [SEP] suffix on encoder tokenizers.
            bool hasEncoderSep = !string.IsNullOrEmpty(tokenizer.SpecialTokens.SepToken);
            opts = new EncodingOptions
            {
                AddSpecialTokens = !hasEncoderSep,
                Padding = false,
                Truncation = false,
                ReturnAttentionMask = false,
                ReturnTokenTypeIds = false
            };
        }

        return tokenizer.Encode(text, opts).TokenIds;
    }

    internal void RunCausalDecodeLoop(
        IReadOnlyList<int> promptIds,
        int maxSteps,
        Func<int, Vector<T>, (int PickedTokenId, bool Continue)> onStep)
    {
        if (Model == null)
            throw new InvalidOperationException("Model is not initialized.");

        bool useKvCache = InferenceOptimizationConfig?.EnableKVCache ?? true;

        if (useKvCache && Model is IIncrementalCausalLanguageModel<T> incrementalLm)
        {
            try
            {
                incrementalLm.ResetCache();
                int lastPicked = -1;
                for (int step = 0; step < maxSteps; step++)
                {
                    var logits = step == 0
                        ? incrementalLm.StartSequence(promptIds)
                        : incrementalLm.AppendToken(lastPicked);

                    var (picked, cont) = onStep(step, logits);
                    if (!cont || picked < 0)
                        break;
                    lastPicked = picked;
                }
            }
            finally
            {
                incrementalLm.ResetCache();
            }

            return;
        }

        if (Model is ICausalLanguageModel<T> causalLm)
        {
            var running = new List<int>(promptIds.Count + maxSteps);
            running.AddRange(promptIds);
            for (int step = 0; step < maxSteps; step++)
            {
                var logits = causalLm.NextTokenLogits(running);
                var (picked, cont) = onStep(step, logits);
                if (!cont || picked < 0)
                    break;
                running.Add(picked);
            }

            return;
        }

        if (Model is NeuralNetworkBase<T> &&
            typeof(TInput) == typeof(Tensor<T>) &&
            typeof(TOutput) == typeof(Tensor<T>))
        {
            var seqConfig = InferenceOptimizationConfig ?? new InferenceOptimizationConfig
            {
                EnableKVCache = true,
                EnablePagedKVCache = true,
                AttentionMasking = AttentionMaskingMode.Causal
            };

            using var sequence = new InferenceSequence(this, seqConfig, multiLoRATask: null);
            var running = new List<int>(promptIds.Count + maxSteps);
            running.AddRange(promptIds);

            for (int step = 0; step < maxSteps; step++)
            {
                // The whole sequence on the first step (or without a KV cache), otherwise just the newest token.
                var stepTensor = step == 0 || !sequence.IsKvCacheActive
                    ? TokenIdTensor(running)
                    : TokenIdTensor(new[] { running[running.Count - 1] });

                var rawOut = sequence.Predict((TInput)(object)stepTensor);
                if (rawOut is not Tensor<T> outTensor)
                {
                    throw new InvalidOperationException("The inference sequence did not return a logits tensor.");
                }

                var logits = ExtractLastPositionLogitsVector(outTensor);

                var (picked, cont) = onStep(step, logits);
                if (!cont || picked < 0)
                    break;
                running.Add(picked);
            }

            return;
        }

        if (Model is ILanguageModel<T> languageModel)
        {
            var running = new List<int>(promptIds.Count + maxSteps);
            running.AddRange(promptIds);
            for (int step = 0; step < maxSteps; step++)
            {
                var logitsTensor = languageModel.ForwardLogits(TokenIdTensor(running));
                var logits = ExtractLastPositionLogitsVector(logitsTensor);
                var (picked, cont) = onStep(step, logits);
                if (!cont || picked < 0)
                    break;
                running.Add(picked);
            }

            return;
        }

        throw new InvalidOperationException(
            $"Model of type '{Model.GetType().Name}' does not support causal token generation. " +
            "Configure a causal language model (e.g. PretrainedSource, Transformer<T>, or IIncrementalCausalLanguageModel<T>).");
    }

    /// <summary>The token ids as a [1, count] tensor of <typeparamref name="T"/>.</summary>
    private static Tensor<T> TokenIdTensor(IReadOnlyList<int> tokenIds)
    {
        return new Tensor<int>(tokenIds.ToArray(), new[] { 1, tokenIds.Count }).Cast<T>();
    }

    internal static Vector<T> ExtractLastPositionLogitsVector(Tensor<T> logitsTensor)
    {
        var shape = logitsTensor.Shape;
        int vocab = shape[shape.Length - 1];
        Tensor<T> row;
        if (shape.Length == 3)
        {
            row = ResultEngine.TensorSlice(logitsTensor, new[] { 0, shape[1] - 1, 0 }, new[] { 1, 1, vocab });
        }
        else if (shape.Length == 2)
        {
            row = ResultEngine.TensorSlice(logitsTensor, new[] { shape[0] - 1, 0 }, new[] { 1, vocab });
        }
        else
        {
            throw new InvalidOperationException($"Expected rank-2 or rank-3 logits tensor; got [{string.Join(",", shape)}].");
        }

        return ResultEngine.Reshape(row, new[] { vocab }).ToVector();
    }

    private static int ArgMaxLogits(Vector<T> logits)
    {
        // The first largest logit wins ties.
        return ResultEngine.TensorArgMax(Tensor<T>.FromVector(logits), 0).ToArray()[0];
    }

    private static int SampleTemperatureLogits(Vector<T> logits, double temperature, Random rng)
    {
        // p = softmax(logits / temperature). The largest logit is subtracted before dividing so a small temperature
        // cannot overflow the scaled logits in low precision. The subtraction runs on halved logits and the factor
        // of two is restored after the division: l - max itself overflows for logits of opposite sign near the top
        // of the range (-60000 - 60000 in Half), which zeroed a token the temperature should have kept.
        var numOps = MathHelper.GetNumericOperations<T>();
        var halved = ResultEngine.TensorMultiplyScalar(Tensor<T>.FromVector(logits), numOps.FromDouble(0.5));
        var shifted = ResultEngine.TensorSubtractScalar(halved, ResultEngine.TensorMaxValue(halved));
        var scaled = ResultEngine.TensorMultiplyScalar(
            ResultEngine.TensorDivideScalar(shifted, numOps.FromDouble(Math.Max(1e-6, temperature))),
            numOps.FromDouble(2.0));
        var cumulative = ResultEngine.TensorCumSum(ResultEngine.TensorSoftmax(scaled, 0), 0);

        // Inverse-CDF sampling: the first token whose cumulative probability reaches r, or the last token when
        // rounding leaves the total just below r.
        var r = new Tensor<T>(new[] { numOps.FromDouble(rng.NextDouble()) }, new[] { 1 });
        int picked = ResultEngine.TensorSearchSorted(cumulative, r).ToArray()[0];
        return Math.Min(picked, logits.Length - 1);
    }

    private static int? TryResolveEosTokenId(AiDotNet.Tokenization.Interfaces.ITokenizer tokenizer)
    {
        string eos = tokenizer.SpecialTokens.EosToken;
        if (!string.IsNullOrEmpty(eos) && tokenizer.Vocabulary.ContainsToken(eos))
            return tokenizer.Vocabulary.GetTokenId(eos);
        return null;
    }

    private static HashSet<int> ResolveAllSpecialTokenIds(AiDotNet.Tokenization.Interfaces.ITokenizer tokenizer)
    {
        var set = new HashSet<int>();
        foreach (var tok in tokenizer.SpecialTokens.GetAllSpecialTokens())
        {
            if (!string.IsNullOrEmpty(tok) && tokenizer.Vocabulary.ContainsToken(tok))
                set.Add(tokenizer.Vocabulary.GetTokenId(tok));
        }

        return set;
    }

    private static string ExtractFirstCleanWord(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return string.Empty;

        int start = 0;
        while (start < text.Length && (char.IsWhiteSpace(text[start]) || IsOuterPunctuation(text[start])))
            start++;

        int end = start;
        while (end < text.Length && !char.IsWhiteSpace(text[end]))
        {
            char c = text[end];
            if (IsOuterPunctuation(c))
            {
                bool internalApostrophe = (c == '\'' || c == '\u2019')
                    && end > start
                    && end + 1 < text.Length
                    && char.IsLetter(text[end + 1]);
                if (!internalApostrophe)
                    break;
            }
            end++;
        }

        return end > start ? text.Substring(start, end - start) : string.Empty;
    }

    private static bool IsOuterPunctuation(char c) =>
        c is '.' or ',' or '!' or '?' or ';' or ':' or '"' or '\'' or '\u201c' or '\u201d' or '\u2018' or '\u2019'
            or '(' or ')' or '[' or ']' or '{' or '}';
}
