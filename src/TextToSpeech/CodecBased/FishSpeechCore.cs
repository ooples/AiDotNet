using AiDotNet.Helpers;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Which Fish Speech release a sequence format follows.</summary>
internal enum FishSpeechVersion
{
    /// <summary>Fish Speech 1.4: one <c>&lt;|semantic|&gt;</c> token per frame, codes stored plus one.</summary>
    V14,

    /// <summary>Fish Speech 1.5: the frame's <c>&lt;|semantic:c0|&gt;</c> token, codes stored as they are.</summary>
    V15,
}

/// <summary>
/// Fish Speech's sequences — rows <c>[1 + codebooks, L]</c>, the token row then one row per codebook — as the releases build
/// them for training (<c>pack_sentences</c>) and for inference (<c>encode_tokens</c>), with the release's tokenizer
/// (<paramref name="encode"/>, special tokens recognized).
/// </summary>
internal sealed class FishSpeechFormat
{
    private readonly Func<string, int[]> _encode;

    public FishSpeechFormat(FishSpeechVersion version, Func<string, int[]> encode, Func<string, int> tokenId, int codebooks, int codebookSize)
    {
        Version = version;
        _encode = encode;
        Codebooks = codebooks;
        CodebookSize = codebookSize;
        ImEnd = tokenId("<|im_end|>");
        if (version == FishSpeechVersion.V15) SemanticBegin = tokenId("<|semantic:0|>");
        else Semantic = tokenId("<|semantic|>");
    }

    public FishSpeechVersion Version { get; }
    public int Codebooks { get; }
    public int CodebookSize { get; }
    public int ImEnd { get; }
    public int Semantic { get; } = -1;
    public int SemanticBegin { get; } = -1;

    /// <summary>Whether a token carries a frame's codes (its codebook embeddings are added).</summary>
    public bool IsSemantic(int token) => Version == FishSpeechVersion.V15
        ? token >= SemanticBegin && token < SemanticBegin + CodebookSize
        : token == Semantic;

    public int[] Encode(string text) => _encode(text);

    // A message's parts as the 1.5 Conversation encodes them: each part separately.
    private sealed class Encoded
    {
        public readonly List<int> Tokens = new(), Labels = new();
        public readonly List<bool> Vq = new();
        public readonly List<int[]> Codes = new();       // per VQ frame, its codebook values
    }

    private void Message15(Encoded e, string role, IEnumerable<object> parts, bool calLoss, bool addImEnd = true)
    {
        var all = new List<object> { $"<|im_start|>{role}\n" };
        all.AddRange(parts);
        if (addImEnd) all.Add("<|im_end|>");
        for (int p = 0; p < all.Count; p++)
        {
            bool first = p == 0;
            if (all[p] is string or int[])
            {
                foreach (int id in all[p] as int[] ?? _encode((string)all[p]))
                {
                    e.Tokens.Add(id);
                    e.Labels.Add(calLoss && !first ? id : -100);
                    e.Vq.Add(false);
                    e.Codes.Add(new int[Codebooks]);
                }
            }
            else
            {
                var codes = (int[,])all[p];
                for (int t = 0; t < codes.GetLength(1); t++)
                {
                    int id = SemanticBegin + codes[0, t];
                    e.Tokens.Add(id);
                    e.Labels.Add(calLoss ? id : -100);
                    e.Vq.Add(true);
                    e.Codes.Add(Enumerable.Range(0, Codebooks).Select(b => codes[b, t]).ToArray());
                }
            }
        }
    }

    private int[,] Rows(Encoded e)
    {
        int l = e.Tokens.Count;
        var rows = new int[Codebooks + 1, l];
        for (int t = 0; t < l; t++)
        {
            rows[0, t] = e.Tokens[t];
            if (e.Vq[t])
                for (int b = 0; b < Codebooks; b++) rows[b + 1, t] = e.Codes[t][b];
        }
        return rows;
    }

    /// <summary>
    /// A training sample (the dataset's <c>pack_sentences</c>): tokens and labels <c>[1 + codebooks, L]</c>, shifted
    /// (labels are the next position's), with -100 where nothing is scored. 1.5: the system prompt, the text and the
    /// assistant's <c>&lt;|voice|&gt;</c> and codes, scored from <c>&lt;|voice|&gt;</c> on. 1.4: the text and
    /// <c>&lt;|semantic|&gt;</c> per frame with codes plus one, the token row scored throughout.
    /// </summary>
    public (int[,] Tokens, int[,] Labels) TrainingSample(string text, int[,] codes) => TrainingSampleOf(text, codes);

    /// <summary><see cref="TrainingSample(string, int[,])"/> with the text already tokenized.</summary>
    public (int[,] Tokens, int[,] Labels) TrainingSample(int[] textIds, int[,] codes) => TrainingSampleOf(textIds, codes);

    // 1.4's user turn and the assistant's header: the release encodes the string whole; token ids are put between the
    // encoded markup (the same ids whenever the text starts with a character the pre-tokenizer splits from "\n").
    private int[] UserPrompt14(object text) => text is string s
        ? _encode($"<|im_start|>user\n{s}<|im_end|><|im_start|>assistant\n")
        : _encode("<|im_start|>user\n").Concat((int[])text).Concat(_encode("<|im_end|><|im_start|>assistant\n")).ToArray();

    private (int[,] Tokens, int[,] Labels) TrainingSampleOf(object text, int[,] codes)
    {
        int frames = codes.GetLength(1), n = Codebooks;
        if (codes.GetLength(0) != n) throw new ArgumentException($"Expected {n} codebooks.", nameof(codes));
        if (Version == FishSpeechVersion.V15)
        {
            var e = new Encoded();
            Message15(e, "system", new object[] { "Speak out the provided text." }, calLoss: false);
            Message15(e, "user", new object[] { text }, calLoss: false);
            Message15(e, "assistant", new object[] { "<|voice|>", codes }, calLoss: true);
            int l = e.Tokens.Count - 1;
            var tokens = new int[n + 1, l];
            var labels = new int[n + 1, l];
            for (int t = 0; t < l; t++)
            {
                tokens[0, t] = e.Tokens[t];
                labels[0, t] = e.Labels[t + 1];
                for (int b = 0; b < n; b++)
                {
                    tokens[b + 1, t] = e.Vq[t] ? e.Codes[t][b] : 0;
                    labels[b + 1, t] = e.Vq[t + 1] ? e.Codes[t + 1][b] : -100;
                }
            }
            for (int b = 0; b < n; b++) labels[b + 1, l - 1] = 0;
            return (tokens, labels);
        }
        else
        {
            var prompt = UserPrompt14(text);
            int p = prompt.Length, full = p + frames + 1;
            var sequence = new int[n + 1, full];
            for (int t = 0; t < p; t++) sequence[0, t] = prompt[t];
            for (int t = 0; t < frames; t++)
            {
                sequence[0, p + t] = Semantic;
                for (int b = 0; b < n; b++) sequence[b + 1, p + t] = codes[b, t] + 1;
            }
            sequence[0, full - 1] = ImEnd;
            int l = full - 1;
            var tokens = new int[n + 1, l];
            var labels = new int[n + 1, l];
            for (int r = 0; r <= n; r++)
                for (int t = 0; t < l; t++)
                {
                    tokens[r, t] = sequence[r, t];
                    labels[r, t] = r > 0 && t + 1 < p ? -100 : sequence[r, t + 1];
                }
            return (tokens, labels);
        }
    }

    /// <summary>1.5's system prompt rows (<c>"Speak out the provided text."</c>); 1.4 has none.</summary>
    public int[,]? SystemPrompt()
    {
        if (Version != FishSpeechVersion.V15) return null;
        var e = new Encoded();
        Message15(e, "system", new object[] { "Speak out the provided text." }, calLoss: false);
        return Rows(e);
    }

    /// <summary>
    /// One segment of an inference prompt (<c>encode_tokens</c>): the user's (already cleaned) text and the assistant's turn,
    /// either a reference's codes <c>[codebooks, frames]</c> closed by <c>&lt;|im_end|&gt;</c> or the open turn to continue.
    /// </summary>
    public int[,] Segment(string text, int[,]? codes) => SegmentOf(text, codes);

    /// <summary><see cref="Segment(string, int[,])"/> with the text already tokenized.</summary>
    public int[,] Segment(int[] textIds, int[,]? codes) => SegmentOf(textIds, codes);

    private int[,] SegmentOf(object text, int[,]? codes)
    {
        int n = Codebooks;
        if (Version == FishSpeechVersion.V15)
        {
            var e = new Encoded();
            Message15(e, "user", new object[] { text }, calLoss: false);
            if (codes is not null) Message15(e, "assistant", new object[] { "<|voice|>", codes }, calLoss: false);
            else Message15(e, "assistant", new object[] { "<|voice|>" }, calLoss: false, addImEnd: false);
            return Rows(e);
        }
        var prompt = UserPrompt14(text);
        int frames = codes?.GetLength(1) ?? 0, l = prompt.Length + (codes is null ? 0 : frames + 1);
        var rows = new int[n + 1, l];
        for (int t = 0; t < prompt.Length; t++) rows[0, t] = prompt[t];
        if (codes is not null)
        {
            for (int t = 0; t < frames; t++)
            {
                rows[0, prompt.Length + t] = Semantic;
                for (int b = 0; b < n; b++) rows[b + 1, prompt.Length + t] = codes[b, t] + 1;
            }
            rows[0, l - 1] = ImEnd;
        }
        return rows;
    }

    /// <summary>The codes <c>generate_long</c> keeps of a generated sequence: 1.5 drops the first generated frame and keeps
    /// the last (<c>y[1:, n + 1:]</c>); 1.4 drops the last and subtracts one (<c>y[1:, n:-1] - 1</c>).</summary>
    public int[,] KeptCodes(int[,] sequence, int promptLength)
    {
        int n = Codebooks, total = sequence.GetLength(1);
        int start = Version == FishSpeechVersion.V15 ? promptLength + 1 : promptLength;
        int end = Version == FishSpeechVersion.V15 ? total : total - 1;
        int frames = Math.Max(0, end - start);
        var codes = new int[n, frames];
        for (int b = 0; b < n; b++)
            for (int t = 0; t < frames; t++)
            {
                codes[b, t] = sequence[b + 1, start + t] - (Version == FishSpeechVersion.V14 ? 1 : 0);
                if (codes[b, t] < 0) throw new InvalidOperationException("A negative code was generated.");
            }
        return codes;
    }

    /// <summary>The release's sampling (<c>logits_to_probs</c>): the repetition penalty over <paramref name="previous"/>
    /// (when given), top-p keeping at least the best token, the temperature, and a draw from the softmax.</summary>
    public static int Sample(double[] logits, IReadOnlyList<int>? previous, double temperature, double topP, double repetitionPenalty, Random random)
    {
        var scores = (double[])logits.Clone();
        if (previous is not null)
            foreach (int t in previous)
                if (t >= 0 && t < scores.Length)
                    scores[t] = logits[t] < 0 ? logits[t] * repetitionPenalty : logits[t] / repetitionPenalty;
        var order = Enumerable.Range(0, scores.Length).OrderByDescending(i => scores[i]).ThenBy(i => i).ToArray();
        double max = scores[order[0]], total = 0;
        foreach (var s in scores) total += Math.Exp(s - max);
        double cumulative = 0;
        var removed = new bool[scores.Length];
        for (int r = 0; r < order.Length; r++)
        {
            cumulative += Math.Exp(scores[order[r]] - max) / total;
            removed[order[r]] = r > 0 && cumulative > topP;
        }
        double t2 = Math.Max(temperature, 1e-5), best = double.NegativeInfinity;
        for (int i = 0; i < scores.Length; i++) if (!removed[i]) best = Math.Max(best, scores[i] / t2);
        var weights = new double[scores.Length];
        double sum = 0;
        for (int i = 0; i < scores.Length; i++)
        {
            weights[i] = removed[i] ? 0 : Math.Exp(scores[i] / t2 - best);
            sum += weights[i];
        }
        double u = random.NextDouble() * sum, acc = 0;
        int last = order[0];
        for (int i = 0; i < scores.Length; i++)
        {
            if (weights[i] <= 0) continue;
            acc += weights[i];
            last = i;
            if (acc > u) return i;
        }
        return last;
    }
}

/// <summary>
/// The releases' decoding (<c>generate</c>, <c>decode_n_tokens</c>, <c>decode_one_token_ar</c>): the slow transformer over
/// the prompt with a key-value cache, then one frame per step — the token sampled from the slow logits, the codebooks from
/// the fast transformer in turn — until <c>&lt;|im_end|&gt;</c> or the length limit. The repetition penalty reads a window
/// of the last 16 decoded frames (the first 16 columns, zeros included, until 16 are decoded); the prefill frame has none.
/// 1.5 penalizes the token too and takes codebook 0 from the semantic token; 1.4 samples every codebook and zeroes them
/// on non-semantic frames.
/// </summary>
internal static class FishSpeechGenerator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private static double[] Row(Tensor<T> logits, int row)
    {
        int v = logits.Shape[1];
        var r = new double[v];
        for (int i = 0; i < v; i++) r[i] = NumOps.ToDouble(logits[row, i]);
        return r;
    }

    // Constrained decoding keeps only speech: the semantic tokens and (1.4) the codes a frame can store, 1 and up.
    private static double[] Speech(double[] logits, Func<int, bool> allowed)
    {
        for (int i = 0; i < logits.Length; i++)
            if (!allowed(i)) logits[i] = double.NegativeInfinity;
        return logits;
    }

    private static int[] Frame(IEngine engine, FishDualAr<T> net, FishSpeechFormat format, Tensor<T> hidden, int[,]? window, double temperature,
        double topP, double repetitionPenalty, Random random, bool constrain)
    {
        int n = format.Codebooks;
        var frame = new int[n + 1];
        IReadOnlyList<int>? Window(int row) => window is null ? null : Enumerable.Range(0, window.GetLength(1)).Select(t => window[row, t]).ToArray();
        var tokenLogits = Row(net.TokenLogits(hidden), 0);
        if (constrain) tokenLogits = Speech(tokenLogits, format.IsSemantic);
        frame[0] = FishSpeechFormat.Sample(tokenLogits,
            format.Version == FishSpeechVersion.V15 ? Window(0) : null, temperature, topP, repetitionPenalty, random);
        var inputs = new List<Tensor<T>> { hidden };
        int first = 0;
        if (format.Version == FishSpeechVersion.V15)
        {
            frame[1] = Math.Max(frame[0] - format.SemanticBegin, 0);
            inputs.Add(net.FastEmbeddings.Forward(Latest.IndexTtsOps.Ids<T>(new[] { frame[1] })));
            first = 1;
        }
        for (int b = first; b < n; b++)
        {
            var logits = Row(net.FastLogits(engine.TensorConcatenate(inputs.ToArray(), 0)), inputs.Count - 1);
            if (constrain && format.Version == FishSpeechVersion.V14) logits = Speech(logits, code => code > 0);
            frame[b + 1] = FishSpeechFormat.Sample(logits, Window(b + 1), temperature, topP, repetitionPenalty, random);
            inputs.Add(net.FastEmbeddings.Forward(Latest.IndexTtsOps.Ids<T>(new[] { frame[b + 1] })));
        }
        if (format.Version == FishSpeechVersion.V14 && frame[0] != format.Semantic)
            for (int b = 1; b <= n; b++) frame[b] = 0;
        return frame;
    }

    /// <summary>The prompt rows <c>[1 + codebooks, L]</c> followed by the decoded frames; with
    /// <paramref name="constrain"/>, every frame is speech and decoding runs to the length limit.</summary>
    public static int[,] Generate(IEngine engine, FishDualAr<T> net, FishSpeechFormat format, int[,] prompt, int maxNewTokens, double temperature,
        double topP, double repetitionPenalty, Random random, bool constrain = false)
    {
        int n = format.Codebooks, length = prompt.GetLength(1), maxSequence = net.Configuration.MaxSequence;
        int newTokens = maxNewTokens > 0 ? Math.Min(maxNewTokens, maxSequence - length) : maxSequence - length;
        if (newTokens < 1) throw new ArgumentException("The prompt fills the model's context.", nameof(prompt));
        var cache = new FishDualAr<T>.Cache();
        var hidden = net.SlowHidden(net.Embed(prompt, format.IsSemantic), Enumerable.Range(0, length).ToArray(), cache);
        var frames = new List<int[]> { Frame(engine, net, format, engine.TensorNarrow(hidden, 0, length - 1, 1), null, temperature, topP, repetitionPenalty, random, constrain) };
        var previous = new int[n + 1, maxSequence];
        for (int i = 0; i < newTokens - 1; i++)
        {
            int from = i < 16 ? 0 : i - 16;
            var window = new int[n + 1, 16];
            for (int r = 0; r <= n; r++)
                for (int t = 0; t < 16; t++) window[r, t] = previous[r, from + t];
            var current = new int[n + 1, 1];
            for (int r = 0; r <= n; r++) current[r, 0] = frames[^1][r];
            var h = net.SlowHidden(net.Embed(current, format.IsSemantic), new[] { length + i }, cache);
            var next = Frame(engine, net, format, h, window, temperature, topP, repetitionPenalty, random, constrain);
            frames.Add(next);
            for (int r = 0; r <= n; r++) previous[r, i] = next[r];
            if (next[0] == format.ImEnd) break;
        }
        var sequence = new int[n + 1, length + frames.Count];
        for (int r = 0; r <= n; r++)
        {
            for (int t = 0; t < length; t++) sequence[r, t] = prompt[r, t];
            for (int t = 0; t < frames.Count; t++) sequence[r, length + t] = frames[t][r];
        }
        return sequence;
    }
}

/// <summary>Fish Speech's training objectives over a teacher-forced pass.</summary>
internal static class FishSpeechObjective<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    // The sum of log p(label) over the labels that are not -100, and their count.
    private static (Tensor<T> Sum, int Count) LogProbabilities(IEngine engine, Tensor<T> logits, IReadOnlyList<int> labels)
    {
        int rows = logits.Shape[0], v = logits.Shape[1];
        var picks = new List<int>();
        for (int i = 0; i < rows; i++)
            if (labels[i] != -100) picks.Add(i * v + labels[i]);
        if (picks.Count == 0) return (new Tensor<T>(new[] { 1 }), 0);
        var flat = engine.Reshape(engine.TensorLogSoftmax(logits, axis: 1), new[] { rows * v, 1 });
        var picked = engine.TensorIndexSelect(flat, new Tensor<int>(new[] { picks.Count }, new Vector<int>(picks.ToArray())), 0);
        return (engine.Reshape(engine.ReduceSum(picked, null), new[] { 1 }), picks.Count);
    }

    private static (int[] Tokens, int[] Codebooks) Labels(int[,] labels)
    {
        int rows = labels.GetLength(0), l = labels.GetLength(1), n = rows - 1;
        var tokens = Enumerable.Range(0, l).Select(t => labels[0, t]).ToArray();
        var codebooks = new int[l * n];
        for (int t = 0; t < l; t++)
            for (int b = 0; b < n; b++) codebooks[t * n + b] = labels[b + 1, t];
        return (tokens, codebooks);
    }

    /// <summary>lit_module's losses: the token cross-entropy and the codebook cross-entropy, each a mean over its scored
    /// labels.</summary>
    public static (Tensor<T> Token, Tensor<T> Codebook) Losses(IEngine engine, FishDualAr<T> net, FishSpeechFormat format, int[,] tokens, int[,] labels,
        bool training = false, Random? random = null)
    {
        int l = tokens.GetLength(1);
        var (tokenLogits, codebookLogits) = net.Forward(tokens, format.IsSemantic, training, random);
        var (tokenLabels, codebookLabels) = Labels(labels);
        var (tokenSum, tokenCount) = LogProbabilities(engine, tokenLogits, tokenLabels);
        var flat = engine.Reshape(codebookLogits, new[] { l * format.Codebooks, format.CodebookSize });
        var (codeSum, codeCount) = LogProbabilities(engine, flat, codebookLabels);
        return (engine.TensorMultiplyScalar(tokenSum, NumOps.FromDouble(-1.0 / Math.Max(1, tokenCount))),
            engine.TensorMultiplyScalar(codeSum, NumOps.FromDouble(-1.0 / Math.Max(1, codeCount))));
    }

    /// <summary>A sequence's log-probability: every scored token and codebook label's log-probability, summed.</summary>
    public static Tensor<T> SequenceLogProbability(IEngine engine, FishDualAr<T> net, FishSpeechFormat format, int[,] tokens, int[,] labels,
        bool training = false, Random? random = null)
    {
        int l = tokens.GetLength(1);
        var (tokenLogits, codebookLogits) = net.Forward(tokens, format.IsSemantic, training, random);
        var (tokenLabels, codebookLabels) = Labels(labels);
        var flat = engine.Reshape(codebookLogits, new[] { l * format.Codebooks, format.CodebookSize });
        return engine.TensorAdd(LogProbabilities(engine, tokenLogits, tokenLabels).Sum, LogProbabilities(engine, flat, codebookLabels).Sum);
    }

    // The negative log-sigmoid of x, softplus(-x), written on the tape as log(1 + exp(-x)).
    private static Tensor<T> NegativeLogSigmoid(IEngine engine, Tensor<T> x) =>
        engine.TensorLog(engine.TensorAddScalar(engine.TensorExp(engine.TensorNegate(x)), NumOps.One));

    /// <summary>Standard DPO (Rafailov et al. 2023): <c>-log sigmoid(beta ((pi_c - ref_c) - (pi_r - ref_r)))</c> with the
    /// reference's log-probabilities fixed.</summary>
    public static Tensor<T> Dpo(IEngine engine, Tensor<T> chosen, Tensor<T> rejected, double referenceChosen, double referenceRejected, double beta)
    {
        var margin = engine.TensorAddScalar(engine.TensorSubtract(chosen, rejected), NumOps.FromDouble(-(referenceChosen - referenceRejected)));
        return NegativeLogSigmoid(engine, engine.TensorMultiplyScalar(margin, NumOps.FromDouble(beta)));
    }

    /// <summary>
    /// Fish Speech's reference-free DPO (git 8093258): per position, the codebook labels' summed log-probabilities of the
    /// chosen and rejected sequences, <c>-log sigmoid(beta (chosen - rejected))</c> averaged over positions. The release
    /// batches the pair, so the shorter sequence is padded (its log-probabilities zero) to the longer one's length.
    /// </summary>
    public static Tensor<T> ReferenceFreeDpo(IEngine engine, FishDualAr<T> net, FishSpeechFormat format, int[,] chosenTokens, int[,] chosenLabels,
        int[,] rejectedTokens, int[,] rejectedLabels, double beta, bool training = false, Random? random = null)
    {
        int n = format.Codebooks, v = format.CodebookSize;
        int l = Math.Max(chosenTokens.GetLength(1), rejectedTokens.GetLength(1));
        Tensor<T> PerPosition(int[,] tokens, int[,] labels)
        {
            int length = tokens.GetLength(1);
            var (_, codebookLogits) = net.Forward(tokens, format.IsSemantic, training, random);
            var logProbabilities = engine.TensorLogSoftmax(engine.Reshape(codebookLogits, new[] { length * n, v }), axis: 1);
            var mask = new Tensor<T>(new[] { length * n, v });
            for (int t = 0; t < length; t++)
                for (int b = 0; b < n; b++)
                {
                    int label = labels[b + 1, t];
                    if (label != -100) mask[t * n + b, label] = NumOps.One;
                }
            var summed = engine.ReduceSum(engine.TensorMultiply(logProbabilities, mask), new[] { 1 }, keepDims: false);
            var positions = engine.ReduceSum(engine.Reshape(summed, new[] { length, n }), new[] { 1 }, keepDims: false);
            return length == l ? positions : engine.TensorConcatenate(new[] { positions, new Tensor<T>(new[] { l - length }) }, 0);
        }
        var difference = engine.TensorSubtract(PerPosition(chosenTokens, chosenLabels), PerPosition(rejectedTokens, rejectedLabels));
        var losses = NegativeLogSigmoid(engine, engine.TensorMultiplyScalar(difference, NumOps.FromDouble(beta)));
        return engine.Reshape(engine.TensorMultiplyScalar(engine.ReduceSum(losses, null), NumOps.FromDouble(1.0 / l)), new[] { 1 });
    }
}
