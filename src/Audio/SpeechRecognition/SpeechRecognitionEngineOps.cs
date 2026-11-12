namespace AiDotNet.Audio.SpeechRecognition;

/// <summary>
/// Whole-tensor helpers shared by the speech recognition models (CTC greedy decoding, streaming buffers).
/// </summary>
internal static class SpeechRecognitionEngineOps
{
    /// <summary>
    /// Returns the arg-max vocabulary index of every frame of <c>[frames, vocab]</c> logits
    /// (a rank-1 <c>[vocab]</c> tensor is treated as a single frame).
    /// </summary>
    /// <remarks>
    /// NaN logits never win (they are mapped to negative infinity first) and ties go to the lowest index,
    /// so a frame whose logits are all NaN or negative infinity decodes to index 0.
    /// </remarks>
    internal static int[] FrameArgMax<T>(Tensor<T> logits)
    {
        var engine = AiDotNetEngine.Current;
        int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1;
        int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0];
        var frames = logits.Reshape([numFrames, vocabSize]);
        var cleaned = engine.TensorNanToNum(frames, double.NegativeInfinity, double.PositiveInfinity, double.NegativeInfinity);
        return engine.TensorArgMax(cleaned, 1).ToArray();
    }

    /// <summary>
    /// CTC greedy collapse: drops a frame's token when it is the blank or repeats the previous frame's token.
    /// </summary>
    internal static List<int> CollapseCtcFrames(int[] frameTokens, int blankIndex)
    {
        var tokens = new List<int>();
        int prevToken = -1;
        foreach (int token in frameTokens)
        {
            if (token != blankIndex && token != prevToken)
            {
                tokens.Add(token);
            }

            prevToken = token;
        }

        return tokens;
    }

    /// <summary>
    /// Concatenates buffered audio chunks (each read in flat order) into one new rank-1 tensor.
    /// </summary>
    internal static Tensor<T> ConcatFlat<T>(IReadOnlyList<Tensor<T>> chunks)
    {
        if (chunks.Count == 1)
        {
            return new Tensor<T>(chunks[0].ToArray(), [chunks[0].Length]);
        }

        var flat = new List<Tensor<T>>(chunks.Count);
        foreach (var chunk in chunks)
        {
            flat.Add(chunk.Reshape([chunk.Length]));
        }

        return AiDotNetEngine.Current.Concat(flat, 0);
    }

    /// <summary>
    /// CTC prefix beam search (Graves et al. 2006; Hannun et al. 2014, "First-Pass Large Vocabulary
    /// Continuous Speech Recognition using Bi-Directional Recurrent DNNs", Algorithm 1).
    /// </summary>
    /// <remarks>
    /// Every prefix keeps two log-probabilities: paths ending in blank (p_b) and paths ending in a
    /// symbol (p_nb). Per frame a prefix is extended by blank (stays, into p_b), by repeating its last
    /// symbol (stays, collapsed, into p_nb), or by a new symbol (grows; a repeat of the last symbol only
    /// counts from p_b because it needs a separating blank). Candidates that reach the same prefix are
    /// merged with log-sum-exp and the <paramref name="beamWidth"/> best prefixes by p_b + p_nb survive.
    /// Symbol extensions are restricted to the frame's <paramref name="beamWidth"/> most likely
    /// non-blank symbols. Logits are <c>[frames, vocab]</c> (rank 1 = one frame).
    /// </remarks>
    internal static List<int> CtcPrefixBeamSearch<T>(Tensor<T> logits, int beamWidth, int blankIndex)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1;
        int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0];
        if (beamWidth < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(beamWidth), "Beam width must be at least 1.");
        }

        if (blankIndex < 0 || blankIndex >= vocabSize)
        {
            throw new ArgumentOutOfRangeException(nameof(blankIndex), "Blank index must be inside the vocabulary.");
        }

        // log(0) is represented by the most negative finite value (sums are floored there again) so that
        // differences inside log-add-exp stay finite.
        T logZero = numOps.MinValue;
        var frames = logits.Reshape([numFrames, vocabSize]);
        var logProbs = engine.TensorClamp(engine.TensorLogSoftmax(frames, 1), logZero, numOps.Zero);

        int charCount = Math.Min(beamWidth, vocabSize - 1);
        int[] topChars = Array.Empty<int>();
        if (charCount > 0)
        {
            var blankMask = new bool[numFrames * vocabSize];
            for (int t = 0; t < numFrames; t++)
            {
                blankMask[t * vocabSize + blankIndex] = true;
            }

            var nonBlank = engine.TensorMaskedFill(logProbs, blankMask, logZero);
            engine.TensorTopK(nonBlank, charCount, 1, out var topIndices);
            topChars = topIndices.ToArray();
        }

        var prefixes = new List<int[]> { Array.Empty<int>() };
        var pb = new Tensor<T>(new[] { numOps.Zero }, [1]);
        var pnb = new Tensor<T>(new[] { logZero }, [1]);

        for (int t = 0; t < numFrames; t++)
        {
            var frameLogProbs = engine.TensorSelect(logProbs, 0, t);
            int beamCount = prefixes.Count;
            int extCount = beamCount * charCount;
            int candidateCount = 2 * beamCount + extCount;

            // Candidate layout: [0, B) blank extension, [B, 2B) collapsed repeat, [2B, 2B + B*K) new symbol.
            var lastIdx = new int[beamCount];
            var hasLast = new bool[beamCount];
            var blankIdx = new int[beamCount];
            var charIdx = new int[extCount];
            var charBeam = new int[extCount];
            var isRepeat = new bool[extCount];
            var groupOf = new int[candidateCount];
            var groupIds = new Dictionary<string, int>(StringComparer.Ordinal);
            var groupPrefixes = new List<int[]>();

            for (int b = 0; b < beamCount; b++)
            {
                var prefix = prefixes[b];
                hasLast[b] = prefix.Length > 0;
                lastIdx[b] = hasLast[b] ? prefix[prefix.Length - 1] : blankIndex;
                blankIdx[b] = blankIndex;
                int group = GroupOf(prefix, groupIds, groupPrefixes);
                groupOf[b] = group;
                groupOf[beamCount + b] = group;
                for (int k = 0; k < charCount; k++)
                {
                    int e = b * charCount + k;
                    int c = topChars[t * charCount + k];
                    charIdx[e] = c;
                    charBeam[e] = b;
                    isRepeat[e] = hasLast[b] && c == lastIdx[b];
                    if (c == blankIndex)
                    {
                        groupOf[2 * beamCount + e] = -1;
                        continue;
                    }

                    var extended = new int[prefix.Length + 1];
                    Array.Copy(prefix, extended, prefix.Length);
                    extended[prefix.Length] = c;
                    groupOf[2 * beamCount + e] = GroupOf(extended, groupIds, groupPrefixes);
                }
            }

            var total = engine.TensorLogAddExp(pb, pnb);
            var pbBlank = engine.TensorMax(engine.TensorAdd(total, engine.TensorIndexSelect(frameLogProbs, IntVector(blankIdx), 0)), logZero);
            var pnbRepeat = engine.TensorWhere(
                hasLast,
                engine.TensorMax(engine.TensorAdd(pnb, engine.TensorIndexSelect(frameLogProbs, IntVector(lastIdx), 0)), logZero),
                Filled(beamCount, logZero));

            var pbParts = new List<Tensor<T>> { pbBlank, Filled(beamCount + extCount, logZero) };
            var pnbParts = new List<Tensor<T>> { Filled(beamCount, logZero), pnbRepeat };
            if (extCount > 0)
            {
                var beamIndex = IntVector(charBeam);
                var extBase = engine.TensorWhere(
                    isRepeat,
                    engine.TensorIndexSelect(pb, beamIndex, 0),
                    engine.TensorIndexSelect(total, beamIndex, 0));
                pnbParts.Add(engine.TensorMax(engine.TensorAdd(extBase, engine.TensorIndexSelect(frameLogProbs, IntVector(charIdx), 0)), logZero));
            }

            var candidatePb = engine.Concat(pbParts, 0);
            var candidatePnb = engine.Concat(pnbParts, 0);

            // Merge candidates reaching the same prefix: log-sum-exp over each group's members.
            int groupCount = groupPrefixes.Count;
            var member = new bool[groupCount * candidateCount];
            for (int j = 0; j < candidateCount; j++)
            {
                if (groupOf[j] >= 0)
                {
                    member[groupOf[j] * candidateCount + j] = true;
                }
            }

            var groupPb = GroupLogSumExp(engine, candidatePb, member, groupCount, candidateCount, logZero);
            var groupPnb = GroupLogSumExp(engine, candidatePnb, member, groupCount, candidateCount, logZero);
            var groupTotal = engine.TensorLogAddExp(groupPb, groupPnb);

            int keep = Math.Min(beamWidth, groupCount);
            engine.TensorTopK(groupTotal.Reshape([1, groupCount]), keep, 1, out var bestGroups);
            int[] selected = bestGroups.ToArray();
            var selectedIndex = IntVector(selected);

            // Copy the survivors out of engine buffers: they are read again on the next frame.
            pb = new Tensor<T>(engine.TensorIndexSelect(groupPb, selectedIndex, 0).ToArray(), [keep]);
            pnb = new Tensor<T>(engine.TensorIndexSelect(groupPnb, selectedIndex, 0).ToArray(), [keep]);
            var survivors = new List<int[]>(keep);
            foreach (int g in selected)
            {
                survivors.Add(groupPrefixes[g]);
            }

            prefixes = survivors;
        }

        return new List<int>(prefixes[0]);
    }

    private static int GroupOf(int[] prefix, Dictionary<string, int> groupIds, List<int[]> groupPrefixes)
    {
        string key = string.Join(",", prefix);
        if (!groupIds.TryGetValue(key, out int group))
        {
            group = groupPrefixes.Count;
            groupIds[key] = group;
            groupPrefixes.Add(prefix);
        }

        return group;
    }

    private static Tensor<T> GroupLogSumExp<T>(IEngine engine, Tensor<T> candidates, bool[] member, int groupCount, int candidateCount, T logZero)
    {
        var tiled = engine.TensorTile(candidates.Reshape([1, candidateCount]), [groupCount, 1]);
        var fill = Filled(groupCount * candidateCount, logZero).Reshape([groupCount, candidateCount]);
        var masked = engine.TensorWhere(member, tiled, fill);
        return engine.TensorLogSumExp(masked, 1, false);
    }

    private static Tensor<int> IntVector(int[] values) => new Tensor<int>(values, [values.Length]);

    private static Tensor<T> Filled<T>(int length, T value)
    {
        var tensor = new Tensor<T>([length]);
        tensor.Fill(value);
        return tensor;
    }
}
