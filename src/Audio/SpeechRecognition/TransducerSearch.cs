namespace AiDotNet.Audio.SpeechRecognition;

/// <summary>A transducer hypothesis: the labels emitted so far and log Pr(labels) after the frames read.</summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal sealed class TransducerHypothesis<T>
{
    internal TransducerHypothesis(int[] labels, T logProbability)
    {
        Labels = labels;
        LogProbability = logProbability;
    }

    /// <summary>The emitted labels, blanks excluded.</summary>
    internal int[] Labels { get; }

    /// <summary>log Pr(labels | frames read), summed over every alignment the search kept.</summary>
    internal T LogProbability { get; }
}

/// <summary>
/// Transducer decoding over the (t, u) lattice (Graves 2012, "Sequence Transduction with Recurrent Neural
/// Networks", section 2.2). The joint network is read through <c>logProbs(labels, t)</c>: the log-probabilities
/// <c>[V]</c> of every symbol at frame <c>t</c> after the prediction network has read <c>labels</c>. Any number
/// of labels can be emitted at one frame (up to a per-frame cap); the blank moves to the next frame.
/// </summary>
internal static class TransducerSearch
{
    /// <summary>
    /// Greedy decoding: at each frame emit the most likely symbol and feed it to the prediction network, until
    /// the blank (or <paramref name="maxSymbolsPerFrame"/> labels) ends the frame.
    /// </summary>
    internal static List<int> Greedy<T>(int frames, Func<IReadOnlyList<int>, int, Tensor<T>> logProbs, int blank, int maxSymbolsPerFrame)
    {
        if (logProbs is null) throw new ArgumentNullException(nameof(logProbs));
        var labels = new List<int>();
        for (int t = 0; t < frames; t++)
        {
            for (int s = 0; s < maxSymbolsPerFrame; s++)
            {
                int best = ArgMax(logProbs(labels, t));
                if (best == blank)
                {
                    break;
                }

                labels.Add(best);
            }
        }

        return labels;
    }

    /// <summary>
    /// The output sequence beam search of Graves 2012 (Algorithm 1). At each frame the kept hypotheses (B) first
    /// absorb the probability of reaching them from their own kept prefixes within the frame (the prefix merge);
    /// then the most probable hypothesis of A is moved to B with the blank and extended by each label, until B
    /// holds <paramref name="beamWidth"/> hypotheses more probable than the best left in A. B is cut to the beam
    /// at the end of each frame. The result is ordered by log Pr(y) / |y|, the paper's final criterion
    /// (the empty sequence is divided by one).
    /// </summary>
    /// <remarks>
    /// A hypothesis that was already kept is not extended into again within the frame (the prefix merge has
    /// counted that path), and each label extension has a single parent, so every alignment the search reaches
    /// is counted once. Each pop extends only the <paramref name="beamWidth"/> most likely labels, and at most
    /// <paramref name="maxSymbolsPerFrame"/> labels are emitted at one frame.
    /// </remarks>
    internal static List<TransducerHypothesis<T>> BeamSearch<T>(
        int frames,
        Func<IReadOnlyList<int>, int, Tensor<T>> logProbs,
        int vocabularySize,
        int blank,
        int beamWidth,
        int maxSymbolsPerFrame)
    {
        if (logProbs is null) throw new ArgumentNullException(nameof(logProbs));
        if (beamWidth < 1) throw new ArgumentOutOfRangeException(nameof(beamWidth), "The beam must keep at least one hypothesis.");
        if (vocabularySize < 2) throw new ArgumentOutOfRangeException(nameof(vocabularySize), "The vocabulary needs the blank and a label.");
        if (blank < 0 || blank >= vocabularySize) throw new ArgumentOutOfRangeException(nameof(blank));

        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        T negativeInfinity = numOps.FromDouble(double.NegativeInfinity);
        var blankMask = new bool[vocabularySize];
        blankMask[blank] = true;
        int extensions = Math.Min(beamWidth, vocabularySize - 1);

        var kept = new List<Node<T>> { new Node<T>(Array.Empty<int>(), numOps.Zero, 0) };
        for (int t = 0; t < frames; t++)
        {
            int frame = t;
            var cache = new Dictionary<string, Tensor<T>>(StringComparer.Ordinal);
            Tensor<T> Step(int[] labels)
            {
                string key = Key(labels);
                if (!cache.TryGetValue(key, out var scores))
                {
                    var produced = logProbs(labels, frame);
                    if (produced.Length != vocabularySize)
                    {
                        throw new InvalidOperationException($"The joint network gave {produced.Length} scores for a vocabulary of {vocabularySize}.");
                    }

                    // Kept across calls, so it is a copy rather than an engine buffer.
                    scores = new Tensor<T>(produced.ToArray(), new[] { vocabularySize });
                    cache[key] = scores;
                }

                return scores;
            }

            // A: the kept hypotheses, each with the probability of reaching it from a kept prefix in this frame.
            var initial = kept.OrderBy(h => h.Labels.Length).ToList();
            var open = new Dictionary<string, Node<T>>(StringComparer.Ordinal);
            foreach (var y in initial)
            {
                var terms = new List<T> { y.Score };
                foreach (var prefix in initial)
                {
                    int gap = y.Labels.Length - prefix.Labels.Length;
                    if (gap <= 0 || gap > maxSymbolsPerFrame || !IsPrefix(prefix.Labels, y.Labels))
                    {
                        continue;
                    }

                    // log Pr(prefix) + sum of log Pr(y_j | y_<j, t) over the labels y adds to the prefix
                    var parts = new T[gap + 1];
                    parts[0] = prefix.Score;
                    for (int j = prefix.Labels.Length; j < y.Labels.Length; j++)
                    {
                        parts[j - prefix.Labels.Length + 1] = Step(Take(y.Labels, j)).GetFlat(y.Labels[j]);
                    }

                    terms.Add(engine.TensorSum(new Tensor<T>(parts, new[] { parts.Length })));
                }

                open[y.Key] = new Node<T>(y.Labels, LogSumExp(engine, terms), 0);
            }

            var initialKeys = new HashSet<string>(open.Keys, StringComparer.Ordinal);
            var next = new List<Node<T>>();
            while (open.Count > 0)
            {
                var candidates = open.Values.ToList();
                var openScores = new Tensor<T>(candidates.Select(h => h.Score).ToArray(), new[] { candidates.Count });
                int bestIndex = ArgMax(openScores);
                var star = candidates[bestIndex];
                if (next.Count >= beamWidth && CountAbove(engine, next, star.Score) >= beamWidth)
                {
                    break;
                }

                open.Remove(star.Key);

                // log Pr(y*) + log Pr(k | y*, t) for every symbol k
                var extended = engine.TensorAddScalar(Step(star.Labels), star.Score);

                // y* then the blank ends the frame
                next.Add(new Node<T>(star.Labels, extended.GetFlat(blank), 0));
                if (star.Emitted >= maxSymbolsPerFrame)
                {
                    continue;
                }

                var labelScores = engine.TensorMaskedFill(extended, blankMask, negativeInfinity);
                engine.TensorTopK(labelScores, extensions, 0, out var topIndices);
                foreach (int k in topIndices.ToArray())
                {
                    if (k == blank)
                    {
                        continue;
                    }

                    var child = Append(star.Labels, k);
                    string childKey = Key(child);
                    if (initialKeys.Contains(childKey) || open.ContainsKey(childKey))
                    {
                        continue;
                    }

                    open[childKey] = new Node<T>(child, extended.GetFlat(k), star.Emitted + 1);
                }
            }

            kept = Prune(engine, next, beamWidth);
        }

        return RankByLength(engine, kept);
    }

    private sealed class Node<T>
    {
        internal Node(int[] labels, T score, int emitted)
        {
            Labels = labels;
            Score = score;
            Emitted = emitted;
            Key = TransducerSearch.Key(labels);
        }

        internal int[] Labels { get; }

        internal T Score { get; }

        internal int Emitted { get; }

        internal string Key { get; }
    }

    private static string Key(int[] labels) => string.Join(",", labels);

    private static bool IsPrefix(int[] prefix, int[] sequence)
    {
        for (int i = 0; i < prefix.Length; i++)
        {
            if (prefix[i] != sequence[i])
            {
                return false;
            }
        }

        return true;
    }

    private static int[] Take(int[] labels, int count)
    {
        var taken = new int[count];
        Array.Copy(labels, taken, count);
        return taken;
    }

    private static int[] Append(int[] labels, int label)
    {
        var appended = new int[labels.Length + 1];
        Array.Copy(labels, appended, labels.Length);
        appended[labels.Length] = label;
        return appended;
    }

    /// <summary>The index of the largest score; NaN never wins and ties go to the lowest index.</summary>
    private static int ArgMax<T>(Tensor<T> scores)
    {
        var engine = AiDotNetEngine.Current;
        var flat = scores.Reshape(new[] { scores.Length });
        var cleaned = engine.TensorNanToNum(flat, double.NegativeInfinity, double.PositiveInfinity, double.NegativeInfinity);
        return engine.TensorArgMax(cleaned, 0).GetFlat(0);
    }

    private static T LogSumExp<T>(IEngine engine, List<T> terms)
    {
        return engine.TensorLogSumExp(new Tensor<T>(terms.ToArray(), new[] { terms.Count })).GetFlat(0);
    }

    private static int CountAbove<T>(IEngine engine, List<Node<T>> nodes, T threshold)
    {
        var scores = new Tensor<T>(nodes.Select(h => h.Score).ToArray(), new[] { nodes.Count });
        return engine.TensorSum(engine.TensorGreaterThan(scores, threshold).Cast<int>());
    }

    private static List<Node<T>> Prune<T>(IEngine engine, List<Node<T>> nodes, int beamWidth)
    {
        if (nodes.Count <= beamWidth)
        {
            return nodes;
        }

        var scores = new Tensor<T>(nodes.Select(h => h.Score).ToArray(), new[] { nodes.Count });
        engine.TensorTopK(scores, beamWidth, 0, out var indices);
        return indices.ToArray().Select(i => nodes[i]).ToList();
    }

    private static List<TransducerHypothesis<T>> RankByLength<T>(IEngine engine, List<Node<T>> nodes)
    {
        var scores = new Tensor<T>(nodes.Select(h => h.Score).ToArray(), new[] { nodes.Count });
        var lengths = new Tensor<int>(nodes.Select(h => Math.Max(1, h.Labels.Length)).ToArray(), new[] { nodes.Count }).Cast<T>();
        engine.TensorTopK(engine.TensorDivide(scores, lengths), nodes.Count, 0, out var order);
        return order.ToArray().Select(i => new TransducerHypothesis<T>(nodes[i].Labels, nodes[i].Score)).ToList();
    }
}
