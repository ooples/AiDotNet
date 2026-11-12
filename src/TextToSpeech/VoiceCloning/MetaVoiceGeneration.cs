namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>MetaVoice's original uncached, single-utterance generation rules. Sampling
/// receives explicit uniform draws so policy and controller behavior can be audited independently.</summary>
internal static class MetaVoiceGeneration
{
    internal static double[] Probabilities(double[] logits, double temperature, int? topK, double? topP)
    {
        if (temperature <= 0 || double.IsNaN(temperature) || double.IsInfinity(temperature) ||
            (topK is { } k && k <= 0) || (topP is { } p && (p < 0 || p > 1 || double.IsNaN(p))) ||
            (topK.HasValue && topP.HasValue) || logits.Length == 0)
            throw new ArgumentException("Invalid MetaVoice sampling configuration.");
        var values = logits.Select(x => x / temperature).ToArray();
        if (topK is { } keep)
        {
            double threshold = values.OrderByDescending(x => x).ElementAt(Math.Min(keep, values.Length) - 1);
            for (int i = 0; i < values.Length; i++) if (values[i] < threshold) values[i] = double.NegativeInfinity;
        }
        double maximum = values.Max();
        var probabilities = values.Select(x => Math.Exp(x - maximum)).ToArray();
        double total = probabilities.Sum();
        for (int i = 0; i < probabilities.Length; i++) probabilities[i] /= total;
        if (topP is { } nucleus)
        {
            var order = Enumerable.Range(0, values.Length).OrderByDescending(i => probabilities[i]).ToArray();
            double cumulative = 0; bool remove = false;
            foreach (int i in order)
            {
                if (remove) probabilities[i] = 0;
                else { cumulative += probabilities[i]; remove = cumulative > nucleus; }
            }
            total = probabilities.Sum();
            for (int i = 0; i < probabilities.Length; i++) probabilities[i] /= total;
        }
        if (probabilities.Any(x => double.IsNaN(x) || double.IsInfinity(x)))
            throw new ArithmeticException("MetaVoice sampling probabilities are not finite.");
        return probabilities;
    }

    private static int Draw(double[] probabilities, double uniform)
    {
        if (uniform < 0 || uniform >= 1 || double.IsNaN(uniform)) throw new ArgumentOutOfRangeException(nameof(uniform));
        double cumulative = 0; int last = -1;
        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (probabilities[i] > 0) last = i;
            if (uniform < cumulative) return i;
        }
        return last >= 0 ? last : throw new ArithmeticException("MetaVoice sampling distribution is empty.");
    }

    internal static int[] FirstStage(int[] prompt, int maximumNewTokens, int context,
        Func<int[], bool, double[]> forward, Func<int, double> uniform, bool speakerConditioned,
        double temperature = 1, int? topK = null, double? topP = .95,
        (double Speaker, double Prompt)? guidance = null, int endAudio = 2048, int endText = 2561,
        Action<int, double[]>? observeProbabilities = null)
    {
        if (prompt.Length == 0 || maximumNewTokens < 1 || prompt.Length + maximumNewTokens >= context)
            throw new ArgumentException("MetaVoice prompt and generation budget must fit strictly inside the context.");
        if (guidance is { } scale && (!speakerConditioned || scale.Speaker < 1 || scale.Prompt < 1))
            throw new ArgumentException("MetaVoice guidance requires a conditioned speaker and scales at least one.");
        var output = Enumerable.Repeat(endAudio, prompt.Length + maximumNewTokens).ToArray();
        Array.Copy(prompt, output, prompt.Length);
        for (int step = 0; step < maximumNewTokens; step++)
        {
            var prefix = output.Take(prompt.Length + step).ToArray();
            var logits = forward(prefix, speakerConditioned);
            if (guidance is { } cfg)
            {
                var unconditionalSpeaker = forward(prefix, false);
                var unconditionalPrompt = cfg.Prompt > 1
                    ? forward(prefix.Select(token => token > endAudio ? endText : token).ToArray(), true) : null;
                for (int i = 0; i < logits.Length; i++) logits[i] = (cfg.Speaker + cfg.Prompt - 1) * logits[i]
                    + (1 - cfg.Speaker) * unconditionalSpeaker[i] + (unconditionalPrompt is null ? 0 : (1 - cfg.Prompt) * unconditionalPrompt[i]);
            }
            var probabilities = Probabilities(logits, temperature, topK, topP);
            observeProbabilities?.Invoke(step, probabilities);
            int next = Draw(probabilities, uniform(step));
            output[prompt.Length + step] = next;
            if (next == endAudio) break;
        }
        return output;
    }

    internal static int[][] SecondStageInput(int[] text, int[][] firstAudio, int context = 1024, int pad = 1024)
    {
        if (firstAudio.Length != 2 || firstAudio[0].Length != firstAudio[1].Length || context <= 0)
            throw new ArgumentException("Second-stage conditioning requires two equal-length first-stage codebooks.");
        return new[]
        {
            text.Concat(firstAudio[0]).Concat(new[] { pad }).Concat(Enumerable.Repeat(pad, context)).Take(context).ToArray(),
            Enumerable.Repeat(pad, text.Length).Concat(firstAudio[1]).Concat(new[] { pad }).Concat(Enumerable.Repeat(pad, context)).Take(context).ToArray()
        };
    }

    internal static int[][] NonCausal(int heads, int time, Func<int, int, double[]> logits,
        Func<int, int, double> uniform, double temperature = 1, int? topK = 200,
        Action<int, int, double[]>? observeProbabilities = null)
    {
        var output = new int[heads][];
        for (int head = 0; head < heads; head++)
        {
            output[head] = new int[time];
            for (int frame = 0; frame < time; frame++)
            {
                var probabilities = Probabilities(logits(head, frame), temperature, topK, null);
                observeProbabilities?.Invoke(head, frame, probabilities);
                output[head][frame] = Draw(probabilities, uniform(head, frame));
            }
        }
        return output;
    }
}
