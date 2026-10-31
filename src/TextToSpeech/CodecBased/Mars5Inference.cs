using AiDotNet.Helpers;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The NAR's diffusion sampling settings (<c>DSH</c> in <c>mars5/diffuser.py</c>), as <c>Mars5TTS.tts</c> sets
/// them. <c>tts</c> sets <c>last_greedy=True</c>, but <c>perform_simple_inference</c> never passes it to
/// <c>reverse_diffusion</c>, so the release samples at t = 0 too: <see cref="LastGreedy"/> is false to match.</summary>
internal sealed record Mars5DiffusionSampling(int JumpLength = 1, int JumpSamples = 1, bool LastGreedy = false, double X0Temperature = 0.7,
    double GuidanceWeight = 3, bool ScaledInference = true, bool DeepClone = true, int Q0OverrideSteps = 20);

/// <summary>
/// MARS5's multinomial diffusion (<c>MultinomialDiffusion</c>, after Hoogeboom et al. 2021 by way of TransFusion): a cosine
/// schedule over <c>T</c> steps on categorical codes of <c>K</c> classes, the posteriors in the log domain, Gumbel
/// sampling, the variational loss with the KL prior, and the RePaint-style inference of <c>perform_simple_inference</c>
/// with the first codebook held to the AR's codes.
/// </summary>
internal sealed class Mars5Diffusion
{
    /// <summary>MIN_LOG_ARG: the one-hot encodings' floor.</summary>
    public const double MinLogArg = 1e-7;

    private readonly double[] _logAlpha, _log1MinAlpha, _logCumprodAlpha, _log1MinCumprodAlpha;

    public Mars5Diffusion(int classes, int timesteps, double s = 0.008)
    {
        Classes = classes;
        Timesteps = timesteps;
        // cosine_beta_schedule: ᾱ from cos², α_t = ᾱ_t/ᾱ_{t−1} clamped to [0.001, 1], then its square root; logs in float64,
        // stored in float32.
        var cumprod = new double[timesteps + 1];
        for (int i = 0; i <= timesteps; i++)
        {
            double x = (double)i / timesteps;
            double c = Math.Cos((x + s) / (1 + s) * Math.PI * 0.5);
            cumprod[i] = c * c;
        }
        for (int i = timesteps; i >= 0; i--) cumprod[i] /= cumprod[0];
        _logAlpha = new double[timesteps];
        _log1MinAlpha = new double[timesteps];
        _logCumprodAlpha = new double[timesteps];
        _log1MinCumprodAlpha = new double[timesteps];
        double running = 0;
        var la = new double[timesteps];
        for (int t = 0; t < timesteps; t++)
        {
            double alpha = Math.Sqrt(Math.Min(1.0, Math.Max(0.001, cumprod[t + 1] / cumprod[t])));
            la[t] = Math.Log(alpha);
        }
        for (int t = 0; t < timesteps; t++)
        {
            running += la[t];
            _logAlpha[t] = (float)la[t];
            _log1MinAlpha[t] = (float)Log1MinA(la[t]);
            _logCumprodAlpha[t] = (float)running;
            _log1MinCumprodAlpha[t] = (float)Log1MinA(running);
        }
    }

    public int Classes { get; }
    public int Timesteps { get; }

    private static double Log1MinA(double a) => Math.Log(Math.Max(1 - Math.Exp(a), 1e-30));

    private static double LogAddExp(double a, double b)
    {
        double m = Math.Max(a, b);
        return m + Math.Log(Math.Exp(a - m) + Math.Exp(b - m));
    }

    /// <summary><c>index_to_log_onehot</c>: 0 at the code, log(1e-7) elsewhere.</summary>
    public static double LogOneHot(int code, int k) => code == k ? 0.0 : Math.Log(MinLogArg);

    // q_pred: log(ᾱ_t · x₀ + (1 − ᾱ_t)/K) over one row of K log-probabilities.
    private void QPred(double[] row, int offset, int t, double[] output)
    {
        double a = _logCumprodAlpha[t], b = _log1MinCumprodAlpha[t] - Math.Log(Classes);
        for (int k = 0; k < Classes; k++) output[k] = LogAddExp(row[offset + k] + a, b);
    }

    // q_pred_one_timestep: log(α_t · x_t + (1 − α_t)/K).
    private void QPredOne(double[] row, int offset, int t, double[] output)
    {
        double a = _logAlpha[t], b = _log1MinAlpha[t] - Math.Log(Classes);
        for (int k = 0; k < Classes; k++) output[k] = LogAddExp(row[offset + k] + a, b);
    }

    /// <summary>
    /// <c>q_posterior(log_x0, log_xt, t)</c> for rows <c>[n, K]</c> flattened: <c>q(x_{t−1} | x₀)</c> (x₀ itself at t = 0)
    /// times <c>q(x_t | x_{t−1})</c>, normalized.
    /// </summary>
    public double[] QPosterior(double[] logX0, double[] logXt, int t)
    {
        int n = logX0.Length / Classes;
        var output = new double[logX0.Length];
        var ev = new double[Classes];
        var step = new double[Classes];
        for (int r = 0; r < n; r++)
        {
            int o = r * Classes;
            if (t == 0) Array.Copy(logX0, o, ev, 0, Classes);
            else QPred(logX0, o, t - 1, ev);
            QPredOne(logXt, o, t, step);
            double max = double.NegativeInfinity;
            for (int k = 0; k < Classes; k++) max = Math.Max(max, ev[k] += step[k]);
            double sum = 0;
            for (int k = 0; k < Classes; k++) sum += Math.Exp(ev[k] - max);
            double norm = max + Math.Log(sum);
            for (int k = 0; k < Classes; k++) output[o + k] = ev[k] - norm;
        }
        return output;
    }

    /// <summary><c>q_pred(log_x0, t)</c> for flattened rows.</summary>
    public double[] QPredRows(double[] logX0, int t)
    {
        var output = new double[logX0.Length];
        var row = new double[Classes];
        for (int r = 0; r < logX0.Length / Classes; r++)
        {
            QPred(logX0, r * Classes, t, row);
            Array.Copy(row, 0, output, r * Classes, Classes);
        }
        return output;
    }

    /// <summary><c>log_sample_categorical</c>: argmax of the log-probabilities plus Gumbel noise from the uniform draws
    /// <paramref name="uniform"/> (one per entry; <c>torch.rand_like</c>).</summary>
    public int[] SampleCategorical(double[] logProbabilities, double[] uniform)
    {
        int n = logProbabilities.Length / Classes;
        var sample = new int[n];
        for (int r = 0; r < n; r++)
        {
            double best = double.NegativeInfinity;
            int arg = 0;
            for (int k = 0; k < Classes; k++)
            {
                int i = r * Classes + k;
                double gumbel = -Math.Log(Math.Max(-Math.Log(Math.Max(uniform[i], MinLogArg)), MinLogArg));
                double v = gumbel + logProbabilities[i];
                if (v > best) { best = v; arg = k; }
            }
            sample[r] = arg;
        }
        return sample;
    }

    /// <summary>The scaled forward step (<c>q_pred_one_timestep_scaled</c>, "sequentially progressive diffusion"): each
    /// position's α raised by a sigmoid ramp along the sequence at RePaint offset <paramref name="c"/>.</summary>
    public double[] QPredOneScaled(double[] logXt, int t, double c, int jumpLength, int length, int codebooks)
    {
        var output = new double[logXt.Length];
        double aa = length * (c / jumpLength);
        for (int p = 0; p < length; p++)
        {
            double sig = 1 / (1 + Math.Exp(-(p - aa + 20) / 8));
            double la = Math.Min(Math.Log(1 / sig) + _logAlpha[t], 0);
            double lb = Math.Log(sig) + _log1MinAlpha[t] - Math.Log(Classes);
            for (int q = 0; q < codebooks; q++)
                for (int k = 0; k < Classes; k++)
                {
                    int i = (p * codebooks + q) * Classes + k;
                    output[i] = LogAddExp(logXt[i] + la, lb);
                }
        }
        return output;
    }

    /// <summary><c>q_pred_one_timestep</c> for flattened rows.</summary>
    public double[] QPredOneRows(double[] logXt, int t)
    {
        var output = new double[logXt.Length];
        var row = new double[Classes];
        for (int r = 0; r < logXt.Length / Classes; r++)
        {
            QPredOne(logXt, r * Classes, t, row);
            Array.Copy(row, 0, output, r * Classes, Classes);
        }
        return output;
    }

    /// <summary>The log one-hot rows of codes.</summary>
    public double[] OneHotRows(int[] codes)
    {
        var rows = new double[codes.Length * Classes];
        for (int r = 0; r < codes.Length; r++)
        {
            if (codes[r] < 0 || codes[r] >= Classes) throw new ArgumentOutOfRangeException(nameof(codes), $"Classes lie in [0, {Classes}).");
            for (int k = 0; k < Classes; k++) rows[r * Classes + k] = LogOneHot(codes[r], k);
        }
        return rows;
    }

    /// <summary><c>get_schedule(T, jump_len, jump_n_sample)</c>: RePaint's step sequence ending in −1.</summary>
    public static List<int> Schedule(int steps, int jumpLength, int jumpSamples)
    {
        var jumps = new Dictionary<int, int>();
        for (int j = 0; j < steps - jumpLength; j += jumpLength) jumps[j] = jumpSamples - 1;
        int t = steps;
        var ts = new List<int>();
        while (t >= 1)
        {
            t--;
            ts.Add(t);
            if (jumps.TryGetValue(t, out int left) && left > 0)
            {
                jumps[t] = left - 1;
                for (int i = 0; i < jumpLength; i++) ts.Add(++t);
            }
        }
        ts.Add(-1);
        return ts;
    }

    /// <summary>
    /// <c>kl_prior</c>: <c>KL(q(x_T | x₀) ‖ uniform)</c> summed over the rows of a log one-hot x₀.
    /// </summary>
    public double KlPrior(double[] logX0)
    {
        var q = QPredRows(logX0, Timesteps - 1);
        double logUniform = -Math.Log(Classes), kl = 0;
        for (int i = 0; i < q.Length; i++) kl += Math.Exp(q[i]) * (q[i] - logUniform);
        return kl;
    }

    /// <summary>
    /// <c>compute_Lt</c> with its default <c>include_kl_prior=True</c> on the tape: the KL from the true posterior
    /// <c>q(x_{t−1} | x_t, x₀)</c> to the model's <c>q(x_{t−1} | x_t, x̂₀)</c>, summed over the rows, plus the KL prior (the
    /// default replaces the t = 0 decoder term with that sum). <paramref name="logX0Predicted"/> <c>[n, K]</c> is the
    /// model's log-softmax.
    /// </summary>
    public Tensor<T> Loss<T>(IEngine engine, double[] logX0, double[] logXt, Tensor<T> logX0Predicted, int t, bool includeKlPrior = true)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int n = logX0.Length / Classes;
        var truth = QPosterior(logX0, logXt, t);
        // The model's posterior: log_softmax(q_pred(x̂₀, t−1) + q_pred_one_timestep(x_t, t)), x̂₀ itself at t = 0.
        Tensor<T> ev;
        if (t == 0) ev = logX0Predicted;
        else
        {
            // log(e^{p + a} + e^{b}) = b + softplus(p + a − b)
            double a = _logCumprodAlpha[t - 1], b = _log1MinCumprodAlpha[t - 1] - Math.Log(Classes);
            ev = engine.TensorAddScalar(engine.Softplus(engine.TensorAddScalar(logX0Predicted, ops.FromDouble(a - b))), ops.FromDouble(b));
        }
        var step = QPredOneRows(logXt, t);
        var stepTensor = new Tensor<T>(new[] { n, Classes }, new Vector<T>(step.Select(v => ops.FromDouble(v)).ToArray()));
        var model = engine.TensorLogSoftmax(engine.TensorAdd(ev, stepTensor), axis: 1);
        // KL(true ‖ model) = Σ e^true (true − model)
        var weights = new Tensor<T>(new[] { n, Classes }, new Vector<T>(truth.Select(v => ops.FromDouble(Math.Exp(v))).ToArray()));
        double constant = 0;
        for (int i = 0; i < truth.Length; i++) constant += Math.Exp(truth[i]) * truth[i];
        var kl = engine.TensorAddScalar(engine.TensorNegate(engine.ReduceSum(engine.TensorMultiply(weights, model), null)), ops.FromDouble(constant));
        if (!includeKlPrior)
        {
            if (t != 0) return kl;
            // L₀: −Σ x₀ log p(x₀ | x₁)
            var target = new Tensor<T>(new[] { n, Classes }, new Vector<T>(logX0.Select(v => ops.FromDouble(Math.Exp(v))).ToArray()));
            return engine.TensorNegate(engine.ReduceSum(engine.TensorMultiply(target, model), null));
        }
        return engine.TensorAddScalar(kl, ops.FromDouble(KlPrior(logX0)));
    }

    /// <summary>
    /// <c>perform_simple_inference</c> with <c>retain_quant0</c>: the codes <c>[L, codebooks]</c> whose first codebook is
    /// <paramref name="quant0"/>, from uniform random codes through the RePaint schedule. <paramref name="predict"/> gives the
    /// model's x₀ logits <c>[L, codebooks, K]</c> flattened for codes and a step (conditioned, or with the conditioning
    /// dropped); a deep clone prefixes <paramref name="prompt"/>'s codes, known throughout, and crops them off.
    /// <paramref name="uniform"/> draws torch.rand_like's values and <paramref name="randomCode"/> torch.randint's.
    /// </summary>
    public int[,] Infer(int[] quant0, int codebooks, int[,]? prompt, Func<int[,], int, bool, double[]> predict, Mars5DiffusionSampling dsh,
        Func<int, double[]> uniform, Func<int> randomCode)
    {
        int length = quant0.Length;
        int offset = dsh.DeepClone && prompt is not null ? prompt.GetLength(0) : 0;
        int total = offset + length;
        var x = new int[total, codebooks];
        var known = new int[total, codebooks];
        var mask = new bool[total, codebooks];
        var q0 = new int[total];
        // torch.randint over the generated part, then its first codebook set to the AR's codes.
        var draws = new int[length, codebooks];
        for (int i = 0; i < length; i++)
            for (int q = 0; q < codebooks; q++) draws[i, q] = randomCode();
        for (int i = 0; i < total; i++)
            for (int q = 0; q < codebooks; q++)
            {
                if (i < offset)
                {
                    x[i, q] = known[i, q] = prompt![i, q];
                    mask[i, q] = true;
                }
                else
                {
                    x[i, q] = q == 0 ? quant0[i - offset] : draws[i - offset, q];
                    if (q == 0) { known[i, q] = quant0[i - offset]; mask[i, q] = true; }
                }
                if (q == 0) q0[i] = x[i, 0];
            }
        var times = Schedule(Timesteps, dsh.JumpLength, dsh.JumpSamples);
        double c = 0;
        for (int s = 0; s + 1 < times.Count; s++)
        {
            int tLast = times[s], tCur = times[s + 1];
            if (tCur < tLast)
            {
                if (c > dsh.JumpSamples) c = 0;
                c += 1.0 / dsh.JumpLength;
                x = ReverseStep(x, known, mask, tLast, codebooks, predict, dsh, uniform);
            }
            else
            {
                var logX = OneHotRows(Flatten(x));
                var probabilities = dsh.ScaledInference ? QPredOneScaled(logX, tLast, c, dsh.JumpLength, total, codebooks) : QPredOneRows(logX, tLast);
                x = Unflatten(SampleCategorical(probabilities, uniform(probabilities.Length)), total, codebooks);
            }
            if (dsh.Q0OverrideSteps < tLast)
                for (int i = 0; i < total; i++) x[i, 0] = q0[i];
        }
        var result = new int[length, codebooks];
        for (int i = 0; i < length; i++)
            for (int q = 0; q < codebooks; q++) result[i, q] = x[offset + i, q];
        return result;
    }

    // reverse_diffusion: x_{t−1} from the model's x̂₀ (guided, tempered), the known codes noised to step t (themselves at
    // t = 0) where the mask is set.
    private int[,] ReverseStep(int[,] x, int[,] known, bool[,] mask, int t, int codebooks, Func<int[,], int, bool, double[]> predict,
        Mars5DiffusionSampling dsh, Func<int, double[]> uniform)
    {
        int total = x.GetLength(0), n = total * codebooks;
        var logits = predict(x, t, false);
        if (dsh.GuidanceWeight != 1)
        {
            var unconditional = predict(x, t, true);
            for (int i = 0; i < logits.Length; i++) logits[i] = dsh.GuidanceWeight * logits[i] + (1 - dsh.GuidanceWeight) * unconditional[i];
        }
        var logX0 = new double[logits.Length];
        for (int r = 0; r < n; r++)
        {
            int o = r * Classes;
            double max = double.NegativeInfinity;
            for (int k = 0; k < Classes; k++) max = Math.Max(max, logits[o + k] /= dsh.X0Temperature);
            double sum = 0;
            for (int k = 0; k < Classes; k++) sum += Math.Exp(logits[o + k] - max);
            double norm = max + Math.Log(sum);
            for (int k = 0; k < Classes; k++) logX0[o + k] = logits[o + k] - norm;
        }
        var posterior = QPosterior(logX0, OneHotRows(Flatten(x)), t);
        int[] unknown;
        if (t == 0 && dsh.LastGreedy)
        {
            unknown = new int[n];
            for (int r = 0; r < n; r++)
            {
                int arg = 0;
                for (int k = 1; k < Classes; k++) if (posterior[r * Classes + k] > posterior[r * Classes + arg]) arg = k;
                unknown[r] = arg;
            }
        }
        else unknown = SampleCategorical(posterior, uniform(posterior.Length));
        int[] knownStep;
        if (t == 0) knownStep = Flatten(known);
        else
        {
            var q = QPredRows(OneHotRows(Flatten(known)), t);
            knownStep = SampleCategorical(q, uniform(q.Length));
        }
        var next = new int[total, codebooks];
        for (int i = 0; i < total; i++)
            for (int qb = 0; qb < codebooks; qb++)
            {
                int r = i * codebooks + qb;
                next[i, qb] = mask[i, qb] ? knownStep[r] : unknown[r];
            }
        return next;
    }

    private static int[] Flatten(int[,] x)
    {
        int rows = x.GetLength(0), cols = x.GetLength(1);
        var flat = new int[rows * cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) flat[i * cols + j] = x[i, j];
        return flat;
    }

    private static int[,] Unflatten(int[] flat, int rows, int cols)
    {
        var x = new int[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) x[i, j] = flat[i * cols + j];
        return x;
    }
}

/// <summary>MARS5's AR sampling settings (<c>InferenceConfig</c>'s AR fields).</summary>
internal sealed record Mars5ArSampling(double Temperature = 0.7, int TopK = 200, double TopP = 0.2, double TypicalP = 1.0, double FrequencyPenalty = 3,
    double PresencePenalty = 0.4, int PenaltyWindow = 80, double EosPenaltyDecay = 0.5, double EosPenaltyFactor = 1, int MaxLength = 2000);

/// <summary>MARS5's logit processors (<c>mars5/samplers.py</c>) and the AR decoding loop (<c>ar_generate</c>, beam width 1).</summary>
internal static class Mars5Sampler
{
    /// <summary><c>freq_rep_penalty</c>: each logit less α_f times its count among the last tokens and α_p if present.</summary>
    public static void FrequencyPenalty(double[] logits, IReadOnlyList<int> previous, double frequency, double presence, int window)
    {
        var counts = new Dictionary<int, int>();
        foreach (int id in previous.Skip(Math.Max(0, previous.Count - window))) counts[id] = counts.TryGetValue(id, out int c) ? c + 1 : 1;
        foreach (var kv in counts) logits[kv.Key] -= kv.Value * frequency + presence;
    }

    /// <summary><c>early_eos_penalty</c>: the end token's logit less factor·max(est − n, 1)^decay while n ≤ est.</summary>
    public static void EosPenalty(double[] logits, int generated, int estimated, double decay, double factor, int eos)
    {
        if (generated > estimated) return;
        logits[eos] -= factor * Math.Pow(Math.Max(estimated - generated, 1), decay);
    }

    /// <summary><c>top_k_top_p_filtering</c> (min_tokens 1).</summary>
    public static void TopKTopP(double[] logits, int topK, double topP)
    {
        if (topK > 0)
        {
            int k = Math.Min(Math.Max(topK, 1), logits.Length);
            double kth = logits.OrderByDescending(v => v).ElementAt(k - 1);
            for (int i = 0; i < logits.Length; i++) if (logits[i] < kth) logits[i] = double.NegativeInfinity;
        }
        if (topP < 1.0)
        {
            // A stable descending sort, as torch.sort ties keep their order on CPU.
            var order = Enumerable.Range(0, logits.Length).OrderByDescending(i => logits[i]).ToArray();
            double max = logits[order[0]], sum = 0;
            foreach (int i in order) sum += Math.Exp(logits[i] - max);
            double cumulative = 0;
            var remove = new bool[order.Length];
            for (int r = 0; r < order.Length; r++)
            {
                cumulative += Math.Exp(logits[order[r]] - max) / sum;
                remove[r] = cumulative > topP;
            }
            // Shift right: the first token above the threshold stays.
            for (int r = order.Length - 1; r >= 1; r--) remove[r] = remove[r - 1];
            remove[0] = false;
            for (int r = 0; r < order.Length; r++) if (remove[r]) logits[order[r]] = double.NegativeInfinity;
        }
    }

    /// <summary><c>apply_typical_p</c> (Meister et al. 2022): keep the tokens whose surprisal is closest to the entropy up
    /// to the mass.</summary>
    public static void TypicalP(double[] logits, double mass)
    {
        if (mass > 0.999) return;
        double max = logits.Max(), sum = logits.Sum(v => Math.Exp(v - max));
        var normalized = logits.Select(v => v - max - Math.Log(sum)).ToArray();
        double entropy = 0;
        foreach (var v in normalized) if (!double.IsNegativeInfinity(v)) entropy -= v * Math.Exp(v);
        var shifted = normalized.Select(v => Math.Abs(-v - entropy)).ToArray();
        var order = Enumerable.Range(0, logits.Length).OrderBy(i => shifted[i]).ToArray();
        double sMax = order.Max(i => logits[i]), sSum = order.Sum(i => Math.Exp(logits[i] - sMax));
        double cumulative = 0;
        int last = 0;
        foreach (int i in order)
        {
            cumulative += Math.Exp(logits[i] - sMax) / sSum;
            if (cumulative < mass) last++;
        }
        double threshold = shifted[order[Math.Min(last, order.Length - 1)]];
        for (int i = 0; i < logits.Length; i++) if (shifted[i] > threshold) logits[i] = double.NegativeInfinity;
    }
}

/// <summary>
/// The release's silence trimming (<c>mars5/trim.py</c>, a torch port of <c>librosa.effects.trim</c>): frames of 2048
/// samples every 512 (reflect-padded by half a frame), their mean power in dB below the loudest frame's, the audio kept
/// from the first to past the last frame within <c>top_db</c>.
/// </summary>
internal static class Mars5Trim
{
    public static (double[] Audio, int Start, int End) Trim(double[] y, double topDb = 60, int frameLength = 2048, int hopLength = 512)
    {
        int pad = frameLength / 2, n = y.Length;
        if (n == 0) return (y, 0, 0);
        var padded = new double[n + 2 * pad];
        for (int i = 0; i < padded.Length; i++)
        {
            int j = i - pad;
            // numpy/torch reflect: mirror without repeating the edge sample.
            while (j < 0 || j >= n) j = j < 0 ? -j : 2 * (n - 1) - j;
            padded[i] = y[j];
        }
        int frames = 1 + (padded.Length - frameLength) / hopLength;
        var power = new double[frames];
        for (int f = 0; f < frames; f++)
        {
            double s = 0;
            for (int k = 0; k < frameLength; k++) s += padded[f * hopLength + k] * padded[f * hopLength + k];
            double rms = Math.Sqrt(s / frameLength);
            power[f] = rms * rms;
        }
        double reference = Math.Max(1e-10, power.Max());
        int first = -1, lastFrame = -1;
        for (int f = 0; f < frames; f++)
        {
            double db = 10 * Math.Log10(Math.Max(1e-10, power[f])) - 10 * Math.Log10(reference);
            if (db > -topDb)
            {
                if (first < 0) first = f;
                lastFrame = f;
            }
        }
        if (first < 0) return (Array.Empty<double>(), 0, 0);
        int start = first * hopLength, end = Math.Min(n, (lastFrame + 1) * hopLength);
        return (y.Skip(start).Take(end - start).ToArray(), start, end);
    }
}
