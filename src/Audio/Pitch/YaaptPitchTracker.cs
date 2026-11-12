namespace AiDotNet.Audio.Pitch;

/// <summary>The parameters of <see cref="YaaptPitchTracker"/> (AMFM_decompy's <c>yaapt</c> keyword arguments).</summary>
internal sealed record YaaptParameters
{
    public double FrameLength { get; init; } = 35.0;
    public double TdaFrameLength { get; init; } = 35.0;
    public double FrameSpace { get; init; } = 10.0;
    public double F0Min { get; init; } = 60.0;
    public double F0Max { get; init; } = 400.0;
    public int FftLength { get; init; } = 8192;
    public int BandpassOrder { get; init; } = 150;
    public double BandpassLow { get; init; } = 50.0;
    public double BandpassHigh { get; init; } = 1500.0;
    public double NlferThreshold1 { get; init; } = 0.75;
    public double NlferThreshold2 { get; init; } = 0.1;
    public int ShcHarmonics { get; init; } = 3;
    public double ShcWindow { get; init; } = 40.0;
    public int ShcMaxPeaks { get; init; } = 4;
    public double ShcPeakWidth { get; init; } = 50.0;
    public double ShcThreshold1 { get; init; } = 5.0;
    public double ShcThreshold2 { get; init; } = 1.25;
    public double Dp5K1 { get; init; } = 11.0;
    public double NccfThreshold1 { get; init; } = 0.3;
    public double NccfThreshold2 { get; init; } = 0.9;
    public int NccfMaxCandidates { get; init; } = 3;
    public int NccfPeakWidth { get; init; } = 5;
    public double MeritBoost { get; init; } = 0.20;
    public double MeritPivot { get; init; } = 0.99;
    public int MedianOrder { get; init; } = 7;
    public double DpW1 { get; init; } = 0.15;
    public double DpW2 { get; init; } = 0.5;
    public double DpW3 { get; init; } = 0.1;
    public double DpW4 { get; init; } = 0.9;
    public double SpecPitchMinStd { get; init; } = 0.05;

    /// <summary>SpiritLM's (and speech-resynthesis') call: 20 ms frames every 5 ms at 16 kHz, a 25 ms time-domain frame
    /// and an NCCF peak threshold of 0.25.</summary>
    public static YaaptParameters SpiritLm => new() { FrameLength = 20.0, FrameSpace = 5.0, NccfThreshold1 = 0.25, TdaFrameLength = 25.0 };
}

/// <summary>
/// YAAPT, Yet Another Algorithm for Pitch Tracking (Zahorian and Hu 2008), as AMFM_decompy's <c>pYAAPT.yaapt</c> computes
/// it: the signal and its square band-passed (a 151-tap Hamming FIR, 50–1500 Hz); the normalized low-frequency energy
/// ratio for voicing; a spectral pitch track from the squared signal's spectral harmonics correlation smoothed by dynamic
/// programming; NCCF candidates from both signals around it; refined candidates; and the final dynamic program.
/// </summary>
/// <remarks>
/// <para>SpiritLM offers it as its paper's F0 extractor for the pitch VQ-VAE (FCPE is the release's default for the
/// language model's tokens). <see cref="SpiritLmF0"/> is the release's <c>pYAAPTF0Extractor</c>: the waveform cut to whole
/// hops, peak-normalized to 0.95, padded by half a frame, the tracked F0 linearly interpolated over unvoiced frames (the
/// ends held), with the energy-based voicing.</para>
/// <para>The port keeps the reference's arithmetic where it decides anything: each time-domain frame has its mean removed
/// in place in a buffer the overlapping frames share, as numpy's strided views do; peak picking returns no pitch when it
/// finds exactly the maximum number of peaks; ties go to the first index in arg-maxima and to the last in the dynamic
/// program's arg-minima; medians pad with zeros (scipy's <c>medfilt</c>); and the spectral track is interpolated by scipy's
/// PCHIP.</para>
/// </remarks>
internal static class YaaptPitchTracker
{
    /// <summary>The tracked pitch per frame (0 when unvoiced) and the energy-based voicing.</summary>
    internal sealed record Track(double[] Pitch, bool[] Voiced, double[] Energy);

    /// <summary>Runs YAAPT on <paramref name="data"/> at <paramref name="sampleRate"/> Hz.</summary>
    public static Track Run(IReadOnlyList<double> data, double sampleRate, YaaptParameters p)
    {
        var b = Firwin(p.BandpassOrder + 1, p.BandpassLow / (sampleRate / 2), p.BandpassHigh / (sampleRate / 2));
        var signal = data.ToArray();
        var squared = signal.Select(v => v * v).ToArray();
        var filtered = Fir(b, signal);
        var filteredSquared = Fir(b, squared);
        int frameSize = (int)Math.Truncate(p.FrameLength * sampleRate / 1000);
        int frameJump = (int)Math.Truncate(p.FrameSpace * sampleRate / 1000);
        if (frameSize <= 15) throw new ArgumentException($"Frame length value {frameSize} is too short.");
        if (frameSize >= 2048) throw new ArgumentException($"Frame length value {frameSize} exceeds the limit.");
        int size = signal.Length;

        var (energy, voiced) = Nlfer(filtered, size, sampleRate, frameSize, frameJump, p);
        int frames = energy.Length;
        var (specPitch, pitchStd) = SpecTrack(filteredSquared, size, sampleRate, frameSize, frameJump, frames, voiced, p);
        var (tp1, tm1) = TimeTrack(filtered, size, sampleRate, specPitch, pitchStd, frameJump, p);
        var (tp2, tm2) = TimeTrack(filteredSquared, size, sampleRate, specPitch, pitchStd, frameJump, p);
        int candidates = p.NccfMaxCandidates;
        double[,] Pad(double[,] m)
        {
            if (m.GetLength(1) >= frames) return m;
            var padded = new double[candidates, frames];
            for (int r = 0; r < candidates; r++)
                for (int c = 0; c < m.GetLength(1); c++) padded[r, c] = m[r, c];
            return padded;
        }
        tp1 = Pad(tp1); tm1 = Pad(tm1); tp2 = Pad(tp2); tm2 = Pad(tm2);
        var (refPitch, refMerit) = Refine(tp1, tm1, tp2, tm2, specPitch, energy, voiced, frames, p);
        var final = Dynamic(refPitch, refMerit, energy, frames, p);
        return new Track(final, voiced, energy);
    }

    /// <summary>
    /// The release's <c>pYAAPTF0Extractor</c> on 16 kHz audio with an 80-sample hop: the F0 per frame, unvoiced frames
    /// interpolated, and the voicing (1 or 0).
    /// </summary>
    public static (double[] F0, double[] Voiced) SpiritLmF0(IReadOnlyList<double> audio, int sampleRate = 16000, int hop = 80)
    {
        var track = SpiritLmTrack(audio, sampleRate, hop);
        return (Interpolate(track.Pitch), track.Voiced.Select(v => v ? 1.0 : 0.0).ToArray());
    }

    /// <summary>YAAPT's track of the audio as the release's <c>get_pitch</c> prepares it (whole hops, peak 0.95, half a
    /// frame of zeros on each side), before interpolation.</summary>
    public static Track SpiritLmTrack(IReadOnlyList<double> audio, int sampleRate = 16000, int hop = 80)
    {
        int count = audio.Count / hop * hop;
        if (count == 0) throw new ArgumentException("The audio is shorter than one hop.", nameof(audio));
        double peak = 0;
        for (int i = 0; i < count; i++) peak = Math.Max(peak, Math.Abs(audio[i]));
        if (peak < 2.2250738585072014e-308) peak = 1.0;      // librosa's normalize leaves silence unscaled.
        const double frameLength = 20.0;
        int pad = hop == 80 ? (int)(frameLength / 1000 * sampleRate) / 2 : 0;
        var padded = new double[count + 2 * pad];
        for (int i = 0; i < count; i++) padded[pad + i] = audio[i] / peak * 0.95;
        var p = YaaptParameters.SpiritLm with { FrameSpace = (double)hop / sampleRate * 1000 };
        return Run(padded, sampleRate, p);
    }

    /// <summary>The release's <c>interpolate_f0</c>: linear between nonzero frames, the ends held.</summary>
    public static double[] Interpolate(double[] f0)
    {
        var result = (double[])f0.Clone();
        var voiced = Enumerable.Range(0, f0.Length).Where(i => f0[i] != 0).ToArray();
        if (voiced.Length <= 1) return result;
        for (int t = voiced[0]; t <= voiced[^1]; t++)
        {
            // scipy interp1d: hi = searchsorted(x, t) clipped to [1, n − 1], lo = hi − 1, slope · (t − x_lo) + y_lo (a
            // voiced point past the first too).
            int hi = Array.BinarySearch(voiced, t);
            if (hi < 0) hi = ~hi;
            hi = Math.Min(Math.Max(hi, 1), voiced.Length - 1);
            int x0 = voiced[hi - 1], x1 = voiced[hi];
            double slope = (f0[x1] - f0[x0]) / (x1 - x0);
            result[t] = slope * (t - x0) + f0[x0];
        }
        for (int t = 0; t < voiced[0]; t++) result[t] = result[voiced[0]];
        for (int t = voiced[^1] + 1; t < f0.Length; t++) result[t] = result[voiced[^1]];
        return result;
    }

    // ------------------------------------------------------------------ filters and windows

    /// <summary>scipy's <c>firwin(taps, [f1, f2], pass_zero=False)</c>: a Hamming-windowed band-pass, scaled to unit gain at
    /// the band's centre.</summary>
    internal static double[] Firwin(int taps, double f1, double f2)
    {
        double alpha = 0.5 * (taps - 1);
        var h = new double[taps];
        for (int n = 0; n < taps; n++)
        {
            double m = n - alpha;
            h[n] = f2 * Sinc(f2 * m) - f1 * Sinc(f1 * m);
            h[n] *= 0.54 - 0.46 * Math.Cos(2 * Math.PI * n / (taps - 1));
        }
        double centre = 0.5 * (f1 + f2), s = 0;
        for (int n = 0; n < taps; n++) s += h[n] * Math.Cos(Math.PI * (n - alpha) * centre);
        for (int n = 0; n < taps; n++) h[n] /= s;
        return h;
    }

    private static double Sinc(double x) => x == 0 ? 1.0 : Math.Sin(Math.PI * x) / (Math.PI * x);

    /// <summary><c>lfilter(b, 1, x)</c>: the causal FIR filter from rest.</summary>
    internal static double[] Fir(double[] b, double[] x)
    {
        var y = new double[x.Length];
        for (int n = 0; n < x.Length; n++)
        {
            double sum = 0;
            for (int k = Math.Min(b.Length - 1, n); k >= 0; k--) sum += b[k] * x[n - k];
            y[n] = sum;
        }
        return y;
    }

    private static double[] SymmetricHann(int m)
    {
        var w = new double[m];
        for (int n = 0; n < m; n++) w[n] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / (m - 1));
        return w;
    }

    private static double[] Kaiser(int m, double beta)
    {
        var w = new double[m];
        double alpha = (m - 1) / 2.0, denominator = BesselI0(beta);
        for (int n = 0; n < m; n++)
        {
            double r = (n - alpha) / alpha;
            w[n] = BesselI0(beta * Math.Sqrt(Math.Max(0, 1 - r * r))) / denominator;
        }
        return w;
    }

    private static double BesselI0(double x)
    {
        double sum = 1, term = 1, half = x / 2;
        for (int k = 1; k < 500; k++)
        {
            term *= half * half / ((double)k * k);
            sum += term;
            if (term < sum * 1e-17) break;
        }
        return sum;
    }

    /// <summary>scipy's <c>medfilt</c>: the median of each zero-padded window.</summary>
    internal static double[] MedianFilter(double[] x, int kernel)
    {
        int half = kernel / 2;
        var result = new double[x.Length];
        var window = new double[kernel];
        for (int i = 0; i < x.Length; i++)
        {
            for (int k = 0; k < kernel; k++)
            {
                int j = i - half + k;
                window[k] = j >= 0 && j < x.Length ? x[j] : 0.0;
            }
            Array.Sort(window);
            result[i] = window[half];
        }
        return result;
    }

    /// <summary>scipy's <c>PchipInterpolator(x, y)</c> evaluated at <paramref name="at"/> (extrapolating with the end
    /// pieces).</summary>
    internal static double[] Pchip(double[] x, double[] y, IReadOnlyList<double> at)
    {
        int n = x.Length;
        if (n < 2) throw new ArgumentException("PCHIP needs two points.");
        var h = new double[n - 1];
        var m = new double[n - 1];
        for (int i = 0; i < n - 1; i++) { h[i] = x[i + 1] - x[i]; m[i] = (y[i + 1] - y[i]) / h[i]; }
        var d = new double[n];
        if (n == 2) { d[0] = d[1] = m[0]; }
        else
        {
            for (int k = 1; k < n - 1; k++)
            {
                bool flat = Math.Sign(m[k]) != Math.Sign(m[k - 1]) || m[k] == 0 || m[k - 1] == 0;
                if (flat) { d[k] = 0; continue; }
                double w1 = 2 * h[k] + h[k - 1], w2 = h[k] + 2 * h[k - 1];
                double whmean = (w1 / m[k - 1] + w2 / m[k]) / (w1 + w2);
                d[k] = 1.0 / whmean;
            }
            d[0] = Edge(h[0], h[1], m[0], m[1]);
            d[n - 1] = Edge(h[n - 2], h[n - 3], m[n - 2], m[n - 3]);
        }
        var result = new double[at.Count];
        for (int q = 0; q < at.Count; q++)
        {
            double v = at[q];
            int i;
            if (v < x[0]) i = 0;
            else if (v >= x[n - 1]) i = n - 2;
            else
            {
                i = Array.BinarySearch(x, v);
                i = i >= 0 ? Math.Min(i, n - 2) : ~i - 1;
            }
            double dx = h[i], slope = m[i];
            double t2 = (d[i] + d[i + 1] - 2 * slope) / dx;
            double c0 = t2 / dx, c1 = (slope - d[i]) / dx - t2, c2 = d[i], c3 = y[i];
            double s = v - x[i];
            // scipy's evaluate_poly1: c3 + c2·s + c1·s² + c0·s³, accumulated in that order.
            double result0 = c3, z = s;
            result0 += c2 * z; z *= s;
            result0 += c1 * z; z *= s;
            result0 += c0 * z;
            result[q] = result0;
        }
        return result;
    }

    private static double Edge(double h0, double h1, double m0, double m1)
    {
        double d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
        if (Math.Sign(d) != Math.Sign(m0)) return 0;
        if (Math.Sign(m0) != Math.Sign(m1) && Math.Abs(d) > 3 * Math.Abs(m0)) return 3 * m0;
        return d;
    }

    private static double[] Magnitude(double[] frame, int nfft)
    {
        var input = new Vector<double>(nfft);
        for (int i = 0; i < Math.Min(frame.Length, nfft); i++) input[i] = frame[i];
        var spectrum = new FastFourierTransform<double>().Forward(input);
        var result = new double[nfft / 2 + 1];
        for (int k = 0; k < result.Length; k++)
        {
            double re = spectrum[k].Real, im = spectrum[k].Imaginary;
            result[k] = Math.Sqrt(re * re + im * im);
        }
        return result;
    }

    private static double Mean(IEnumerable<double> values)
    {
        double sum = 0;
        int count = 0;
        foreach (var v in values) { sum += v; count++; }
        return sum / count;
    }

    private static double Std(double[] values)
    {
        double mean = Mean(values), sum = 0;
        foreach (var v in values) sum += (v - mean) * (v - mean);
        return Math.Sqrt(sum / values.Length);
    }

    // ------------------------------------------------------------------ the algorithm

    private static (double[] Energy, bool[] Voiced) Nlfer(double[] filtered, int size, double fs, int frameSize, int frameJump, YaaptParameters p)
    {
        int nfft = p.FftLength;
        int nMin = (int)Math.Round(p.F0Min * 2 / fs * nfft, MidpointRounding.ToEven);
        int nMax = (int)Math.Round(p.F0Max / fs * nfft, MidpointRounding.ToEven);
        var window = SymmetricHann(frameSize + 2).Skip(1).Take(frameSize).ToArray();
        int half = frameSize / 2, frames = 0;
        for (int s = half; s < size - half; s += frameJump) frames++;
        var energy = new double[frames];
        var frame = new double[frameSize];
        for (int f = 0; f < frames; f++)
        {
            for (int n = 0; n < frameSize; n++) frame[n] = filtered[f * frameJump + n] * window[n];
            var magnitude = Magnitude(frame, nfft);
            double sum = 0;
            for (int k = nMin - 1; k < nMax; k++) sum += magnitude[k];
            energy[f] = sum;
        }
        double mean = Mean(energy);
        var voiced = new bool[frames];
        for (int f = 0; f < frames; f++)
        {
            energy[f] /= mean;
            voiced[f] = energy[f] > p.NlferThreshold1;
        }
        return (energy, voiced);
    }

    private static (double[] Pitch, double[] Merit) Peaks(double[] data, double delta, int maxPeaks, YaaptParameters p)
    {
        const double epsilon = 1e-14;
        int width = (int)Math.Truncate(p.ShcPeakWidth / delta);
        if (width % 2 == 0) width++;
        int center = (int)Math.Ceiling(width / 2.0);
        int minLag = (int)Math.Truncate(p.F0Min / delta - center);
        int maxLag = (int)Math.Truncate(p.F0Max / delta + center);
        if (minLag < 1) minLag = 1;
        if (maxLag > data.Length - width) maxLag = data.Length - width;
        double Max(int from, int to) { double m = double.NegativeInfinity; for (int i = from; i < to; i++) m = Math.Max(m, data[i]); return m; }
        double maxData = Max(minLag, maxLag + 1);
        var d = data;
        if (maxData > epsilon) d = data.Select(v => v / maxData).ToArray();
        double avg = 0;
        for (int i = minLag; i <= maxLag; i++) avg += d[i];
        avg /= maxLag - minLag + 1;
        (double[], double[]) None() => (new double[maxPeaks], Enumerable.Repeat(1.0, maxPeaks).ToArray());
        if (avg > 1 / p.ShcThreshold1) return None();
        var pitch = new List<double>(new double[maxPeaks]);
        var merit = new List<double>(new double[maxPeaks]);
        int numPeaks = 0;
        for (int n = minLag + center + 1; n < maxLag - center + 1; n++)
        {
            if (!(d[n] > d[n - 1] && d[n] > d[n + 1] && d[n] > p.ShcThreshold2 * avg)) continue;
            int arg = 0;
            for (int k = 1; k < 2 * center + 1; k++) if (d[n - center + k] > d[n - center + arg]) arg = k;
            if (arg != center) continue;
            if (numPeaks >= maxPeaks) { pitch.Add(0); merit.Add(0); }
            pitch[numPeaks] = n * delta;
            merit[numPeaks] = d[n];
            numPeaks++;
        }
        if (merit.Max() / avg < p.ShcThreshold1) return None();
        var order = Enumerable.Range(0, merit.Count).OrderBy(i => -merit[i]).ToArray();
        var sortedPitch = order.Select(i => pitch[i]).ToArray();
        var sortedMerit = order.Select(i => merit[i]).ToArray();
        numPeaks = Math.Min(numPeaks, maxPeaks);
        var outPitch = new double[maxPeaks];
        var outMerit = new double[maxPeaks];
        for (int i = 0; i < numPeaks; i++) { outPitch[i] = sortedPitch[i]; outMerit[i] = sortedMerit[i]; }
        if (numPeaks > 0 && numPeaks < maxPeaks)
        {
            for (int i = numPeaks; i < maxPeaks; i++) { outPitch[i] = outPitch[0]; outMerit[i] = outMerit[0]; }
            return (outPitch, outMerit);
        }
        return None();
    }

    // The lowest-cost path through local costs [candidates, frames] and transitions [from, to, frames].
    private static int[] Path(double[,] local, double[,,] trans, int candidates, int frames)
    {
        var pred = new int[candidates, frames];
        var path = Enumerable.Repeat(1, frames).ToArray();
        var pSmall = new int[frames];
        var pcost = new double[candidates];
        for (int i = 0; i < candidates; i++) pcost[i] = local[i, 0];
        var ccost = new double[candidates];
        for (int t = 1; t < frames; t++)
        {
            for (int a = 0; a < candidates; a++)
            {
                // The last of equal minima, as argmin over the reversed row picks.
                int best = 0;
                double bestValue = double.PositiveInfinity;
                bool first = true;
                for (int bIndex = 0; bIndex < candidates; bIndex++)
                {
                    double v = pcost[bIndex] + trans[bIndex, a, t];
                    if (first || LessOrEqualForLast(v, bestValue)) { bestValue = v; best = bIndex; }
                    first = false;
                }
                pred[a, t] = best;
                ccost[a] = pcost[best] + trans[best, a, t] + local[a, t];
            }
            Array.Copy(ccost, pcost, candidates);
            int j = 0;
            double jValue = double.PositiveInfinity;
            for (int a = 0; a < candidates; a++)
                if (a == 0 || LessOrEqualForLast(pcost[a], jValue)) { jValue = pcost[a]; j = a; }
            pSmall[t] = j;
        }
        path[frames - 1] = pSmall[frames - 1];
        for (int t = frames - 2; t >= 0; t--) path[t] = pred[path[t + 1], t + 1];
        return path;
    }

    // numpy's argmin takes the first NaN, else the first minimum; over a reversed row that is the last NaN or last minimum.
    private static bool LessOrEqualForLast(double v, double best) =>
        double.IsNaN(v) || !double.IsNaN(best) && v <= best;

    private static double[] Dynamic5(double[,] pitch, double[,] merit, double k1, double f0Min)
    {
        int candidates = pitch.GetLength(0), frames = pitch.GetLength(1);
        var local = new double[candidates, frames];
        for (int i = 0; i < candidates; i++)
            for (int t = 0; t < frames; t++) local[i, t] = 1 - merit[i, t];
        var trans = new double[candidates, candidates, frames];
        for (int t = 1; t < frames; t++)
            for (int i = 0; i < candidates; i++)
                for (int j = 0; j < candidates; j++)
                {
                    double v = Math.Abs(pitch[j, t] - pitch[i, t - 1]) / f0Min;
                    trans[i, j, t] = k1 * (0.05 * v + v * v);
                }
        var path = Path(local, trans, candidates, frames);
        return Enumerable.Range(0, frames).Select(t => pitch[path[t], t]).ToArray();
    }

    private static (double[] SpecPitch, double PitchStd) SpecTrack(double[] filtered, int size, double fs, int frameSize, int frameJump, int frames,
        bool[] voiced, YaaptParameters p)
    {
        int nframeSize = frameSize * 2, maxPeaks = p.ShcMaxPeaks, nfft = p.FftLength;
        double delta = fs / nfft;
        int windowLength = (int)Math.Truncate(p.ShcWindow / delta);
        int halfWindow = (int)Math.Truncate(windowLength / 2.0);
        if (windowLength % 2 == 0) windowLength++;
        int maxShc = (int)Math.Truncate((p.F0Max + p.ShcPeakWidth * 2) / delta);
        int minShc = (int)Math.Ceiling(p.F0Min / delta);
        int harmonics = p.ShcHarmonics;
        var candPitch = new double[maxPeaks, frames];
        var candMerit = new double[maxPeaks, frames];
        for (int i = 0; i < maxPeaks; i++) for (int f = 0; f < frames; f++) candMerit[i, f] = 1;
        int dataLength = nframeSize + (frames - 1) * frameJump;
        var data = new double[Math.Max(dataLength, size)];
        Array.Copy(filtered, data, Math.Min(size, data.Length));
        var window = Kaiser(nframeSize, 0.5);
        var shc = new double[maxShc];
        var magnitude = new double[halfWindow + nfft / 2 + 1];
        var slice = new double[nframeSize];
        for (int f = 0; f < frames; f++)
        {
            if (!voiced[f]) continue;
            int start = f * frameJump;
            for (int n = 0; n < nframeSize; n++) slice[n] = data[start + n] * window[n];
            double mean = Mean(slice);
            for (int n = 0; n < nframeSize; n++) slice[n] -= mean;
            var mag = Magnitude(slice, nfft);
            Array.Copy(mag, 0, magnitude, halfWindow, mag.Length);
            for (int r = 0; r <= maxShc - minShc; r++)
            {
                double sum = 0;
                for (int c = 0; c < windowLength; c++)
                {
                    double product = 1;
                    for (int h = 0; h <= harmonics; h++) product *= magnitude[minShc * (h + 1) + r * (h + 1) + c];
                    sum += product;
                }
                shc[minShc - 1 + r] = sum;
            }
            var (pp, pm) = Peaks(shc, delta, maxPeaks, p);
            for (int i = 0; i < maxPeaks; i++) { candPitch[i, f] = pp[i]; candMerit[i, f] = pm[i]; }
        }
        var voicedFrames = Enumerable.Range(0, frames).Where(f => candPitch[0, f] > 0).ToArray();
        int numVoiced = voicedFrames.Length;
        if (numVoiced == 0) throw new InvalidOperationException("YAAPT found no voiced spectral candidate (the reference fails here too).");
        var vcp = new double[maxPeaks, numVoiced];
        var vcm = new double[maxPeaks, numVoiced];
        for (int i = 0; i < maxPeaks; i++)
            for (int v = 0; v < numVoiced; v++) { vcp[i, v] = candPitch[i, voicedFrames[v]]; vcm[i, v] = candMerit[i, voicedFrames[v]]; }
        var firstRow = Enumerable.Range(0, numVoiced).Select(v => vcp[0, v]).ToArray();
        double avgVoiced = Mean(firstRow), stdVoiced = Std(firstRow);
        var index = new int[numVoiced];
        var minPeak = new double[numVoiced];
        var minMerit = new double[numVoiced];
        for (int v = 0; v < numVoiced; v++)
        {
            int best = 0;
            double bestValue = double.PositiveInfinity;
            for (int i = 0; i < maxPeaks; i++)
            {
                double value = Math.Abs(vcp[i, v] - 0.8 * avgVoiced) * (3 - vcm[i, v]);
                if (i == 0 || value < bestValue) { bestValue = value; best = i; }
            }
            index[v] = best;
            minPeak[v] = vcp[best, v];
            minMerit[v] = vcm[best, v];
        }
        minPeak = MedianFilter(minPeak, Math.Max(1, p.MedianOrder - 2));
        for (int v = 0; v < numVoiced; v++) { vcp[index[v], v] = minPeak[v]; vcm[index[v], v] = minMerit[v]; }
        double weightTrans = p.Dp5K1 * stdVoiced / avgVoiced;
        double[] voicedPitch;
        var specPitch = Enumerable.Range(0, frames).Select(f => candPitch[0, f]).ToArray();
        if (numVoiced > 2)
            voicedPitch = MedianFilter(Dynamic5(vcp, vcm, weightTrans, p.F0Min), Math.Max(1, p.MedianOrder - 2));
        else
            voicedPitch = Enumerable.Repeat(150.0, numVoiced).ToArray();
        double pitchAvg = Mean(voicedPitch);
        double pitchStd = Math.Max(Std(voicedPitch), pitchAvg * p.SpecPitchMinStd);
        for (int v = 0; v < numVoiced; v++) specPitch[voicedFrames[v]] = voicedPitch[v];
        if (specPitch[0] < pitchAvg / 2) specPitch[0] = pitchAvg;
        if (specPitch[frames - 1] < pitchAvg / 2) specPitch[frames - 1] = pitchAvg;
        var nonzero = Enumerable.Range(0, frames).Where(f => specPitch[f] != 0).ToArray();
        specPitch = Pchip(nonzero.Select(f => (double)f).ToArray(), nonzero.Select(f => specPitch[f]).ToArray(),
            Enumerable.Range(0, frames).Select(f => (double)f).ToArray());
        specPitch = Fir(new[] { 1.0 / 3, 1.0 / 3, 1.0 / 3 }, specPitch);
        if (frames > 3)
        {
            specPitch[0] = specPitch[2];
            specPitch[1] = specPitch[3];
        }
        return (specPitch, pitchStd);
    }

    private static double[] CrossCorrelation(double[] data, int offset, int length, int lagMin, int lagMax)
    {
        int n = length - lagMax;
        if (n <= 0) throw new InvalidOperationException("Negative index in the cross correlation of the time-domain analysis; increase the time-domain frame length.");
        var phi = new double[length];
        // data -= mean(data), in place in the buffer the overlapping frames share.
        double mean = 0;
        for (int i = 0; i < length; i++) mean += data[offset + i];
        mean /= length;
        for (int i = 0; i < length; i++) data[offset + i] -= mean;
        double pEnergy = 0;
        for (int i = 0; i < n; i++) pEnergy += data[offset + i] * data[offset + i];
        for (int lag = lagMin; lag < lagMax; lag++)
        {
            double numerator = 0, energy = 0;
            for (int i = 0; i < n; i++)
            {
                double v = data[offset + lag + i];
                numerator += v * data[offset + i];
                energy += v * v;
            }
            phi[lag] = numerator / Math.Sqrt(energy * pEnergy);
        }
        return phi;
    }

    private static (double[] Pitch, double[] Merit) CompareRate(double[] phi, double fs, int maxCandidates, int lagMin, int lagMax, YaaptParameters p)
    {
        int center = (int)Math.Truncate(p.NccfPeakWidth / 2.0);
        var peaks = new List<int>();
        for (int n = lagMin + center; n < lagMax - center + 1; n++)
            if (phi[n] > phi[n - 1] && phi[n] > phi[n + 1] && phi[n] > p.NccfThreshold1) peaks.Add(n);
        var pitch = new List<double>(new double[maxCandidates]);
        var merit = new List<double>(new double[maxCandidates]);
        int numPeaks = 0;
        double phiMax = double.NegativeInfinity;
        bool phiNaN = false;
        foreach (var v in phi) { if (double.IsNaN(v)) phiNaN = true; else phiMax = Math.Max(phiMax, v); }
        if (!phiNaN && phiMax > p.NccfThreshold2 && peaks.Count > 0)
        {
            int maxPoint = peaks[0];
            for (int i = 1; i < peaks.Count; i++) if (phi[peaks[i]] > phi[maxPoint]) maxPoint = peaks[i];
            pitch[0] = fs / (maxPoint + 1);
            merit[0] = peaks.Max(i => phi[i]);
            numPeaks = 1;
        }
        else
        {
            foreach (int n in peaks)
            {
                int arg = 0;
                for (int k = 1; k < 2 * center + 1; k++) if (phi[n - center + k] > phi[n - center + arg]) arg = k;
                if (arg != center) continue;
                if (numPeaks < pitch.Count) { pitch[numPeaks] = fs / (n + 1); merit[numPeaks] = phi[n]; }
                else { pitch.Add(fs / (n + 1)); merit.Add(phi[n]); }
                numPeaks++;
            }
        }
        var order = Enumerable.Range(0, merit.Count).OrderBy(i => -merit[i]).Take(maxCandidates).ToArray();
        var outPitch = order.Select(i => pitch[i]).ToArray();
        var outMerit = order.Select(i => merit[i]).ToArray();
        double top = outMerit.Max();
        if (top > 1.0) for (int i = 0; i < outMerit.Length; i++) outMerit[i] /= top;
        return (outPitch, outMerit);
    }

    private static (double[,] Pitch, double[,] Merit) TimeTrack(double[] filtered, int size, double fs, double[] specPitch, double pitchStd, int frameJump,
        YaaptParameters p)
    {
        int tdaLength = (int)(p.TdaFrameLength * fs / 1000);
        int tdaOverlap = tdaLength - frameJump;
        int tdaFrames = (int)((double)(size - tdaOverlap) / frameJump);
        int spectral = specPitch.Length;
        if (tdaFrames > spectral) tdaFrames = spectral;
        int maxCandidates = p.NccfMaxCandidates;
        double freqThreshold = 5.0 * pitchStd;
        var timePitch = new double[maxCandidates, tdaFrames];
        var timeMerit = new double[maxCandidates, tdaFrames];
        var data = new double[Math.Max(size, (tdaFrames - 1) * frameJump + tdaLength)];
        Array.Copy(filtered, data, size);
        int halfWidth = (int)Math.Truncate(p.NccfPeakWidth / 2.0);
        for (int f = 0; f < tdaFrames; f++)
        {
            double low = Math.Max(specPitch[f] - 2.0 * pitchStd, p.F0Min);
            double high = Math.Min(specPitch[f] + 2.0 * pitchStd, p.F0Max);
            int lagMin = (int)Math.Truncate(fs / high) - halfWidth;
            int lagMax = (int)Math.Truncate(fs / low) + halfWidth;
            var phi = CrossCorrelation(data, f * frameJump, tdaLength, lagMin, lagMax);
            var (pitch, merit) = CompareRate(phi, fs, maxCandidates, lagMin, lagMax, p);
            for (int i = 0; i < maxCandidates; i++) { timePitch[i, f] = pitch[i]; timeMerit[i, f] = merit[i]; }
        }
        for (int i = 0; i < maxCandidates; i++)
            for (int f = 0; f < tdaFrames; f++)
            {
                double diff = Math.Abs(timePitch[i, f] - specPitch[f]);
                double match = diff < freqThreshold ? 1 - diff / freqThreshold : 0;
                timeMerit[i, f] = (1 + p.MeritBoost) * timeMerit[i, f] * match;
            }
        return (timePitch, timeMerit);
    }

    private static (double[,] Pitch, double[,] Merit) Refine(double[,] tp1, double[,] tm1, double[,] tp2, double[,] tm2, double[] specPitch,
        double[] energy, bool[] voiced, int frames, YaaptParameters p)
    {
        int half = tp1.GetLength(0), candidates = half * 2;
        var pitch = new double[candidates, frames];
        var merit = new double[candidates, frames];
        for (int f = 0; f < frames; f++)
        {
            var columnPitch = new double[candidates];
            var columnMerit = new double[candidates];
            for (int i = 0; i < half; i++)
            {
                columnPitch[i] = tp1[i, f]; columnMerit[i] = tm1[i, f];
                columnPitch[half + i] = tp2[i, f]; columnMerit[half + i] = tm2[i, f];
            }
            var order = Enumerable.Range(0, candidates).OrderBy(i => -columnMerit[i]).ToArray();
            var sortedMerit = columnMerit.OrderBy(v => v).Reverse().ToArray();
            for (int i = 0; i < candidates; i++) { pitch[i, f] = columnPitch[order[i]]; merit[i, f] = sortedMerit[i]; }
        }
        var bestPitch = MedianFilter(Enumerable.Range(0, frames).Select(f => pitch[0, f]).ToArray(), p.MedianOrder);
        for (int f = 0; f < frames; f++) bestPitch[f] *= voiced[f] ? 1 : 0;
        var meritMat = new bool[candidates, frames];
        for (int f = 0; f < frames; f++)
        {
            bool idx2 = energy[f] > p.NlferThreshold2 && pitch[0, f] > 0;
            for (int i = 1; i < candidates - 1; i++) meritMat[i, f] = pitch[i, f] == 0 && idx2;
        }
        for (int f = 0; f < frames; f++)
        {
            bool idx1 = energy[f] <= p.NlferThreshold2;
            bool idx2 = energy[f] > p.NlferThreshold2 && pitch[0, f] > 0;
            bool idx3 = energy[f] > p.NlferThreshold2 && pitch[0, f] <= 0;
            if (idx1)
                for (int i = 0; i < candidates; i++) { pitch[i, f] = 0; merit[i, f] = p.MeritPivot; }
            if (idx2) { pitch[candidates - 1, f] = 0; merit[candidates - 1, f] = 1.0 - merit[0, f]; }
            for (int i = 0; i < candidates; i++) if (meritMat[i, f]) merit[i, f] = 0;
            if (idx3)
            {
                pitch[0, f] = specPitch[f];
                merit[0, f] = Math.Min(1, energy[f] / 2.0);
                for (int i = 1; i < candidates; i++) { pitch[i, f] = 0; merit[i, f] = 1.0 - merit[0, f]; }
            }
            pitch[candidates - 2, f] = bestPitch[f];
            merit[candidates - 2, f] = bestPitch[f] > 0 ? merit[0, f] : 1.0 - Math.Min(1, energy[f] / 2.0);
            pitch[candidates - 3, f] = specPitch[f];
            merit[candidates - 3, f] = energy[f] / 5.0;
        }
        return (pitch, merit);
    }

    private static double[] Dynamic(double[,] refPitch, double[,] refMerit, double[] energy, int frames, YaaptParameters p)
    {
        int candidates = refPitch.GetLength(0);
        var best = Enumerable.Range(0, frames).Select(f => refPitch[candidates - 2, f]).Where(v => v > 0).ToArray();
        double meanPitch = best.Length > 0 ? Mean(best) : double.NaN;
        var local = new double[candidates, frames];
        for (int i = 0; i < candidates; i++)
            for (int f = 0; f < frames; f++) local[i, f] = 1 - refMerit[i, f];
        var trans = new double[candidates, candidates, frames];
        for (int i = 0; i < candidates; i++)
            for (int j = 0; j < candidates; j++)
            {
                trans[i, j, 0] = 1.0 / p.DpW4;
                for (int t = 1; t < frames; t++)
                {
                    double current = refPitch[j, t], previous = refPitch[i, t - 1], value;
                    if (current > 0 && previous > 0) value = p.DpW1 * (Math.Abs(current - previous) / meanPitch);
                    else if (current == 0 && previous > 0 || current > 0 && previous == 0)
                        value = p.DpW2 * (1 - Math.Min(1, Math.Abs(energy[t - 1] - energy[t])));
                    else if (current == 0 && previous == 0) value = p.DpW3;
                    else value = 1;
                    trans[i, j, t] = value / p.DpW4;
                }
            }
        var path = Path(local, trans, candidates, frames);
        return Enumerable.Range(0, frames).Select(t => refPitch[path[t], t]).ToArray();
    }
}
