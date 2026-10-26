namespace AiDotNet.Audio.Resampling;

/// <summary>
/// libsoxr's constant-rate resampler (the SoX Resampler Library 0.1.3, as python-soxr 1.1.0 vendors it) at its "HQ"
/// quality, for one channel: linear-phase stages planned from the rate ratio (half-band decimators, DFT-filter stages that
/// interpolate by L and decimate by M through overlap-save FFT convolution, a poly-phase FIR for the remaining ratio),
/// Kaiser-window filters designed for 20-bit precision, and the stream's FIFOs, flush and output length.
/// </summary>
/// <remarks>
/// libsoxr computes a quality of 20 bits or fewer in single precision; this port computes in double precision, as libsoxr
/// does under its <c>SOXR_DOUBLE_PRECISION</c> flag (the same filters and stages). <see cref="LibrosaResample"/> is
/// librosa 0.10's default <c>resample</c> (<c>res_type = "soxr_hq"</c>) around it.
/// </remarks>
internal sealed class SoxrResampler
{
    private const double Mult32 = 65536.0 * 65536.0;
    private const int MinDftLog2 = 10, LargeDftLog2 = 17, CoefSizeKBytes = 400, SizeofReal = 8, FifoMinItems = 0x4000 / SizeofReal;

    /// <summary>librosa.resample(y, orig_sr, target_sr) with its default <c>res_type="soxr_hq"</c> and <c>fix=True</c>: the
    /// resampled signal padded with zeros or cut to ceil(n · target / orig) samples; the input itself at equal rates.</summary>
    public static double[] LibrosaResample(double[] y, double origRate, double targetRate)
    {
        if (origRate == targetRate) return (double[])y.Clone();
        int size = (int)Math.Ceiling(y.Length * (targetRate / origRate));
        var resampled = Resample(y, origRate, targetRate);
        var fixedLength = new double[size];
        Array.Copy(resampled, fixedLength, Math.Min(size, resampled.Length));
        return fixedLength;
    }

    /// <summary>
    /// python-soxr's <c>resample(x, in_rate, out_rate, "HQ")</c> on a mono signal: libsoxr fed in blocks of
    /// max(1000, 48000 · in / out) samples into an output of n · out / in + 1 samples, then flushed; the stream yields
    /// round(n · out / in) samples.
    /// </summary>
    public static double[] Resample(double[] x, double inRate, double outRate)
    {
        if (inRate <= 0 || outRate <= 0) throw new ArgumentException("Sample rate should be over 0");
        int olen = (int)(x.Length * outRate / inRate + 1);
        var y = new double[olen];
        var rate = new Rate(inRate / outRate);
        int blockLength = (int)Math.Max(1000.0, 48000 * inRate / outRate);
        int outPos = 0;
        for (int index = 0; index < x.Length; index += blockLength)
        {
            int n = Math.Min(blockLength, x.Length - index);
            rate.Input(x, index, n);
            outPos += rate.Output(y, outPos, olen - outPos);
        }
        rate.Flush();
        outPos += rate.Output(y, outPos, olen - outPos);
        Array.Resize(ref y, outPos);
        return y;
    }

    // ------------------------------------------------------------------ FIFO (fifo.h)

    private sealed class Fifo
    {
        public double[] Data = new double[FifoMinItems];
        public int Begin, End;

        public int Occupancy => End - Begin;

        public int Reserve(int n)
        {
            if (Begin == End) Begin = End = 0;
            while (true)
            {
                if (End + n <= Data.Length)
                {
                    int at = End;
                    End += n;
                    return at;
                }
                if (Begin > FifoMinItems)
                {
                    Array.Copy(Data, Begin, Data, 0, End - Begin);
                    End -= Begin;
                    Begin = 0;
                    continue;
                }
                Array.Resize(ref Data, Data.Length + n);
            }
        }

        public void TrimBy(int n) => End -= n;

        /// <summary>Advances the read position by n; the old read position, or −1 when fewer than n are held.</summary>
        public int Read(int n)
        {
            if (n > End - Begin) return -1;
            int at = Begin;
            Begin += n;
            return at;
        }
    }

    // ------------------------------------------------------------------ stages (cr.h)

    private enum StageKind { Half, Dft, Poly0, Poly1, Poly2, Poly3 }

    private sealed class DftFilter
    {
        public int DftLength, NumTaps, PostPeak;
        public double[] Coefs = Array.Empty<double>();
    }

    private sealed class Stage
    {
        public StageKind Kind;
        public Fifo Fifo = new();
        public int Pre, PrePost, Preload, InputSize = 8192;
        public double OutInRatio;
        public bool IsInput;
        public DftFilter? Filter;
        public double[] Coefs = Array.Empty<double>();
        public long At, Step;           // 32.32 fixed point (step_t's whole); the DFT stage uses the integer parts only
        public int AtInteger, StepInteger;
        public int L, RemM, N, PhaseBits;
        public double Phase0;

        public int Occupancy => Math.Max(0, Fifo.Occupancy - PrePost);
        public int ReadIndex => Fifo.Begin + Pre;
    }

    private sealed class Rate
    {
        private readonly double _ioRatio;
        private readonly Stage[] _stages;
        private readonly int _numStages;
        private long _samplesIn, _samplesOut;
        private bool _flushing;

        public Rate(double ioRatio)
        {
            _ioRatio = ioRatio;
            _stages = Init(ioRatio, out _numStages);
        }

        public void Input(double[] samples, int offset, int n)
        {
            if (_flushing) return;
            _samplesIn += n;
            var fifo = _stages[0].Fifo;
            int at = fifo.Reserve(n);
            Array.Copy(samples, offset, fifo.Data, at, n);
        }

        public void Flush()
        {
            if (_flushing) return;
            _samplesOut -= (long)(_samplesIn / _ioRatio + .5);
            _samplesIn = 0;
            _flushing = true;
        }

        // soxr_output_1ch: process, then take what the last FIFO holds (up to len).
        public int Output(double[] destination, int offset, int len)
        {
            Process(len);
            var fifo = _stages[_numStages].Fifo;
            int n = _flushing ? (int)Math.Min(-_samplesOut, len) : len;
            n = Math.Min(n, fifo.Occupancy);
            _samplesOut += n;
            int at = fifo.Read(n);
            Array.Copy(fifo.Data, at, destination, offset, n);
            return n;
        }

        private void Process(int olen)
        {
            int n = _flushing ? (int)Math.Min(-_samplesOut, olen) : olen;
            var fifo = _stages[_numStages].Fifo;
            bool done = false;
            while (!done && fifo.Occupancy < n)
                done = _numStages == 0 || StageProcess(_numStages - 1, _flushing);
        }

        private bool StageProcess(int index, bool flushing)
        {
            var stage = _stages[index];
            var fifo = stage.Fifo;
            bool done = false;
            int want;
            while (!done && (want = stage.InputSize - fifo.Occupancy) > 0)
            {
                if (stage.IsInput)
                {
                    if (flushing)
                    {
                        int at = fifo.Reserve(want);
                        Array.Clear(fifo.Data, at, want);
                    }
                    else done = true;
                }
                else done = StageProcess(index - 1, flushing);
            }
            Run(stage, _stages[index + 1].Fifo);
            return done && fifo.Occupancy < stage.InputSize;
        }
    }

    private static void Run(Stage stage, Fifo output)
    {
        switch (stage.Kind)
        {
            case StageKind.Half: HalfStage(stage, output); break;
            case StageKind.Dft: DftStage(stage, output); break;
            case StageKind.Poly0: Poly0Stage(stage, output); break;
            default: PolyStage(stage, output, stage.Kind - StageKind.Poly0); break;
        }
    }

    // ------------------------------------------------------------------ stage planning (_soxr_init)

    private static double LinearToDb(double x) => Math.Log10(x) * 20;

    private static Stage[] Init(double ioRatio, out int numStages)
    {
        // soxr_quality_spec(SOXR_HQ, 0): 20 bits, linear phase, pass-band to 1 − 0.05 / (3 dB point), stop-band at 1.
        double bits = 20;
        double rej = bits * LinearToDb(2.0);
        double Fp0 = 1 - .05 / (1 - InvFResp(-3.0, rej)), Fs0 = 1, phaseResponse = 50, tbw0 = Fs0 - Fp0;
        const int rolloff = 0;            // SOXR_ROLLOFF_SMALL
        const int interpolator = -1;      // SOXR_COEF_INTERP_AUTO
        const bool iOpt = true;
        double tbwTighten = 1;
        double Tighten(double x) => Fs0 - (Fs0 - x) * tbwTighten;

        double arbM = ioRatio, Fn1, Fp1 = Fp0, Fs1 = Fs0, bits1 = Math.Min(bits, 33);
        double att = (bits1 + 1) * LinearToDb(2.0), attArb = att;
        int preL = 1, preM = 1, shr = 0, arbL = 1, postL = 1, postM = 1;
        bool upsample = false, rational = false;
        int n = 0, i;
        int mode = ((int)Math.Ceiling(bits1) - 6) / 4;   // not the 16-bit medium-roll-off case

        if (ioRatio < 1 && Fs0 - 1 > 1 - Fp0 / (1 + 1e-5)) throw new InvalidOperationException("imaging greater than rolloff");
        if (ioRatio <= 0) throw new ArgumentOutOfRangeException(nameof(ioRatio), "resampling factor not positive");

        while (n++ == 0)
        {
            int maxL = interpolator > 0 ? 1 : mode != 0 ? 2048 : (int)Math.Ceiling(CoefSizeKBytes * 1000.0 / (42 * SizeofReal));
            double d, epsilon = 0, frac;
            upsample = arbM < 1;
            for (i = (int)(.5 * arbM), shr = 0; (i >>= 1) != 0; arbM *= .5, ++shr) { }
            preM = upsample || (arbM > 1.5 && arbM < 2) ? 1 : 0;
            postM = 1 + (arbM > 1 && preM != 0 ? 1 : 0);
            arbM /= postM;
            preL = 1 + (preM == 0 && arbM < 2 ? 1 : 0) + (upsample && mode != 0 ? 1 : 0);
            arbM *= preL;
            if ((frac = arbM - (int)arbM) != 0) epsilon = Math.Abs(Math.Floor(frac * Mult32 + .5) / (frac * Mult32) - 1);
            for (i = 1, rational = frac == 0; i <= maxL && !rational; ++i)
            {
                d = frac * i;
                int attempt = (int)(d + .5);
                if (rational = Math.Abs(attempt / d - 1) <= epsilon)
                {
                    if (attempt == i)
                    {
                        arbM = Math.Ceiling(arbM);
                        int x2 = arbM > 3 ? 1 : 0;
                        shr += x2;
                        arbM /= 1 + x2;
                    }
                    else
                    {
                        arbM = i * (int)arbM + attempt;
                        arbL = i;
                    }
                }
            }
            int L = preL * arbL, M = (int)(arbM * postM), x = (L | M) & 1;
            if (x == 0) { L >>= 1; M >>= 1; }
            if (iOpt && postL == 1 && (d = preL * arbL / arbM) > 4 && d != 5)
            {
                for (postL = 4, i = (int)(d / 16); (i >>= 1) != 0 && postL < 256; postL <<= 1) { }
                arbM = arbM * postL / arbL / preL;
                arbL = 1;
                n = 0;
            }
            else if (rational && (Math.Max(L, M) < 3 + 2 * (iOpt ? 1 : 0) || L * M < 6 * (iOpt ? 1 : 0)))
            {
                preL = L;
                preM = M;
                arbM = arbL = postM = 1;
            }
            if (mode == 0 && (!rational || n == 0)) { ++mode; n = 0; }
        }

        bool havePre = preM * preL != 1, haveArb = arbM * arbL != 1, havePost = postM * postL != 1;
        numStages = shr + (havePre ? 1 : 0) + (haveArb ? 1 : 0) + (havePost ? 1 : 0);
        var stages = new Stage[numStages + 1];
        for (i = 0; i <= numStages; i++) stages[i] = new Stage();
        stages[0].IsInput = true;
        var shared = new[] { new DftFilter(), new DftFilter() };
        double multiplier = 1;
        double alpha = postM / (ioRatio * postL);

        if ((n = numStages) > 1)
        {
            if (haveArb) { att += LinearToDb(2.0); attArb = att; --n; }
            att += LinearToDb(n);
        }

        int halfIndex = 0;
        while (halfIndex + 1 < SoxrTables.HalfFirs.Length && att > SoxrTables.HalfFirs[halfIndex].Attenuation) halfIndex++;
        int s = 0;
        for (i = 0; i < shr; ++i, ++s)
        {
            var st = stages[s];
            st.Kind = StageKind.Half;
            st.Coefs = SoxrTables.HalfFirs[halfIndex].Coefs;
            st.N = st.Coefs.Length;
            st.PrePost = 4 * st.N;
            st.Preload = st.Pre = st.PrePost >> 1;
        }

        if (havePre)
        {
            Fn1 = preM != 0 ? Math.Max(preL, preM) : arbM / arbL;
            DftStageInit(shared[0], Tighten(Fp1), Fs1, Fn1, att, phaseResponse, stages[s++], preL, Math.Max(preM, 1), ref multiplier);
            Fp1 /= Fn1;
            Fs1 /= Fn1;
        }

        if (haveArb)
        {
            var st = stages[s];
            float[] rolloffs = { -.01f, -.3f, 0, -.103f };
            var f = PolyFirs[6 * ((upsample ? 1 : 0) + (preM != 0 ? 1 : 0)) + mode - (upsample ? 0 : 1)];
            int order, numCoefs = (int)f.Interp[0].Scalar, phaseBits, phases;
            long coefsSize;
            double at, Fp = Fp1, Fs, Fn, mult = upsample ? 1 : arbM / arbL;
            if (!upsample && preM != 0) { Fn = 2 * mult; Fs = 3 + Math.Abs(Fs1 - 1); }
            else { Fn = 1; Fs = 2 - (mode != 0 ? Fp1 + (Fs1 - Fp1) * .7 : Fs1); }
            if (mode != 0) Fp = Fs - (Fs - Fp) / (1 - InvFResp(rolloffs[rolloff], attArb));
            i = (interpolator < 0 ? (!rational ? 1 : 0) : Math.Max(interpolator, !rational ? 1 : 0)) - 1;
            (float Scalar, StageKind? Kind) f1;
            do
            {
                f1 = f.Interp[++i];
                if (i != 0) { arbM /= arbL; arbL = 1; rational = false; }
                phaseBits = (int)Math.Ceiling(f1.Scalar - Math.Log(mult) / Math.Log(2.0));
                phases = !rational ? 1 << phaseBits : arbL;
                if (f.Interp[0].Scalar == 0)
                {
                    int phases0 = Math.Max(phases, 19), n0 = 0;
                    DesignLpf(Fp, Fs, -Fn, attArb, ref n0, phases0, f.Beta);
                    numCoefs = n0 / phases0 + 1;
                    numCoefs += numCoefs & (preM == 0 ? 1 : 0);
                }
                if ((numCoefs & 1) != 0 && rational && (arbL & 1) != 0) { phases <<= 1; arbL <<= 1; arbM *= 2; }
                at = arbL * (st.Phase0 = .5 * (numCoefs & 1));
                order = i + (i != 0 && mode > 4 ? 1 : 0);
                coefsSize = (long)numCoefs * phases * (order + 1) * SizeofReal;
            } while (interpolator < 0 && i < 2 && f.Interp[i + 1].Kind is not null && coefsSize / 1000 > CoefSizeKBytes);

            int numTaps = numCoefs * phases - 1;
            var coefs = DesignLpf(Fp, Fs, Fn, attArb, ref numTaps, phases, f.Beta)!;
            st.Coefs = PreparePolyFirCoefs(coefs, numCoefs, phases, order, multiplier);
            multiplier = 1;
            st.Kind = f1.Kind!.Value;
            st.PrePost = numCoefs - 1;
            st.Preload = (numCoefs - 1) >> 1;
            st.N = numCoefs;
            st.PhaseBits = phaseBits;
            st.L = arbL;
            st.At = (long)(at * Mult32 + .5);
            st.Step = (long)(arbM * Mult32 + .5);
            st.OutInRatio = Mult32 * arbL / st.Step;
            ++s;
        }

        if (havePost)
            DftStageInit(shared[1], Tighten(Fp0 / (upsample ? alpha : 1)), upsample ? Math.Max(2 - Fs0 / alpha, 1) : Fs0, Math.Max(postL, postM), att,
                phaseResponse, stages[s++], postL, postM, ref multiplier);

        for (i = 0; i < numStages; ++i)
        {
            var fifo = stages[i].Fifo;
            int at = fifo.Reserve(stages[i].Preload);
            Array.Clear(fifo.Data, at, stages[i].Preload);
        }
        return stages;
    }

    // poly_firs (cr-core.c, without SIMD): the poly-phase filter per (up/down-sampling, pre-stage, mode), each with its
    // coefficient-interpolation variants (the phase-bits scalar and the interpolation order); a fixed-length entry's
    // first scalar is its length. A null kind is an empty slot.
    private sealed record PolyFir(float Beta, (float Scalar, StageKind? Kind)[] Interp);

    private static readonly PolyFir[] PolyFirs =
    {
        P(-1, (0, StageKind.Poly0), (7.2f, StageKind.Poly1), (5.0f, StageKind.Poly2)),
        P(-1, (0, StageKind.Poly0), (9.4f, StageKind.Poly1), (6.7f, StageKind.Poly2)),
        P(-1, (0, StageKind.Poly0), (12.4f, StageKind.Poly1), (7.8f, StageKind.Poly2)),
        P(-1, (0, StageKind.Poly0), (13.6f, StageKind.Poly1), (9.3f, StageKind.Poly2)),
        P(-1, (0, StageKind.Poly0), (10.5f, StageKind.Poly2), (8.4f, StageKind.Poly3)),
        P(-1, (0, StageKind.Poly0), (11.85f, StageKind.Poly2), (9.0f, StageKind.Poly3)),
        P(-1, (0, StageKind.Poly0), (8.0f, StageKind.Poly1), (5.3f, StageKind.Poly2)),
        P(-1, (0, StageKind.Poly0), (8.6f, StageKind.Poly1), (5.7f, StageKind.Poly2)),
        P(-1, (0, StageKind.Poly0), (10.6f, StageKind.Poly1), (6.75f, StageKind.Poly2)),
        P(-1, (0, StageKind.Poly0), (12.6f, StageKind.Poly1), (8.6f, StageKind.Poly2)),
        P(-1, (0, StageKind.Poly0), (9.6f, StageKind.Poly2), (7.6f, StageKind.Poly3)),
        P(-1, (0, StageKind.Poly0), (11.4f, StageKind.Poly2), (8.65f, StageKind.Poly3)),
        P(10.62f, (42, StageKind.Poly0), (0, null), (0, null)),
        P(11.28f, (11, StageKind.Poly0), (8, StageKind.Poly1), (6, StageKind.Poly2)),
        P(-1, (0, StageKind.Poly0), (9, StageKind.Poly1), (6, StageKind.Poly2)),
        P(-1, (0, StageKind.Poly0), (11, StageKind.Poly1), (7, StageKind.Poly2)),
        P(-1, (0, StageKind.Poly0), (13, StageKind.Poly1), (8, StageKind.Poly2)),
        P(-1, (0, StageKind.Poly0), (10, StageKind.Poly2), (8, StageKind.Poly3)),
        P(-1, (0, StageKind.Poly0), (12, StageKind.Poly2), (9, StageKind.Poly3)),
    };

    private static PolyFir P(float beta, params (float, StageKind?)[] interp) => new(beta, interp);

    // prepare_poly_fir_coefs: per phase and tap, the coefficient and its interpolation polynomial (order 0–3), stored
    // [phase][tap][order..0] with the taps reversed.
    private static double[] PreparePolyFirCoefs(double[] coefs, int numCoefs, int numPhases, int order, double multiplier)
    {
        var result = new double[numCoefs * numPhases * (order + 1)];
        double fm1 = coefs[0], f1 = 0, f2 = 0;
        int Index(int phase, int interp, int tap) => numCoefs * (order + 1) * phase + (order + 1) * tap + (order - interp);
        for (int i = numCoefs - 1; i >= 0; --i)
            for (int j = numPhases - 1; j >= 0; --j)
            {
                double f0 = fm1, b = 0, c = 0, d = 0;
                int pos = i * numPhases + j - 1;
                fm1 = pos > 0 ? coefs[pos - 1] * multiplier : 0;
                switch (order)
                {
                    case 1: b = f1 - f0; break;
                    case 2: b = f1 - (.5 * (f2 + f0) - f1) - f0; c = .5 * (f2 + f0) - f1; break;
                    case 3: c = .5 * (f1 + fm1) - f0; d = (1 / 6.0) * (f2 - f1 + fm1 - f0 - 4 * c); b = f1 - f0 - d - c; break;
                }
                int tap = numCoefs - 1 - i;
                if (order > 2) result[Index(j, 3, tap)] = d;
                if (order > 1) result[Index(j, 2, tap)] = c;
                if (order > 0) result[Index(j, 1, tap)] = b;
                result[Index(j, 0, tap)] = f0;
                f2 = f1;
                f1 = f0;
            }
        return result;
    }

    // set_dft_length: four times the nearest power of two of the taps, or half that when large.
    private static int SetDftLength(int numTaps, int min, int large)
    {
        double d = Math.Log(numTaps) / Math.Log(2.0);
        return 1 << Math.Min(Math.Max((int)(d + 2.77), min), Math.Max((int)(d + 1.77), large));
    }

    private static bool IsPowerOf2(int x) => !(x < 2 || (x & (x - 1)) != 0);

    private static void DftStageInit(DftFilter f, double Fp, double Fs, double Fn, double att, double phaseResponse, Stage p, int L, int M,
        ref double multiplier)
    {
        int numTaps = 0, dftLength = f.DftLength;
        bool fDomainM = Math.Abs(3 - M) == 1 && Fs <= 1;
        if (dftLength == 0)
        {
            int k = phaseResponse == 50 && IsPowerOf2(L) && Fn == L ? L << 1 : 4;
            var h = DesignLpf(Fp, Fs, Fn, att, ref numTaps, -k, -1.0)!;
            f.PostPeak = numTaps / 2;   // linear phase
            dftLength = SetDftLength(numTaps, MinDftLog2, LargeDftLog2);
            f.Coefs = new double[dftLength];
            int offset = dftLength - numTaps + 1;
            double m = (1.0 / dftLength) * 2 * L * multiplier;
            for (int i = 0; i < numTaps; ++i) f.Coefs[(i + offset) & (dftLength - 1)] = h[i] * m;
            Rdft(f.Coefs, 0, dftLength, forward: true);
            f.NumTaps = numTaps;
            f.DftLength = dftLength;
        }
        multiplier = 1;
        p.OutInRatio = (double)L / M;
        p.Kind = StageKind.Dft;
        p.Filter = f;
        p.Preload = f.PostPeak / L;
        p.AtInteger = f.PostPeak % L;
        p.L = L;
        p.StepInteger = fDomainM ? -M / 2 : M;
        p.Phase0 = p.AtInteger / p.L;
        p.InputSize = (f.DftLength - p.AtInteger + p.L - 1) / p.L;
    }

    // ------------------------------------------------------------------ stage functions

    // half-fir.h: decimate by 2 with an odd-length symmetric FIR (the centre tap 0.5).
    private static void HalfStage(Stage p, Fifo outputFifo)
    {
        var data = p.Fifo.Data;
        int input = p.ReadIndex;
        int numIn = Math.Min(p.Occupancy, p.InputSize);
        int numOut = (numIn + 1) >> 1;
        int output = outputFifo.Reserve(numOut);
        data = p.Fifo.Data;
        var o = outputFifo.Data;
        var coefs = p.Coefs;
        for (int i = 0; i < numOut; ++i, input += 2)
        {
            double sum = data[input] * .5;
            for (int j = 0; j < coefs.Length; j++) sum += (data[input - (2 * j + 1)] + data[input + (2 * j + 1)]) * coefs[j];
            o[output + i] = sum;
        }
        p.Fifo.Read(2 * numOut);
    }

    // poly-fir0.h: a rational poly-phase FIR (no coefficient interpolation).
    private static void Poly0Stage(Stage p, Fifo outputFifo)
    {
        int numIn = Math.Min(p.Occupancy, p.InputSize);
        if (numIn == 0) return;
        int input = p.ReadIndex;
        int at = (int)(p.At >> 32), step = (int)(p.Step >> 32);
        int numOut = (numIn * p.L - at + step - 1) / step;
        int output = outputFifo.Reserve(numOut);
        var data = p.Fifo.Data;
        var o = outputFifo.Data;
        int n = p.N;
        int i;
        for (i = 0; at < numIn * p.L; ++i, at += step)
        {
            int div = at / p.L, rem = at % p.L;
            int x = input + div, c = n * rem;
            double sum = 0;
            for (int j = 0; j < n; j++) sum += p.Coefs[c + j] * data[x + j];
            o[output + i] = sum;
        }
        p.Fifo.Read(at / p.L);
        p.At = (long)(at % p.L) << 32 | (uint)p.At;
    }

    // poly-fir.h (standard-precision clock): an irrational poly-phase FIR, the coefficients interpolated between phases.
    private static void PolyStage(Stage p, Fifo outputFifo, int order)
    {
        int input = p.ReadIndex;
        int numIn = Math.Min(p.Occupancy, p.InputSize);
        int maxNumOut = 1 + (int)(numIn * p.OutInRatio);
        int output = outputFifo.Reserve(maxNumOut);
        var data = p.Fifo.Data;
        var o = outputFifo.Data;
        int n = p.N, bits = p.PhaseBits;
        long at = p.At;
        int i;
        for (i = 0; (int)(at >> 32) < numIn; ++i, at += p.Step)
        {
            int x = input + (int)(at >> 32);
            uint frac = (uint)at;
            int phase = (int)(frac >> (32 - bits));
            double xf = (uint)(frac << bits) * (1 / Mult32);
            double sum = 0;
            for (int j = 0; j < n; j++)
            {
                int c = (order + 1) * (n * phase + j);
                double coef = order switch
                {
                    1 => p.Coefs[c] * xf + p.Coefs[c + 1],
                    2 => (p.Coefs[c] * xf + p.Coefs[c + 1]) * xf + p.Coefs[c + 2],
                    _ => ((p.Coefs[c] * xf + p.Coefs[c + 1]) * xf + p.Coefs[c + 2]) * xf + p.Coefs[c + 3],
                };
                sum += coef * data[x + j];
            }
            o[output + i] = sum;
        }
        outputFifo.TrimBy(maxNumOut - i);
        p.Fifo.Read((int)(at >> 32));
        p.At = (uint)at;
    }

    // dft_stage_fn: overlap-save convolution with the stage's filter, interpolating by L (zero-stuffing, or replicating the
    // spectrum when L is a power of two) and decimating by M (in time, or for M of 2 or 4 by keeping the low band).
    private static void DftStage(Stage p, Fifo outputFifo)
    {
        int numIn = Math.Max(0, p.Fifo.Occupancy);
        var f = p.Filter!;
        int overlap = f.NumTaps - 1, dftLength = f.DftLength;
        if (p.AtInteger + p.L * numIn >= dftLength)
        {
            int numerator = dftLength - overlap - p.AtInteger + p.L - 1;
            int quot = numerator / p.L, rem = numerator % p.L;
            int input = p.Fifo.Begin;
            p.Fifo.Read(quot);
            int output = outputFifo.Reserve(dftLength);
            var inData = p.Fifo.Data;
            var o = outputFifo.Data;
            int i, j;
            if (IsPowerOf2(p.L))
            {
                // F-domain interpolation.
                int portion = dftLength / p.L;
                Array.Copy(inData, input, o, output, portion);
                Rdft(o, output, portion, forward: true);
                for (i = portion + 2; i < (portion << 1); i += 2)
                {
                    o[output + i] = o[output + (portion << 1) - i];
                    o[output + i + 1] = -o[output + (portion << 1) - i + 1];
                }
                o[output + portion] = o[output + 1];
                o[output + portion + 1] = 0;
                o[output + 1] = o[output];
                for (portion <<= 1; i < dftLength; i += portion, portion <<= 1)
                {
                    Array.Copy(o, output, o, output + i, portion);
                    o[output + i + 1] = 0;
                }
            }
            else
            {
                if (p.L == 1) Array.Copy(inData, input, o, output, dftLength);
                else
                {
                    Array.Clear(o, output, dftLength);
                    for (j = 0, i = p.AtInteger; i < dftLength; ++j, i += p.L) o[output + i] = inData[input + j];
                    p.AtInteger = p.L - 1 - rem;
                }
                Rdft(o, output, dftLength, forward: true);
            }

            if (p.StepInteger > 0)
            {
                OrderedConvolve(o, output, dftLength, f.Coefs);
                Rdft(o, output, dftLength, forward: false);
                if (p.StepInteger != 1)
                {
                    for (j = 0, i = p.RemM; i < dftLength - overlap; ++j, i += p.StepInteger) o[output + j] = o[output + i];
                    p.RemM = i - (dftLength - overlap);
                    outputFifo.TrimBy(dftLength - j);
                }
                else outputFifo.TrimBy(overlap);
            }
            else
            {
                // F-domain decimation.
                int m = -p.StepInteger;
                OrderedPartialConvolve(o, output, dftLength >> m, f.Coefs);
                Rdft(o, output, dftLength >> m, forward: false);
                outputFifo.TrimBy((((1 << m) - 1) * dftLength + overlap) >> m);
            }
        }
        p.InputSize = (dftLength - p.AtInteger + p.L - 1) / p.L;
    }

    // _soxr_ordered_convolve: the packed spectra multiplied (DC and Nyquist real).
    private static void OrderedConvolve(double[] a, int offset, int n, double[] b)
    {
        a[offset] *= b[0];
        a[offset + 1] *= b[1];
        for (int i = 2; i < n; i += 2)
        {
            double tmp = a[offset + i];
            a[offset + i] = b[i] * tmp - b[i + 1] * a[offset + i + 1];
            a[offset + i + 1] = b[i + 1] * tmp + b[i] * a[offset + i + 1];
        }
    }

    // _soxr_ordered_partial_convolve: the low n bins multiplied, the Nyquist slot from the product at bin n/2.
    private static void OrderedPartialConvolve(double[] a, int offset, int n, double[] b)
    {
        int i;
        a[offset] *= b[0];
        for (i = 2; i < n; i += 2)
        {
            double tmp = a[offset + i];
            a[offset + i] = b[i] * tmp - b[i + 1] * a[offset + i + 1];
            a[offset + i + 1] = b[i + 1] * tmp + b[i] * a[offset + i + 1];
        }
        a[offset + 1] = b[i] * a[offset + i] - b[i + 1] * a[offset + i + 1];
    }

    // ------------------------------------------------------------------ filter design (filter.c, dbesi0.c)

    // lsx_design_lpf: a Kaiser-windowed sinc low-pass from the pass-band end, stop-band start and attenuation; k > 0
    // phases, k < 0 taps = 1 (mod -k); Fn < 0 only sizes it.
    private static double[]? DesignLpf(double Fp, double Fs, double Fn, double att, ref int numTaps, int k, double beta)
    {
        int n = numTaps, phases = Math.Max(k, 1), modulo = Math.Max(-k, 1);
        double rho = phases == 1 ? .5 : att < 120 ? .63 : .75;
        Fp /= Math.Abs(Fn);
        Fs /= Math.Abs(Fn);
        double trBw = .5 * (Fs - Fp);
        trBw /= phases;
        Fs /= phases;
        trBw = Math.Min(trBw, .5 * Fs);
        double Fc = Fs - trBw;
        // lsx_kaiser_params
        beta = beta < 0 ? KaiserBeta(att, trBw * .5 / Fc) : beta;
        double a = att < 60 ? (att - 7.95) / (2.285 * Math.PI * 2) : ((.0007528358 - 1.577737e-05 * beta) * beta + .6248022) * beta + .06186902;
        numTaps = numTaps == 0 ? (int)Math.Ceiling(a / trBw + 1) : numTaps;
        if (n == 0)
            numTaps = phases > 1 ? numTaps / phases * phases + phases - 1 : (numTaps + modulo - 2) / modulo * modulo + 1;
        return Fn < 0 ? null : MakeLpf(numTaps, Fc, beta, rho, phases);
    }

    private static readonly double[][] KaiserBetaCoefs =
    {
        new[] { -6.784957e-10, 1.02856e-05, 0.1087556, -0.8988365 + .001 },
        new[] { -6.897885e-10, 1.027433e-05, 0.10876, -0.8994658 + .002 },
        new[] { -1.000683e-09, 1.030092e-05, 0.1087677, -0.9007898 + .003 },
        new[] { -3.654474e-10, 1.040631e-05, 0.1087085, -0.8977766 + .006 },
        new[] { 8.106988e-09, 6.983091e-06, 0.1091387, -0.9172048 + .015 },
        new[] { 9.519571e-09, 7.272678e-06, 0.1090068, -0.9140768 + .025 },
        new[] { -5.626821e-09, 1.342186e-05, 0.1083999, -0.9065452 + .05 },
        new[] { -9.965946e-08, 5.073548e-05, 0.1040967, -0.7672778 + .085 },
        new[] { 1.604808e-07, -5.856462e-05, 0.1185998, -1.34824 + .1 },
        new[] { -1.511964e-07, 6.363034e-05, 0.1064627, -0.9876665 + .18 },
    };

    // lsx_kaiser_beta: above 60 dB, a cubic in the attenuation interpolated over log2 of the transition width.
    private static double KaiserBeta(double att, double trBw)
    {
        if (att >= 60)
        {
            double realm = Math.Log(trBw / .0005) / Math.Log(2.0);
            int last = KaiserBetaCoefs.Length - 1;
            var c0 = KaiserBetaCoefs[Math.Min(Math.Max((int)realm, 0), last)];
            var c1 = KaiserBetaCoefs[Math.Min(Math.Max(1 + (int)realm, 0), last)];
            double b0 = ((c0[0] * att + c0[1]) * att + c0[2]) * att + c0[3];
            double b1 = ((c1[0] * att + c1[1]) * att + c1[2]) * att + c1[3];
            return b0 + (b1 - b0) * (realm - (int)realm);
        }
        if (att > 50) return .1102 * (att - 8.7);
        if (att > 20.96) return .58417 * Math.Pow(att - 20.96, .4) + .07886 * (att - 20.96);
        return 0;
    }

    // lsx_make_lpf: the windowed sinc, symmetric.
    private static double[] MakeLpf(int numTaps, double Fc, double beta, double rho, double scale)
    {
        int m = numTaps - 1;
        var h = new double[numTaps];
        double mult = scale / BesselI0(beta), mult1 = 1 / (.5 * m + rho);
        for (int i = 0; i <= m / 2; ++i)
        {
            double z = i - .5 * m, x = z * Math.PI, y = z * mult1;
            h[i] = x != 0 ? Math.Sin(Fc * x) / x : Fc;
            h[i] *= BesselI0(beta * Math.Sqrt(1 - y * y)) * mult;
            if (m - i != i) h[m - i] = h[i];
        }
        return h;
    }

    // lsx_inv_f_resp: the normalized frequency at which a filter of attenuation a drops by `drop` dB.
    private static double InvFResp(double drop, double a)
    {
        double x = ((2.0517e-07 * a - 1.1303e-04) * a + .023154) * a + .55924;
        drop = Math.Exp(drop * (2.30258509299404568402 * 0.05));
        double s = drop > .5 ? 1 - drop : drop;
        double sinePow = Math.Log(.5) / Math.Log(Math.Sin(x * .5));
        x = Math.Asin(Math.Pow(s, 1 / sinePow)) / x;
        return drop > .5 ? x : 1 - x;
    }

    // dbesi0 (Ooura): I0 by piecewise polynomials below 12.5, an asymptotic form above.
    private static double BesselI0(double x)
    {
        double w = Math.Abs(x), t, y;
        int k;
        if (w < 8.5)
        {
            var a = SoxrTables.BesselI0A;
            t = w * w * 0.0625;
            k = 13 * (int)t;
            y = a[k];
            for (int j = 1; j <= 12; j++) y = y * t + a[k + j];
        }
        else if (w < 12.5)
        {
            var b = SoxrTables.BesselI0B;
            k = (int)w;
            t = w - k;
            k = 14 * (k - 8);
            y = b[k];
            for (int j = 1; j <= 13; j++) y = y * t + b[k + j];
        }
        else
        {
            var c = SoxrTables.BesselI0C;
            t = 60 / w;
            k = 9 * (int)t;
            y = c[k];
            for (int j = 1; j <= 8; j++) y = y * t + c[k + j];
            y *= Math.Sqrt(t) * Math.Exp(w);
        }
        return y;
    }

    // ------------------------------------------------------------------ real DFT in Ooura's packed layout (fft4g rdft)

    // forward: a[2k] = sum x_j cos(2 pi jk/n), a[2k+1] = sum x_j sin(2 pi jk/n), a[1] = the Nyquist term; backward
    // (unscaled): x_k = (a[0] + a[1] cos(pi k)) / 2 + sum a[2j] cos(2 pi jk/n) + a[2j+1] sin(2 pi jk/n), n/2 times the
    // inverse DFT.
    private static void Rdft(double[] a, int offset, int n, bool forward)
    {
        var re = new double[n];
        var im = new double[n];
        int half = n / 2;
        if (forward)
        {
            for (int i = 0; i < n; i++) re[i] = a[offset + i];
            Fft(re, im, inverse: false);
            a[offset] = re[0];
            a[offset + 1] = re[half];
            for (int k = 1; k < half; k++)
            {
                a[offset + 2 * k] = re[k];
                a[offset + 2 * k + 1] = -im[k];
            }
        }
        else
        {
            re[0] = a[offset];
            re[half] = a[offset + 1];
            for (int k = 1; k < half; k++)
            {
                re[k] = a[offset + 2 * k];
                im[k] = -a[offset + 2 * k + 1];
                re[n - k] = re[k];
                im[n - k] = -im[k];
            }
            Fft(re, im, inverse: true);
            for (int i = 0; i < n; i++) a[offset + i] = 0.5 * re[i];
        }
    }

    // An in-place iterative radix-2 complex FFT (exp(-/+ 2 pi i jk/n), unscaled) with directly computed twiddles.
    private static void Fft(double[] re, double[] im, bool inverse)
    {
        int n = re.Length;
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }
        var (cos, sin) = Twiddles(n);
        double sign = inverse ? 1 : -1;
        for (int size = 2; size <= n; size <<= 1)
        {
            int halfSize = size >> 1, stride = n / size;
            for (int start = 0; start < n; start += size)
                for (int k = 0; k < halfSize; k++)
                {
                    double wr = cos[k * stride], wi = sign * sin[k * stride];
                    int u = start + k, v = u + halfSize;
                    double tr = re[v] * wr - im[v] * wi, ti = re[v] * wi + im[v] * wr;
                    re[v] = re[u] - tr;
                    im[v] = im[u] - ti;
                    re[u] += tr;
                    im[u] += ti;
                }
        }
    }

    private static readonly Dictionary<int, (double[] Cos, double[] Sin)> TwiddleCache = new();

    private static (double[] Cos, double[] Sin) Twiddles(int n)
    {
        lock (TwiddleCache)
        {
            if (TwiddleCache.TryGetValue(n, out var cached)) return cached;
            var cos = new double[n / 2];
            var sin = new double[n / 2];
            for (int k = 0; k < n / 2; k++)
            {
                cos[k] = Math.Cos(2 * Math.PI * k / n);
                sin[k] = Math.Sin(2 * Math.PI * k / n);
            }
            return TwiddleCache[n] = (cos, sin);
        }
    }
}
