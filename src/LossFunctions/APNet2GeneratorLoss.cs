using System.Linq;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.LossFunctions;

/// <summary>
/// APNet / APNet2's generator objective: amplitude-spectrum loss, anti-wrapping phase loss and
/// reconstructed-STFT loss.
/// </summary>
/// <remarks>
/// <para>
/// Implements <c>L_G = lambda_A * L_A + lambda_P * L_P + lambda_S * L_S</c> from Ai &amp; Ling 2023
/// (APNet, arXiv:2305.07952, Eq. 37), carried forward unchanged by Du et al. 2023 (APNet2,
/// arXiv:2311.11545). The paper's coefficients are <c>lambda_A = 45</c>, <c>lambda_P = 100</c> and
/// <c>lambda_S = 20</c>.
/// </para>
/// <para>
/// <b>Why this exists.</b> Without it the vocoder trained under a generic squared error, which was
/// applied to phase directly. Phase is an angle: a prediction of <c>-pi + e</c> against a target of
/// <c>pi - e</c> is nearly correct but scores as though it were maximally wrong, and the gradient
/// pushes the prediction the long way round. Avoiding exactly that is the paper's central
/// contribution, and the anti-wrapping function below is what does it:
/// <c>AW(x) = x - 2*pi*round(x / (2*pi))</c>, which maps any angular error into
/// <c>(-pi, pi]</c> before it is penalised.
/// </para>
/// <para>
/// <b>What is not included.</b> The paper's fourth term <c>L_W</c> (mel-spectrogram loss, feature
/// matching, and MPD/MRD hinge GAN losses) needs a discriminator pair and an adversarial training
/// loop, which this vocoder has no path for; it is deliberately absent rather than approximated.
/// The STFT-consistency half of <c>L_S</c> is also absent for a concrete reason: it requires
/// <c>STFT(ISTFT(S))</c>, and <c>IEngine.STFT</c> returns its magnitude and phase through
/// <c>out</c> parameters, placing it outside the autodiff graph. The L1 real/imaginary half of
/// <c>L_S</c> is implemented in full.
/// </para>
/// <para><b>For Beginners:</b> A sound can be described by how loud each frequency is (amplitude)
/// and where each frequency's wave starts (phase). Phase wraps around like a clock face, so 11
/// o'clock and 1 o'clock are two hours apart, not ten. This objective measures phase errors the
/// short way round the clock, which is what lets the model learn phase at all.</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
public sealed class APNet2GeneratorLoss<T> : LossFunctionBase<T>
{
    private const double TwoPi = 2.0 * Math.PI;
    private const double HalfPi = Math.PI / 2.0;

    private readonly double _lambdaAmplitude;
    private readonly double _lambdaPhase;
    private readonly double _lambdaStft;
    private readonly double _lambdaMel;
    private readonly double _phaseEpsilon;
    private readonly int _fftSize;
    private readonly int _hopSize;
    private readonly int _sampleRate;
    private readonly int _melChannels;

    // Constant bases, built once. They carry no gradient, and rebuilding a [nFft, bins] pair on
    // every training step would cost more than the loss itself.
    private Tensor<T>? _window;
    private Tensor<T>? _dftCos;
    private Tensor<T>? _dftSin;
    private Tensor<T>? _melTransposed;

    /// <summary>Guards the paired publication of <see cref="_dftCos"/> and <see cref="_dftSin"/>.</summary>
    private readonly object _dftBasisLock = new();

    private int _melTransposedBins = -1;

    /// <summary>Creates the generator objective.</summary>
    /// <param name="lambdaAmplitude">Weight of the amplitude-spectrum loss. Paper value 45.</param>
    /// <param name="lambdaPhase">Weight of the anti-wrapping phase loss. Paper value 100.</param>
    /// <param name="lambdaStft">Weight of the reconstructed-STFT loss. Paper value 20.</param>
    /// <param name="lambdaMel">
    /// Weight of the mel-spectrogram loss inside <c>L_W</c>. Paper value 45.
    /// </param>
    /// <param name="fftSize">FFT size, needed to rebuild the waveform for the consistency term.</param>
    /// <param name="hopSize">Hop length between frames.</param>
    /// <param name="sampleRate">Sample rate, for the mel filterbank.</param>
    /// <param name="melChannels">Number of mel bands.</param>
    /// <param name="phaseEpsilon">
    /// Floor applied to the magnitude of the pseudo-real component before dividing by it, so a bin
    /// that lands exactly on the imaginary axis yields a finite gradient instead of an infinity.
    /// </param>
    public APNet2GeneratorLoss(
        double lambdaAmplitude = 45.0,
        double lambdaPhase = 100.0,
        double lambdaStft = 20.0,
        double lambdaMel = 45.0,
        int fftSize = 1024,
        int hopSize = 256,
        int sampleRate = 22050,
        int melChannels = 80,
        double phaseEpsilon = 1e-7)
    {
        _lambdaAmplitude = lambdaAmplitude;
        _lambdaPhase = lambdaPhase;
        _lambdaStft = lambdaStft;
        _lambdaMel = lambdaMel;
        _fftSize = fftSize;
        _hopSize = hopSize;
        _sampleRate = sampleRate;
        _melChannels = melChannels;
        _phaseEpsilon = phaseEpsilon;
    }

    /// <summary>Weight of the mel-spectrogram loss.</summary>
    public double LambdaMel => _lambdaMel;

    /// <summary>Weight of the amplitude-spectrum loss.</summary>
    public double LambdaAmplitude => _lambdaAmplitude;

    /// <summary>Weight of the anti-wrapping phase loss.</summary>
    public double LambdaPhase => _lambdaPhase;

    /// <summary>Weight of the reconstructed-STFT loss.</summary>
    public double LambdaStft => _lambdaStft;

    /// <inheritdoc/>
    /// <remarks>
    /// Both tensors carry APNet2's dual-branch output along their last axis, as
    /// <c>[log-amplitude | pseudo-real | pseudo-imaginary]</c>, each block <c>fftBins</c> wide.
    /// That is the layout <c>APNet2.ForwardDualBranch</c> produces.
    /// </remarks>
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        int bins = BlockWidth(predicted, nameof(predicted));
        if (BlockWidth(target, nameof(target)) != bins)
        {
            throw new ArgumentException(
                $"APNet2's objective needs prediction and target to describe the same number of " +
                $"frequency bins; got {bins} and {BlockWidth(target, nameof(target))}.",
                nameof(target));
        }

        int last = predicted.Rank - 1;

        var logAmplitude = Engine.TensorNarrow(predicted, last, 0, bins);
        var real = Engine.TensorNarrow(predicted, last, bins, bins);
        var imaginary = Engine.TensorNarrow(predicted, last, 2 * bins, bins);

        var logAmplitudeTarget = Engine.TensorNarrow(target, last, 0, bins);
        var realTarget = Engine.TensorNarrow(target, last, bins, bins);
        var imaginaryTarget = Engine.TensorNarrow(target, last, 2 * bins, bins);

        var amplitudeLoss = AmplitudeLoss(logAmplitude, logAmplitudeTarget);
        var phaseLoss = PhaseLoss(real, imaginary, realTarget, imaginaryTarget, last);
        var stftLoss = StftLoss(
            logAmplitude, real, imaginary,
            logAmplitudeTarget, realTarget, imaginaryTarget);

        var melLoss = MelLoss(logAmplitude, logAmplitudeTarget);

        var total = Engine.TensorAdd(
            Engine.TensorAdd(Scale(amplitudeLoss, _lambdaAmplitude), Scale(phaseLoss, _lambdaPhase)),
            Scale(stftLoss, _lambdaStft));

        return Engine.TensorAdd(total, Scale(melLoss, _lambdaMel));
    }

    /// <summary>L_A: squared error between predicted and natural log-amplitude spectra.</summary>
    private Tensor<T> AmplitudeLoss(Tensor<T> logAmplitude, Tensor<T> logAmplitudeTarget)
    {
        var difference = Engine.TensorSubtract(logAmplitude, logAmplitudeTarget);
        return Mean(Engine.TensorMultiply(difference, difference));
    }

    /// <summary>
    /// L_P: the sum of instantaneous-phase, group-delay and phase-time-difference losses, each
    /// passed through the anti-wrapping function.
    /// </summary>
    /// <remarks>
    /// Group delay differences run along the frequency axis and phase-time differences along the
    /// frame axis, per the paper. A tensor with a single frame or a single bin has no differences
    /// to take along that axis, so the corresponding term is skipped rather than fabricated.
    /// </remarks>
    private Tensor<T> PhaseLoss(
        Tensor<T> real, Tensor<T> imaginary,
        Tensor<T> realTarget, Tensor<T> imaginaryTarget,
        int frequencyAxis)
    {
        var phase = Phase(real, imaginary);
        var phaseTarget = Phase(realTarget, imaginaryTarget);

        // L_IP: direct phase error.
        var loss = MeanAbs(AntiWrap(Engine.TensorSubtract(phase, phaseTarget)));

        // L_GD: along frequency.
        int bins = phase.Shape[frequencyAxis];
        if (bins > 1)
        {
            var groupDelay = Difference(phase, frequencyAxis);
            var groupDelayTarget = Difference(phaseTarget, frequencyAxis);
            loss = Engine.TensorAdd(
                loss,
                MeanAbs(AntiWrap(Engine.TensorSubtract(groupDelay, groupDelayTarget))));
        }

        // L_PTD: along time. Frames are the axis immediately before the frequency axis.
        int timeAxis = frequencyAxis - 1;
        if (timeAxis >= 0 && phase.Shape[timeAxis] > 1)
        {
            var timeDifference = Difference(phase, timeAxis);
            var timeDifferenceTarget = Difference(phaseTarget, timeAxis);
            loss = Engine.TensorAdd(
                loss,
                MeanAbs(AntiWrap(Engine.TensorSubtract(timeDifference, timeDifferenceTarget))));
        }

        return loss;
    }

    /// <summary>
    /// L_S: L1 distance between the predicted and natural reconstructed STFT spectra.
    /// </summary>
    /// <remarks>
    /// The complex spectrum is rebuilt from the branches as <c>S = exp(logA) * (R + jI)</c>. The
    /// pseudo-real/imaginary pair is unit-normalised by the model, so <c>R</c> and <c>I</c> are
    /// already <c>cos(phi)</c> and <c>sin(phi)</c> and no trigonometry is needed here.
    /// </remarks>
    private Tensor<T> StftLoss(
        Tensor<T> logAmplitude, Tensor<T> real, Tensor<T> imaginary,
        Tensor<T> logAmplitudeTarget, Tensor<T> realTarget, Tensor<T> imaginaryTarget)
    {
        var amplitude = Engine.TensorExp(logAmplitude);
        var amplitudeTarget = Engine.TensorExp(logAmplitudeTarget);

        var realPart = Engine.TensorMultiply(amplitude, real);
        var imaginaryPart = Engine.TensorMultiply(amplitude, imaginary);
        var realPartTarget = Engine.TensorMultiply(amplitudeTarget, realTarget);
        var imaginaryPartTarget = Engine.TensorMultiply(amplitudeTarget, imaginaryTarget);

        var l1 = Engine.TensorAdd(
            MeanAbs(Engine.TensorSubtract(realPart, realPartTarget)),
            MeanAbs(Engine.TensorSubtract(imaginaryPart, imaginaryPartTarget)));

        var consistency = ConsistencyLoss(amplitude, real, imaginary, realPart, imaginaryPart);
        return consistency is null ? l1 : Engine.TensorAdd(l1, consistency);
    }

    /// <summary>
    /// The STFT-consistency half of <c>L_S</c>: how far the predicted spectrum is from being the
    /// STFT of an actual signal.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A neural network can emit an amplitude/phase pair that no waveform could ever produce,
    /// because neighbouring frames overlap and therefore constrain one another. The paper measures
    /// that by resynthesising and re-analysing: <c>||S - STFT(ISTFT(S))||</c>.
    /// </para>
    /// <para>
    /// <c>IEngine.ISTFT</c> returns a tensor and so stays on the tape, but <c>IEngine.STFT</c>
    /// reports its results through <c>out</c> parameters and would detach the graph. The forward
    /// analysis is therefore done explicitly here as what it mathematically is — a framing, a
    /// windowing and a projection onto the DFT basis — all of which are ordinary tape-tracked
    /// tensor operations.
    /// </para>
    /// </remarks>
    private Tensor<T>? ConsistencyLoss(
        Tensor<T> amplitude, Tensor<T> real, Tensor<T> imaginary,
        Tensor<T> realPart, Tensor<T> imaginaryPart)
    {
        int bins = amplitude.Shape[amplitude.Rank - 1];
        if (amplitude.Rank != 2 || bins != (_fftSize / 2) + 1 || _hopSize <= 0) return null;

        var window = Window();
        var phase = Phase(real, imaginary);

        var waveform = Engine.ISTFT(
            amplitude, phase, _fftSize, _hopSize, window, center: false, length: null);

        var analysed = ForwardStft(waveform, window, amplitude.Shape[0], bins);
        if (analysed is null) return null;

        return Engine.TensorAdd(
            MeanAbs(Engine.TensorSubtract(realPart, analysed.Value.Real)),
            MeanAbs(Engine.TensorSubtract(imaginaryPart, analysed.Value.Imaginary)));
    }

    /// <summary>
    /// Frames, windows and projects a waveform onto the DFT basis, entirely with tape-tracked ops.
    /// </summary>
    private (Tensor<T> Real, Tensor<T> Imaginary)? ForwardStft(
        Tensor<T> waveform, Tensor<T> window, int expectedFrames, int bins)
    {
        int samples = waveform.Shape[waveform.Rank - 1];
        if (samples < _fftSize) return null;

        int frames = 1 + ((samples - _fftSize) / _hopSize);
        if (frames <= 0) return null;

        var windowed = new Tensor<T>[frames];
        for (int f = 0; f < frames; f++)
        {
            var frame = Engine.TensorNarrow(waveform, waveform.Rank - 1, f * _hopSize, _fftSize);
            windowed[f] = Engine.TensorMultiply(frame, window);
        }

        var stacked = frames == 1
            ? Engine.Reshape(windowed[0], new[] { 1, _fftSize })
            : Engine.TensorStack(windowed, axis: 0);

        EnsureDftBasis(bins);

        // real = x . cos, imaginary = -x . sin  (the standard analysis convention).
        var realOut = Engine.TensorMatMul(stacked, _dftCos!);
        var imaginaryOut = Engine.TensorNegate(Engine.TensorMatMul(stacked, _dftSin!));

        if (realOut.Shape[0] != expectedFrames) return null;

        return (realOut, imaginaryOut);
    }

    /// <summary>Builds the periodic Hann window once.</summary>
    private Tensor<T> Window()
    {
        if (_window is not null) return _window;

        var window = new Tensor<T>(new[] { _fftSize });
        for (int n = 0; n < _fftSize; n++)
        {
            double value = 0.5 - (0.5 * Math.Cos(TwoPi * n / _fftSize));
            window[n] = NumOps.FromDouble(value);
        }

        _window = window;
        return window;
    }

    /// <summary>Builds the real and imaginary DFT bases once, shaped <c>[nFft, bins]</c>.</summary>
    /// <remarks>
    /// BOTH BASES ARE PUBLISHED TOGETHER, under a lock. The test read only <c>_dftCos</c> and the
    /// assignments ran in order, so a second thread calling ComputeTapeLoss on the same instance
    /// could observe <c>_dftCos</c> non-null with a matching bin count while <c>_dftSin</c> was
    /// still null, and then dereference it. This is the only field pair here that can tear --
    /// <c>_window</c> and <c>_melTransposed</c> share the unsynchronized-lazy pattern but each is a
    /// single field, so a reader sees either the old value or the new one.
    /// </remarks>
    private void EnsureDftBasis(int bins)
    {
        lock (_dftBasisLock)
        {
            if (_dftCos is not null && _dftCos.Shape[1] == bins) return;
        }

        var cos = new Tensor<T>(new[] { _fftSize, bins });
        var sin = new Tensor<T>(new[] { _fftSize, bins });
        for (int n = 0; n < _fftSize; n++)
        {
            for (int k = 0; k < bins; k++)
            {
                double angle = TwoPi * n * k / _fftSize;
                cos[(n * bins) + k] = NumOps.FromDouble(Math.Cos(angle));
                sin[(n * bins) + k] = NumOps.FromDouble(Math.Sin(angle));
            }
        }

        lock (_dftBasisLock)
        {
            // Another thread may have finished the same size while this one was building. Either
            // pair is equally valid; publishing both together is what matters.
            _dftCos = cos;
            _dftSin = sin;
        }
    }

    /// <summary>
    /// The mel-spectrogram loss from <c>L_W</c>: L1 distance between the log-mel spectrograms of
    /// the predicted and natural amplitude spectra.
    /// </summary>
    /// <remarks>
    /// <para>
    /// No STFT is involved. The mel spectrogram is a fixed linear projection of the amplitude
    /// spectrum, and this model predicts that spectrum directly, so the term is
    /// <c>melFilterbank * exp(logA)</c> — a matrix multiply against a constant, which the tape
    /// differentiates exactly. Going via a waveform round-trip would add error and cost for a
    /// value already in hand.
    /// </para>
    /// </remarks>
    private Tensor<T> MelLoss(Tensor<T> logAmplitude, Tensor<T> logAmplitudeTarget)
    {
        int bins = logAmplitude.Shape[logAmplitude.Rank - 1];
        var filterbankTransposed = MelFilterbankTransposed(bins);
        if (filterbankTransposed is null) return Const(Zero(), 0.0);

        var mel = Engine.TensorMatMul(Engine.TensorExp(logAmplitude), filterbankTransposed);
        var melTarget = Engine.TensorMatMul(Engine.TensorExp(logAmplitudeTarget), filterbankTransposed);

        var floor = NumOps.FromDouble(1e-5);
        var logMel = Engine.TensorLog(Engine.TensorAddScalar(mel, floor));
        var logMelTarget = Engine.TensorLog(Engine.TensorAddScalar(melTarget, floor));

        return MeanAbs(Engine.TensorSubtract(logMel, logMelTarget));
    }

    /// <summary>
    /// The mel filterbank as <c>[bins, nMels]</c>, ready to right-multiply a <c>[frames, bins]</c>
    /// amplitude spectrum.
    /// </summary>
    /// <remarks>
    /// Transposed by hand rather than with a tensor op because the filterbank is a constant: it
    /// carries no gradient, so keeping it off the tape avoids recording a transpose on every step.
    /// Returns <c>null</c> when the spectrum's bin count does not match the configured FFT size,
    /// in which case the mel term is skipped rather than computed against a mismatched basis.
    /// </remarks>
    private Tensor<T>? MelFilterbankTransposed(int bins)
    {
        if (_melChannels <= 0 || bins != (_fftSize / 2) + 1) return null;
        if (_melTransposed is not null && _melTransposedBins == bins) return _melTransposed;

        var filterbank = Engine.CreateMelFilterbank(
            _melChannels, _fftSize, _sampleRate, NumOps.Zero, NumOps.FromDouble(_sampleRate / 2.0));

        var transposed = new Tensor<T>(new[] { bins, _melChannels });
        for (int m = 0; m < _melChannels; m++)
        {
            for (int b = 0; b < bins; b++)
            {
                transposed[(b * _melChannels) + m] = filterbank[(m * bins) + b];
            }
        }

        _melTransposed = transposed;
        _melTransposedBins = bins;
        return transposed;
    }

    /// <summary>A scalar-shaped tensor, used when a term does not apply.</summary>
    private Tensor<T> Zero() => new Tensor<T>(new[] { 1 });

    /// <summary>
    /// The paper's phase computation:
    /// <c>Phi(R, I) = arctan(I / R) - (pi/2) * Sgn*(I) * [Sgn*(R) - 1]</c>.
    /// </summary>
    /// <remarks>
    /// The correction term is what turns a two-quadrant arctangent into the four-quadrant principal
    /// value: it is zero when <c>R &gt;= 0</c> and adds <c>+/- pi</c> according to the sign of
    /// <c>I</c> when <c>R &lt; 0</c>.
    /// </remarks>
    private Tensor<T> Phase(Tensor<T> real, Tensor<T> imaginary)
    {
        var signReal = SignStar(real);
        var signImaginary = SignStar(imaginary);

        // Keep |R| away from zero so the quotient stays finite, without moving R across zero.
        var safeReal = Engine.TensorAdd(real, Scale(signReal, _phaseEpsilon));

        var principal = Atan(Engine.TensorDivide(imaginary, safeReal));

        var correction = Scale(
            Engine.TensorMultiply(
                signImaginary,
                Engine.TensorSubtract(signReal, Const(signReal, 1.0))),
            HalfPi);

        return Engine.TensorSubtract(principal, correction);
    }

    /// <summary>The paper's anti-wrapping function <c>AW(x) = x - 2*pi*round(x / (2*pi))</c>.</summary>
    /// <remarks>
    /// Rounding has zero derivative almost everywhere, which is exactly the required behaviour: the
    /// <c>2*pi*k</c> offset is a constant shift, so <c>d/dx AW(x) = 1</c> and the gradient is that
    /// of the unwrapped error.
    /// </remarks>
    private Tensor<T> AntiWrap(Tensor<T> x)
    {
        var turns = Engine.TensorRound(Scale(x, 1.0 / TwoPi));
        return Engine.TensorSubtract(x, Scale(turns, TwoPi));
    }

    /// <summary>
    /// <c>Sgn*(x)</c>: the paper's sign function, which returns 1 at zero rather than 0.
    /// </summary>
    private Tensor<T> SignStar(Tensor<T> x)
    {
        var sign = Engine.TensorSign(x);
        // sign + (1 - |sign|) maps 0 to 1 and leaves +/-1 untouched.
        return Engine.TensorAdd(
            sign,
            Engine.TensorSubtract(Const(sign, 1.0), Engine.TensorAbs(sign)));
    }

    /// <summary>
    /// Elementwise arctangent, composed only of tape-tracked tensor operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <c>AiDotNet.Tensors</c> exposes <c>Atan</c> for <c>Vector&lt;T&gt;</c> only, which is outside
    /// the autodiff graph, so using it here would silently sever the gradient. This is the standard
    /// range-reduced minimax evaluation instead: fold the argument into <c>[0, 1]</c> using
    /// <c>atan(x) = pi/2 - atan(1/x)</c> for <c>|x| &gt; 1</c>, apply a degree-9 odd polynomial, and
    /// restore the sign. This is the classic Hastings degree-9 polynomial, whose maximum absolute
    /// error on the reduced range is about 1.15e-5 -- NOT the 1e-7 previously claimed here. Raise the
    /// degree if a run needs tighter phase precision than that. Every step is an <c>IEngine</c> op,
    /// so the tape records the whole thing.
    /// </para>
    /// </remarks>
    private Tensor<T> Atan(Tensor<T> x)
    {
        var sign = Engine.TensorSign(x);
        var magnitude = Engine.TensorAbs(x);

        var one = Const(x, 1.0);
        var isLarge = Engine.TensorGreaterThan(magnitude, one);

        // Reciprocal of a value floored at 1, so the unused branch can never divide by zero.
        var reciprocal = Engine.TensorReciprocal(Engine.TensorClampMin(magnitude, NumOps.One));
        var reduced = Engine.TensorWhere(isLarge, reciprocal, magnitude);

        var polynomial = AtanUnitInterval(reduced);

        // atan(|x|) = poly(r) when |x| <= 1, and pi/2 - poly(1/|x|) otherwise.
        var large = Engine.TensorSubtract(Const(x, HalfPi), polynomial);
        var magnitudeAtan = Engine.TensorWhere(isLarge, large, polynomial);

        return Engine.TensorMultiply(sign, magnitudeAtan);
    }

    /// <summary>Minimax odd polynomial for <c>atan</c> on <c>[0, 1]</c> (Hastings).</summary>
    private Tensor<T> AtanUnitInterval(Tensor<T> r)
    {
        var rSquared = Engine.TensorMultiply(r, r);

        // Horner: r * (c1 + s*(c3 + s*(c5 + s*(c7 + s*c9))), s = r^2.
        var accumulator = Const(r, 0.0208351);
        accumulator = Engine.TensorAdd(Engine.TensorMultiply(accumulator, rSquared), Const(r, -0.0851330));
        accumulator = Engine.TensorAdd(Engine.TensorMultiply(accumulator, rSquared), Const(r, 0.1801410));
        accumulator = Engine.TensorAdd(Engine.TensorMultiply(accumulator, rSquared), Const(r, -0.3302995));
        accumulator = Engine.TensorAdd(Engine.TensorMultiply(accumulator, rSquared), Const(r, 0.9998660));

        return Engine.TensorMultiply(r, accumulator);
    }

    /// <summary>First difference along <paramref name="axis"/>.</summary>
    private Tensor<T> Difference(Tensor<T> x, int axis)
    {
        int length = x.Shape[axis];
        var ahead = Engine.TensorNarrow(x, axis, 1, length - 1);
        var behind = Engine.TensorNarrow(x, axis, 0, length - 1);
        return Engine.TensorSubtract(ahead, behind);
    }

    private Tensor<T> MeanAbs(Tensor<T> x) => Mean(Engine.TensorAbs(x));

    /// <summary>Mean over every axis, staying on the tape.</summary>
    /// <remarks>
    /// <c>Engine.TensorMean</c> returns a bare <c>T</c>, which would detach the result from the
    /// autodiff graph and silently zero the gradient; <c>ReduceMean</c> returns a tensor the tape
    /// still owns.
    /// </remarks>
    private Tensor<T> Mean(Tensor<T> x)
    {
        var allAxes = Enumerable.Range(0, x.Shape.Length).ToArray();
        return Engine.ReduceMean(x, allAxes, keepDims: false);
    }

    /// <summary>A constant tensor shaped like <paramref name="like"/>, staying on the tape.</summary>
    private Tensor<T> Const(Tensor<T> like, double value)
    {
        var constant = new Tensor<T>(like.Shape.ToArray());
        var scalar = NumOps.FromDouble(value);
        for (int i = 0; i < constant.Length; i++) constant[i] = scalar;
        return constant;
    }

    private Tensor<T> Scale(Tensor<T> x, double weight)
        => weight == 1.0 ? x : Engine.TensorMultiply(x, Const(x, weight));

    private static int BlockWidth(Tensor<T> tensor, string name)
    {
        if (tensor.Rank < 1)
        {
            throw new ArgumentException(
                $"APNet2's objective needs the dual-branch output, got rank {tensor.Rank}.", name);
        }

        int width = tensor.Shape[tensor.Rank - 1];
        if (width % 3 != 0)
        {
            throw new ArgumentException(
                $"APNet2's objective expects {name}'s last axis to hold log-amplitude, " +
                $"pseudo-real and pseudo-imaginary blocks of equal width, so it must be divisible " +
                $"by 3; got {width}.",
                name);
        }

        return width / 3;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Evaluates the same objective without the tape, for callers that only need the scalar value.
    /// </remarks>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        var predictedTensor = Tensor<T>.FromVector(predicted);
        var actualTensor = Tensor<T>.FromVector(actual);
        var loss = ComputeTapeLoss(predictedTensor, actualTensor);
        return loss.Length > 0 ? loss[loss.Length - 1] : NumOps.Zero;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Not available: this objective is differentiated by the tape, which records the anti-wrapping
    /// phase computation as it runs. Returning a hand-rolled gradient here would mean maintaining a
    /// second derivation of the same maths that could silently drift from the forward one, so the
    /// tape path is the only supported route.
    /// </remarks>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
        => throw new NotSupportedException(
            "APNet2GeneratorLoss is differentiated through the autodiff tape; call ComputeTapeLoss " +
            "via TrainWithTape rather than requesting an explicit derivative vector.");
}
