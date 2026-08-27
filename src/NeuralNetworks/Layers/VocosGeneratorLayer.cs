using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// The native Vocos generator: an isotropic ConvNeXt backbone followed by a differentiable
/// complex-STFT head and inverse STFT synthesis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This implements the architecture from Siuzdak, "Vocos: Closing the Gap between Time-Domain
/// and Fourier-Based Neural Vocoders for High-Quality Audio Synthesis" (ICLR 2024). Mel frames
/// stay at their original temporal resolution throughout the ConvNeXt backbone. The final linear
/// projection predicts log magnitude and phase in parallel; a Hann-window inverse STFT produces
/// the waveform without transposed-convolution upsampling.
/// </para>
/// <para>
/// Input is <c>[batch, mel, frames]</c> or <c>[mel, frames]</c>. Output is
/// <c>[batch, frames * hopLength]</c> or <c>[frames * hopLength]</c>, respectively. All tensor
/// transformations, including ISTFT, go through the engine so the gradient tape sees the complete
/// mel-to-waveform graph.
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, Cost = ComputeCost.High,
    TestInputShape = "1, 8, 4", TestConstructorArgs = "8, 16, 2, 32, 16, 4")]
// Roles and ranks from this layer's own guard in ForwardTraced - "requires [mel, frames] or
// [batch, mel, frames] input" - which throws for every other rank. The mel axis is named Channels
// rather than Features because that is what the code itself calls it: it is the input channel count of
// _inputEmbedding (a Conv1DLayer(numMels -> hiddenDim)), and the line after that forward is commented
// "// [B, C, T]". Naming it Channels also makes the hand-off from a mel front-end legible.
//
// THE RANK DROPS BY ONE. This layer is a vocoder: it consumes a time-frequency map and emits a
// WAVEFORM, so the mel axis does not survive - [B, mel, frames] leaves as [B, samples]. One output
// declaration with BatchOptional covers both the rank-2 and rank-1 results, mirroring the input pair.
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Time,
    BatchOptional = true, Direction = TensorLayoutDirection.Input,
    Note = "Mel spectrogram: the channel axis is the mel filterbank, the last axis is frames.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time,
    BatchOptional = true, Direction = TensorLayoutDirection.Output,
    Note = "Raw waveform samples; the mel axis is consumed by the Fourier head and inverse STFT.")]
[AutoParameters]
public partial class VocosGeneratorLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Read off the two lines that close <c>ForwardTraced</c>:
    /// <c>int waveformLength = frames * _hopLength;</c> then
    /// <c>Reshape(waveform, new[] { waveformLength })</c> for the unbatched case, or the batched
    /// <c>waveform</c> ([B, frames * hopLength]) as it stands. So the single surviving non-batch axis is
    /// the frame count multiplied by the hop length - <c>Scaled(Time, _hopLength)</c>, with the factor
    /// read off the constructor argument.
    /// </para>
    /// <para>
    /// <c>Scaled</c> and not <c>Window</c>: overlap-add synthesis advances by exactly one hop per frame,
    /// and nothing here rounds. The class remarks state the same relation directly - "Output is
    /// [batch, frames * hopLength] or [frames * hopLength]".
    /// </para>
    /// <para>
    /// THE MEL AXIS HAS NO OUTPUT COUNTERPART, which is why this is hand-written and why the returned
    /// list is one shorter than the input rank. Its width is not resized, it is CONSUMED: the backbone
    /// has already replaced it with <c>_hiddenDim</c> at the very first convolution, the Fourier head
    /// projects that to <c>n_fft + 2</c> magnitude/phase coefficients, and the inverse STFT turns those
    /// into samples along time. Declaring any relation for it would invent an axis the output does not
    /// have.
    /// </para>
    /// <para>
    /// Note also that <c>_nFft</c> does NOT appear in the relation, which is easy to expect and wrong.
    /// It sizes the analysis window and therefore the per-frame overlap, but the same-padded ISTFT
    /// trims back to one hop per frame, so the FFT size cancels out of the length entirely.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank is not (2 or 3) || _hopLength <= 0) return null;

        var samples = new OutputAxisContract(
            TensorAxis.Time, AxisRelation.Scaled(TensorAxis.Time, _hopLength));

        return inputRank == 2
            ? new[] { samples }
            : new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                samples,
            };
    }

    private readonly int _numMels;
    private readonly int _hiddenDim;
    private readonly int _numBackboneBlocks;
    private readonly int _intermediateDim;
    private readonly int _nFft;
    private readonly int _hopLength;

    private readonly Conv1DLayer<T> _inputEmbedding;
    private readonly LayerNormalizationLayer<T> _inputNormalization;
    private readonly DepthwiseConv1DLayer<T>[] _depthwiseConvolutions;
    private readonly LayerNormalizationLayer<T>[] _blockNormalizations;
    private readonly FullyConnectedLayer<T>[] _blockExpansions;
    private readonly FullyConnectedLayer<T>[] _blockProjections;
    private readonly Tensor<T>[] _layerScales;
    private readonly LayerNormalizationLayer<T> _outputNormalization;
    private readonly FullyConnectedLayer<T> _fourierProjection;
    private readonly Tensor<T> _window;
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Tensor<int> _inverseFftInteriorReverseIndices;
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Tensor<int> _inverseFftBitReverseIndices;
    private readonly Tensor<T>[] _inverseFftCosines;
    private readonly Tensor<T>[] _inverseFftSines;

    /// <summary>Creates a native Vocos generator.</summary>
    public VocosGeneratorLayer(
        int numMels = 100,
        int hiddenDim = 512,
        int numBackboneBlocks = 8,
        int intermediateDim = 1536,
        int nFft = 1024,
        int hopLength = 256)
        : base(new[] { numMels, -1 }, new[] { -1 })
    {
        if (numMels <= 0) throw new ArgumentOutOfRangeException(nameof(numMels));
        if (hiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenDim));
        if (numBackboneBlocks <= 0) throw new ArgumentOutOfRangeException(nameof(numBackboneBlocks));
        if (intermediateDim <= 0) throw new ArgumentOutOfRangeException(nameof(intermediateDim));
        if (nFft < 2 || (nFft & (nFft - 1)) != 0)
            throw new ArgumentOutOfRangeException(nameof(nFft), "FFT size must be a positive power of two.");
        if (hopLength <= 0 || hopLength > nFft) throw new ArgumentOutOfRangeException(nameof(hopLength));

        _numMels = numMels;
        _hiddenDim = hiddenDim;
        _numBackboneBlocks = numBackboneBlocks;
        _intermediateDim = intermediateDim;
        _nFft = nFft;
        _hopLength = hopLength;

        var identity = new IdentityActivation<T>();
        var gelu = new GELUActivation<T>();

        // Official Vocos: Conv1d(num_mels, dim, kernel_size=7, padding=3).
        _inputEmbedding = new Conv1DLayer<T>(numMels, hiddenDim, kernelSize: 7, padding: 3, activation: identity);
        // The reference backbone normalizes the embedded features before the first
        // ConvNeXt block, in addition to each block's own pre-MLP normalization.
        _inputNormalization = new LayerNormalizationLayer<T>(hiddenDim, epsilon: 1e-6);
        RegisterSubLayer(_inputEmbedding);
        RegisterSubLayer(_inputNormalization);

        _depthwiseConvolutions = new DepthwiseConv1DLayer<T>[numBackboneBlocks];
        _blockNormalizations = new LayerNormalizationLayer<T>[numBackboneBlocks];
        _blockExpansions = new FullyConnectedLayer<T>[numBackboneBlocks];
        _blockProjections = new FullyConnectedLayer<T>[numBackboneBlocks];
        _layerScales = new Tensor<T>[numBackboneBlocks];

        T initialScale = NumOps.FromDouble(1.0 / numBackboneBlocks);
        for (int i = 0; i < numBackboneBlocks; i++)
        {
            _depthwiseConvolutions[i] = new DepthwiseConv1DLayer<T>(hiddenDim, kernelSize: 7, padding: 3, activation: identity);
            _blockNormalizations[i] = new LayerNormalizationLayer<T>(hiddenDim, epsilon: 1e-6);
            _blockExpansions[i] = new FullyConnectedLayer<T>(hiddenDim, intermediateDim, gelu);
            _blockProjections[i] = new FullyConnectedLayer<T>(intermediateDim, hiddenDim, identity);
            _layerScales[i] = new Tensor<T>([hiddenDim]);
            _layerScales[i].Fill(initialScale);

            RegisterSubLayer(_depthwiseConvolutions[i]);
            RegisterSubLayer(_blockNormalizations[i]);
            RegisterSubLayer(_blockExpansions[i]);
            RegisterSubLayer(_blockProjections[i]);
            RegisterTrainableParameter(_layerScales[i], PersistentTensorRole.Weights);
        }

        _outputNormalization = new LayerNormalizationLayer<T>(hiddenDim, epsilon: 1e-6);
        _fourierProjection = new FullyConnectedLayer<T>(hiddenDim, nFft + 2, identity);
        RegisterSubLayer(_outputNormalization);
        RegisterSubLayer(_fourierProjection);

        _window = Engine.CreateWindow<T>("hann", nFft);
        RegisterBuffer(_window, "istft_hann_window");

        int interiorCount = nFft / 2 - 1;
        var reverseInterior = new int[interiorCount];
        for (int i = 0; i < interiorCount; i++) reverseInterior[i] = interiorCount - i;
        _inverseFftInteriorReverseIndices = new Tensor<int>(reverseInterior, [interiorCount]);

        // Math.Log2 is unavailable on net471; nFft is already validated as a power of two.
        int bits = (int)Math.Round(Math.Log(nFft, 2.0));
        var bitReverse = new int[nFft];
        for (int value = 0; value < nFft; value++)
        {
            int source = value;
            int reversed = 0;
            for (int bit = 0; bit < bits; bit++)
            {
                reversed = (reversed << 1) | (source & 1);
                source >>= 1;
            }
            bitReverse[value] = reversed;
        }
        _inverseFftBitReverseIndices = new Tensor<int>(bitReverse, [nFft]);

        _inverseFftCosines = new Tensor<T>[bits];
        _inverseFftSines = new Tensor<T>[bits];
        for (int stage = 0, length = 2; stage < bits; stage++, length *= 2)
        {
            int half = length / 2;
            _inverseFftCosines[stage] = new Tensor<T>([1, 1, 1, half]);
            _inverseFftSines[stage] = new Tensor<T>([1, 1, 1, half]);
            for (int j = 0; j < half; j++)
            {
                double angle = 2.0 * Math.PI * j / length;
                _inverseFftCosines[stage][j] = NumOps.FromDouble(Math.Cos(angle));
                _inverseFftSines[stage][j] = NumOps.FromDouble(Math.Sin(angle));
            }
            RegisterBuffer(_inverseFftCosines[stage], $"ifft_cos_{length}");
            RegisterBuffer(_inverseFftSines[stage], $"ifft_sin_{length}");
        }
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        bool unbatched = input.Shape.Length == 2;
        if (!unbatched && input.Shape.Length != 3)
        {
            throw new ArgumentException(
                $"VocosGeneratorLayer requires [mel, frames] or [batch, mel, frames] input; got rank {input.Shape.Length}.",
                nameof(input));
        }

        int melAxis = unbatched ? 0 : 1;
        if (input.Shape[melAxis] != _numMels)
        {
            throw new ArgumentException(
                $"VocosGeneratorLayer expected {_numMels} mel channels, got {input.Shape[melAxis]}.",
                nameof(input));
        }

        int batch = unbatched ? 1 : input.Shape[0];
        int frames = input.Shape[^1];
        var x = unbatched
            ? Engine.Reshape(input, new[] { 1, _numMels, frames })
            : input;

        x = _inputEmbedding.Forward(x); // [B, C, T]
        x = Engine.TensorPermute(x, new[] { 0, 2, 1 });
        x = _inputNormalization.Forward(x);
        x = Engine.TensorPermute(x, new[] { 0, 2, 1 });
        for (int i = 0; i < _numBackboneBlocks; i++)
        {
            // ConvNeXt block: DWConv7 -> channels-last LN -> Linear/GELU/Linear ->
            // learnable per-channel layer scale -> residual.
            var residual = x;
            var block = _depthwiseConvolutions[i].Forward(x);
            block = Engine.TensorPermute(block, new[] { 0, 2, 1 });
            block = _blockNormalizations[i].Forward(block);
            block = _blockExpansions[i].Forward(block);
            block = _blockProjections[i].Forward(block);
            block = Engine.TensorPermute(block, new[] { 0, 2, 1 });
            var scale = Engine.Reshape(_layerScales[i], new[] { 1, _hiddenDim, 1 });
            block = Engine.TensorMultiply(block, scale);
            x = Engine.TensorAdd(residual, block);
        }

        // [B,C,T] -> [B,T,C], exactly the official Vocos head layout.
        var features = Engine.TensorPermute(x, new[] { 0, 2, 1 });
        features = _outputNormalization.Forward(features);
        var coefficients = _fourierProjection.Forward(features); // [B,T,n_fft+2]

        int bins = _nFft / 2 + 1;
        var logMagnitude = Engine.TensorSlice(
            coefficients,
            new[] { 0, 0, 0 },
            new[] { batch, frames, bins });
        var phase = Engine.TensorSlice(
            coefficients,
            new[] { 0, 0, bins },
            new[] { batch, frames, bins });

        // The reference implementation exponentiates magnitude, caps it at 100, and wraps
        // phase via cos/sin. IEngine.ISTFT accepts the equivalent magnitude/phase polar form.
        var magnitude = Engine.TensorClamp(
            Engine.TensorExp(logMagnitude),
            NumOps.Zero,
            NumOps.FromDouble(100.0));

        var waveform = DifferentiableIstftSame(magnitude, phase, batch, frames);
        int waveformLength = frames * _hopLength;

        return unbatched
            ? Engine.Reshape(waveform, new[] { waveformLength })
            : waveform;
    }

    /// <summary>
    /// Differentiable equivalent of the reference implementation's custom same-padded ISTFT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Tensors 0.120.5 exposes <c>IEngine.ISTFT</c>, but that convenience op is intentionally
    /// classified as non-differentiable in its operation registry. Calling it would detach the
    /// entire Fourier head from the tape. Compose the same transform from radix-2 butterfly,
    /// Pad, Add, Slice, and broadcast operations instead. This is the same strategy PyTorch uses:
    /// complex synthesis and overlap-add remain ordinary graph operations.
    /// </para>
    /// </remarks>
    private Tensor<T> DifferentiableIstftSame(
        Tensor<T> magnitude,
        Tensor<T> phase,
        int batch,
        int frames)
    {
        var real = Engine.TensorMultiply(magnitude, Engine.TensorCos(phase));
        var imaginary = Engine.TensorMultiply(magnitude, Engine.TensorSin(phase));

        var timeFrames = DifferentiableInverseRealFft(real, imaginary, batch, frames);
        var window = Engine.Reshape(_window, new[] { 1, 1, _nFft });
        timeFrames = Engine.TensorMultiply(timeFrames, window);

        int paddedLength = (frames - 1) * _hopLength + _nFft;
        Tensor<T>? overlapAdded = null;
        for (int frame = 0; frame < frames; frame++)
        {
            var frameAudio = Engine.TensorSlice(
                timeFrames,
                new[] { 0, frame, 0 },
                new[] { batch, 1, _nFft });
            frameAudio = Engine.Reshape(frameAudio, new[] { batch, 1, 1, _nFft });
            int left = frame * _hopLength;
            int right = paddedLength - _nFft - left;
            var positioned = Engine.Pad(frameAudio, 0, 0, left, right, NumOps.Zero);
            overlapAdded = overlapAdded is null
                ? positioned
                : Engine.TensorAdd(overlapAdded, positioned);
        }

        if (overlapAdded is null)
            throw new InvalidOperationException("Vocos ISTFT requires at least one frame.");

        // Normalize by the Hann-window squared envelope. This tensor is a fixed coefficient,
        // independent of the trainable graph; multiplying by it preserves every upstream gradient.
        var inverseEnvelope = new Tensor<T>([1, 1, 1, paddedLength]);
        var envelope = new double[paddedLength];
        for (int frame = 0; frame < frames; frame++)
        {
            int offset = frame * _hopLength;
            for (int sample = 0; sample < _nFft; sample++)
            {
                double w = NumOps.ToDouble(_window[sample]);
                envelope[offset + sample] += w * w;
            }
        }
        for (int sample = 0; sample < paddedLength; sample++)
        {
            inverseEnvelope[sample] = NumOps.FromDouble(
                envelope[sample] > 1e-8 ? 1.0 / envelope[sample] : 0.0);
        }
        var normalized = Engine.TensorMultiply(overlapAdded, inverseEnvelope);

        int samePadding = (_nFft - _hopLength) / 2;
        int waveformLength = frames * _hopLength;
        var trimmed = Engine.TensorSlice(
            normalized,
            new[] { 0, 0, 0, samePadding },
            new[] { batch, 1, 1, waveformLength });
        return Engine.Reshape(trimmed, new[] { batch, waveformLength });
    }

    /// <summary>
    /// Reconstructs the Hermitian spectrum and evaluates a radix-2 inverse FFT entirely from
    /// tape-recorded tensor primitives.
    /// </summary>
    /// <remarks>
    /// Tensors 0.120.5 records IRFFT, but its backward currently applies an unscaled RFFT and omits
    /// the one-sided spectrum's endpoint/interior adjoint weights. That makes analytical gradients
    /// disagree with the actual inverse transform. Expressing the butterflies directly is O(N log N),
    /// remains accelerator-compatible, and gives autodiff the exact derivative of the forward graph.
    /// </remarks>
    private Tensor<T> DifferentiableInverseRealFft(
        Tensor<T> positiveReal,
        Tensor<T> positiveImaginary,
        int batch,
        int frames)
    {
        var reversedReal = Engine.TensorGather(
            positiveReal, _inverseFftInteriorReverseIndices, axis: 2);
        var reversedImaginary = Engine.TensorMultiplyScalar(
            Engine.TensorGather(positiveImaginary, _inverseFftInteriorReverseIndices, axis: 2),
            NumOps.FromDouble(-1.0));

        var real = Engine.TensorConcatenate(new[] { positiveReal, reversedReal }, axis: 2);
        var imaginary = Engine.TensorConcatenate(new[] { positiveImaginary, reversedImaginary }, axis: 2);
        real = Engine.TensorGather(real, _inverseFftBitReverseIndices, axis: 2);
        imaginary = Engine.TensorGather(imaginary, _inverseFftBitReverseIndices, axis: 2);

        for (int stage = 0, length = 2; stage < _inverseFftCosines.Length; stage++, length *= 2)
        {
            int half = length / 2;
            int blocks = _nFft / length;
            var realBlocks = Engine.Reshape(real, new[] { batch, frames, blocks, length });
            var imaginaryBlocks = Engine.Reshape(imaginary, new[] { batch, frames, blocks, length });
            var evenReal = Engine.TensorSlice(
                realBlocks, new[] { 0, 0, 0, 0 }, new[] { batch, frames, blocks, half });
            var oddReal = Engine.TensorSlice(
                realBlocks, new[] { 0, 0, 0, half }, new[] { batch, frames, blocks, half });
            var evenImaginary = Engine.TensorSlice(
                imaginaryBlocks, new[] { 0, 0, 0, 0 }, new[] { batch, frames, blocks, half });
            var oddImaginary = Engine.TensorSlice(
                imaginaryBlocks, new[] { 0, 0, 0, half }, new[] { batch, frames, blocks, half });

            var twiddledReal = Engine.TensorSubtract(
                Engine.TensorMultiply(oddReal, _inverseFftCosines[stage]),
                Engine.TensorMultiply(oddImaginary, _inverseFftSines[stage]));
            var twiddledImaginary = Engine.TensorAdd(
                Engine.TensorMultiply(oddReal, _inverseFftSines[stage]),
                Engine.TensorMultiply(oddImaginary, _inverseFftCosines[stage]));

            real = Engine.Reshape(
                Engine.TensorConcatenate(
                    new[]
                    {
                        Engine.TensorAdd(evenReal, twiddledReal),
                        Engine.TensorSubtract(evenReal, twiddledReal)
                    },
                    axis: 3),
                new[] { batch, frames, _nFft });
            imaginary = Engine.Reshape(
                Engine.TensorConcatenate(
                    new[]
                    {
                        Engine.TensorAdd(evenImaginary, twiddledImaginary),
                        Engine.TensorSubtract(evenImaginary, twiddledImaginary)
                    },
                    axis: 3),
                new[] { batch, frames, _nFft });
        }

        return Engine.TensorMultiplyScalar(real, NumOps.FromDouble(1.0 / _nFft));
    }

    private IEnumerable<(ILayer<T>? Layer, Tensor<T>? Scale)> OrderedParameterParts()
    {
        yield return (_inputEmbedding, null);
        yield return (_inputNormalization, null);
        for (int i = 0; i < _numBackboneBlocks; i++)
        {
            yield return (_depthwiseConvolutions[i], null);
            yield return (_blockNormalizations[i], null);
            yield return (_blockExpansions[i], null);
            yield return (_blockProjections[i], null);
            yield return (null, _layerScales[i]);
        }
        yield return (_outputNormalization, null);
        yield return (_fourierProjection, null);
    }

    /// <inheritdoc/>
    public override IReadOnlyList<Tensor<T>> GetTrainableParameters() => _layerScales;

    /// <inheritdoc/>
    /// <remarks>
    /// Copy-on-write cloning and contiguous parameter buffers replace tensor objects rather
    /// than copying values. Rebind the array consumed by <see cref="Forward"/> as well as the
    /// base registration list; otherwise a trained clone keeps forwarding with its freshly
    /// initialized layer scales even though the framework supplied the trained tensors.
    /// </remarks>
    public override void SetTrainableParameters(IReadOnlyList<Tensor<T>> parameters)
    {
        if (parameters.Count != _layerScales.Length)
        {
            throw new ArgumentException(
                $"Expected {_layerScales.Length} Vocos layer-scale tensors, got {parameters.Count}.",
                nameof(parameters));
        }

        for (int i = 0; i < parameters.Count; i++)
        {
            if (parameters[i].Rank != 1 || parameters[i].Length != _hiddenDim)
            {
                throw new ArgumentException(
                    $"Vocos layer scale {i} must have shape [{_hiddenDim}], got " +
                    $"[{string.Join(",", parameters[i].Shape)}].",
                    nameof(parameters));
            }
        }

        ClearRegisteredParameters();
        for (int i = 0; i < parameters.Count; i++)
        {
            _layerScales[i] = parameters[i];
            AppendTrainableParameter(_layerScales[i], PersistentTensorRole.Weights);
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        // Tape-based optimizers consume the registered tensors directly. Preserve the same flattened
        // ordering for legacy callers; child-layer manual gradients are still exposed where present.
        var values = new List<T>((int)ParameterCount);
        foreach (var (layer, scale) in OrderedParameterParts())
        {
            if (layer is not null)
            {
                var gradients = layer.GetParameterGradients();
                for (int i = 0; i < gradients.Length; i++) values.Add(gradients[i]);
            }
            else if (scale is not null)
            {
                for (int i = 0; i < scale.Length; i++) values.Add(NumOps.Zero);
            }
        }
        return new Vector<T>(values.ToArray());
    }

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        base.ClearGradients();
        foreach (var (layer, _) in OrderedParameterParts()) layer?.ClearGradients();
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var (layer, _) in OrderedParameterParts()) layer?.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var (layer, _) in OrderedParameterParts()) layer?.ResetState();
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var ci = System.Globalization.CultureInfo.InvariantCulture;
        metadata["NumMels"] = _numMels.ToString(ci);
        metadata["HiddenDim"] = _hiddenDim.ToString(ci);
        metadata["NumBackboneBlocks"] = _numBackboneBlocks.ToString(ci);
        metadata["IntermediateDim"] = _intermediateDim.ToString(ci);
        metadata["NFft"] = _nFft.ToString(ci);
        metadata["HopLength"] = _hopLength.ToString(ci);
        return metadata;
    }
}
