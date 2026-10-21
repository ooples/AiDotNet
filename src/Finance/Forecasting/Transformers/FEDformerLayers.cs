using AiDotNet.Attributes;
using AiDotNet.Extensions;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Finance.Forecasting.Transformers;

public partial class FEDformer<T>
{
    /// <summary>
    /// FEDformer-f as the official MAZiqing/FEDformer code builds it (Zhou et al., ICML 2022), as one layer from a window
    /// [B, SequenceLength, F] to the forecast [B, PredictionHorizon, F].
    /// </summary>
    /// <remarks>
    /// <para>
    /// The input is decomposed into seasonal and trend parts by a moving average. The encoder embeds the window (a
    /// circular convolution) and runs frequency-enhanced blocks, each followed by decomposition, and a seasonal norm. The
    /// decoder starts from the last LabelLength seasonal steps followed by zeros and runs frequency-enhanced self- and
    /// cross-attention and a feed-forward, decomposing after each; the trends it removes, projected, accumulate onto the
    /// last LabelLength trend steps followed by the input's mean. Its seasonal output, normed and projected, plus that
    /// trend, ends in the forecast.
    /// </para>
    /// <para>
    /// As in the reference, one Fourier block serves every encoder layer, and one self- and one cross-attention block
    /// every decoder layer: their complex weights are shared, while each layer has its own projections. Every weight is
    /// kept in the reference's PyTorch layout (<see cref="NamedTensors"/> gives the reference names), and every operation
    /// is an engine op the gradient tape records.
    /// </para>
    /// </remarks>
    [TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
    private sealed partial class FedformerNetwork : LayerBase<T>, IShapeContract
    {
        private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

        private readonly int _sequenceLength;
        private readonly int _labelLength;
        private readonly int _predictionHorizon;
        private readonly int _features;
        private readonly int _modelDimension;
        private readonly int _feedForward;
        private readonly int _heads;
        private readonly int _encoderLayers;
        private readonly int _decoderLayers;
        private readonly int _modes;
        private readonly FourierModeSelection _modeSelection;
        private readonly int _modeSelectionSeed;
        private readonly int _movingAverage;
        private readonly double _dropout;

        private readonly List<(string Name, Tensor<T> Tensor)> _named = new();
        private Tensor<T> _encoderEmbedding;
        private Tensor<T> _decoderEmbedding;
        private readonly List<EncoderBlock> _encoder = new();
        private (Tensor<T> Gamma, Tensor<T> Beta) _encoderNorm;
        private readonly List<DecoderBlock> _decoder = new();
        private readonly List<SpectralWeights> _spectral = new();
        private (Tensor<T> Gamma, Tensor<T> Beta) _decoderNorm;
        private Tensor<T> _projectionWeight;
        private Tensor<T> _projectionBias;

        internal FedformerNetwork(int sequenceLength, int labelLength, int predictionHorizon, int features, int modelDimension,
            int heads, int feedForward, int encoderLayers, int decoderLayers, int modes, FourierModeSelection modeSelection,
            int modeSelectionSeed, int movingAverage, double dropout)
            : base(new[] { sequenceLength, features }, new[] { predictionHorizon, features })
        {
            if (heads < 1 || modelDimension % heads != 0)
                throw new ArgumentException($"FEDformer's Fourier blocks split the model dimension into heads; {modelDimension} does not divide into {heads}.", nameof(heads));
            if (dropout < 0 || dropout >= 1 || double.IsNaN(dropout))
                throw new ArgumentOutOfRangeException(nameof(dropout), "Dropout must lie in [0, 1).");
            _sequenceLength = sequenceLength;
            _labelLength = labelLength;
            _predictionHorizon = predictionHorizon;
            _features = features;
            _modelDimension = modelDimension;
            _feedForward = feedForward;
            _heads = heads;
            _encoderLayers = encoderLayers;
            _decoderLayers = decoderLayers;
            _modes = modes;
            _modeSelection = modeSelection;
            _modeSelectionSeed = modeSelectionSeed;
            _movingAverage = movingAverage;
            _dropout = dropout;

            // The modes in the reference's order: encoder, decoder self-attention, then the cross-attention's queries and
            // keys, from one seeded generator, so a rebuilt network reads the same frequencies.
            int decoderLength = labelLength + predictionHorizon;
            var random = RandomHelper.CreateSeededRandom(modeSelectionSeed);
            var encoderModes = SelectModes(sequenceLength, random);
            var decoderModes = SelectModes(decoderLength, random);
            var crossQueryModes = SelectModes(decoderLength, random);
            var crossKeyModes = SelectModes(sequenceLength, random);

            _encoderEmbedding = KaimingConv(modelDimension, features);
            _decoderEmbedding = KaimingConv(modelDimension, features);
            var encoderSpectral = new SpectralWeights(modelDimension, heads, encoderModes);
            var decoderSpectral = new SpectralWeights(modelDimension, heads, decoderModes);
            var crossSpectral = new SpectralWeights(modelDimension, heads, crossQueryModes);
            _spectral.AddRange(new[] { encoderSpectral, decoderSpectral, crossSpectral });
            for (int i = 0; i < _encoderLayers; i++) _encoder.Add(new EncoderBlock(this, encoderSpectral));
            _encoderNorm = (Ones(modelDimension), new Tensor<T>(new[] { modelDimension }));
            for (int i = 0; i < _decoderLayers; i++) _decoder.Add(new DecoderBlock(this, decoderSpectral, crossSpectral, crossKeyModes));
            _decoderNorm = (Ones(modelDimension), new Tensor<T>(new[] { modelDimension }));
            _projectionWeight = Uniform(new[] { features, modelDimension }, modelDimension);
            _projectionBias = Uniform(new[] { features }, modelDimension);

            // Registered in the reference's state-dict order; a shared Fourier weight is registered once (with its first
            // user, where the reference's parameters() yields it) and named under every block that uses it.
            Register("enc_embedding.value_embedding.tokenConv.weight", _encoderEmbedding);
            Register("dec_embedding.value_embedding.tokenConv.weight", _decoderEmbedding);
            for (int i = 0; i < _encoder.Count; i++) _encoder[i].Register($"encoder.attn_layers.{i}.", first: i == 0);
            Register("encoder.norm.layernorm.weight", _encoderNorm.Gamma);
            Register("encoder.norm.layernorm.bias", _encoderNorm.Beta);
            for (int i = 0; i < _decoder.Count; i++) _decoder[i].Register($"decoder.layers.{i}.", first: i == 0);
            Register("decoder.norm.layernorm.weight", _decoderNorm.Gamma);
            Register("decoder.norm.layernorm.bias", _decoderNorm.Beta);
            Register("decoder.projection.weight", _projectionWeight);
            Register("decoder.projection.bias", _projectionBias);
        }

        /// <inheritdoc />
        public override bool SupportsTraining => true;

        /// <inheritdoc />
        public override void ResetState()
        {
        }

        /// <inheritdoc />
        /// <remarks>
        /// The blocks read their weights through their own references, which the base class's field rebinding cannot
        /// reach; every reference to a replaced tensor (a copy-on-write clone's shared storage, a parameter buffer's
        /// view) is repointed here, shared Fourier weights once.
        /// </remarks>
        public override void SetTrainableParameters(IReadOnlyList<Tensor<T>> parameters)
        {
            var previous = GetTrainableParameters().ToArray();
            base.SetTrainableParameters(parameters);
            if (previous.Length != parameters.Count) return;
            Tensor<T> Map(Tensor<T> tensor)
            {
                for (int i = 0; i < previous.Length; i++)
                    if (ReferenceEquals(previous[i], tensor)) return parameters[i];
                return tensor;
            }
            _encoderEmbedding = Map(_encoderEmbedding);
            _decoderEmbedding = Map(_decoderEmbedding);
            _encoderNorm = (Map(_encoderNorm.Gamma), Map(_encoderNorm.Beta));
            _decoderNorm = (Map(_decoderNorm.Gamma), Map(_decoderNorm.Beta));
            _projectionWeight = Map(_projectionWeight);
            _projectionBias = Map(_projectionBias);
            foreach (var spectral in _spectral) spectral.Rebind(Map);
            foreach (var block in _encoder) block.Rebind(Map);
            foreach (var block in _decoder) block.Rebind(Map);
            for (int i = 0; i < _named.Count; i++) _named[i] = (_named[i].Name, Map(_named[i].Tensor));
        }

        /// <inheritdoc />
        /// <remarks>[batch, SequenceLength, F] becomes [batch, PredictionHorizon, F].</remarks>
        public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank != 3 ? null : new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Fixed(_predictionHorizon)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_features)),
        };

        /// <summary>Every weight under its reference state-dict name; shared Fourier weights appear under each user.</summary>
        internal IReadOnlyList<(string Name, Tensor<T> Tensor)> NamedTensors => _named;

        /// <inheritdoc />
        protected override Tensor<T> ForwardTraced(Tensor<T> input)
        {
            if (input.Rank != 3 || input.Shape[1] != _sequenceLength || input.Shape[2] != _features)
                throw new ArgumentException(
                    $"FEDformer expects [batch, {_sequenceLength}, {_features}]; got [{string.Join(", ", input.Shape.ToArray())}].", nameof(input));
            int batch = input.Shape[0];
            var mean = Engine.TensorBroadcastTo(Engine.ReduceMean(input, new[] { 1 }, true), new[] { batch, _predictionHorizon, _features });
            var (seasonal, trend) = Decompose(input);
            var trendSum = Engine.TensorConcatenate(new[] { LastSteps(trend, _labelLength), mean }, 1);
            var seasonalInit = Engine.TensorConcatenate(
                new[] { LastSteps(seasonal, _labelLength), new Tensor<T>(new[] { batch, _predictionHorizon, _features }) }, 1);

            var encoded = Drop(CircularConv3(input, _encoderEmbedding));
            foreach (var block in _encoder) encoded = block.Forward(encoded);
            encoded = SeasonalNorm(encoded, _encoderNorm);

            var decoded = Drop(CircularConv3(seasonalInit, _decoderEmbedding));
            foreach (var block in _decoder)
            {
                var (seasonalPart, residualTrend) = block.Forward(decoded, encoded);
                decoded = seasonalPart;
                trendSum = Engine.TensorAdd(trendSum, residualTrend);
            }
            decoded = Linear(SeasonalNorm(decoded, _decoderNorm), _projectionWeight, _projectionBias);
            return LastSteps(Engine.TensorAdd(trendSum, decoded), _predictionHorizon);
        }

        // get_frequency_modes: min(modes, length / 2) of the frequencies below length / 2, at random (sorted) or the
        // lowest.
        private int[] SelectModes(int length, Random random)
        {
            int count = Math.Min(_modes, length / 2);
            if (_modeSelection == FourierModeSelection.Lowest) return Enumerable.Range(0, count).ToArray();
            var all = Enumerable.Range(0, length / 2).ToArray();
            for (int i = all.Length - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (all[i], all[j]) = (all[j], all[i]);
            }
            return all.Take(count).OrderBy(m => m).ToArray();
        }

        private void Register(string name, Tensor<T> tensor)
        {
            RegisterTrainableParameter(tensor, PersistentTensorRole.Weights);
            _named.Add((name, tensor));
        }

        private void Alias(string name, Tensor<T> tensor) => _named.Add((name, tensor));

        private static Tensor<T> Ones(int size)
        {
            var t = new Tensor<T>(new[] { size });
            t.Fill(Ops.One);
            return t;
        }

        // torch.nn.Linear and Conv1d's default initialization: uniform in +-1/sqrt(fan_in).
        private static Tensor<T> Uniform(int[] shape, int fanIn)
        {
            var tensor = new Tensor<T>(shape);
            var random = LayerInitializationSeedScope.NextRandom();
            double bound = 1.0 / Math.Sqrt(fanIn);
            for (int i = 0; i < tensor.Length; i++) tensor[i] = Ops.FromDouble((random.NextDouble() * 2 - 1) * bound);
            return tensor;
        }

        // TokenEmbedding's kaiming_normal_(mode='fan_in', nonlinearity='leaky_relu') for a [d_model, features, 3] kernel.
        private static Tensor<T> KaimingConv(int outChannels, int inChannels)
        {
            var tensor = new Tensor<T>(new[] { outChannels, inChannels, 3 });
            var random = LayerInitializationSeedScope.NextRandom();
            double std = Math.Sqrt(2.0 / (1.0 + 0.01 * 0.01)) / Math.Sqrt(inChannels * 3);
            for (int i = 0; i < tensor.Length; i++) tensor[i] = Ops.FromDouble(random.NextGaussian() * std);
            return tensor;
        }

        // Dropout in training mode only, as nn.Dropout: a fresh inverted-dropout mask per call, multiplied in so the tape
        // records it. As DropoutLayer does, the mask seed mixes the layer's RandomSeed with a call counter, so two runs at
        // the same seed drop the same elements.
        private Tensor<T> Drop(Tensor<T> x)
        {
            if (!IsTrainingMode || _dropout <= 0) return x;
            ulong call = unchecked((ulong)System.Threading.Interlocked.Increment(ref _dropoutCalls) - 1);
            int? seed = RandomSeed.HasValue ? unchecked((int)(((uint)RandomSeed.Value * 2654435761u) ^ (uint)call)) : (int?)null;
            var mask = Engine.TensorDropoutMask<T>(x._shape, Ops.FromDouble(_dropout), Ops.FromDouble(1.0 / (1.0 - _dropout)), seed);
            return Engine.TensorMultiply(x, mask);
        }

        private long _dropoutCalls;

        // x [..., in] times a PyTorch [out, in] weight (plus a bias), over the last axis.
        private Tensor<T> Linear(Tensor<T> x, Tensor<T> weight, Tensor<T>? bias)
        {
            int inFeatures = weight.Shape[1], outFeatures = weight.Shape[0];
            int rows = x.Length / inFeatures;
            var y = Engine.TensorMatMul(Engine.Reshape(x, new[] { rows, inFeatures }), Engine.TensorPermute(weight, new[] { 1, 0 }));
            if (bias is not null)
                y = Engine.TensorAdd(y, Engine.TensorBroadcastTo(Engine.Reshape(bias, new[] { 1, outFeatures }), new[] { rows, outFeatures }));
            var shape = x.Shape.ToArray();
            shape[shape.Length - 1] = outFeatures;
            return Engine.Reshape(y, shape);
        }

        // F.gelu's exact form, 0.5 x (1 + erf(x / sqrt 2)), as the reference's activation='gelu'.
        private Tensor<T> Gelu(Tensor<T> x) => Engine.TensorMultiply(Engine.TensorMultiplyScalar(x, Ops.FromDouble(0.5)),
            Engine.TensorAddScalar(Engine.TensorErf(Engine.TensorMultiplyScalar(x, Ops.FromDouble(1.0 / Math.Sqrt(2.0)))), Ops.One));

        // A bias-free 1x1 Conv1d ([out, in, 1] weight) over [B, L, in].
        private Tensor<T> Pointwise(Tensor<T> x, Tensor<T> weight) =>
            Linear(x, Engine.Reshape(weight, new[] { weight.Shape[0], weight.Shape[1] }), null);

        // A circular Conv1d (kernel 3, padding 1) over time on [B, L, C], with a PyTorch [out, C, 3] weight and no bias:
        // out[t] = sum_j W[:, :, j] x[(t + j - 1) mod L].
        private Tensor<T> CircularConv3(Tensor<T> x, Tensor<T> weight)
        {
            var s = x.Shape;
            int batch = s[0], length = s[1], channels = s[2], outChannels = weight.Shape[0];
            var previous = Engine.TensorConcatenate(new[]
            {
                Engine.TensorSlice(x, new[] { 0, length - 1, 0 }, new[] { batch, 1, channels }),
                Engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { batch, length - 1, channels }),
            }, 1);
            var next = Engine.TensorConcatenate(new[]
            {
                Engine.TensorSlice(x, new[] { 0, 1, 0 }, new[] { batch, length - 1, channels }),
                Engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { batch, 1, channels }),
            }, 1);
            // [x(t-1) | x(t) | x(t+1)] against the kernel taps 0, 1, 2.
            var stacked = Engine.Reshape(Engine.TensorConcatenate(new[] { previous, x, next }, 2), new[] { batch * length, 3 * channels });
            var kernel = Engine.Reshape(Engine.TensorPermute(weight, new[] { 2, 1, 0 }), new[] { 3 * channels, outChannels });
            return Engine.Reshape(Engine.TensorMatMul(stacked, kernel), new[] { batch, length, outChannels });
        }

        // my_Layernorm: LayerNorm over the features, then the mean over time subtracted.
        private Tensor<T> SeasonalNorm(Tensor<T> x, (Tensor<T> Gamma, Tensor<T> Beta) norm)
        {
            var normalized = Engine.LayerNorm(x, norm.Gamma, norm.Beta, 1e-5, out _, out _);
            var mean = Engine.ReduceMean(normalized, new[] { 1 }, true);
            return Engine.TensorSubtract(normalized, Engine.TensorBroadcastTo(mean, normalized.Shape.ToArray()));
        }

        // The last count time steps of [B, L, F] (an empty [B, 0, F] for none).
        private Tensor<T> LastSteps(Tensor<T> x, int count)
        {
            var s = x.Shape;
            return Engine.TensorSlice(x, new[] { 0, s[1] - count, 0 }, new[] { s[0], count, s[2] });
        }

        // series_decomp: the trend is a moving average over time with the ends replicated (Autoformer's moving_avg),
        // the seasonal part the rest. The reference pads kernel - 1 - (kernel - 1) / 2 copies of the first step in front
        // and (kernel - 1) / 2 of the last behind, then averages each window: A[t, s] is x[s]'s weight in the trend at t.
        private (Tensor<T> Seasonal, Tensor<T> Trend) Decompose(Tensor<T> x)
        {
            var s = x.Shape;
            int batch = s[0], length = s[1], channels = s[2], kernel = _movingAverage;
            int front = kernel - 1 - (kernel - 1) / 2;
            var average = new Tensor<T>(new[] { length, length });
            var share = Ops.FromDouble(1.0 / kernel);
            for (int t = 0; t < length; t++)
                for (int j = 0; j < kernel; j++)
                {
                    int source = Math.Min(length - 1, Math.Max(0, t - front + j));
                    average[t, source] = Ops.Add(average[t, source], share);
                }
            var timeMajor = Engine.Reshape(Engine.TensorPermute(x, new[] { 1, 0, 2 }), new[] { length, batch * channels });
            var trend = Engine.TensorPermute(
                Engine.Reshape(Engine.TensorMatMul(average, timeMajor), new[] { length, batch, channels }), new[] { 1, 0, 2 });
            return (Engine.TensorSubtract(x, trend), trend);
        }

        /// <summary>The complex weights of one Fourier block and the frequencies it reads; shared by its users.</summary>
        private sealed class SpectralWeights
        {
            internal SpectralWeights(int modelDimension, int heads, int[] modes)
            {
                Modes = modes;
                int e = modelDimension / heads;
                // scale * torch.rand(8, in / 8, out / 8, modes, dtype=cfloat), scale = 1 / (in * out): both parts uniform.
                double scale = 1.0 / ((double)modelDimension * modelDimension);
                var random = LayerInitializationSeedScope.NextRandom();
                Real = new Tensor<T>(new[] { heads, e, e, modes.Length });
                Imaginary = new Tensor<T>(new[] { heads, e, e, modes.Length });
                for (int i = 0; i < Real.Length; i++) Real[i] = Ops.FromDouble(scale * random.NextDouble());
                for (int i = 0; i < Imaginary.Length; i++) Imaginary[i] = Ops.FromDouble(scale * random.NextDouble());
            }

            internal int[] Modes { get; }
            internal Tensor<T> Real { get; private set; }
            internal Tensor<T> Imaginary { get; private set; }

            internal void Rebind(Func<Tensor<T>, Tensor<T>> map)
            {
                Real = map(Real);
                Imaginary = map(Imaginary);
            }
        }

        /// <summary>
        /// The frequency-enhanced attention of one AutoCorrelationLayer: query, key, value and output projections around
        /// FEB-f (self) or FEA-f (cross), per head. The reference fixes eight heads in its weights' shape
        /// [8, d / 8, d / 8, modes]; here the head count is NumHeads (eight by default), which is that shape at eight.
        /// </summary>
        /// <remarks>
        /// <para>
        /// FEB-f: the queries' spectrum at the selected modes is multiplied, per head, by the learned complex matrix and
        /// written into the first <c>modes</c> frequency bins, as the reference's <c>out_ft[..., wi]</c> does, then
        /// transformed back. FEA-f: the selected modes of queries and keys form a complex attention <c>tanh(Q^T K)</c>
        /// that mixes the keys' modes, which are multiplied by the learned matrix, written into the queries' bins,
        /// divided by d_model squared and transformed back.
        /// </para>
        /// <para>
        /// The real FFT and its inverse restricted to those modes are products with cosine and sine tables, which keeps
        /// the block on the tape. As in the reference, the head-major result [B, H, E, L] is read as [B, L, H * E]
        /// without a transpose, and the projections the reference computes but never reads (keys and values of FEB-f,
        /// values of FEA-f) are kept as parameters.
        /// </para>
        /// </remarks>
        private sealed class FourierAttention
        {
            private readonly FedformerNetwork _owner;
            private readonly SpectralWeights _spectral;
            private readonly int[]? _keyModes;
            private readonly int _d;
            private int Heads { get; }
            private Tensor<T> _queryWeight;
            private Tensor<T> _queryBias;
            private Tensor<T> _keyWeight;
            private Tensor<T> _keyBias;
            private Tensor<T> _valueWeight;
            private Tensor<T> _valueBias;
            private Tensor<T> _outWeight;
            private Tensor<T> _outBias;

            internal FourierAttention(FedformerNetwork owner, SpectralWeights spectral, int[]? keyModes)
            {
                _owner = owner;
                _spectral = spectral;
                _keyModes = keyModes;
                _d = owner._modelDimension;
                Heads = owner._heads;
                _queryWeight = Uniform(new[] { _d, _d }, _d);
                _queryBias = Uniform(new[] { _d }, _d);
                _keyWeight = Uniform(new[] { _d, _d }, _d);
                _keyBias = Uniform(new[] { _d }, _d);
                _valueWeight = Uniform(new[] { _d, _d }, _d);
                _valueBias = Uniform(new[] { _d }, _d);
                _outWeight = Uniform(new[] { _d, _d }, _d);
                _outBias = Uniform(new[] { _d }, _d);
            }

            private IEngine Engine => _owner.Engine;

            internal void Rebind(Func<Tensor<T>, Tensor<T>> map)
            {
                _queryWeight = map(_queryWeight);
                _queryBias = map(_queryBias);
                _keyWeight = map(_keyWeight);
                _keyBias = map(_keyBias);
                _valueWeight = map(_valueWeight);
                _valueBias = map(_valueBias);
                _outWeight = map(_outWeight);
                _outBias = map(_outBias);
            }

            internal void Register(string prefix, bool ownsSpectral)
            {
                if (ownsSpectral)
                {
                    _owner.Register(prefix + ".inner_correlation.weights1.real", _spectral.Real);
                    _owner.Register(prefix + ".inner_correlation.weights1.imag", _spectral.Imaginary);
                }
                else
                {
                    _owner.Alias(prefix + ".inner_correlation.weights1.real", _spectral.Real);
                    _owner.Alias(prefix + ".inner_correlation.weights1.imag", _spectral.Imaginary);
                }
                _owner.Register(prefix + ".query_projection.weight", _queryWeight);
                _owner.Register(prefix + ".query_projection.bias", _queryBias);
                _owner.Register(prefix + ".key_projection.weight", _keyWeight);
                _owner.Register(prefix + ".key_projection.bias", _keyBias);
                _owner.Register(prefix + ".value_projection.weight", _valueWeight);
                _owner.Register(prefix + ".value_projection.bias", _valueBias);
                _owner.Register(prefix + ".out_projection.weight", _outWeight);
                _owner.Register(prefix + ".out_projection.bias", _outBias);
            }

            /// <summary>FEB-f over x [B, L, D].</summary>
            internal Tensor<T> SelfAttend(Tensor<T> x)
            {
                int batch = x.Shape[0], length = x.Shape[1];
                var (real, imaginary) = Spectrum(_owner.Linear(x, _queryWeight, _queryBias), length, _spectral.Modes);
                var (yReal, yImaginary) = MixModes(real, imaginary, batch);
                // The reference writes mode wi into bin wi, whichever frequency it was read from.
                var bins = Enumerable.Range(0, _spectral.Modes.Length).ToArray();
                return _owner.Linear(InverseSpectrum(yReal, yImaginary, bins, length, batch, 1.0), _outWeight, _outBias);
            }

            /// <summary>FEA-f: queries from x [B, Lq, D], keys from cross [B, Lkv, D].</summary>
            internal Tensor<T> CrossAttend(Tensor<T> x, Tensor<T> cross)
            {
                if (_keyModes is null) throw new InvalidOperationException("This is a self-attention block.");
                int batch = x.Shape[0], queryLength = x.Shape[1], keyLength = cross.Shape[1];
                int e = _d / Heads, mq = _spectral.Modes.Length, mk = _keyModes.Length;
                var (qr, qi) = Spectrum(_owner.Linear(x, _queryWeight, _queryBias), queryLength, _spectral.Modes);
                var (kr, ki) = Spectrum(_owner.Linear(cross, _keyWeight, _keyBias), keyLength, _keyModes);

                // [B * H, E, M] views; Q^T K without conjugation, as the reference's einsum.
                var qrT = Engine.TensorPermute(Engine.Reshape(qr, new[] { batch * Heads, e, mq }), new[] { 0, 2, 1 });
                var qiT = Engine.TensorPermute(Engine.Reshape(qi, new[] { batch * Heads, e, mq }), new[] { 0, 2, 1 });
                var kr3 = Engine.Reshape(kr, new[] { batch * Heads, e, mk });
                var ki3 = Engine.Reshape(ki, new[] { batch * Heads, e, mk });
                var ar = Engine.TensorSubtract(Engine.BatchMatMul(qrT, kr3), Engine.BatchMatMul(qiT, ki3));
                var ai = Engine.TensorAdd(Engine.BatchMatMul(qrT, ki3), Engine.BatchMatMul(qiT, kr3));
                var (tr, ti) = ComplexTanh(ar, ai);

                // Mix the keys' modes: V[e, x] = sum_y A[x, y] K[e, y].
                var trT = Engine.TensorPermute(tr, new[] { 0, 2, 1 });
                var tiT = Engine.TensorPermute(ti, new[] { 0, 2, 1 });
                var vr = Engine.TensorSubtract(Engine.BatchMatMul(kr3, trT), Engine.BatchMatMul(ki3, tiT));
                var vi = Engine.TensorAdd(Engine.BatchMatMul(kr3, tiT), Engine.BatchMatMul(ki3, trT));
                var (yReal, yImaginary) = MixModes(Engine.Reshape(vr, new[] { batch, Heads, e, mq }),
                    Engine.Reshape(vi, new[] { batch, Heads, e, mq }), batch);
                double scale = 1.0 / ((double)_d * _d);
                return _owner.Linear(InverseSpectrum(yReal, yImaginary, _spectral.Modes, queryLength, batch, scale), _outWeight, _outBias);
            }

            // The rfft of the head-split projection [B, L, D] at the given indices: ([B, H, E, M] real, imaginary).
            private (Tensor<T> Real, Tensor<T> Imaginary) Spectrum(Tensor<T> projected, int length, int[] modes)
            {
                int batch = projected.Shape[0], e = _d / Heads, m = modes.Length;
                var series = Engine.Reshape(
                    Engine.TensorPermute(Engine.Reshape(projected, new[] { batch, length, Heads, e }), new[] { 0, 2, 3, 1 }),
                    new[] { batch * _d, length });
                var cos = new Tensor<T>(new[] { length, m });
                var negSin = new Tensor<T>(new[] { length, m });
                for (int t = 0; t < length; t++)
                    for (int k = 0; k < m; k++)
                    {
                        double angle = 2.0 * Math.PI * modes[k] * t / length;
                        cos[t, k] = Ops.FromDouble(Math.Cos(angle));
                        negSin[t, k] = Ops.FromDouble(-Math.Sin(angle));
                    }
                var shape = new[] { batch, Heads, e, m };
                return (Engine.Reshape(Engine.TensorMatMul(series, cos), shape), Engine.Reshape(Engine.TensorMatMul(series, negSin), shape));
            }

            // Y[b, h, o, m] = sum_i X[b, h, i, m] W[h, i, o, m], complex, for X given as [B, H, E, M] parts.
            private (Tensor<T> Real, Tensor<T> Imaginary) MixModes(Tensor<T> xr, Tensor<T> xi, int batch)
            {
                int e = _d / Heads, m = _spectral.Real.Shape[3];
                Tensor<T> ByMode(Tensor<T> x) =>
                    Engine.Reshape(Engine.TensorPermute(x, new[] { 1, 3, 0, 2 }), new[] { Heads * m, batch, e });
                Tensor<T> WeightByMode(Tensor<T> w) =>
                    Engine.Reshape(Engine.TensorPermute(w, new[] { 0, 3, 1, 2 }), new[] { Heads * m, e, e });
                Tensor<T> Back(Tensor<T> y) =>
                    Engine.TensorPermute(Engine.Reshape(y, new[] { Heads, m, batch, e }), new[] { 2, 0, 3, 1 });
                var ar = ByMode(xr);
                var ai = ByMode(xi);
                var wr = WeightByMode(_spectral.Real);
                var wi = WeightByMode(_spectral.Imaginary);
                var real = Engine.TensorSubtract(Engine.BatchMatMul(ar, wr), Engine.BatchMatMul(ai, wi));
                var imaginary = Engine.TensorAdd(Engine.BatchMatMul(ar, wi), Engine.BatchMatMul(ai, wr));
                return (Back(real), Back(imaginary));
            }

            // irfft(n = length) of a spectrum that is zero except at the given bins, times scale; the head-major
            // [B, H, E, L] result read as [B, L, D].
            private Tensor<T> InverseSpectrum(Tensor<T> real, Tensor<T> imaginary, int[] bins, int length, int batch, double scale)
            {
                int m = bins.Length;
                var cos = new Tensor<T>(new[] { m, length });
                var negSin = new Tensor<T>(new[] { m, length });
                for (int k = 0; k < m; k++)
                {
                    int bin = bins[k];
                    // irfft counts each interior bin twice (its conjugate mirror) and drops the imaginary part of bins
                    // 0 and n / 2.
                    bool edge = bin == 0 || 2 * bin == length;
                    double weight = (edge ? 1.0 : 2.0) * scale / length;
                    for (int t = 0; t < length; t++)
                    {
                        double angle = 2.0 * Math.PI * bin * t / length;
                        cos[k, t] = Ops.FromDouble(weight * Math.Cos(angle));
                        negSin[k, t] = Ops.FromDouble(edge ? 0.0 : -weight * Math.Sin(angle));
                    }
                }
                int rows = batch * _d;
                var series = Engine.TensorAdd(
                    Engine.TensorMatMul(Engine.Reshape(real, new[] { rows, m }), cos),
                    Engine.TensorMatMul(Engine.Reshape(imaginary, new[] { rows, m }), negSin));
                return Engine.Reshape(series, new[] { batch, length, _d });
            }

            // tanh(a + ib) = (tanh 2a + i sin 2b sech 2a) / (1 + cos 2b sech 2a), sech 2a = 2 e^{-2|a|} / (1 + e^{-4|a|}),
            // which stays finite for large |a|.
            private (Tensor<T> Real, Tensor<T> Imaginary) ComplexTanh(Tensor<T> a, Tensor<T> b)
            {
                var twoA = Engine.TensorMultiplyScalar(a, Ops.FromDouble(2.0));
                var twoB = Engine.TensorMultiplyScalar(b, Ops.FromDouble(2.0));
                var decay = Engine.TensorExp(Engine.TensorNegate(Engine.TensorAbs(twoA)));
                var sech = Engine.TensorDivide(Engine.TensorMultiplyScalar(decay, Ops.FromDouble(2.0)),
                    Engine.TensorAddScalar(Engine.TensorMultiply(decay, decay), Ops.One));
                var denominator = Engine.TensorAddScalar(Engine.TensorMultiply(Engine.TensorCos(twoB), sech), Ops.One);
                return (Engine.TensorDivide(Engine.TensorTanh(twoA), denominator),
                    Engine.TensorDivide(Engine.TensorMultiply(Engine.TensorSin(twoB), sech), denominator));
            }
        }

        /// <summary>
        /// An encoder layer: x + FEB-f(x), decomposed; plus the bias-free 1x1-convolution feed-forward with GELU,
        /// decomposed again. Only the seasonal parts go on.
        /// </summary>
        private sealed class EncoderBlock
        {
            private readonly FedformerNetwork _owner;
            private readonly FourierAttention _attention;
            private Tensor<T> _conv1;
            private Tensor<T> _conv2;

            internal EncoderBlock(FedformerNetwork owner, SpectralWeights spectral)
            {
                _owner = owner;
                _attention = new FourierAttention(owner, spectral, null);
                _conv1 = Uniform(new[] { owner._feedForward, owner._modelDimension, 1 }, owner._modelDimension);
                _conv2 = Uniform(new[] { owner._modelDimension, owner._feedForward, 1 }, owner._feedForward);
            }

            internal void Rebind(Func<Tensor<T>, Tensor<T>> map)
            {
                _attention.Rebind(map);
                _conv1 = map(_conv1);
                _conv2 = map(_conv2);
            }

            internal void Register(string prefix, bool first)
            {
                _attention.Register(prefix + "attention", ownsSpectral: first);
                _owner.Register(prefix + "conv1.weight", _conv1);
                _owner.Register(prefix + "conv2.weight", _conv2);
            }

            internal Tensor<T> Forward(Tensor<T> input)
            {
                var engine = _owner.Engine;
                var x = engine.TensorAdd(input, _owner.Drop(_attention.SelfAttend(input)));
                x = _owner.Decompose(x).Seasonal;
                var y = _owner.Drop(_owner.Gelu(_owner.Pointwise(x, _conv1)));
                y = _owner.Drop(_owner.Pointwise(y, _conv2));
                return _owner.Decompose(engine.TensorAdd(x, y)).Seasonal;
            }
        }

        /// <summary>
        /// A decoder layer: self FEB-f, cross FEA-f against the encoder output and the feed-forward, each followed by a
        /// decomposition; the three trends, summed and projected by a bias-free circular Conv1d (kernel 3) to the
        /// output features, are the layer's contribution to the trend.
        /// </summary>
        private sealed class DecoderBlock
        {
            private readonly FedformerNetwork _owner;
            private readonly FourierAttention _selfAttention;
            private readonly FourierAttention _crossAttention;
            private Tensor<T> _conv1;
            private Tensor<T> _conv2;
            private Tensor<T> _trendProjection;

            internal DecoderBlock(FedformerNetwork owner, SpectralWeights selfSpectral, SpectralWeights crossSpectral, int[] crossKeyModes)
            {
                _owner = owner;
                _selfAttention = new FourierAttention(owner, selfSpectral, null);
                _crossAttention = new FourierAttention(owner, crossSpectral, crossKeyModes);
                _conv1 = Uniform(new[] { owner._feedForward, owner._modelDimension, 1 }, owner._modelDimension);
                _conv2 = Uniform(new[] { owner._modelDimension, owner._feedForward, 1 }, owner._feedForward);
                _trendProjection = Uniform(new[] { owner._features, owner._modelDimension, 3 }, owner._modelDimension * 3);
            }

            internal void Rebind(Func<Tensor<T>, Tensor<T>> map)
            {
                _selfAttention.Rebind(map);
                _crossAttention.Rebind(map);
                _conv1 = map(_conv1);
                _conv2 = map(_conv2);
                _trendProjection = map(_trendProjection);
            }

            internal void Register(string prefix, bool first)
            {
                _selfAttention.Register(prefix + "self_attention", ownsSpectral: first);
                _crossAttention.Register(prefix + "cross_attention", ownsSpectral: first);
                _owner.Register(prefix + "conv1.weight", _conv1);
                _owner.Register(prefix + "conv2.weight", _conv2);
                _owner.Register(prefix + "projection.weight", _trendProjection);
            }

            internal (Tensor<T> Seasonal, Tensor<T> Trend) Forward(Tensor<T> x, Tensor<T> encoderOutput)
            {
                var engine = _owner.Engine;
                x = engine.TensorAdd(x, _owner.Drop(_selfAttention.SelfAttend(x)));
                var (s1, t1) = _owner.Decompose(x);
                x = engine.TensorAdd(s1, _owner.Drop(_crossAttention.CrossAttend(s1, encoderOutput)));
                var (s2, t2) = _owner.Decompose(x);
                var y = _owner.Drop(_owner.Gelu(_owner.Pointwise(s2, _conv1)));
                y = _owner.Drop(_owner.Pointwise(y, _conv2));
                var (s3, t3) = _owner.Decompose(engine.TensorAdd(s2, y));
                var trend = _owner.CircularConv3(engine.TensorAdd(engine.TensorAdd(t1, t2), t3), _trendProjection);
                return (s3, trend);
            }
        }
    }
}
