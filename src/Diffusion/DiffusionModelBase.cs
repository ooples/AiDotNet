using System.Linq;
using AiDotNet.Autodiff;
using AiDotNet.Deployment.Optimization.Quantization;
using AiDotNet.Deployment.Optimization.Quantization.Training;
using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Diffusion;

/// <summary>
/// Base class for diffusion-based generative models providing common functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This abstract base class implements the common behavior for all diffusion models,
/// including the generation loop, noise addition, loss computation, and state management.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all diffusion models build upon.
/// It handles the common tasks that every diffusion model needs:
/// <list type="bullet">
/// <item><description>The generation loop (iteratively denoising from noise)</description></item>
/// <item><description>Adding noise during training</description></item>
/// <item><description>Computing the training loss</description></item>
/// <item><description>Saving and loading the model</description></item>
/// </list>
/// Specific diffusion models (like DDPM, Latent Diffusion) extend this base to implement
/// their unique noise prediction architectures.</para>
/// </remarks>
public abstract class DiffusionModelBase<T> : IDiffusionModel<T>, IConfigurableModel<T>, IModelShape, IDisposable
{
    /// <summary>
    /// Concrete diffusion models can override this method to yield the components
    /// they own that hold disposable resources — typically the noise predictor
    /// (DiT, UNet, MMDiT) plus, for latent diffusion, the VAE and conditioner.
    /// </summary>
    /// <remarks>
    /// <para>
    /// We use a method-based opt-in rather than a <c>NoisePredictor</c> property
    /// to avoid name collision with <see cref="ILatentDiffusionModel{T}.NoisePredictor"/>
    /// (which returns the non-nullable interface type and is part of an existing
    /// contract). Subclasses of <c>LatentDiffusionModelBase</c> can override
    /// this method to surface the same predictor for Dispose cleanup without
    /// changing their interface obligations.
    /// </para>
    /// <para>
    /// <b>Default behavior</b>: when a subclass does NOT override this, the base
    /// performs a reflection walk over its own and the subclass's instance
    /// fields and yields anything that implements <see cref="IDisposable"/>.
    /// This catches the common case (a private predictor field) without forcing
    /// every existing concrete model to override — but for predictable cleanup
    /// in performance-sensitive code, an explicit override remains preferred.
    /// </para>
    /// <example>
    /// <code>
    /// public class DDPMModel&lt;T&gt; : DiffusionModelBase&lt;T&gt; {
    ///     private readonly UNetNoisePredictor&lt;T&gt; _unet;
    ///     protected override IEnumerable&lt;IDisposable&gt; EnumerateDisposableComponents() {
    ///         yield return _unet;
    ///     }
    /// }
    /// </code>
    /// </example>
    /// </remarks>
    protected virtual IEnumerable<IDisposable> EnumerateDisposableComponents()
    {
        // Default: yield nothing. Concrete diffusion models that own disposable
        // components (noise predictor, VAE, conditioner, scheduler, etc.) must
        // override this and explicitly yield only the components THIS model owns.
        // We do not reflect over instance fields here because that would also
        // tear down injected/shared dependencies (e.g., a tokenizer or text
        // encoder shared across multiple pipelines), creating cross-instance
        // lifecycle breakage.
        yield break;
    }

    private bool _disposed;

    /// <summary>
    /// Provides access to the hardware-accelerated tensor engine.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Random number generator for noise sampling.
    /// </summary>
    protected Random RandomGenerator;

    /// <summary>
    /// The step scheduler controlling the diffusion process.
    /// </summary>
    private readonly INoiseScheduler<T> _scheduler;

    /// <summary>
    /// The loss function used for training (typically MSE for noise prediction).
    /// </summary>
    protected readonly ILossFunction<T> LossFunction;

    /// <summary>
    /// The configuration options for this diffusion model.
    /// </summary>
    private readonly DiffusionModelOptions<T> _options;

    /// <summary>
    /// Gets the configuration options for this model.
    /// </summary>
    protected ModelOptions Options => _options;

    /// <inheritdoc/>
    public virtual ModelOptions GetOptions() => _options;

    /// <summary>
    /// The optional neural network architecture blueprint for custom layer configuration.
    /// </summary>
    private readonly NeuralNetworkArchitecture<T>? _architecture;

    /// <summary>
    /// Active feature indices used by the model.
    /// </summary>
    private HashSet<int> _activeFeatureIndices = new HashSet<int>();

    /// <summary>
    /// G5 (#1624) quantization-aware-training hook, created lazily on the first <see cref="Train"/> step
    /// once <see cref="IsQuantizationAwareTrainingEnabled"/> is true.
    /// </summary>
    private QATTrainingHook<T>? _qatHook;

    /// <summary>QAT config (explicit or default). Captured by <see cref="EnableQuantizationAwareTraining"/>.</summary>
    private QuantizationConfiguration? _qatConfig;

    /// <summary>
    /// Explicit QAT override: <c>null</c> = auto (engage by the parameter-count threshold),
    /// <c>true</c>/<c>false</c> = forced on/off.
    /// </summary>
    private bool? _qatExplicit;

    /// <summary>
    /// Parameter-count threshold at/above which QAT engages by DEFAULT (G5, #1624). Foundation-scale
    /// diffusion models are the int8-deployment targets, so they train quantization-aware by default;
    /// smaller models stay full-precision unless QAT is requested explicitly.
    /// </summary>
    private const long DefaultQatThresholdParams = 500_000_000L;

    /// <summary>Test/diagnostic override for <see cref="DefaultQatThresholdParams"/>. Process-global.</summary>
    internal static long? QatThresholdOverride { get; set; }

    /// <summary>
    /// The learning rate converted to type T for training computations.
    /// </summary>
    protected T LearningRate;

    /// <summary>
    /// Cached result of the reflection walk that discovers trainable parameter tensors.
    /// The walk was called per Train step, consuming a non-trivial amount of time on
    /// large models. Tensor references are stable (DenseLayer.SetParameters modifies in
    /// place) so caching is safe. Subclasses that swap layer references at runtime can
    /// invalidate via InvalidateTrainableParametersCache.
    /// </summary>
    private Tensor<T>[]? _cachedTrainableParameters;

    // ── Copy-on-write weight sharing (cheap Clone of large models) ───────────────────────────────
    //
    // Clone() shares the parent's weight TENSORS by reference instead of deep-copying every
    // parameter (a 600M-param model would otherwise copy ~2.4 GB on every clone). The clone gets its
    // OWN layer objects — and therefore its own forward activation caches — but each layer's weight
    // tensors point at the parent's, so Predict on either model reads identical weights at O(1) clone
    // cost. The first time EITHER model WRITES a weight (training), it transparently copies its own
    // weights first (copy-on-write) so neither can corrupt the other. A shared set is reference-
    // counted: when a member detaches, the remaining members keep sharing, and once only one member
    // is left it owns the tensors outright and never needs to copy.

    /// <summary>Reference-counted membership token for a set of models sharing weight tensors.</summary>
    private sealed class WeightShareGroup
    {
        private int _count;
        public WeightShareGroup(int initialCount) => _count = initialCount;
        /// <summary>A new model joins the shared set.</summary>
        public void Join() { lock (this) _count++; }
        /// <summary>Decrements the sharer count; returns true if a copy is required (others remain).</summary>
        public bool LeaveAndNeedsCopy()
        {
            lock (this)
            {
                bool othersRemain = _count > 1;
                if (_count > 0) _count--;
                return othersRemain;
            }
        }
    }

    private WeightShareGroup? _shareGroup;

    /// <summary>
    /// Reflection-walks the model graph (same traversal as <see cref="CollectTrainableParameters"/>)
    /// and returns every <see cref="Interfaces.ITrainableLayer{T}"/> in a stable order, so a clone's
    /// layers can be paired positionally with its parent's.
    /// </summary>
    private List<Interfaces.ITrainableLayer<T>> CollectTrainableLayers()
    {
        var layers = new List<Interfaces.ITrainableLayer<T>>();
        CollectLayersInto(this, layers, new HashSet<object>(AiDotNet.Helpers.TensorReferenceComparer<object>.Instance));
        return layers;
    }

    private void CollectLayersInto(object? obj, List<Interfaces.ITrainableLayer<T>> layers, HashSet<object> visited)
    {
        if (obj is null || !visited.Add(obj)) return;
        if (obj is Interfaces.ITrainableLayer<T> trainable) layers.Add(trainable);
        foreach (var field in obj.GetType().GetFields(
                     System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic))
        {
            Type ft = field.FieldType;
            if (ft.IsValueType || ft == typeof(string)) continue;
            object? value;
            try { value = field.GetValue(obj); } catch { continue; }
            if (value is null) continue;
            if (value is System.Collections.IEnumerable en && value is not Interfaces.ITrainableLayer<T>)
                foreach (var item in en) CollectLayersInto(item, layers, visited);
            else
                CollectLayersInto(value, layers, visited);
        }
    }

    /// <summary>
    /// Makes this (freshly-constructed) model SHARE <paramref name="parent"/>'s weight tensors by
    /// reference — the copy-on-write fast path for <see cref="Clone"/>. Both models join a shared
    /// reference-counted set; the first weight write on either triggers a private copy. The two layer
    /// lists must have identical structure (same model class + config), which Clone guarantees.
    /// </summary>
    protected void ShareWeightsFrom(DiffusionModelBase<T> parent)
    {
        var parentLayers = parent.CollectTrainableLayers();
        var myLayers = CollectTrainableLayers();
        if (parentLayers.Count != myLayers.Count)
            throw new InvalidOperationException(
                $"ShareWeightsFrom: layer count mismatch (parent {parentLayers.Count} vs clone {myLayers.Count}). " +
                "Copy-on-write clone requires identical structure.");

        for (int i = 0; i < myLayers.Count; i++)
            myLayers[i].SetTrainableParameters(parentLayers[i].GetTrainableParameters());

        if (parent._shareGroup is null)
        {
            // Parent was a sole owner; it and this clone now form a shared set of two. The parent must
            // also copy-on-write before its next weight write, so it joins the group too.
            parent._shareGroup = new WeightShareGroup(2);
        }
        else
        {
            // Parent already shares with one or more clones; this one joins the existing set.
            parent._shareGroup.Join();
        }
        _shareGroup = parent._shareGroup;
        InvalidateTrainableParametersCache();
    }

    /// <summary>
    /// Copy-on-write guard: if this model is sharing its weight tensors with another model, give it a
    /// private deep copy of every weight tensor BEFORE the caller mutates any of them. Called at the
    /// top of every weight-mutating entry point (<see cref="Train"/>, <see cref="SetParameters"/>).
    /// No-op once the model owns its weights outright.
    /// </summary>
    protected void EnsureOwnWeights()
    {
        var group = _shareGroup;
        if (group is null) return;
        _shareGroup = null;
        if (group.LeaveAndNeedsCopy())
        {
            foreach (var layer in CollectTrainableLayers())
            {
                var shared = layer.GetTrainableParameters();
                if (shared is null) continue;
                var owned = new Tensor<T>[shared.Count];
                for (int i = 0; i < shared.Count; i++) owned[i] = shared[i].Clone();
                layer.SetTrainableParameters(owned);
            }
        }
        InvalidateTrainableParametersCache();
    }

    /// <inheritdoc />
    public INoiseScheduler<T> Scheduler => _scheduler;

    /// <inheritdoc />
    public abstract long ParameterCount { get; }

    /// <summary>
    /// Streams the diffusion stack's trainable weight tensors per-tensor.
    /// Default implementation is empty; concrete subclasses (latent
    /// diffusion, cascade, dual-encoder) override to yield from their
    /// noise predictor / VAE / conditioner sub-models. Foundation-scale
    /// stacks overflow <see cref="int.MaxValue"/> in the aggregate
    /// <see cref="ParameterCount"/>, so callers walking these chunks
    /// accumulate length into a <see cref="long"/>.
    /// </summary>
    public virtual IEnumerable<Tensor<T>> GetParameterChunks() => System.Linq.Enumerable.Empty<Tensor<T>>();

    /// <summary>
    /// Streaming counterpart to <see cref="SetParameters"/>: assigns weights from per-tensor
    /// chunks in <see cref="GetParameterChunks"/> order without materializing a flat aggregate.
    /// Default buffers the chunks into one flat <see cref="Vector{T}"/> and delegates to
    /// <see cref="SetParameters"/> (back-compatible for tractable stacks); foundation-scale
    /// subclasses override to consume one chunk at a time and stay flat-free.
    /// </summary>
    public virtual void SetParameterChunks(IEnumerable<Tensor<T>> chunks)
    {
        if (chunks is null) throw new ArgumentNullException(nameof(chunks));
        // Detach any copy-on-write-shared weights before mutating in place — SetParameters writes
        // through raw spans that bypass the COW write barrier, so without this a chunk assignment
        // could corrupt a sibling clone. Same guard Train() applies.
        EnsureOwnWeights();

        var buffered = new List<Tensor<T>>();
        long total = 0;
        foreach (var chunk in chunks)
        {
            if (chunk is null)
                throw new ArgumentException("Chunk sequence contains a null tensor.", nameof(chunks));
            buffered.Add(chunk);
            total += chunk.Length;
        }

        var flat = new Vector<T>(checked((int)total));
        int offset = 0;
        foreach (var chunk in buffered)
        {
            var v = chunk.ToVector();
            for (int i = 0; i < v.Length; i++) flat[offset++] = v[i];
        }

        SetParameters(flat);
    }

    /// <inheritdoc/>
    public virtual Vector<T> SanitizeParameters(Vector<T> parameters) => parameters;

    /// <inheritdoc/>
    public virtual bool SupportsParameterInitialization => ParameterCount > 0;

    /// <inheritdoc />
    public ILossFunction<T> DefaultLossFunction => LossFunction;

    /// <summary>
    /// Gets the optional neural network architecture used for custom layer configuration.
    /// </summary>
    /// <remarks>
    /// When provided, derived models should check <c>Architecture.Layers</c> first before
    /// creating default layers. This allows users to supply custom layer configurations
    /// via <see cref="NeuralNetworkArchitecture{T}"/>.
    /// </remarks>
    public NeuralNetworkArchitecture<T>? Architecture => _architecture;

    /// <summary>
    /// Initializes a new instance of the DiffusionModelBase class.
    /// </summary>
    /// <param name="options">Configuration options for the diffusion model. If null, uses default options.</param>
    /// <param name="scheduler">Optional custom scheduler. If null, creates one from options.</param>
    /// <param name="architecture">
    /// Optional neural network architecture for custom layer configuration.
    /// When provided, derived models should check <c>Architecture.Layers</c> first before
    /// creating default layers. If null, models use their own research-paper defaults.
    /// </param>
    protected DiffusionModelBase(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        NeuralNetworkArchitecture<T>? architecture = null)
    {
        _architecture = architecture;
        _options = options ?? new DiffusionModelOptions<T>();

        // Create scheduler from options if not provided
        if (scheduler != null)
        {
            _scheduler = scheduler;
        }
        else
        {
            var schedulerConfig = new SchedulerConfig<T>(
                trainTimesteps: _options.TrainTimesteps,
                betaStart: NumOps.FromDouble(_options.BetaStart),
                betaEnd: NumOps.FromDouble(_options.BetaEnd),
                betaSchedule: _options.BetaSchedule,
                clipSample: _options.ClipSample,
                predictionType: _options.PredictionType);
            _scheduler = new DDIMScheduler<T>(schedulerConfig);
        }

        // Set loss function from options or default
        LossFunction = _options.LossFunction ?? new MeanSquaredErrorLoss<T>();

        // Convert learning rate from double to T
        LearningRate = NumOps.FromDouble(_options.LearningRate);

        // Set up random generator
        RandomGenerator = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Creates a reproducible RNG for the INFERENCE / generation path. When the
    /// caller passes no explicit <paramref name="seed"/>, a STABLE seed (the
    /// model's construction seed, else a fixed constant) is used — deliberately
    /// NOT the advancing <see cref="RandomGenerator"/> — so that two
    /// <c>Predict(sameInput)</c> calls draw the identical noise and produce the
    /// identical output (the <c>Predict_ShouldBeDeterministic</c> contract every
    /// diffusion model must honor), while callers wanting sample variety still
    /// pass an explicit seed. Training intentionally keeps using
    /// <see cref="RandomGenerator"/>, which MUST advance across steps to draw
    /// fresh timesteps/noise each iteration.
    /// </summary>
    /// <param name="seed">Optional explicit seed; when null a stable seed is used.</param>
    protected Random CreateInferenceRng(int? seed)
        => RandomHelper.CreateSeededRandom(seed ?? _options.Seed ?? InferenceDefaultSeed);

    /// <summary>
    /// Fixed fallback seed for <see cref="CreateInferenceRng"/> when the model
    /// was constructed without a seed — any constant works; it only needs to be
    /// stable across calls so unseeded generation is reproducible.
    /// </summary>
    private const int InferenceDefaultSeed = 0;

    #region IDiffusionModel<T> Implementation

    /// <inheritdoc />
    public virtual Tensor<T> Generate(int[] shape, int numInferenceSteps = 50, int? seed = null)
        => Generate(shape, numInferenceSteps, seed, initialSample: null);

    /// <summary>
    /// Async overload of <see cref="Generate(int[], int, int?)"/>. Returns a
    /// <see cref="ValueTask{T}"/> so callers on the .NET threadpool can await
    /// completion without blocking a worker thread for the full denoising loop.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Runs the denoising loop through <see cref="GenerateAsyncCore"/>, which
    /// awaits <see cref="PredictNoiseAsync"/> at each timestep so the compiled
    /// async execution path can overlap backend work with host-side scheduler
    /// state updates (timestep embedding, scheduler.Step, RNG advance,
    /// NaN/Inf sanitization). On a CPU engine each step's await completes
    /// inline; on a GPU engine the await polls the stream completion event
    /// without blocking a threadpool worker.
    /// </para>
    /// </remarks>
    public virtual System.Threading.Tasks.ValueTask<Tensor<T>> GenerateAsync(
        int[] shape,
        int numInferenceSteps = 50,
        int? seed = null,
        System.Threading.CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return GenerateAsyncCore(shape, numInferenceSteps, seed, initialSample: null, cancellationToken);
    }

    /// <summary>
    /// True-async denoising loop. Each per-step noise prediction goes through
    /// <see cref="PredictNoiseAsync"/> which routes to the compile host's
    /// <c>ExecuteAsync</c>; awaits between steps let the runtime overlap host
    /// scheduler work with backend tail kernels. Concrete subclasses that
    /// don't yet expose a true-async sampler-update can stay on the inherited
    /// behavior — only the noise-predictor stage gains async semantics today,
    /// which is already the dominant cost on CPU (75% of wall time per the
    /// profile in #1273) and the highest-priority overlap point.
    /// </summary>
    protected virtual async System.Threading.Tasks.ValueTask<Tensor<T>> GenerateAsyncCore(
        int[] shape,
        int numInferenceSteps,
        int? seed,
        Vector<T>? initialSample,
        System.Threading.CancellationToken cancellationToken)
    {
        ValidateGenerateInputs(shape, numInferenceSteps, out long totalElements);

        using var _ = new NoGradScope<T>();

        Vector<T> sample = ResolveInitialSample(shape, numInferenceSteps, seed, initialSample, totalElements);

        _scheduler.SetTimesteps(numInferenceSteps);
        var sampleTensor = new Tensor<T>(shape, sample);
        var noisePredVec = new Vector<T>(sample.Length);

        foreach (var timestep in _scheduler.Timesteps)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var sampleSpan = sample.AsSpan();
            var tensorSpan = sampleTensor.AsWritableSpan();
            sampleSpan.CopyTo(tensorSpan);

            // True-async noise prediction: GPU stream completion polled
            // without blocking; CPU completes inline.
            var noisePrediction = await PredictNoiseAsync(sampleTensor, timestep, conditioning: null, cancellationToken)
                .ConfigureAwait(false);

            var predSpan = noisePrediction.AsSpan();
            if (predSpan.Length != noisePredVec.Length)
                throw new InvalidOperationException(
                    $"PredictNoise output length ({predSpan.Length}) does not match the " +
                    $"latent/sample length ({noisePredVec.Length}). Check that the noise " +
                    $"predictor's output shape matches the input tensor shape.");
            for (int idx = 0; idx < predSpan.Length; idx++)
                noisePredVec[idx] = predSpan[idx];

            sample = _scheduler.Step(noisePredVec, timestep, sample, NumOps.Zero);

            // NaN/Inf guard via shared helper — same contract the sync
            // Generate path uses, centralised so the two surfaces can't drift.
            int sanitizedCount = SanitizeNonFiniteElements(sample);
            if (sanitizedCount > 0)
                System.Diagnostics.Trace.TraceWarning(
                    $"DiffusionModelBase.GenerateAsync: sanitized {sanitizedCount} non-finite element(s) at timestep {timestep}.");
        }

        return new Tensor<T>(shape, sample);
    }

    /// <summary>
    /// Concrete async noise prediction overload for the base path. Routes
    /// through whatever <see cref="PredictNoise(Tensor{T}, int)"/> implementation
    /// the subclass provides; subclasses extending <see cref="NoisePredictorBase{T}"/>
    /// inherit the compile-host-aware async path automatically.
    /// </summary>
    /// <param name="noisySample">The current noisy latent / pixel tensor.</param>
    /// <param name="timestep">The current diffusion timestep.</param>
    /// <param name="conditioning">
    /// Optional cross-attention conditioning tensor (text embedding, image
    /// embedding, etc.). <b>Ignored by this default implementation</b> — the
    /// base unconditional path delegates to <see cref="PredictNoise(Tensor{T}, int)"/>
    /// which has no conditioning slot. The parameter exists for the multi-stage
    /// chain pipeline (latent diffusion's text-conditioner → cross-attention
    /// noise predictor pattern in #1272 / #1273); concrete subclasses that
    /// accept conditioning override this method to thread it through to their
    /// noise predictor (see <see cref="LatentDiffusionModelBase{T}"/>).
    /// <see cref="GenerateAsyncCore"/> currently passes <see langword="null"/>
    /// for the unconditional path; subclasses with conditioning supply it via
    /// their own <c>GenerateAsync</c> override.
    /// </param>
    /// <param name="cancellationToken">Honored before the underlying forward runs.</param>
    protected virtual System.Threading.Tasks.ValueTask<Tensor<T>> PredictNoiseAsync(
        Tensor<T> noisySample,
        int timestep,
        Tensor<T>? conditioning,
        System.Threading.CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return new System.Threading.Tasks.ValueTask<Tensor<T>>(PredictNoise(noisySample, timestep));
    }

    /// <summary>
    /// Internal Generate that optionally accepts an explicit starting sample.
    /// When <paramref name="initialSample"/> is null, fresh Gaussian noise is
    /// drawn (the standard text-to-image / unconditional-generation path).
    /// When non-null, the supplied vector is used as the initial noisy sample
    /// and the denoising loop runs starting from it — this is what
    /// <see cref="Predict(Tensor{T})"/> uses to honour its
    /// "input is the noisy starting point" contract (cf. PyTorch diffusers'
    /// <c>pipeline(latents=...)</c> override). Using a hash of the input for
    /// the RNG seed and then discarding the input would defeat the contract,
    /// since the model's output would no longer depend on the input values
    /// themselves — failing tests like NoiseSchedule_ShouldBeMonotonic that
    /// scale the input and expect the output magnitude to track the scale.
    /// </summary>
    /// <summary>
    /// Shared input validation for the sync and async Generate paths. Throws
    /// the same exceptions both surfaces produce, computes the total element
    /// count with overflow checking. Centralised so a fix only has to land
    /// in one place.
    /// </summary>
    private static void ValidateGenerateInputs(int[] shape, int numInferenceSteps, out long totalElements)
    {
        if (shape == null || shape.Length == 0)
            throw new ArgumentException("Shape must be a non-empty array.", nameof(shape));
        if (numInferenceSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(numInferenceSteps), "Must be positive.");

        var invalidDims = shape.Where(d => d <= 0).ToArray();
        if (invalidDims.Length > 0)
            throw new ArgumentOutOfRangeException(nameof(shape), $"All dimensions must be positive, but found {invalidDims[0]}.");

        totalElements = 1;
        foreach (var dim in shape)
            totalElements = checked(totalElements * dim);
        if (totalElements > int.MaxValue)
            throw new ArgumentException("Total tensor size exceeds maximum supported size.", nameof(shape));
    }

    /// <summary>
    /// Resolves the initial denoising sample: if the caller passed an explicit
    /// <paramref name="initialSample"/>, validates its length and returns it
    /// directly; otherwise draws fresh Gaussian noise via the seeded RNG.
    /// Shared between the sync and async Generate paths.
    /// </summary>
    private Vector<T> ResolveInitialSample(int[] shape, int numInferenceSteps, int? seed, Vector<T>? initialSample, long totalElements)
    {
        if (initialSample is not null)
        {
            if (initialSample.Length != (int)totalElements)
                throw new ArgumentException(
                    $"initialSample.Length ({initialSample.Length}) must equal the product of shape ({(int)totalElements}).",
                    nameof(initialSample));
            return initialSample;
        }
        var rng = CreateInferenceRng(seed);
        return SampleNoise((int)totalElements, rng);
    }

    /// <summary>
    /// Replaces every NaN / Infinity element of <paramref name="sample"/> with
    /// zero (Ho et al. 2020 §3.2 paper-minimum "Predict returns finite tensor"
    /// contract). Returns the sanitised count for diagnostic logging.
    /// </summary>
    private int SanitizeNonFiniteElements(Vector<T> sample)
    {
        int sanitizedCount = 0;
        for (int si = 0; si < sample.Length; si++)
        {
            double v = NumOps.ToDouble(sample[si]);
            if (double.IsNaN(v) || double.IsInfinity(v))
            {
                sample[si] = NumOps.Zero;
                sanitizedCount++;
            }
        }
        return sanitizedCount;
    }

    protected virtual Tensor<T> Generate(int[] shape, int numInferenceSteps, int? seed, Vector<T>? initialSample)
    {
        ValidateGenerateInputs(shape, numInferenceSteps, out long totalElements);

        // Suppress tape recording during inference (like PyTorch torch.no_grad())
        using var _ = new NoGradScope<T>();

        Vector<T> sample = ResolveInitialSample(shape, numInferenceSteps, seed, initialSample, totalElements);

        // Set up the scheduler for inference
        _scheduler.SetTimesteps(numInferenceSteps);

        // Pre-allocate reusable tensor for the denoising loop to avoid
        // creating a new Tensor per step (50 allocations → 1)
        var sampleTensor = new Tensor<T>(shape, sample);

        // Pre-allocate reusable noise prediction vector to avoid per-step allocation
        var noisePredVec = new Vector<T>(sample.Length);

        // Forward caching allocator (Tensors #661 consumer wiring, second boundary
        // after NeuralNetworkBase.Predict): a NoisePredictorBase forward is NOT a
        // NeuralNetworkBase.Predict call, so its per-step intermediate tensors are
        // never recycled by the Predict funnel. Run the whole denoise loop inside one
        // TensorArena and Reset() per step so every step's predictor + scheduler
        // intermediates are recycled rather than GC-churned — ~50 forwards per
        // generation is the dominant inference-time allocation source. The carried
        // latent (`sample`) is the one value that must survive each Reset, so it is
        // detached to a GC-owned buffer below. The pre-allocated `sampleTensor` /
        // `noisePredVec` are created above (outside the arena) and are therefore GC.
        // (The async GenerateAsyncCore path is intentionally NOT wrapped: TensorArena.Current
        // is [ThreadStatic] and `await ...ConfigureAwait(false)` can resume on another thread,
        // which would desynchronise the arena scope — an async-aware arena is a separate change.)
        //
        // Gated by DiffusionDenoiseEnabled (default OFF), NOT the default-on Predict-funnel flag:
        // unlike the single-shot Predict funnel, the multi-step denoise loop is NOT arena-safe.
        // Diffusion forward layers hold cross-forward cached tensors (e.g. DiffusionResBlock's
        // pre-allocated GroupNorm output buffer; attention reshape scratch) first allocated inside
        // this arena scope; the per-step Reset() recycles that memory, so a later step's allocation
        // aliases a still-referenced cached buffer and corrupts its shape — observed as a downsample
        // conv output emerging with a stale [B, H*W, C] attention layout, surfacing as "Input has N
        // channels but layer expects M" (and a native host crash when it corrupts the shared pool).
        // Re-enable only after the layer caches are made arena-safe. See issue #1668.
        using var arena = (InferenceArenaSettings.Enabled && InferenceArenaSettings.DiffusionDenoiseEnabled)
            ? TensorArena.Create()
            : null;

        // Iterative denoising loop
        foreach (var timestep in _scheduler.Timesteps)
        {
            // Update tensor data in-place from sample vector using Span copy
            var sampleSpan = sample.AsSpan();
            var tensorSpan = sampleTensor.AsWritableSpan();
            sampleSpan.CopyTo(tensorSpan);

            // Predict the noise
            var noisePrediction = PredictNoiseStep(sampleTensor, timestep);

            // Copy prediction to pre-allocated vector (avoids ToVector() allocation).
            // Fail fast on length mismatch — silently truncating or leaving stale
            // values in noisePredVec would produce corrupted denoising steps.
            var predSpan = noisePrediction.AsSpan();
            if (predSpan.Length != noisePredVec.Length)
            {
                throw new InvalidOperationException(
                    $"PredictNoise output length ({predSpan.Length}) does not match the " +
                    $"latent/sample length ({noisePredVec.Length}). Check that the noise " +
                    $"predictor's output shape matches the input tensor shape.");
            }
            for (int idx = 0; idx < predSpan.Length; idx++)
                noisePredVec[idx] = predSpan[idx];

            // Perform one denoising step
            // eta=0 for deterministic generation
            sample = _scheduler.Step(
                noisePredVec,
                timestep,
                sample,
                NumOps.Zero);

            // Detach the carried latent to a GC-owned buffer so it survives the
            // arena Reset() below: Vector arithmetic in _scheduler.Step routes its
            // output through TensorAllocator.Rent (arena Tier-0), so without this copy
            // `sample` would point into recycled scratch on the next step. new Vector<T>(len)
            // allocates a plain GC array; arena?.Reset() then recycles every other per-step
            // tensor (predictor intermediates, scheduler temporaries).
            if (arena != null)
            {
                var detached = new Vector<T>(sample.Length);
                sample.AsSpan().CopyTo(detached.AsWritableSpan());
                sample = detached;
            }

            // NaN/Inf guard per Ho et al. 2020 §3.2: a trained DDPM produces
            // bounded predictions ε ≈ N(0, I), but an untrained / randomly-
            // initialized noise predictor can emit values orders of magnitude
            // past that, and the scheduler's α_t / β_t math then accumulates
            // the blow-up into Inf/NaN within a handful of steps. Clip
            // non-finite elements to zero so Predict on an untrained model
            // returns a finite tensor (the documented paper-minimum
            // contract), matching the Song et al. 2020 DDIM paper's
            // "noise-only sampling = finite noise output" invariant.
            int sanitizedCount = SanitizeNonFiniteElements(sample);
            if (sanitizedCount > 0)
            {
                System.Diagnostics.Trace.TraceWarning(
                    $"DiffusionModelBase.Generate: sanitized {sanitizedCount} non-finite element(s) at timestep {timestep}.");
            }

            // Recycle this step's intermediates (predictor activations + scheduler
            // temporaries) for the next step. The only value carried forward —
            // `sample` — was detached to GC above, and sampleTensor/noisePredVec
            // live outside the arena, so the reset is safe. Without this the arena
            // would accumulate all ~50 steps and only free on loop exit.
            arena?.Reset();
        }

        return new Tensor<T>(shape, sample);
    }

    /// <inheritdoc />
    public abstract Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep);

    /// <summary>
    /// Runs one denoising-step noise prediction, optionally inside a GPU deferred execution graph
    /// (AiDotNet.Tensors #642) when <see cref="DiffusionModelOptions{T}.UseGpuExecutionGraph"/> is
    /// enabled and the active engine is a CUDA <c>DirectGpuTensorEngine</c>. Recording the whole
    /// forward into one graph keeps intermediates on-device, fuses kernels, overlaps streams, reuses
    /// buffers, and drops per-op host round-trips — the substrate a CUDA-graph capture replays.
    /// A recoverable failure (or a non-CUDA / unavailable engine) transparently falls back to the
    /// eager <see cref="PredictNoise"/> — but only failures that surface as exceptions are caught;
    /// an op that is not yet deferred-correct can still silently produce wrong-but-finite output, so
    /// this is NOT an unconditional "never worse than eager" guarantee (see
    /// <see cref="DiffusionModelOptions{T}.UseGpuExecutionGraph"/>).
    /// </summary>
    protected Tensor<T> PredictNoiseStep(Tensor<T> noisySample, int timestep)
    {
        if (!_options.UseGpuExecutionGraph)
            return PredictNoise(noisySample, timestep);

        var engine = AiDotNetEngine.Current as AiDotNet.Tensors.Engines.DirectGpuTensorEngine;
        if (engine is null || !engine.IsGpuAvailable)
        {
            // Enabled but there is no CUDA engine to record into → this IS a fallback per the
            // diagnostics contract (the "no GPU" case documented on
            // DiffusionDeferredStepDiagnostics.FellBackCount), so count it rather than returning
            // eager silently and undercounting non-CUDA fallbacks.
            DiffusionDeferredStepDiagnostics.RecordFellBack();
            return PredictNoise(noisySample, timestep);
        }

        try
        {
            using var scope = engine.BeginDeferredScope();
            if (scope is null)
            {
                DiffusionDeferredStepDiagnostics.RecordFellBack();
                return PredictNoise(noisySample, timestep);
            }

            // Ops inside PredictNoise route through DeferredScope.Current.RecordingBackend
            // automatically; Execute() replays the recorded, optimized graph. The result tensor
            // stays GPU-resident (lazy) and is materialised on first CPU read by the caller — its
            // output buffer is owned by the activation cache, so it survives the scope dispose.
            var prediction = PredictNoise(noisySample, timestep);
            scope.Execute();
            // Materialise the result to a GC-owned CPU tensor BEFORE the deferred scope disposes.
            // The lazy result is GPU-resident and its backing buffer is owned by the scope's
            // recording backend / activation cache, which is torn down on scope dispose — reading
            // it AFTER dispose (as the denoise loop does, via AsSpan) yields recycled/garbage
            // values. ToArray() forces the GPU->CPU download while the buffers are still alive,
            // and wrapping in a fresh Tensor detaches from the scope-owned storage. (Mirrors the
            // inference-arena DetachFromArena pattern.)
            var dims = new int[prediction.Rank];
            for (int d = 0; d < dims.Length; d++) dims[d] = prediction.Shape[d];
            var materialized = new Tensor<T>(prediction.ToArray(), dims);
            DiffusionDeferredStepDiagnostics.RecordExecuted();
            return materialized;
        }
        // Fall back to eager only for RECOVERABLE failures (an op not yet deferred-correct, a
        // transient GPU/runtime fault, a deferred-scope rejection). Unrecoverable conditions
        // (host OOM, access violation, stack overflow) are NOT masked — re-running the same
        // forward eagerly cannot help and would hide a real fault behind a "never worse than
        // eager" claim, exactly the silent-wrong-output risk #1650 is meant to avoid.
        catch (System.Exception ex) when (!IsUnrecoverable(ex))
        {
            DiffusionDeferredStepDiagnostics.RecordFellBack();
            System.Diagnostics.Trace.TraceWarning(
                "DiffusionModelBase.PredictNoiseStep: deferred GPU forward failed, falling back to "
              + $"eager PredictNoise (timestep {timestep}): {ex.GetType().Name}: {ex.Message}");
            return PredictNoise(noisySample, timestep);
        }
    }

    private static bool IsUnrecoverable(System.Exception ex) =>
        ex is OutOfMemoryException
           or System.StackOverflowException
           or System.AccessViolationException;

    /// <inheritdoc />
    /// <remarks>
    /// Strict single-timestep contract. Multi-timestep batch processing (different
    /// timesteps per sample) is not implemented at the base level — concrete
    /// diffusion variants that need per-sample timesteps must override this and
    /// loop over the batch. Passing a multi-element <paramref name="timesteps"/>
    /// throws <see cref="NotSupportedException"/> rather than silently using
    /// only <c>timesteps[0]</c> and corrupting the loss.
    /// </remarks>
    public virtual T ComputeLoss(Tensor<T> cleanSamples, Tensor<T> noise, int[] timesteps)
    {
        if (cleanSamples == null)
            throw new ArgumentNullException(nameof(cleanSamples));
        if (noise == null)
            throw new ArgumentNullException(nameof(noise));
        if (timesteps == null || timesteps.Length == 0)
            throw new ArgumentException("Timesteps must be a non-empty array.", nameof(timesteps));
        if (timesteps.Length != 1)
        {
            throw new NotSupportedException(
                $"DiffusionModelBase.ComputeLoss requires exactly one timestep " +
                $"(got {timesteps.Length}). Override this method in subclasses that need " +
                $"per-sample timesteps to loop over the batch and accumulate per-sample loss.");
        }

        var cleanVector = cleanSamples.ToVector();
        var noiseVector = noise.ToVector();

        // Add noise to clean samples at the given timestep.
        var noisySample = _scheduler.AddNoise(cleanVector, noiseVector, timesteps[0]);

        // Create tensor for noise prediction
        var noisySampleTensor = new Tensor<T>(cleanSamples._shape, noisySample);

        // Predict the noise
        var predictedNoise = PredictNoise(noisySampleTensor, timesteps[0]);

        // Compute MSE between predicted and actual noise
        return LossFunction.CalculateLoss(predictedNoise.ToVector(), noiseVector);
    }

    #endregion

    #region IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>> Implementation

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Performs one training step using the denoising score matching objective.
    /// Computes gradients and updates model parameters using the configured learning rate.
    /// </para>
    /// <para><b>For Beginners:</b> This method performs a single training iteration:
    /// <list type="number">
    /// <item><description>Computes how wrong the model's noise predictions are (gradients)</description></item>
    /// <item><description>Adjusts the model's parameters using the learning rate to make better predictions</description></item>
    /// </list>
    /// You can control the step size by setting the LearningRate in the options.</para>
    /// </remarks>
    /// <summary>
    /// Forces quantization-aware training ON (G5, #1624): from the next <see cref="Train"/> step the
    /// forward pass uses fake-quantized weights (quantize→dequantize, simulating int8 inference
    /// precision) while keeping full-precision shadow weights that the optimizer updates (a
    /// straight-through estimator), so the model learns weights that survive post-training int8
    /// quantization. Foundation-scale models already engage this <b>by default</b> (above
    /// <see cref="DefaultQatThresholdParams"/>); call this to force it on a smaller model or to override
    /// the config. QAT is lossy — it changes the training trajectory. Pass a
    /// <see cref="QuantizationConfiguration"/> to tune bit width / symmetry; default is symmetric int8.
    /// </summary>
    public void EnableQuantizationAwareTraining(QuantizationConfiguration? config = null)
    {
        _qatExplicit = true;
        _qatConfig = config;
        _qatHook = null; // rebuilt lazily in Train with the new config
    }

    /// <summary>Forces quantization-aware training OFF, overriding the parameter-count default.</summary>
    public void DisableQuantizationAwareTraining()
    {
        _qatExplicit = false;
        _qatHook = null;
    }

    /// <summary>
    /// Whether quantization-aware training is engaged for <see cref="Train"/> (G5, #1624). Defaults to ON
    /// for foundation-scale models (<see cref="ParameterCount"/> ≥ <see cref="DefaultQatThresholdParams"/>)
    /// and OFF for smaller ones, unless overridden via <see cref="EnableQuantizationAwareTraining"/> /
    /// <see cref="DisableQuantizationAwareTraining"/>.
    /// </summary>
    public bool IsQuantizationAwareTrainingEnabled =>
        // Opt-in, default OFF — matching the Train() contract comment, the project's
        // streaming/activation-checkpointing convention, and PyTorch (torch QAT is always
        // explicit, never engaged by model size). The previous auto-engage-by-parameter-
        // count (>= 500M) silently turned QAT ON for every foundation-scale diffusion model:
        // each train step then fake-quantized ALL ~1.8B weights (a serial ToVector copy of
        // the full weight set + per-element requantize + copy-back), which a CPU profile
        // measured at ~96s/iter of pure overhead — over half a Kandinsky train step — AND,
        // being lossy, changed the training trajectory of the vanilla-DDPM contract tests.
        // Foundation models that target int8 deployment opt in via EnableQuantizationAwareTraining().
        _qatExplicit ?? false;

    public virtual void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Copy-on-write: if this model shares weight tensors with a clone/parent, give it a private
        // copy before we mutate weights in place below, so the other model isn't corrupted.
        EnsureOwnWeights();

        // G5 (#1624) quantization-aware training (opt-in, default OFF). Fake-quantize the weights the
        // forward will use, keeping full-precision shadows to restore before the optimizer update — the
        // straight-through estimator: gradients computed against the quantized values update the
        // full-precision weights. Quantization happens OUTSIDE the gradient tape, so the STE is implicit
        // (the tape treats the quantized values as the leaf weights). On the very first step the lazy
        // layers aren't resolved yet (empty/zero-length collection) so this is a natural no-op warmup.
        Tensor<T>[]? qatParams = null;
        Vector<T>[]? qatShadows = null;
        if (IsQuantizationAwareTrainingEnabled)
        {
            _qatHook ??= new QATTrainingHook<T>(_qatConfig ?? new QuantizationConfiguration { QATWarmupEpochs = 0 });
            qatParams = CollectTrainableParameters();
            if (qatParams.Length > 0)
            {
                _qatHook.Reset(); // fresh per-tensor scales each step, tracking the current weight range
                qatShadows = new Vector<T>[qatParams.Length];
                for (int i = 0; i < qatParams.Length; i++)
                {
                    qatShadows[i] = qatParams[i].ToVector();
                    if (qatShadows[i].Length == 0) continue;
                    var quantized = _qatHook.ApplyFakeQuantization(qatShadows[i], "qat_param_" + i);
                    var span = qatParams[i].Data.Span;
                    for (int k = 0; k < span.Length && k < quantized.Length; k++) span[k] = quantized[k];
                }
            }
        }

        // Tape-based direct per-tensor SGD step. The forward pass records Engine
        // ops onto the thread-local gradient tape, backward returns per-tensor
        // gradients, and we apply them in place via param -= lr * grad. This
        // bypasses the legacy flat-vector round-trip (GetParameters →
        // FlattenGradients → ApplyGradients → SetParameters) entirely — that
        // path doesn't work once the reflection walker discovers more trainable
        // tensors than GetParameters knows about, which is now the norm after
        // migrating layers like FlashAttentionLayer from Matrix<T> to Tensor<T>.

        // Sample a random timestep and build the noisy training sample.
        var timestep = RandomGenerator.Next(_scheduler.Config.TrainTimesteps);
        var inputVector = input.ToVector();
        var noiseVector = SampleNoise(inputVector.Length, RandomGenerator);
        var noisySample = _scheduler.AddNoise(inputVector, noiseVector, timestep);
        var noisySampleTensor = new Tensor<T>(input._shape, noisySample);

        using var tape = new GradientTape<T>();

        // Forward pass — triggers lazy layer initialization, then we walk for
        // trainable parameters. Collection must happen AFTER the forward pass so
        // newly-initialized layers are visible to the walker.
        var predicted = PredictNoise(noisySampleTensor, timestep);
        var paramTensors = CollectTrainableParameters();
        if (paramTensors.Length == 0)
        {
            throw new InvalidOperationException(
                $"{GetType().Name} has no trainable parameters discoverable via " +
                "CollectTrainableParameters. Make sure layers register their weights via " +
                "LayerBase.RegisterTrainableParameter so the gradient tape can reach them.");
        }

        // MSE loss against the true noise — tape-tracked.
        var noiseTensor = new Tensor<T>(predicted._shape, noiseVector);
        var diff = Engine.TensorSubtract(predicted, noiseTensor);
        var sq = Engine.TensorMultiply(diff, diff);
        // DDPM training objective (Ho et al. 2020, the simplified loss L_simple =
        // E[||ε − ε_θ||²]) is the MEAN squared error between predicted and true
        // noise — matching the reference implementations (HuggingFace diffusers
        // trains with F.mse_loss, which uses mean reduction). A SUM here scales
        // the gradient by the element count (e.g. 1024× for a [1,4,16,16] latent),
        // which destabilised the plain-SGD update: Imagen2's
        // Training_ShouldReducePredictionError saw the error RISE (0.317 → 0.321)
        // over 10 steps because each step overshot. Mean keeps the gradient scale
        // invariant to latent size, so the same LearningRate is stable regardless
        // of resolution. d(mean)/dparam = (1/N)·d(sum)/dparam, so this is exactly
        // the mean-MSE gradient.
        var loss = Engine.TensorMultiplyScalar(
            Engine.ReduceSum(sq, null), NumOps.FromDouble(1.0 / sq.Length));

        // Backward pass via graph-based autodiff.
        var grads = tape.ComputeGradients(loss, paramTensors);

        // Global gradient-norm clipping (canonical diffusion training: HuggingFace
        // diffusers and the SVD / Video-Diffusion reference recipes call
        // torch.nn.utils.clip_grad_norm_(params, max_norm=1.0) after every backward).
        // Plain SGD without it can overshoot on a freshly random-initialised model —
        // a single step can move the noise-prediction error the WRONG way before the
        // weights settle, which is exactly what Training_ShouldReducePredictionError
        // caught for FateZero (error rose 0.399 -> 0.456 over 10 steps on an unlucky
        // init, while the seeded sibling models happened to start in a stable basin).
        // Clipping rescales the WHOLE gradient by max_norm/‖g‖ when ‖g‖ exceeds the
        // threshold, so it bounds the step magnitude while preserving its direction —
        // it can only stabilise training, never reverse a descent direction. The norm
        // is computed with Engine reductions (no per-element scalar loop).
        const double MaxGradNorm = 1.0;
        T sumSq = NumOps.Zero;
        foreach (var param in paramTensors)
        {
            if (!grads.TryGetValue(param, out var g) || g is null) continue;
            var gsq = Engine.ReduceSum(Engine.TensorMultiply(g, g), null);
            sumSq = NumOps.Add(sumSq, gsq[0]);
        }
        double gradNorm = Math.Sqrt(Convert.ToDouble(sumSq));
        T clipScale = gradNorm > MaxGradNorm
            ? NumOps.FromDouble(MaxGradNorm / gradNorm)
            : NumOps.One;
        T effectiveLr = NumOps.Multiply(LearningRate, clipScale);

        // G5 (#1624): restore the full-precision shadow weights before the optimizer step, so the
        // update lands on full precision (straight-through estimator). The forward used the quantized
        // values; the gradients are computed against them but applied to the full-precision weights.
        if (qatParams is not null && qatShadows is not null)
        {
            for (int i = 0; i < qatParams.Length; i++)
            {
                var span = qatParams[i].Data.Span;
                var shadow = qatShadows[i];
                for (int k = 0; k < span.Length && k < shadow.Length; k++) span[k] = shadow[k];
            }
        }

        // Per-tensor SGD: param -= effectiveLr * grad, applied in place so registered
        // tensor references stay stable across training steps.
        foreach (var param in paramTensors)
        {
            if (!grads.TryGetValue(param, out var grad) || grad is null) continue;
            var update = Engine.TensorMultiplyScalar(grad, effectiveLr);
            var paramSpan = param.Data.Span;
            var updateSpan = update.AsSpan();
            int n = Math.Min(paramSpan.Length, updateSpan.Length);
            for (int i = 0; i < n; i++)
            {
                paramSpan[i] = NumOps.Subtract(paramSpan[i], updateSpan[i]);
            }
        }
    }

    /// <inheritdoc />
    public virtual Tensor<T> Predict(Tensor<T> input)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        // Diffusion Predict semantics: the input IS the noisy starting sample,
        // not a seed source. The denoising loop starts from `input` and runs
        // backward through the scheduler timesteps. Hashing the input to a
        // seed and then sampling fresh noise (the previous behaviour) makes
        // the output independent of the input *values* — only of the seed —
        // which violates the "scaling the input scales the output" invariant
        // that NoiseSchedule_ShouldBeMonotonic checks. Mirrors PyTorch
        // diffusers' `pipeline(latents=...)` start-from-latents path.
        var initial = new Vector<T>(input.Length);
        var src = input.AsSpan();
        var dst = initial.AsWritableSpan();
        src.CopyTo(dst);
        return Generate(input._shape, _options.DefaultInferenceSteps, seed: null, initialSample: initial);
    }

    /// <inheritdoc />
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = GetType().Name,
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount),
            Complexity = ParameterCount,
            Description = $"Diffusion model with {ParameterCount} parameters using {_scheduler.GetType().Name} scheduler."
        };
    }

    #endregion

    #region IParameterizable<T, Tensor<T>, Tensor<T>> Implementation

    /// <inheritdoc />
    public abstract Vector<T> GetParameters();

    /// <inheritdoc />
    public abstract void SetParameters(Vector<T> parameters);

    /// <inheritdoc />
    public virtual IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        var clone = (DiffusionModelBase<T>)Clone();
        clone.SetParameters(parameters);
        return clone;
    }

    #endregion

    #region IModelSerializer Implementation

    /// <inheritdoc />
    public virtual byte[] Serialize()
    {
        ModelPersistenceGuard.EnforceBeforeSerialize();
        using var stream = new MemoryStream();
        SaveState(stream);
        return stream.ToArray();
    }

    /// <inheritdoc />
    public virtual void Deserialize(byte[] data)
    {
        ModelPersistenceGuard.EnforceBeforeDeserialize();
        using var stream = new MemoryStream(data);
        LoadState(stream);
    }

    /// <inheritdoc/>
    public virtual int[] GetInputShape()
    {
        return new[] { ParameterCountHelper.ToFlatVectorSize(ParameterCount) };
    }

    /// <inheritdoc/>
    public virtual int[] GetOutputShape()
    {
        return new[] { ParameterCountHelper.ToFlatVectorSize(ParameterCount) };
    }

    /// <inheritdoc/>
    public virtual DynamicShapeInfo GetDynamicShapeInfo()
    {
        return DynamicShapeInfo.None;
    }


    /// <inheritdoc />
    public virtual void SaveModel(string filePath)
    {
        var data = Serialize();
        byte[] envelopedData = ModelFileHeader.WrapWithHeader(
            data, this, GetInputShape(), GetOutputShape(), SerializationFormat.Binary);
        File.WriteAllBytes(filePath, envelopedData);
    }

    /// <inheritdoc />
    public virtual void LoadModel(string filePath)
    {
        var data = File.ReadAllBytes(filePath);

        // Extract payload from AIMF envelope
        data = ModelFileHeader.ExtractPayload(data);

        Deserialize(data);
    }

    #endregion

    #region ICheckpointableModel Implementation

    /// <inheritdoc />
    public virtual void SaveState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));
        if (!stream.CanWrite)
            throw new ArgumentException("Stream must be writable.", nameof(stream));

        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        // Save version for future compatibility
        writer.Write(1); // Version 1

        // Save scheduler config
        writer.Write(_scheduler.Config.TrainTimesteps);
        writer.Write(NumOps.ToDouble(_scheduler.Config.BetaStart));
        writer.Write(NumOps.ToDouble(_scheduler.Config.BetaEnd));
        writer.Write((int)_scheduler.Config.BetaSchedule);
        writer.Write((int)_scheduler.Config.PredictionType);
        writer.Write(_scheduler.Config.ClipSample);

        // Save model parameters using SerializationHelper
        SerializationHelper<T>.SerializeVector(writer, GetParameters());

        stream.Flush();
    }

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Loads model state from a stream, including scheduler configuration validation
    /// and model parameters. Throws if the saved scheduler config doesn't match
    /// the current instance's scheduler configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This restores a previously saved model. The scheduler
    /// settings must match between the saved model and this instance to ensure
    /// the loaded parameters work correctly with the noise schedule.</para>
    /// </remarks>
    public virtual void LoadState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));
        if (!stream.CanRead)
            throw new ArgumentException("Stream must be readable.", nameof(stream));

        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        // Read version
        var version = reader.ReadInt32();
        if (version != 1)
            throw new InvalidOperationException($"Unsupported model version: {version}");

        // Read and validate scheduler config
        var savedTrainTimesteps = reader.ReadInt32();
        var savedBetaStart = reader.ReadDouble();
        var savedBetaEnd = reader.ReadDouble();
        var savedBetaSchedule = (BetaSchedule)reader.ReadInt32();
        var savedPredictionType = (DiffusionPredictionType)reader.ReadInt32();
        var savedClipSample = reader.ReadBoolean();

        // Validate critical scheduler parameters match
        if (savedTrainTimesteps != _scheduler.Config.TrainTimesteps)
        {
            throw new InvalidOperationException(
                $"Scheduler config mismatch: saved TrainTimesteps={savedTrainTimesteps}, " +
                $"current={_scheduler.Config.TrainTimesteps}. Create a model with matching scheduler config.");
        }

        if (Math.Abs(savedBetaStart - NumOps.ToDouble(_scheduler.Config.BetaStart)) > 1e-9)
        {
            throw new InvalidOperationException(
                $"Scheduler config mismatch: saved BetaStart={savedBetaStart}, " +
                $"current={NumOps.ToDouble(_scheduler.Config.BetaStart)}. Create a model with matching scheduler config.");
        }

        if (Math.Abs(savedBetaEnd - NumOps.ToDouble(_scheduler.Config.BetaEnd)) > 1e-9)
        {
            throw new InvalidOperationException(
                $"Scheduler config mismatch: saved BetaEnd={savedBetaEnd}, " +
                $"current={NumOps.ToDouble(_scheduler.Config.BetaEnd)}. Create a model with matching scheduler config.");
        }

        if (savedBetaSchedule != _scheduler.Config.BetaSchedule)
        {
            throw new InvalidOperationException(
                $"Scheduler config mismatch: saved BetaSchedule={savedBetaSchedule}, " +
                $"current={_scheduler.Config.BetaSchedule}. Create a model with matching scheduler config.");
        }

        if (savedPredictionType != _scheduler.Config.PredictionType)
        {
            throw new InvalidOperationException(
                $"Scheduler config mismatch: saved PredictionType={savedPredictionType}, " +
                $"current={_scheduler.Config.PredictionType}. Create a model with matching scheduler config.");
        }

        if (savedClipSample != _scheduler.Config.ClipSample)
        {
            throw new InvalidOperationException(
                $"Scheduler config mismatch: saved ClipSample={savedClipSample}, " +
                $"current={_scheduler.Config.ClipSample}. Create a model with matching scheduler config.");
        }

        // Load model parameters using SerializationHelper
        SetParameters(SerializationHelper<T>.DeserializeVector(reader));
    }

    #endregion

    #region IFeatureAware Implementation

    /// <summary>
    /// Ensures active feature indices are initialized with default values if empty.
    /// </summary>
    private void EnsureActiveFeatureIndicesInitialized()
    {
        if (_activeFeatureIndices.Count == 0 && ParameterCount > 0)
        {
            for (int i = 0; i < ParameterCount; i++)
            {
                _activeFeatureIndices.Add(i);
            }
        }
    }

    /// <inheritdoc />
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        EnsureActiveFeatureIndicesInitialized();
        return _activeFeatureIndices;
    }

    /// <inheritdoc />
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        _activeFeatureIndices = new HashSet<int>(featureIndices);
    }

    /// <inheritdoc />
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        EnsureActiveFeatureIndicesInitialized();
        return _activeFeatureIndices.Contains(featureIndex);
    }

    #endregion

    #region IFeatureImportance<T> Implementation

    /// <inheritdoc />
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();
        var uniformImportance = NumOps.FromDouble(1.0 / Math.Max(1, ParameterCount));

        for (int i = 0; i < ParameterCount; i++)
        {
            importance[$"param_{i}"] = uniformImportance;
        }

        return importance;
    }

    #endregion

    #region ICloneable<IFullModel<T, Tensor<T>, Tensor<T>>> Implementation

    /// <inheritdoc />
    public abstract IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy();

    /// <inheritdoc />
    IFullModel<T, Tensor<T>, Tensor<T>> ICloneable<IFullModel<T, Tensor<T>, Tensor<T>>>.Clone()
    {
        return Clone();
    }

    /// <summary>
    /// Creates a deep copy of the model.
    /// </summary>
    /// <returns>A new instance with the same parameters.</returns>
    public abstract IDiffusionModel<T> Clone();

    #endregion

    #region IGradientComputable<T, Tensor<T>, Tensor<T>> Implementation

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Computes gradients for diffusion model training using the denoising score matching objective.
    /// This default implementation uses automatic differentiation via GradientTape when available,
    /// with a fallback to numerical gradients. Derived classes can override for custom gradient computation.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the diffusion model learns:
    /// <list type="bullet">
    /// <item><description>Take a clean sample and add random noise</description></item>
    /// <item><description>Try to predict what noise was added</description></item>
    /// <item><description>Measure how wrong the prediction was (the loss)</description></item>
    /// <item><description>Figure out how to adjust parameters to be less wrong (the gradients)</description></item>
    /// </list></para>
    /// </remarks>
    public virtual Vector<T> ComputeGradients(Tensor<T> input, Tensor<T> target, ILossFunction<T>? lossFunction = null)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (target == null)
            throw new ArgumentNullException(nameof(target));

        // Sample a random timestep
        var timestep = RandomGenerator.Next(_scheduler.Config.TrainTimesteps);

        // Sample noise and build the noisy sample
        var inputVector = input.ToVector();
        var noiseVector = SampleNoise(inputVector.Length, RandomGenerator);
        var noisySample = _scheduler.AddNoise(inputVector, noiseVector, timestep);
        var noisySampleTensor = new Tensor<T>(input._shape, noisySample);

        // Tape-based automatic differentiation. Forward pass runs first so lazy
        // layer initialization (DiTNoisePredictor.EnsureLayersInitialized etc.)
        // fires before we walk for trainable parameters. Every Engine op in the
        // forward records a GradFn entry, so tape.ComputeGradients returns exact
        // per-tensor gradients without requiring a manual backward.
        using var tape = new GradientTape<T>();
        var predicted = PredictNoise(noisySampleTensor, timestep);
        var paramTensors = CollectTrainableParameters();
        if (paramTensors.Length == 0)
        {
            throw new InvalidOperationException(
                $"{GetType().Name} has no trainable parameters discoverable via " +
                "CollectTrainableParameters. Make sure layers register their weights via " +
                "LayerBase.RegisterTrainableParameter so the gradient tape can reach them.");
        }

        var noiseTensor = new Tensor<T>(predicted._shape, noiseVector);
        var diff = Engine.TensorSubtract(predicted, noiseTensor);
        var sq = Engine.TensorMultiply(diff, diff);
        // DDPM training objective (Ho et al. 2020, the simplified loss L_simple =
        // E[||ε − ε_θ||²]) is the MEAN squared error between predicted and true
        // noise — matching the reference implementations (HuggingFace diffusers
        // trains with F.mse_loss, which uses mean reduction). A SUM here scales
        // the gradient by the element count (e.g. 1024× for a [1,4,16,16] latent),
        // which destabilised the plain-SGD update: Imagen2's
        // Training_ShouldReducePredictionError saw the error RISE (0.317 → 0.321)
        // over 10 steps because each step overshot. Mean keeps the gradient scale
        // invariant to latent size, so the same LearningRate is stable regardless
        // of resolution. d(mean)/dparam = (1/N)·d(sum)/dparam, so this is exactly
        // the mean-MSE gradient.
        var loss = Engine.TensorMultiplyScalar(
            Engine.ReduceSum(sq, null), NumOps.FromDouble(1.0 / sq.Length));
        var grads = tape.ComputeGradients(loss, paramTensors);

        // Flatten gradients into a single vector matching the tape-collected
        // parameter order. External callers that go through IGradientComputable
        // still get a flat Vector<T>, though the internal Train path prefers the
        // per-tensor direct apply via TryTapeDirectTrainStep to avoid the flat
        // round-trip entirely.
        return FlattenGradients(paramTensors, grads);
    }

    /// <summary>
    /// Collects all trainable parameter tensors from the noise predictor's layers.
    /// Used by tape-based training to identify which tensors need gradients.
    /// </summary>
    protected virtual Tensor<T>[] CollectTrainableParameters()
    {
        // Cached reflection walk: the walker traverses the full object graph to
        // find every ITrainableLayer's parameter tensors. Layer structure and
        // tensor references are stable after construction (DenseLayer.SetParameters
        // modifies in place), so we only need to walk once per model instance.
        if (_cachedTrainableParameters is not null)
            return _cachedTrainableParameters;

        var allParams = new List<Tensor<T>>();
        CollectLayerParameters(this, allParams, new HashSet<object>(AiDotNet.Helpers.TensorReferenceComparer<object>.Instance));

        // Only cache non-empty results. An empty result usually means lazy
        // initialization hasn't run yet — don't pin that empty list.
        if (allParams.Count > 0)
            _cachedTrainableParameters = allParams.ToArray();

        return allParams.ToArray();
    }

    /// <summary>
    /// Invalidates the cached trainable-parameter walk. Call this from subclasses
    /// that swap layer references at runtime so the next training step re-discovers
    /// the updated structure.
    /// </summary>
    protected void InvalidateTrainableParametersCache()
    {
        _cachedTrainableParameters = null;
    }

    private void CollectLayerParameters(object? obj, List<Tensor<T>> allParams, HashSet<object> visited)
    {
        if (obj is null || !visited.Add(obj)) return;

        if (obj is Interfaces.ITrainableLayer<T> trainable)
        {
            var parameters = trainable.GetTrainableParameters();
            if (parameters is not null)
            {
                foreach (var p in parameters)
                    if (p is not null && p.Length > 0) allParams.Add(p);
            }
        }

        // Recurse into every reference-type instance field so nested composites
        // (e.g., DiffusionModel -> UNetNoisePredictor -> List<Layer>) are fully
        // walked even when the intermediate types don't implement ITrainableLayer.
        // The visited set handles cycles.
        var type = obj.GetType();
        if (type.IsPrimitive || type == typeof(string) || type.IsEnum) return;

        foreach (var field in type.GetFields(
            System.Reflection.BindingFlags.Instance |
            System.Reflection.BindingFlags.NonPublic |
            System.Reflection.BindingFlags.Public))
        {
            // Skip compiler-generated backing fields for non-ref properties and
            // fields whose declared type can't hold a trainable layer.
            if (field.FieldType.IsPrimitive || field.FieldType.IsEnum ||
                field.FieldType == typeof(string) || field.FieldType == typeof(Tensor<T>))
                continue;

            var val = field.GetValue(obj);
            if (val is null) continue;

            if (val is System.Collections.IEnumerable enumerable && val is not string)
            {
                foreach (var item in enumerable)
                    CollectLayerParameters(item, allParams, visited);
            }
            else
            {
                CollectLayerParameters(val, allParams, visited);
            }
        }
    }

    /// <summary>
    /// COW clone lever (#1624): shares each trainable weight tensor's STORAGE with <paramref name="source"/>
    /// via the Tensors copy-on-write <c>Tensor&lt;T&gt;.CloneShared()</c> (O(1)-until-write), instead of the
    /// flat <c>GetParameters()</c> → <c>SetParameters()</c> round-trip that materializes the entire model a
    /// second time (and a giant intermediate flat vector) — the source of the large-diffusion-model Clone
    /// OOM on the 16 GB runner. The first in-place write to either side privatizes that tensor, so the clone
    /// is observationally identical to the flat-copy clone. Fidelity is equivalent to the existing path
    /// because both transfer exactly the model's trainable tensors (diffusion models carry no non-trainable
    /// running statistics / serialization extras) — this just shares them rather than copying.
    ///
    /// <para>Walks <paramref name="source"/> and <c>this</c> in parallel via reflection (identical type ⇒
    /// identical field order ⇒ matching layer order, the same assumption <see cref="CollectTrainableParameters"/>
    /// already relies on) and re-binds each destination layer's parameters through
    /// <see cref="Interfaces.ITrainableLayer{T}.SetTrainableParameters"/>. Returns <c>false</c> if the
    /// trainable-layer structure does not match 1:1 (the caller then falls back to the flat copy), and never
    /// leaves a half-shared clone — structure is fully verified before any sharing.</para>
    /// </summary>
    protected bool TryShareParametersFrom(DiffusionModelBase<T> source)
    {
        if (!AiDotNet.Helpers.CopyOnWriteCloneHelper.TryShareTrainableParameters<T>(source, this))
            return false;
        InvalidateTrainableParametersCache();
        return true;
    }

    /// <summary>
    /// Flattens gradient tensors into a single vector matching GetParameters() layout.
    /// </summary>
    private Vector<T> FlattenGradients(Tensor<T>[] paramTensors, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int totalSize = 0;
        foreach (var p in paramTensors) totalSize += p.Length;

        var flat = new Vector<T>(totalSize);
        int offset = 0;
        foreach (var p in paramTensors)
        {
            if (grads.TryGetValue(p, out var grad))
            {
                var gradSpan = grad.AsSpan();
                for (int i = 0; i < gradSpan.Length; i++)
                    flat[offset + i] = gradSpan[i];
            }
            offset += p.Length;
        }
        return flat;
    }

    /// <inheritdoc />
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        var parameters = GetParameters();

        // Vectorized SGD: params = params - lr * gradients
        var scaledGradients = Engine.Multiply(gradients, learningRate);
        parameters = Engine.Subtract(parameters, scaledGradients);

        SetParameters(parameters);
    }

    #endregion


    #region Helper Methods

    /// <summary>
    /// Samples random noise from a standard normal distribution.
    /// </summary>
    /// <param name="length">The number of elements to sample.</param>
    /// <param name="rng">The random number generator to use.</param>
    /// <returns>A vector of random noise values.</returns>
    protected virtual Vector<T> SampleNoise(int length, Random rng)
    {
        var noise = new Vector<T>(length);

        for (int i = 0; i < length; i++)
        {
            noise[i] = NumOps.FromDouble(rng.NextGaussian());
        }

        return noise;
    }

    #endregion

    #region IDisposable

    /// <inheritdoc />
    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Cascades Dispose to every disposable component the concrete model exposes
    /// via <see cref="EnumerateDisposableComponents"/> (default: reflection walk
    /// over instance fields), plus the owned <c>_scheduler</c>. Concrete diffusion
    /// models that want to constrain WHAT gets disposed (e.g., skip injected
    /// dependencies they don't own) override <see cref="EnumerateDisposableComponents"/>
    /// to return an explicit allow-list. Models that hold additional disposable
    /// composites beyond reflection-walk reach can also override this method and
    /// call <c>base.Dispose(disposing)</c>.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed || !disposing) return;
        _disposed = true;

        // Always dispose the scheduler we own — schedulers may hold buffers
        // (precomputed alpha/beta arrays, native handles for accelerated
        // sampling) that survive model disposal otherwise. Route through the
        // guard so the scheduler instance is disposed at most once even if
        // another owner also cascades into it.
        if (_scheduler is IDisposable disposableScheduler)
        {
            AiDotNet.Helpers.DisposeOnceGuard.TryDispose(disposableScheduler);
        }

        // Cascade to every disposable component the concrete model exposes via
        // EnumerateDisposableComponents (default: reflection walk over instance
        // fields). Shared components (a predictor reused across two diffusion
        // wrappers for ensembling, a VAE loaded once and injected into several
        // models) are common — the guard ensures each instance is disposed
        // exactly once regardless of how many cascades reach it. Many
        // components aren't idempotent on double-Dispose (they'd double-return
        // pooled buffers or crash on null derefs), which is why a plain
        // try/catch around ObjectDisposedException is insufficient.
        foreach (var component in EnumerateDisposableComponents())
        {
            if (component is null) continue;
            AiDotNet.Helpers.DisposeOnceGuard.TryDispose(component);
        }
    }

    #endregion
}

/// <summary>
/// Process-wide observability for the opt-in GPU deferred denoising step
/// (<see cref="DiffusionModelOptions{T}.UseGpuExecutionGraph"/>, #642 / #1650). The step falls back
/// to the eager forward on any recoverable failure, which would otherwise be invisible — a
/// not-yet-deferred-correct op would silently run eagerly and look "fine". These counters let tests
/// (and production telemetry) assert the deferred GPU graph ACTUALLY executed, rather than trusting a
/// bit-equivalence that an eager fallback would also satisfy.
/// </summary>
internal static class DiffusionDeferredStepDiagnostics
{
    private static long _executed;
    private static long _fellBack;

    /// <summary>Number of denoising steps whose forward ran through the deferred GPU graph + Execute().</summary>
    public static long ExecutedCount => System.Threading.Interlocked.Read(ref _executed);

    /// <summary>Number of denoising steps that fell back to the eager forward (no GPU, null scope, or a recoverable deferred failure).</summary>
    public static long FellBackCount => System.Threading.Interlocked.Read(ref _fellBack);

    public static void Reset()
    {
        System.Threading.Interlocked.Exchange(ref _executed, 0);
        System.Threading.Interlocked.Exchange(ref _fellBack, 0);
    }

    internal static void RecordExecuted() => System.Threading.Interlocked.Increment(ref _executed);
    internal static void RecordFellBack() => System.Threading.Interlocked.Increment(ref _fellBack);
}
