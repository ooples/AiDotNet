using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Finance.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Finance.Base;

/// <summary>
/// Abstract base class for time series foundation models that support multiple downstream tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// This base class extends <see cref="ForecastingModelBase{T}"/> with multi-task capabilities
/// defined by <see cref="ITimeSeriesFoundationModel{T}"/>. It provides default implementations
/// that throw <see cref="NotSupportedException"/> for optional tasks, allowing single-task models
/// (e.g., forecasting-only) to inherit without implementing every method.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the foundation that all time series foundation models build upon.
/// It provides:
/// <list type="bullet">
/// <item>Common infrastructure for both ONNX and native mode operation</item>
/// <item>Default "not supported" implementations for optional tasks</item>
/// <item>A <c>ValidateTaskSupported</c> helper to check task compatibility</item>
/// <item>Standard properties for model size, parameter count, and context limits</item>
/// </list>
///
/// Models that support only forecasting (like TimesFM) can inherit this class and only
/// override the forecasting-related methods. Multi-task models (like MOMENT) override
/// the additional task methods they support.
/// </para>
/// </remarks>
public abstract class TimeSeriesFoundationModelBase<T> : ForecastingModelBase<T>, ITimeSeriesFoundationModel<T>
{
    #region Constructors

    /// <summary>
    /// Initializes a new foundation model with deferred configuration.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="lossFunction">Optional loss function override.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor keeps the classic Finance model pattern
    /// where derived classes fill in sequence length and other settings afterward.
    /// </para>
    /// </remarks>
    protected TimeSeriesFoundationModelBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction, maxGradNorm)
    {
    }

    /// <summary>
    /// Initializes a new foundation model in native mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">Input sequence length.</param>
    /// <param name="predictionHorizon">Prediction horizon (future steps to forecast).</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="lossFunction">Optional loss function override.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when training a foundation model from scratch
    /// or fine-tuning with native C# layers.
    /// </para>
    /// </remarks>
    protected TimeSeriesFoundationModelBase(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength,
        int predictionHorizon,
        int numFeatures,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, sequenceLength, predictionHorizon, numFeatures, lossFunction)
    {
    }

    /// <summary>
    /// Initializes a new foundation model in ONNX inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="sequenceLength">Input sequence length expected by the ONNX model.</param>
    /// <param name="predictionHorizon">Prediction horizon expected by the ONNX model.</param>
    /// <param name="numFeatures">Number of input features expected by the ONNX model.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you have a pretrained ONNX model
    /// and only need fast inference.
    /// </para>
    /// </remarks>
    protected TimeSeriesFoundationModelBase(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int sequenceLength,
        int predictionHorizon,
        int numFeatures)
        : base(architecture, onnxModelPath, sequenceLength, predictionHorizon, numFeatures)
    {
    }

    #endregion

    #region ITimeSeriesFoundationModel Properties

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Override this in derived classes to list all tasks the model supports.
    /// The default implementation returns only <see cref="TimeSeriesFoundationModelTask.Forecasting"/>.
    /// </para>
    /// </remarks>
    public virtual IReadOnlyList<TimeSeriesFoundationModelTask> SupportedTasks { get; } =
        new[] { TimeSeriesFoundationModelTask.Forecasting };

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Override this in derived classes if the model supports switching
    /// between tasks at runtime.
    /// </para>
    /// </remarks>
    public virtual TimeSeriesFoundationModelTask CurrentTask { get; } = TimeSeriesFoundationModelTask.Forecasting;

    /// <inheritdoc/>
    public abstract FoundationModelSize ModelSize { get; }

    /// <inheritdoc/>
    public abstract int MaxContextLength { get; }

    /// <inheritdoc/>
    public abstract int MaxPredictionHorizon { get; }

    #endregion

    #region Multi-Task Methods (Default NotSupportedException)

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Default implementation throws <see cref="NotSupportedException"/>.
    /// Override in derived classes that support anomaly detection.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> DetectAnomalies(Tensor<T> series, double? threshold = null)
    {
        ValidateTaskSupported(TimeSeriesFoundationModelTask.AnomalyDetection);
        throw new NotSupportedException($"Override {nameof(DetectAnomalies)} in a derived class to provide an implementation.");
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Default implementation throws <see cref="NotSupportedException"/>.
    /// Override in derived classes that support classification.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> Classify(Tensor<T> series, int numClasses)
    {
        ValidateTaskSupported(TimeSeriesFoundationModelTask.Classification);
        throw new NotSupportedException($"Override {nameof(Classify)} in a derived class to provide an implementation.");
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Default implementation throws <see cref="NotSupportedException"/>.
    /// Override in derived classes that support imputation.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> Impute(Tensor<T> series, Tensor<T> mask)
    {
        ValidateTaskSupported(TimeSeriesFoundationModelTask.Imputation);
        throw new NotSupportedException($"Override {nameof(Impute)} in a derived class to provide an implementation.");
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Default implementation throws <see cref="NotSupportedException"/>.
    /// Override in derived classes that support embedding generation.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> Embed(Tensor<T> series)
    {
        ValidateTaskSupported(TimeSeriesFoundationModelTask.Embedding);
        throw new NotSupportedException($"Override {nameof(Embed)} in a derived class to provide an implementation.");
    }

    #endregion

    #region Helpers

    /// <summary>
    /// Validates that the specified task is supported by this model.
    /// </summary>
    /// <param name="task">The task to validate.</param>
    /// <exception cref="NotSupportedException">
    /// Thrown when the task is not in <see cref="SupportedTasks"/>.
    /// </exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Call this at the start of any task-specific method to give
    /// users a clear error message if they try to use an unsupported task.
    /// </para>
    /// </remarks>
    protected void ValidateTaskSupported(TimeSeriesFoundationModelTask task)
    {
        if (!SupportedTasks.Contains(task))
        {
            throw new NotSupportedException(
                $"{GetType().Name} does not support the '{task}' task. " +
                $"Supported tasks: {string.Join(", ", SupportedTasks)}.");
        }
    }

    /// <summary>
    /// Width of the diffusion-timestep embedding a denoiser packs into its input row. Kollovieh
    /// et al. 2023 encode "the diffusion timesteps into 128-dimensional positional embeddings";
    /// Ho et al. 2020 likewise size the embedding at the backbone's channel width.
    /// </summary>
    protected const int DiffusionTimestepEmbeddingDim = 128;

    /// <summary>
    /// Writes the sinusoidal embedding of a diffusion timestep into <paramref name="destination"/>.
    /// </summary>
    /// <param name="destination">The slots to fill; its length is the embedding width.</param>
    /// <param name="timestep">The diffusion timestep being embedded.</param>
    /// <remarks>
    /// <para>
    /// Ho et al. 2020 Section B specify the diffusion time as "the Transformer sinusoidal position
    /// embedding", and the reference implementations reproduce it the same way
    /// (openai/guided-diffusion timestep_embedding, HuggingFace diffusers
    /// get_timestep_embedding): half the slots hold cos(t*w_i), half hold sin(t*w_i), with
    /// w_i = exp(-ln(10000) * i / half). An odd width leaves its final slot at zero, matching the
    /// zero-padding those implementations apply.
    /// </para>
    /// <para>
    /// A single scalar such as sin(2*pi*t/(T-1)) is NOT a substitute: it is periodic over the
    /// schedule, so two different timesteps collide on one value and the denoiser cannot tell the
    /// two noise levels apart. It then learns the average of the two conditional expectations,
    /// which is a biased estimate of epsilon at both, and the bias compounds across the reverse
    /// chain - training reduces the epsilon loss while the sampled path moves AWAY from the target.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells the denoising network HOW NOISY its input currently is.
    /// Writing that number as a single sine wave makes two different noise levels look identical,
    /// the way a clock hand alone cannot tell morning from evening. The sinusoidal embedding writes
    /// the number at many frequencies at once, so every step is distinguishable.
    /// </para>
    /// </remarks>
    protected void WriteDiffusionTimestepEmbedding(Span<T> destination, int timestep)
    {
        int dim = destination.Length;
        if (dim <= 0) return;

        // A single slot gets the cosine written and then the odd-width zero, so it is always zero.
        if (dim == 1)
        {
            destination.Clear();
            return;
        }

        // [cos(t * w_i) | sin(t * w_i) | 0 when dim is odd], w_i = exp(-ln(10000) * i / half), i in [0, half).
        int half = dim / 2;
        double logMaxPeriod = Math.Log(10000.0);
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var index = Engine.TensorArange<T>(0, half, 1);
        var frequencies = Engine.TensorExp(Engine.TensorMultiplyScalar(index, NumOps.FromDouble(-logMaxPeriod / half)));
        var arguments = Engine.TensorMultiplyScalar(frequencies, NumOps.FromDouble(timestep));
        Engine.TensorCos(arguments).Data.Span.Slice(0, half).CopyTo(destination.Slice(0, half));
        Engine.TensorSin(arguments).Data.Span.Slice(0, half).CopyTo(destination.Slice(half, half));
        if ((dim % 2) != 0) destination.Slice(dim - 1).Clear();
    }


    /// <summary>
    /// Number of (timestep, noise) rows the deterministic denoising objective averages over.
    /// </summary>
    /// <remarks>
    /// Ho et al. 2020 optimize L_simple = E_{t,eps}[||eps - eps_theta(x_t, t)||^2], an expectation
    /// over BOTH the diffusion step and the noise draw. A single draw is a one-sample estimate of
    /// that expectation, so two such draws differ by far more than a few optimizer steps move the
    /// model. Thirty-two rows match the minibatch the training step itself averages over.
    /// </remarks>
    protected const int DeterministicDenoisingObjectiveRows = 32;

    /// <summary>
    /// Builds the fixed (timestep, noise) quadrature used to evaluate a denoising diffusion
    /// model's own training objective without drawing new randomness.
    /// </summary>
    /// <param name="target">The clean signal x_0 the denoiser is trained to recover.</param>
    /// <param name="outputLength">Width of one row of x_0.</param>
    /// <param name="diffusionSteps">The schedule length T.</param>
    /// <param name="sqrtAlphasCumprod">sqrt(alpha_bar_t) for every t in the schedule.</param>
    /// <param name="sqrtOneMinusAlphasCumprod">sqrt(1 - alpha_bar_t) for every t.</param>
    /// <remarks>
    /// <para>
    /// The timesteps are an even grid over the whole schedule and the noise comes from a fixed
    /// seed, so the returned batch is identical on every call. That is what lets a loss-trajectory
    /// probe compare the objective before and after training: the only thing that changed between
    /// two measurements is the model.
    /// </para>
    /// <para><b>For Beginners:</b> a diffusion model is never asked to predict the answer in one
    /// go - it is asked "given this partially noised signal, which noise was added?". Scoring it
    /// therefore means posing that question at a fixed set of noise levels, always with the same
    /// noise, and averaging how far off it was.</para>
    /// </remarks>
    protected (Tensor<T> NoisyTargets, Tensor<T> Noise, int[] Timesteps) BuildDeterministicDenoisingBatch(
        Tensor<T> target,
        int outputLength,
        int diffusionSteps,
        Vector<T> sqrtAlphasCumprod,
        Vector<T> sqrtOneMinusAlphasCumprod)
    {
        if (outputLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputLength), outputLength, "Output length must be positive.");
        if (diffusionSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(diffusionSteps), diffusionSteps, "Diffusion steps must be positive.");

        int rows = DeterministicDenoisingObjectiveRows;
        var timesteps = new int[rows];
        var alphaIndex = new int[rows];
        var oneMinusIndex = new int[rows];
        var noise = new Tensor<T>(new[] { rows, outputLength });
        var random = RandomHelper.CreateSeededRandom(DeterministicDenoisingObjectiveSeed);

        // The draws keep their original row-major order, so the fixed seed reproduces the same noise.
        var noiseValues = noise.Data.Span;
        for (int row = 0; row < rows; row++)
        {
            int step = (int)((long)row * diffusionSteps / rows);
            if (step >= diffusionSteps) step = diffusionSteps - 1;
            timesteps[row] = step;

            // A step past the end of a schedule reads the fallback slot appended to each table below
            // (one for sqrt(alpha_bar), zero for sqrt(1 - alpha_bar)).
            alphaIndex[row] = Math.Min(step, sqrtAlphasCumprod.Length);
            oneMinusIndex[row] = Math.Min(step, sqrtOneMinusAlphasCumprod.Length);

            for (int i = 0; i < outputLength; i++)
            {
                noiseValues[row * outputLength + i] = NextStandardNormal(random);
            }
        }

        // noisy[row, i] = sqrt(alpha_bar_step) * clean[i] + sqrt(1 - alpha_bar_step) * noise[row, i], detached
        // like the noise itself.
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var alphaTable = Engine.TensorConcatenate(
            new[] { Tensor<T>.FromVector(sqrtAlphasCumprod), Tensor<T>.FromVector(new Vector<T>(new[] { NumOps.One })) }, 0);
        var oneMinusTable = Engine.TensorConcatenate(
            new[] { Tensor<T>.FromVector(sqrtOneMinusAlphasCumprod), Tensor<T>.FromVector(new Vector<T>(new[] { NumOps.Zero })) }, 0);
        var rowAlpha = Engine.Reshape(
            Engine.TensorIndexSelect(alphaTable, new Tensor<int>(new[] { rows }, new Vector<int>(alphaIndex)), 0),
            new[] { rows, 1 });
        var rowOneMinus = Engine.Reshape(
            Engine.TensorIndexSelect(oneMinusTable, new Tensor<int>(new[] { rows }, new Vector<int>(oneMinusIndex)), 0),
            new[] { rows, 1 });

        // The clean signal is the target's first outputLength values, zero-padded when the target is shorter.
        var clean = new Tensor<T>(new[] { 1, outputLength });
        int cleanCount = Math.Min(outputLength, target.Length);
        if (cleanCount > 0)
            target.AsSpan().Slice(0, cleanCount).CopyTo(clean.Data.Span);

        var noisy = Engine.TensorAdd(
            Engine.TensorMultiply(rowAlpha, clean),
            Engine.TensorMultiply(rowOneMinus, noise));

        return (noisy, noise, timesteps);
    }

    /// <summary>Fixed seed behind <see cref="BuildDeterministicDenoisingBatch"/>.</summary>
    private const int DeterministicDenoisingObjectiveSeed = 20200619;

    /// <summary>Box-Muller standard normal draw from a caller-owned generator.</summary>
    private T NextStandardNormal(Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        return NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }
    #endregion
}
