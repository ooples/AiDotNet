using System;
using System.Collections.Generic;
// AiDotNet.Attributes is REQUIRED for [TensorLayout] to bind to the right type: two other Tensors
// namespaces declare a TensorLayout, and without this using the attribute silently resolves to one
// of those and the contract is never seen.
using AiDotNet.Attributes;
using AiDotNet.Finance.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Finance.Base;

namespace AiDotNet.Finance.Base;

/// <summary>
/// Base class for financial forecasting models, adding forecasting-specific behavior
/// on top of the core financial model infrastructure.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// This base class layers forecasting-specific requirements (like multi-step prediction
/// and instance normalization) on top of the shared financial model base.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this as the "forecasting toolkit" that all time series
/// models share. It defines what every forecasting model must expose so the rest of the
/// library can treat them consistently.
/// </para>
/// </remarks>
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input,
    Note = "A window of history: SequenceLength past steps, NumFeatures variables per step.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output,
    Note = "The forecast: PredictionHorizon future steps, NumFeatures variables per step.")]
public abstract class ForecastingModelBase<T> : FinancialModelBase<T>, IForecastingModel<T>, IShapeContract
{
    /// <summary>
    /// The forecasting family's output law: <c>[Batch, PredictionHorizon, NumFeatures]</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is one declaration for the whole family because the two quantities it needs are already
    /// on the base - <see cref="FinancialModelBase{T}.PredictionHorizon"/> and
    /// <see cref="FinancialModelBase{T}.NumFeatures"/> - and 71 of the concrete finance models
    /// OVERRIDE them with their own option values (Chronos returns <c>_forecastHorizon</c> and a
    /// NumFeatures of 1; TFT, Informer, PatchTST and the rest do the same). So the same declaration
    /// resolves to a different, correct pair per model without any of them writing a contract.
    /// </para>
    /// <para>
    /// Evidence for the layout, from this class's own code rather than inference:
    /// <see cref="ConcatenatePredictions"/> builds
    /// <c>new Tensor&lt;T&gt;([batchSize, totalSteps, features])</c>, and
    /// <see cref="FinancialModelBase{T}.PredictCore"/> is a straight delegation to <c>Forecast</c>,
    /// so the inference path and that constructor are the same shape.
    /// </para>
    /// <para>
    /// IT IS NOT THE DEFAULT, because measurement says it cannot be. Sweeping all 71 finance models
    /// against it gave 3 agreed and <b>34 DISAGREED</b>, and the disagreements are not noise - the
    /// horizon is usually RIGHT and the RANK is wrong, four different ways:
    /// </para>
    /// <list type="bullet">
    /// <item><description><c>[1,24,1]</c> vs <c>[1,24]</c> - CCDM, LagLlama, MGTSD, NHiTSFinance,
    /// DeepState: the feature axis is dropped when NumFeatures is 1.</description></item>
    /// <item><description><c>[1,24,1]</c> vs <c>[24]</c> - DiffusionTS, Chronos, GraphWaveNet, MTGNN,
    /// STGNN: batch AND feature dropped.</description></item>
    /// <item><description><c>[1,96,7]</c> vs <c>[1,64,7]</c> - FEDformer returns SequenceLength steps,
    /// not PredictionHorizon steps.</description></item>
    /// <item><description><c>[1,96,1]</c> vs <c>[1,64,96]</c> - FlowState transposes the two.</description></item>
    /// </list>
    /// <para>
    /// Models in ONE family, all implementing <c>IForecastingModel</c>, answer the same question at
    /// four different ranks. That is a finding about the family, not about this contract, and the
    /// honest response is to decline by default and let each model that has been MEASURED opt in by
    /// overriding this with <see cref="ForecastHorizonContract"/>. Making the majority law the default
    /// would attach a false contract to 34 models to gain 3.
    /// </para>
    /// </remarks>
    public virtual IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => null;

    /// <summary>The family law, as a helper so a model with a different rank can still reuse it.</summary>
    protected IReadOnlyList<OutputAxisContract>? ForecastHorizonContract(int inputRank)
    {
        if (inputRank != 3 || PredictionHorizon <= 0 || NumFeatures <= 0) return null;
        return
        [
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Fixed(PredictionHorizon)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(NumFeatures)),
        ];
    }

    /// <summary>
    /// Initializes a new forecasting model with deferred configuration.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="lossFunction">Optional loss function override.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor keeps the classic Finance model pattern
    /// where derived classes fill in the sequence length and other settings afterward.
    /// </para>
    /// </remarks>
    protected ForecastingModelBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction, maxGradNorm)
    {
        Options = new ForecastingModelOptions();
    }

    /// <summary>
    /// Initializes a new forecasting model in native mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">Input sequence length.</param>
    /// <param name="predictionHorizon">Prediction horizon (future steps to forecast).</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="lossFunction">Optional loss function override.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you want to train a forecasting model from scratch
    /// using native C# layers.
    /// </para>
    /// </remarks>
    protected ForecastingModelBase(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength,
        int predictionHorizon,
        int numFeatures,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, sequenceLength, predictionHorizon, numFeatures, lossFunction)
    {
        Options = new ForecastingModelOptions();
    }

    /// <summary>
    /// Initializes a new forecasting model in ONNX inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="sequenceLength">Input sequence length expected by the ONNX model.</param>
    /// <param name="predictionHorizon">Prediction horizon expected by the ONNX model.</param>
    /// <param name="numFeatures">Number of input features expected by the ONNX model.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you already have a pretrained ONNX model
    /// and only need fast inference.
    /// </para>
    /// </remarks>
    protected ForecastingModelBase(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int sequenceLength,
        int predictionHorizon,
        int numFeatures)
        : base(architecture, onnxModelPath, sequenceLength, predictionHorizon, numFeatures)
    {
        Options = new ForecastingModelOptions();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Patch size tells the model how many time steps are grouped
    /// together into one chunk when using patch-based forecasting.
    /// </para>
    /// </remarks>
    public abstract int PatchSize { get; }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Stride is how far the patch window moves each step.
    /// Smaller strides mean overlapping patches; larger strides mean fewer patches.
    /// </para>
    /// </remarks>
    public abstract int Stride { get; }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Channel-independent models process each variable separately
    /// with shared weights, which can improve generalization on multivariate data.
    /// </para>
    /// </remarks>
    public abstract bool IsChannelIndependent { get; }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method lets the model predict further into the future
    /// by feeding its own predictions back as new input.
    /// </para>
    /// </remarks>
    public abstract Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps);

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This calculates common error metrics (like MAE and RMSE)
    /// so you can see how accurate the forecasts are.
    /// </para>
    /// </remarks>
    public abstract Dictionary<string, T> Evaluate(Tensor<T> inputs, Tensor<T> targets);

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This normalizes each input sequence so the model is less
    /// sensitive to shifts in scale or level over time.
    /// </para>
    /// </remarks>
    public abstract Tensor<T> ApplyInstanceNormalization(Tensor<T> input);

    /// <summary>
    /// The variance floor RevIN adds before taking a square root, so a constant (zero-variance)
    /// series cannot divide by zero. Kim et al. 2022 use 1e-5.
    /// </summary>
    /// <remarks>
    /// Models that expose the floor through their own options should pass it explicitly to the
    /// normalization helpers instead of relying on this fallback.
    /// </remarks>
    protected const double DefaultRevInEpsilon = 1e-5;

    /// <summary>
    /// RevIN forward over each INSTANCE: every non-batch element of a row is normalized together,
    /// giving one mean/std per batch row.
    /// </summary>
    /// <param name="input">The input series, whose leading dimension is the batch.</param>
    /// <param name="epsilon">Variance floor; see <see cref="DefaultRevInEpsilon"/>.</param>
    /// <param name="mean">Receives the per-instance means, for the reverse step.</param>
    /// <param name="std">Receives the per-instance standard deviations, for the reverse step.</param>
    /// <returns>The normalized tensor, still connected to the autodiff tape.</returns>
    /// <remarks>
    /// <para>
    /// Composed entirely from <c>Engine</c> ops so RevIN is DIFFERENTIABLE. The hand-rolled versions
    /// this replaces accumulated the statistics with scalar <c>NumOps</c> arithmetic and wrote the
    /// output through <c>result.Data.Span[...]</c>, which the tape cannot see -- the normalized tensor
    /// came back as a LEAF, so nothing downstream could differentiate through the normalization.
    /// RevIN is a layer in Kim et al. 2022, not a preprocessing step, and every model whose forward
    /// begins with it was silently training against a detached input.
    /// </para>
    /// <para>
    /// The statistics are returned as <see cref="Vector{T}"/> because that is what the reverse
    /// (denormalize) step of each model already consumes; only the forward needed to become
    /// differentiable for gradients to reach the model's parameters.
    /// </para>
    /// </remarks>
    protected Tensor<T> NormalizeInstanceOnTape(
        Tensor<T> input, double epsilon, out Vector<T> mean, out Vector<T> std)
    {
        int batchSize = input.Shape.Length > 1 ? input.Shape[0] : 1;
        int instanceSize = batchSize > 0 ? input.Length / batchSize : input.Length;
        if (instanceSize <= 0)
        {
            mean = new Vector<T>(0);
            std = new Vector<T>(0);
            return input;
        }

        var flat = Engine.Reshape(input, new[] { batchSize, instanceSize });
        var normalized = NormalizeAlongLastAxis(flat, epsilon, out mean, out std);
        return Engine.Reshape(normalized, (int[])input._shape.Clone());
    }

    /// <summary>
    /// RevIN forward per FEATURE: the trailing dimension is treated as the feature axis and each
    /// feature is normalized over the leading time axis, giving one mean/std per feature.
    /// </summary>
    /// <param name="input">The series, laid out row-major as [steps, features] (or flat).</param>
    /// <param name="epsilon">Variance floor; see <see cref="DefaultRevInEpsilon"/>.</param>
    /// <param name="mean">Receives the per-feature means, for the reverse step.</param>
    /// <param name="std">Receives the per-feature standard deviations, for the reverse step.</param>
    /// <returns>The normalized tensor, still connected to the autodiff tape.</returns>
    /// <remarks>
    /// Same tape rationale as <see cref="NormalizeInstanceOnTape"/>. This is the convention for
    /// multivariate models where the LEADING dimension is time rather than a batch, so statistics
    /// must be taken down the time axis for each channel independently.
    /// </remarks>
    protected Tensor<T> NormalizePerFeatureOnTape(
        Tensor<T> input, double epsilon, out Vector<T> mean, out Vector<T> std)
    {
        int features = input.Rank > 1 ? input.Shape[input.Rank - 1] : 1;
        int steps = features > 0 ? input.Length / features : input.Length;
        if (features <= 0 || steps <= 0)
        {
            mean = new Vector<T>(0);
            std = new Vector<T>(0);
            return input;
        }

        // Reduce down the TIME axis (axis 0) rather than the feature axis, then broadcast the
        // per-feature statistics back across time.
        var flat = Engine.Reshape(input, new[] { steps, features });
        var reduceAxis = new[] { 0 };

        var meanT = Engine.ReduceMean(flat, reduceAxis, keepDims: true);          // [1, features]
        var centered = Engine.TensorSubtract(flat, meanT);
        var varianceT = Engine.ReduceMean(
            Engine.TensorMultiply(centered, centered), reduceAxis, keepDims: true);
        var stdT = Engine.TensorSqrt(Engine.TensorAddScalar(varianceT, NumOps.FromDouble(epsilon)));

        mean = ToVectorOffTape(meanT, features);
        std = ToVectorOffTape(stdT, features);

        var normalized = Engine.TensorDivide(centered, stdT);
        return Engine.Reshape(normalized, (int[])input._shape.Clone());
    }

    /// <summary>Normalizes each row of a [rows, cols] tensor over its columns, on the tape.</summary>
    private Tensor<T> NormalizeAlongLastAxis(
        Tensor<T> flat, double epsilon, out Vector<T> mean, out Vector<T> std)
    {
        int rows = flat.Shape[0];
        var reduceAxis = new[] { 1 };

        var meanT = Engine.ReduceMean(flat, reduceAxis, keepDims: true);          // [rows, 1]
        var centered = Engine.TensorSubtract(flat, meanT);
        var varianceT = Engine.ReduceMean(
            Engine.TensorMultiply(centered, centered), reduceAxis, keepDims: true);
        var stdT = Engine.TensorSqrt(Engine.TensorAddScalar(varianceT, NumOps.FromDouble(epsilon)));

        mean = ToVectorOffTape(meanT, rows);
        std = ToVectorOffTape(stdT, rows);

        return Engine.TensorDivide(centered, stdT);
    }

    /// <summary>
    /// Copies a statistics tensor's values into a <see cref="Vector{T}"/> for the reverse step.
    /// </summary>
    /// <remarks>
    /// Reading the values out is deliberate and does NOT affect the forward's differentiability: the
    /// tensor returned to the caller stays on the tape, and only these copies are detached.
    /// </remarks>
    private static Vector<T> ToVectorOffTape(Tensor<T> stats, int count)
    {
        var v = new Vector<T>(count);
        for (int i = 0; i < count && i < stats.Length; i++)
        {
            v[i] = stats.Data.Span[i];
        }

        return v;
    }

    /// <summary>
    /// Shifts the input window forward by replacing the oldest steps with predictions.
    /// </summary>
    /// <param name="input">Original input tensor.</param>
    /// <param name="predictions">Predictions to append.</param>
    /// <param name="stepsToShift">Number of time steps to shift.</param>
    /// <returns>Shifted input tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Autoregressive forecasting predicts a few steps, then slides
    /// the input window forward to predict more. This method performs that "slide" by
    /// dropping old data and adding new predictions.
    /// </para>
    /// </remarks>
    protected virtual Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int stepsToShift)
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape[1];
        int features = input.Shape.Length > 2 ? input.Shape[2] : NumFeatures;

        int stepsUsed = Math.Min(stepsToShift, seqLen);
        var shifted = new Tensor<T>(input._shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < features; f++)
            {
                for (int t = 0; t < seqLen - stepsUsed; t++)
                {
                    int srcIdx = b * seqLen * features + (t + stepsUsed) * features + f;
                    int dstIdx = b * seqLen * features + t * features + f;
                    if (srcIdx < input.Length && dstIdx < shifted.Length)
                        shifted.Data.Span[dstIdx] = input.Data.Span[srcIdx];
                }

                for (int t = seqLen - stepsUsed; t < seqLen; t++)
                {
                    int predIdx = b * stepsUsed * features + (t - (seqLen - stepsUsed)) * features + f;
                    int dstIdx = b * seqLen * features + t * features + f;
                    if (predIdx < predictions.Length && dstIdx < shifted.Length)
                        shifted.Data.Span[dstIdx] = predictions.Data.Span[predIdx];
                }
            }
        }

        return shifted;
    }

    /// <summary>
    /// Combines multiple prediction chunks into a single long forecast tensor.
    /// </summary>
    /// <param name="predictions">List of prediction tensors.</param>
    /// <param name="totalSteps">Total number of steps requested.</param>
    /// <returns>Combined forecast tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When the model predicts in smaller chunks, we stitch those
    /// chunks together so you get one continuous forecast sequence.
    /// </para>
    /// </remarks>
    protected virtual Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions, int totalSteps)
    {
        if (predictions.Count == 0)
            return new Tensor<T>(new[] { 1, totalSteps, NumFeatures });

        int batchSize = predictions[0].Shape[0];
        int features = predictions[0].Shape.Length > 2 ? predictions[0].Shape[2] : NumFeatures;

        var result = new Tensor<T>(new[] { batchSize, totalSteps, features });
        int currentStep = 0;

        foreach (var pred in predictions)
        {
            int predSteps = pred.Shape.Length > 1 ? pred.Shape[1] : 1;
            int stepsToCopy = Math.Min(predSteps, totalSteps - currentStep);

            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < stepsToCopy; t++)
                {
                    for (int f = 0; f < features; f++)
                    {
                        int srcIdx = b * predSteps * features + t * features + f;
                        int dstIdx = b * totalSteps * features + (currentStep + t) * features + f;
                        if (srcIdx < pred.Length && dstIdx < result.Length)
                            result.Data.Span[dstIdx] = pred.Data.Span[srcIdx];
                    }
                }
            }

            currentStep += stepsToCopy;
            if (currentStep >= totalSteps)
                break;
        }

        return result;
    }
    /// <summary>
    /// Walks the layer stack the way this family's forward passes do: instance-normalize the input and
    /// promote a bare rank-1 series to [1, N] before it reaches the first layer.
    /// </summary>
    /// <remarks>
    /// The generic implementation feeds the raw input straight into Layers[0]. Every forecasting model
    /// here normalizes first and lifts a rank-1 series to a batched one, so the leading ReshapeLayer
    /// received a rank-1 tensor and threw "ReshapeLayer per-sample input element count (1) does not
    /// match output element count". MOMENT, YingLong and TinyTimeMixers each carried a private copy of
    /// this override for exactly that reason and TOTO then hit it too, so it belongs here once rather
    /// than being re-added per model as each shard run surfaces the next one. Models whose forward
    /// differs still override it themselves.
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        var activations = new Dictionary<string, Tensor<T>>();
        if (!UseNativeMode || Layers.Count == 0)
            return activations;

        var current = ApplyInstanceNormalization(input);
        if (current.Rank == 1)
            current = current.Reshape(new[] { 1, current.Length });

        for (int i = 0; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
            activations[$"Layer_{i}_{Layers[i].GetType().Name}"] = current.Clone();
        }

        return activations;
    }

}
