using System;
using System.Collections.Generic;
using AiDotNet.Finance.Interfaces;
using AiDotNet.LossFunctions;
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
public abstract class ForecastingModelBase<T> : FinancialModelBase<T>, IForecastingModel<T>
{
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
        var shifted = new Tensor<T>(input.Shape);

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
}
