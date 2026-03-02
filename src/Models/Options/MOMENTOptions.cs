using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for MOMENT (Multi-task Optimization through Masked Encoding for
/// Time series) foundation model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// MOMENT is a multi-task time series foundation model from Carnegie Mellon University.
/// It uses a T5-based encoder-only transformer with patch embeddings and RevIN
/// (Reversible Instance Normalization) to handle diverse time series tasks including
/// forecasting, anomaly detection, classification, imputation, and embedding generation.
/// </para>
/// <para><b>For Beginners:</b> MOMENT is like a Swiss Army knife for time series:
///
/// <b>Key Innovation — Multi-Task Architecture:</b>
/// Unlike most time series models that only do forecasting, MOMENT handles 5 tasks:
/// 1. <b>Forecasting</b>: Predict future values
/// 2. <b>Anomaly Detection</b>: Find unusual patterns via reconstruction
/// 3. <b>Classification</b>: Label time series segments
/// 4. <b>Imputation</b>: Fill in missing values
/// 5. <b>Embedding</b>: Generate vector representations
///
/// <b>How It Works:</b>
/// - Divides input into patches (like Vision Transformer for images)
/// - Applies RevIN to handle different scales and distributions
/// - Uses a T5-style transformer encoder to process patches
/// - Task-specific heads generate outputs for each task type
///
/// <b>Model Sizes:</b>
/// - Small (~40M params): Fast experiments
/// - Base (~385M params): Strong general-purpose performance
/// - Large (~1B+ params): Maximum capacity
/// </para>
/// <para>
/// <b>Reference:</b> Goswami et al., "MOMENT: A Family of Open Time-Series Foundation Models",
/// ICML 2024. https://arxiv.org/abs/2402.03885
/// </para>
/// </remarks>
public class MOMENTOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public MOMENTOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MOMENTOptions(MOMENTOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        ContextLength = other.ContextLength;
        ForecastHorizon = other.ForecastHorizon;
        PatchLength = other.PatchLength;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        IntermediateSize = other.IntermediateSize;
        DropoutRate = other.DropoutRate;
        ModelSize = other.ModelSize;
        Task = other.Task;
        NumClasses = other.NumClasses;
        MaskRatio = other.MaskRatio;
    }

    /// <summary>
    /// Gets or sets the context length (input sequence length).
    /// </summary>
    /// <value>Defaults to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps the model can look at.
    /// MOMENT works best with 512 time steps.
    /// </para>
    /// </remarks>
    public int ContextLength { get; set; } = 512;

    /// <summary>
    /// Gets or sets the forecast horizon (prediction length).
    /// </summary>
    /// <value>Defaults to 96.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far into the future to predict.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 96;

    /// <summary>
    /// Gets or sets the patch length for input tokenization.
    /// </summary>
    /// <value>Defaults to 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> MOMENT groups consecutive time steps into patches.
    /// With context=512 and patch=64, the model processes 8 patches.
    /// </para>
    /// </remarks>
    public int PatchLength { get; set; } = 64;

    /// <summary>
    /// Gets or sets the hidden dimension of the T5 transformer.
    /// </summary>
    /// <value>Defaults to 1024 (Base variant).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The internal representation size of the transformer.
    /// Larger values increase capacity but require more memory.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the number of transformer encoder layers.
    /// </summary>
    /// <value>Defaults to 24 (Base variant).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How deep the transformer stack is.
    /// More layers = more capacity but more computation.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 24;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>Defaults to 16 (Base variant).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Multi-head attention allows the model to attend to
    /// different patterns simultaneously.
    /// </para>
    /// </remarks>
    public int NumHeads { get; set; } = 16;

    /// <summary>
    /// Gets or sets the intermediate size for the feed-forward network.
    /// </summary>
    /// <value>Defaults to 4096 (4x hidden dimension).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The expansion factor in the MLP blocks.
    /// </para>
    /// </remarks>
    public int IntermediateSize { get; set; } = 4096;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>Defaults to 0.1.</value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the model size variant.
    /// </summary>
    /// <value>Defaults to <see cref="FoundationModelSize.Base"/>.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> MOMENT comes in different sizes:
    /// - Small (~40M params): Fast, lightweight
    /// - Base (~385M params): Strong general-purpose
    /// - Large (~1B+ params): Maximum capacity
    /// </para>
    /// </remarks>
    public FoundationModelSize ModelSize { get; set; } = FoundationModelSize.Base;

    /// <summary>
    /// Gets or sets the active task for the model.
    /// </summary>
    /// <value>Defaults to <see cref="TimeSeriesFoundationModelTask.Forecasting"/>.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> MOMENT supports multiple tasks. Set this to control
    /// which task-specific head is used for inference.
    /// </para>
    /// </remarks>
    public TimeSeriesFoundationModelTask Task { get; set; } = TimeSeriesFoundationModelTask.Forecasting;

    /// <summary>
    /// Gets or sets the number of classification classes (for classification task only).
    /// </summary>
    /// <value>Defaults to null (not applicable unless using classification task).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Only needed when <see cref="Task"/> is
    /// <see cref="TimeSeriesFoundationModelTask.Classification"/>. Set this to the
    /// number of categories you want to classify into.
    /// </para>
    /// </remarks>
    public int? NumClasses { get; set; }

    /// <summary>
    /// Gets or sets the mask ratio for pretraining and imputation tasks.
    /// </summary>
    /// <value>Defaults to 0.3 (30% of patches are masked).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> During pretraining and imputation, a portion of patches
    /// are masked and the model learns to reconstruct them. 0.3 means 30% are masked.
    /// </para>
    /// </remarks>
    public double MaskRatio { get; set; } = 0.3;
}
