using AiDotNet.LinearAlgebra;

namespace AiDotNet.Models.Inputs;

/// <summary>
/// Provides optional calibration data for uncertainty quantification features.
/// </summary>
/// <typeparam name="TInput">The input type used by the model.</typeparam>
/// <typeparam name="TOutput">The output type used by the model.</typeparam>
/// <remarks>
/// <para>
/// Calibration data is used by certain uncertainty features that require a held-out dataset separate from training data.
/// Examples include conformal prediction and probability calibration.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of calibration data as a small "reality check" dataset.
/// The model is trained on training data, then calibration data is used to tune uncertainty-related behavior without overfitting.
/// </para>
/// </remarks>
public sealed class UncertaintyCalibrationData<TInput, TOutput>
{
    /// <summary>
    /// Gets the calibration inputs.
    /// </summary>
    public TInput X { get; }

    /// <summary>
    /// Gets the calibration targets (regression-style calibration).
    /// </summary>
    public TOutput Y { get; }

    /// <summary>
    /// Gets whether regression targets were provided.
    /// </summary>
    public bool HasTargets { get; }

    /// <summary>
    /// Gets the calibration class labels (classification-style calibration).
    /// </summary>
    /// <remarks>
    /// When provided, the calibration logic treats <see cref="Y"/> as optional and uses these labels.
    /// </remarks>
    public Vector<int>? Labels { get; }

    /// <summary>
    /// Gets whether classification labels were provided.
    /// </summary>
    public bool HasLabels => Labels != null;

    /// <summary>
    /// Initializes a new calibration data container.
    /// </summary>
    /// <param name="x">Calibration inputs.</param>
    /// <param name="hasTargets">Whether regression targets are present.</param>
    /// <param name="y">Calibration targets (used only when <paramref name="hasTargets"/> is true).</param>
    /// <param name="labels">Calibration labels (used only for classification calibration).</param>
    private UncertaintyCalibrationData(TInput x, bool hasTargets, TOutput y, Vector<int>? labels)
    {
        X = x;
        Y = y;
        HasTargets = hasTargets;
        Labels = labels;
    }

    /// <summary>
    /// Creates calibration data for regression-style calibration (e.g., conformal prediction intervals).
    /// </summary>
    /// <param name="xCalibration">Calibration inputs.</param>
    /// <param name="yCalibration">Calibration targets.</param>
    /// <returns>A calibration data instance.</returns>
    public static UncertaintyCalibrationData<TInput, TOutput> ForRegression(TInput xCalibration, TOutput yCalibration)
        => new(xCalibration, hasTargets: true, y: yCalibration, labels: null);

    /// <summary>
    /// Creates calibration data for classification-style calibration (e.g., temperature scaling and conformal prediction sets).
    /// </summary>
    /// <param name="xCalibration">Calibration inputs.</param>
    /// <param name="labels">True class labels for calibration samples.</param>
    /// <returns>A calibration data instance.</returns>
    public static UncertaintyCalibrationData<TInput, TOutput> ForClassification(TInput xCalibration, Vector<int> labels)
        => new(xCalibration, hasTargets: false, y: default!, labels: labels);
}
