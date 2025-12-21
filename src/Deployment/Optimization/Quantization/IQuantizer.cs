using System.Collections.Generic;
using AiDotNet.Deployment.Export;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Deployment.Optimization.Quantization;

/// <summary>
/// Interface for model quantization strategies.
/// Properly integrates with AiDotNet's IFullModel architecture.
/// </summary>
/// <typeparam name="T">The numeric type used in the model</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public interface IQuantizer<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the quantization mode (Int8, Float16, etc.).
    /// </summary>
    QuantizationMode Mode { get; }

    /// <summary>
    /// Gets the target bit width for quantization.
    /// </summary>
    int BitWidth { get; }

    /// <summary>
    /// Quantizes the model parameters using IFullModel architecture.
    /// </summary>
    /// <param name="model">The model to quantize (must implement IFullModel)</param>
    /// <param name="config">Quantization configuration</param>
    /// <returns>A new quantized model instance</returns>
    IFullModel<T, TInput, TOutput> Quantize(IFullModel<T, TInput, TOutput> model, QuantizationConfiguration config);

    /// <summary>
    /// Calibrates the quantizer using calibration data by running forward passes through the model.
    /// This collects activation statistics needed for accurate quantization.
    /// </summary>
    /// <param name="model">The model to calibrate</param>
    /// <param name="calibrationData">Data samples for calibration</param>
    void Calibrate(IFullModel<T, TInput, TOutput> model, IEnumerable<TInput> calibrationData);

    /// <summary>
    /// Gets the scale factor for a specific layer or parameter.
    /// </summary>
    /// <param name="layerName">Name of the layer</param>
    /// <returns>The scale factor</returns>
    double GetScaleFactor(string layerName);

    /// <summary>
    /// Gets the zero point for a specific layer or parameter.
    /// </summary>
    /// <param name="layerName">Name of the layer</param>
    /// <returns>The zero point</returns>
    int GetZeroPoint(string layerName);
}
