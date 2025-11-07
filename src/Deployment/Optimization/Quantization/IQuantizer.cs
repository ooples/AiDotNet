namespace AiDotNet.Deployment.Optimization.Quantization;

/// <summary>
/// Interface for model quantization strategies.
/// </summary>
/// <typeparam name="T">The numeric type used in the model</typeparam>
public interface IQuantizer<T> where T : struct
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
    /// Quantizes the model parameters.
    /// </summary>
    /// <param name="model">The model to quantize</param>
    /// <param name="config">Quantization configuration</param>
    /// <returns>The quantized model</returns>
    object Quantize(object model, QuantizationConfiguration config);

    /// <summary>
    /// Calibrates the quantizer using calibration data.
    /// </summary>
    /// <param name="calibrationData">Data samples for calibration</param>
    void Calibrate(IEnumerable<T[]> calibrationData);

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
