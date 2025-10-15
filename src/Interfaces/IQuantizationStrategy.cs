using System;
using System.Threading.Tasks;
using AiDotNet.Deployment.Techniques;
using AiDotNet.NeuralNetworks.Layers;
using DeploymentQuantizationConfig = AiDotNet.Deployment.Techniques.QuantizationConfig;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Interface for quantization strategies.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <typeparam name="TInput">The input type for the model.</typeparam>
    /// <typeparam name="TOutput">The output type for the model.</typeparam>
    public interface IQuantizationStrategy<T, TInput, TOutput>
        where T : struct, IComparable<T>, IConvertible, IEquatable<T>
    {
        /// <summary>
        /// Gets a value indicating whether this strategy requires calibration data.
        /// </summary>
        bool RequiresCalibration { get; }
        
        /// <summary>
        /// Quantizes a model using this strategy.
        /// </summary>
        /// <param name="model">The model to quantize.</param>
        /// <param name="config">The quantization configuration.</param>
        /// <param name="calibrationData">The calibration data (if required).</param>
        /// <returns>The quantized model.</returns>
        Task<IFullModel<T, TInput, TOutput>> QuantizeAsync(
            IFullModel<T, TInput, TOutput> model, 
            DeploymentQuantizationConfig config, 
            CalibrationData<T> calibrationData);
        
        /// <summary>
        /// Quantizes a single layer using this strategy.
        /// </summary>
        /// <param name="layer">The layer to quantize.</param>
        /// <param name="config">The quantization configuration.</param>
        /// <returns>The quantized layer.</returns>
        Task<ILayer<T>> QuantizeLayerAsync(ILayer<T> layer, DeploymentQuantizationConfig config);
        
        /// <summary>
        /// Analyzes a model to provide recommendations for this strategy.
        /// </summary>
        /// <param name="model">The model to analyze.</param>
        /// <param name="config">The quantization configuration.</param>
        /// <returns>Strategy recommendations.</returns>
        StrategyRecommendation AnalyzeModel(IFullModel<T, TInput, TOutput> model, DeploymentQuantizationConfig config);
    }
}