namespace AiDotNet.Interpretability.Interfaces;

/// <summary>
/// Interface for convolutional neural networks that support Grad-CAM explanation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Grad-CAM creates visual explanations showing which parts of
/// an image the CNN focused on. This interface provides the methods needed to extract
/// feature maps (what the CNN "sees") and their gradients (what matters for the prediction).</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("ConvolutionalNetwork")]
public interface IConvolutionalNetwork<T, TInput, TOutput>
{
    /// <summary>
    /// Gets feature maps and their gradients from the last convolutional layer.
    /// </summary>
    /// <param name="input">The input tensor (typically an image).</param>
    /// <param name="targetClass">The class to explain (which prediction to analyze).</param>
    /// <returns>A tuple containing feature maps and their gradients.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Feature maps are what CNN layers detect - edges, textures,
    /// patterns, etc. The gradients tell us which of these detections mattered most for
    /// predicting the target class.</para>
    ///
    /// <para>Grad-CAM combines these to create a heatmap showing which image regions
    /// were most important for the prediction.</para>
    /// </remarks>
    (Tensor<T> featureMaps, Tensor<T> gradients) GetFeatureMapsAndGradients(Tensor<T> input, int targetClass);
}
