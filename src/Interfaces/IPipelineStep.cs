using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Represents a step in a data processing pipeline
    /// </summary>
    /// <typeparam name="T">The numeric type for computations</typeparam>
    /// <typeparam name="TInput">The input data type for pipeline operations</typeparam>
    /// <typeparam name="TOutput">The output data type for pipeline operations</typeparam>
    /// <remarks>
    /// <para><b>For Beginners:</b> A pipeline step is a modular component that processes data in stages.
    /// Each step can fit (learn from data), transform (process data), or both. This pattern allows you to
    /// chain multiple processing steps together to create complex data processing workflows.</para>
    /// <para>The generic parameters allow this interface to work with different types of data while maintaining
    /// type safety. T is typically a numeric type (like double or float) used for calculations, while TInput
    /// and TOutput define what types of data the step accepts and produces.</para>
    /// </remarks>
    [AiDotNet.Configuration.YamlConfigurable("PipelineStep")]
    public interface IPipelineStep<T, TInput, TOutput>
    {
        /// <summary>
        /// Fits/trains this pipeline step on the provided data
        /// </summary>
        /// <param name="inputs">Input data for training</param>
        /// <param name="targets">Target data for supervised learning (optional)</param>
        /// <returns>Task representing the asynchronous operation</returns>
        Task FitAsync(TInput inputs, TOutput? targets = default);

        /// <summary>
        /// Transforms the input data using the fitted model
        /// </summary>
        /// <param name="inputs">Input data to transform</param>
        /// <returns>Transformed output data</returns>
        Task<TOutput> TransformAsync(TInput inputs);

        /// <summary>
        /// Fits and transforms in a single operation (convenience method)
        /// </summary>
        /// <param name="inputs">Input data</param>
        /// <param name="targets">Target data (optional)</param>
        /// <returns>Transformed output data</returns>
        Task<TOutput> FitTransformAsync(TInput inputs, TOutput? targets = default);

        /// <summary>
        /// Gets the parameters of this pipeline step
        /// </summary>
        /// <returns>Dictionary of parameter names and values</returns>
        Dictionary<string, object> GetParameters();

        /// <summary>
        /// Sets the parameters of this pipeline step
        /// </summary>
        /// <param name="parameters">Dictionary of parameter names and values</param>
        void SetParameters(Dictionary<string, object> parameters);

        /// <summary>
        /// Validates that this step can process the given input
        /// </summary>
        /// <param name="inputs">Input data to validate</param>
        /// <returns>True if valid, false otherwise</returns>
        bool ValidateInput(TInput inputs);

        /// <summary>
        /// Gets metadata about this pipeline step
        /// </summary>
        /// <returns>Metadata dictionary</returns>
        Dictionary<string, string> GetMetadata();
    }
}
