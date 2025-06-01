using AiDotNet.Models;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Base interface for pipeline steps in ML workflows
    /// </summary>
    public interface IPipelineStep
    {
        /// <summary>
        /// Gets the name of this pipeline step
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Gets whether this step is fitted/trained
        /// </summary>
        bool IsFitted { get; }

        /// <summary>
        /// Fits/trains this pipeline step on the provided data
        /// </summary>
        /// <param name="inputs">Input data</param>
        /// <param name="targets">Target data (optional for unsupervised steps)</param>
        /// <returns>Task representing the asynchronous operation</returns>
        Task FitAsync(double[][] inputs, double[]? targets = null);

        /// <summary>
        /// Transforms the input data
        /// </summary>
        /// <param name="inputs">Input data to transform</param>
        /// <returns>Transformed data</returns>
        Task<double[][]> TransformAsync(double[][] inputs);

        /// <summary>
        /// Fits and transforms in a single operation
        /// </summary>
        /// <param name="inputs">Input data</param>
        /// <param name="targets">Target data (optional)</param>
        /// <returns>Transformed data</returns>
        Task<double[][]> FitTransformAsync(double[][] inputs, double[]? targets = null);

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
        bool ValidateInput(double[][] inputs);

        /// <summary>
        /// Gets metadata about this pipeline step
        /// </summary>
        /// <returns>Metadata dictionary</returns>
        Dictionary<string, string> GetMetadata();
    }
}