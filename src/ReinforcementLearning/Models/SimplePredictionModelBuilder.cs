using System;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;

namespace AiDotNet.ReinforcementLearning.Models
{
    /// <summary>
    /// A simplified version of PredictionModelBuilder that uses Tensor<double>&lt;T&gt; for both input and output types.
    /// </summary>
    /// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
    /// <remarks>
    /// <para>
    /// This class serves as a wrapper around the more general PredictionModelBuilder&lt;T, TInput, TOutput&gt;,
    /// providing a simplified interface when you're working with tensor-based models such as neural networks
    /// and reinforcement learning models.
    /// </para>
    /// <para>
    /// It automatically uses Tensor<double>&lt;T&gt; for both input and output types, which is the common case
    /// for most deep learning and reinforcement learning models.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of this as a simpler recipe builder for creating AI models that work
    /// with tensors (multidimensional arrays) as both inputs and outputs. This is especially useful
    /// for neural networks and reinforcement learning models.
    /// </para>
    /// </remarks>
    public class PredictionModelBuilder<T>
    {
        private readonly PredictionModelBuilder<T, Tensor<T>, Tensor<T>> _innerBuilder = default!;

        /// <summary>
        /// Initializes a new instance of the PredictionModelBuilder class.
        /// </summary>
        /// <param name="model">Optional model to use. If not provided, a model must be set later.</param>
        /// <remarks>
        /// <para>
        /// This constructor creates a new PredictionModelBuilder with an optional model.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b> This is how you start building your machine learning model.
        /// You can either provide a model right away or add one later using the various helper methods.
        /// </para>
        /// </remarks>
        public PredictionModelBuilder(IFullModel<T, Tensor<T>, Tensor<T>>? model = null)
        {
            _innerBuilder = new PredictionModelBuilder<T, Tensor<T>, Tensor<T>>(model);
        }

        /// <summary>
        /// Sets the model to be used by this builder.
        /// </summary>
        /// <param name="model">The model to use.</param>
        /// <returns>This builder instance for method chaining.</returns>
        /// <remarks>
        /// <para>
        /// This method sets the machine learning model that will be trained and used for predictions.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b> Use this to specify which machine learning algorithm you want to use.
        /// Different models are better for different types of problems.
        /// </para>
        /// </remarks>
        public PredictionModelBuilder<T> SetModel(IFullModel<T, Tensor<T>, Tensor<T>> model)
        {
            _innerBuilder.SetModel(model);
            return this;
        }

        /// <summary>
        /// Builds the model using the provided configuration and returns a full model ready for use.
        /// </summary>
        /// <returns>A fully built and configured model.</returns>
        /// <remarks>
        /// <para>
        /// This method finalizes the model construction process and returns the model ready for use.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b> Call this method when you've finished configuring your model
        /// to get the final model that you can use for training and predictions.
        /// </para>
        /// </remarks>
        public IFullModel<T, Tensor<T>, Tensor<T>> BuildFullModel()
        {
            // Get the model from the inner builder
            // This assumes there's a way to retrieve the model from the builder
            // If not, you might need to store it separately or use reflection
            return _innerBuilder.GetModel();
        }
    }
}