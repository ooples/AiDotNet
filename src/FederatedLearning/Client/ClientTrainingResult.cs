using System.Collections.Generic;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.FederatedLearning.Client
{
    /// <summary>
    /// Client training result containing the outcome of a local training round
    /// </summary>
    /// <typeparam name="T">The numeric type</typeparam>
    public class ClientTrainingResult<T>
    {
        /// <summary>
        /// Gets or sets the client identifier
        /// </summary>
        public string ClientId { get; set; }

        /// <summary>
        /// Gets or sets the parameter updates calculated during training
        /// </summary>
        public Dictionary<string, Vector<T>> ParameterUpdates { get; set; }

        /// <summary>
        /// Gets or sets the training loss achieved
        /// </summary>
        public T TrainingLoss { get; set; }

        /// <summary>
        /// Gets or sets the data size used for training
        /// </summary>
        public int DataSize { get; set; }

        /// <summary>
        /// Gets or sets whether the training was successful
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// Gets or sets the error message if training failed
        /// </summary>
        public string ErrorMessage { get; set; }
    }
}