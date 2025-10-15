using AiDotNet.Enums;
using System;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Non-generic base interface for production monitoring
    /// </summary>
    public interface IProductionMonitor
    {
        /// <summary>
        /// Configures drift detection settings
        /// </summary>
        /// <param name="method">The drift detection method to use</param>
        /// <param name="threshold">The threshold for drift detection</param>
        void ConfigureDriftDetection(DriftDetectionMethod method, double threshold);
        
        /// <summary>
        /// Configures automatic retraining settings
        /// </summary>
        /// <param name="enabled">Whether automatic retraining is enabled</param>
        /// <param name="checkInterval">How often to check if retraining is needed</param>
        void ConfigureRetraining(bool enabled, TimeSpan checkInterval);
    }
}