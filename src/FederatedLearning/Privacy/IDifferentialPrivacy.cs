using System.Collections.Generic;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.FederatedLearning.Privacy
{
    /// <summary>
    /// Interface for implementing differential privacy mechanisms in federated learning.
    /// </summary>
    public interface IDifferentialPrivacy
    {
        /// <summary>
        /// Applies differential privacy to model parameters.
        /// </summary>
        /// <param name="parameters">The model parameters to apply privacy to.</param>
        /// <param name="epsilon">Privacy budget parameter (smaller values mean more privacy).</param>
        /// <param name="delta">Privacy failure probability.</param>
        /// <param name="privacySettings">Additional privacy settings.</param>
        /// <returns>Parameters with differential privacy applied.</returns>
        Dictionary<string, Vector<double>> ApplyPrivacy(
            Dictionary<string, Vector<double>> parameters,
            double epsilon,
            double delta,
            PrivacySettings privacySettings);

        /// <summary>
        /// Calculates the noise scale for a given privacy budget.
        /// </summary>
        /// <param name="epsilon">Privacy budget parameter.</param>
        /// <param name="delta">Privacy failure probability.</param>
        /// <param name="sensitivity">Sensitivity of the function.</param>
        /// <returns>The noise scale to apply.</returns>
        double CalculateNoiseScale(double epsilon, double delta, double sensitivity);

        /// <summary>
        /// Clips gradients to ensure bounded sensitivity.
        /// </summary>
        /// <param name="gradients">The gradients to clip.</param>
        /// <param name="maxNorm">Maximum allowed norm.</param>
        /// <returns>Clipped gradients.</returns>
        Dictionary<string, Vector<double>> ClipGradients(
            Dictionary<string, Vector<double>> gradients,
            double maxNorm);
    }
}