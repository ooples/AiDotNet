using System;
using System.Collections.Generic;

namespace AiDotNet.PhysicsInformed.Interfaces
{
    /// <summary>
    /// Defines the interface for inverse problems in physics-informed neural networks.
    /// </summary>
    /// <typeparam name="T">The numeric type (float, double, etc.) used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// An inverse problem is about finding unknown causes from observed effects.
    ///
    /// Forward Problem (typical):
    /// - Known: Initial conditions, boundary conditions, physical parameters
    /// - Find: Solution at all points in space and time
    /// - Example: Given thermal conductivity k, find temperature distribution T(x,t)
    ///
    /// Inverse Problem:
    /// - Known: Some observations of the solution
    /// - Find: Unknown physical parameters or hidden fields
    /// - Example: Given temperature measurements, find thermal conductivity k
    ///
    /// Types of Inverse Problems:
    ///
    /// 1. Parameter Identification:
    ///    - Find unknown constants in the PDE
    ///    - Example: Identify diffusion coefficient from concentration data
    ///
    /// 2. Source Identification:
    ///    - Find unknown source terms
    ///    - Example: Locate pollution source from downstream measurements
    ///
    /// 3. Boundary Identification:
    ///    - Determine unknown boundary conditions
    ///    - Example: Infer surface heat flux from internal temperature sensors
    ///
    /// 4. Geometry Identification:
    ///    - Find unknown shape of domain
    ///    - Example: Detect tumor location from external measurements
    ///
    /// Challenges:
    /// 1. Ill-posedness: Small noise in data → large errors in parameters
    /// 2. Non-uniqueness: Multiple parameter values may fit the data
    /// 3. Regularization: Need to impose constraints for stable solutions
    ///
    /// PINN Advantage for Inverse Problems:
    /// - Learns solution AND parameters simultaneously
    /// - Physics constraints act as regularization
    /// - Can handle noisy and sparse data
    /// - No need for iterative PDE solves
    /// </remarks>
    public interface IInverseProblem<T>
    {
        /// <summary>
        /// Gets the names of the unknown parameters to identify.
        /// </summary>
        /// <remarks>
        /// Example parameter names:
        /// - "thermal_conductivity"
        /// - "diffusion_coefficient"
        /// - "wave_speed"
        /// - "viscosity"
        /// </remarks>
        string[] ParameterNames { get; }

        /// <summary>
        /// Gets the number of unknown parameters.
        /// </summary>
        int NumberOfParameters { get; }

        /// <summary>
        /// Gets initial guesses for the unknown parameters.
        /// </summary>
        /// <remarks>
        /// For Beginners:
        /// Initial guesses help the optimization start in a reasonable region.
        /// - If you have prior knowledge, use it!
        /// - Otherwise, use typical values for the problem type
        /// - The PINN will refine these during training
        /// </remarks>
        T[] InitialParameterGuesses { get; }

        /// <summary>
        /// Gets lower bounds for the parameters (for constrained optimization).
        /// </summary>
        /// <remarks>
        /// Physical constraints often impose bounds:
        /// - Diffusion coefficient > 0
        /// - Density > 0
        /// - Some ratios between 0 and 1
        /// Null means no lower bound.
        /// </remarks>
        T[]? ParameterLowerBounds { get; }

        /// <summary>
        /// Gets upper bounds for the parameters.
        /// </summary>
        /// <remarks>
        /// Upper bounds can prevent non-physical solutions:
        /// - Speed of sound can't exceed material limit
        /// - Concentration can't exceed saturation
        /// Null means no upper bound.
        /// </remarks>
        T[]? ParameterUpperBounds { get; }

        /// <summary>
        /// Gets the observation data points.
        /// </summary>
        /// <remarks>
        /// For Beginners:
        /// Observations are measurements of the solution at specific locations.
        /// More observations generally lead to better parameter estimates.
        /// The quality and distribution of observations matters!
        ///
        /// Returns a list of (location, observed_value) pairs.
        /// </remarks>
        IReadOnlyList<(T[] location, T[] value)> Observations { get; }

        /// <summary>
        /// Gets whether the measurement noise level is known.
        /// </summary>
        bool HasMeasurementNoiseLevel { get; }

        /// <summary>
        /// Gets the measurement noise level (if known).
        /// </summary>
        /// <remarks>
        /// Knowing the noise level helps with:
        /// - Choosing appropriate regularization
        /// - Estimating uncertainty in parameters
        /// - Weighing data vs physics loss
        /// Check HasMeasurementNoiseLevel before accessing this property.
        /// Returns default(T) if unknown.
        /// </remarks>
        T MeasurementNoiseLevel { get; }

        /// <summary>
        /// Validates that the parameter values are physically meaningful.
        /// </summary>
        /// <param name="parameters">The parameter values to validate.</param>
        /// <returns>True if parameters are valid, false otherwise.</returns>
        /// <remarks>
        /// Beyond simple bounds, this can check:
        /// - Parameter combinations (e.g., CFL condition)
        /// - Physical consistency (e.g., energy conservation constraints)
        /// - Material property relationships
        /// </remarks>
        bool ValidateParameters(T[] parameters);

        /// <summary>
        /// Applies parameters to the underlying PDE and returns the modified PDE specification.
        /// </summary>
        /// <param name="parameters">The parameter values to apply.</param>
        /// <returns>A PDE specification with the given parameters.</returns>
        /// <remarks>
        /// For Beginners:
        /// This creates a "configured" version of the PDE with specific parameter values.
        /// During training:
        /// 1. Parameters are updated
        /// 2. This method creates a new PDE with those parameters
        /// 3. The PINN evaluates residuals using the new PDE
        /// 4. Gradients flow back to update parameters
        /// </remarks>
        IPDESpecification<T> CreateParameterizedPDE(T[] parameters);
    }

    /// <summary>
    /// Provides gradient information for inverse problem parameters.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    public interface IInverseProblemGradient<T> : IInverseProblem<T>
    {
        /// <summary>
        /// Computes gradients of the PDE residual with respect to the unknown parameters.
        /// </summary>
        /// <param name="parameters">Current parameter values.</param>
        /// <param name="inputs">The spatial and temporal coordinates.</param>
        /// <param name="outputs">The predicted solution values.</param>
        /// <param name="derivatives">The derivatives of the solution.</param>
        /// <returns>Gradient of residual with respect to each parameter.</returns>
        /// <remarks>
        /// For Beginners:
        /// This tells us how changing each parameter affects the PDE residual.
        /// - If dR/d(parameter) is large: parameter has strong effect on physics
        /// - If dR/d(parameter) is small: parameter has weak effect (hard to identify)
        ///
        /// The gradients are used to update parameters during training.
        /// </remarks>
        T[] ComputeParameterGradients(
            T[] parameters,
            T[] inputs,
            T[] outputs,
            PDEDerivatives<T> derivatives);
    }

    /// <summary>
    /// Specifies the type of regularization for inverse problems.
    /// </summary>
    public enum InverseProblemRegularization
    {
        /// <summary>
        /// No regularization (may be unstable).
        /// </summary>
        None,

        /// <summary>
        /// L2 (Tikhonov) regularization: Prefers small parameter values.
        /// </summary>
        /// <remarks>
        /// Adds term: λ * Σᵢ (pᵢ)²
        /// Effect: Pulls parameters toward zero
        /// Good when: Expected parameters are small
        /// </remarks>
        L2Tikhonov,

        /// <summary>
        /// L1 (Lasso) regularization: Prefers sparse parameters.
        /// </summary>
        /// <remarks>
        /// Adds term: λ * Σᵢ |pᵢ|
        /// Effect: Encourages some parameters to be exactly zero
        /// Good when: Only few parameters are actually relevant
        /// </remarks>
        L1Lasso,

        /// <summary>
        /// Total Variation regularization: Prefers smooth parameter fields.
        /// </summary>
        /// <remarks>
        /// Adds term: λ * ∫|∇p|
        /// Effect: Encourages piecewise constant parameter distributions
        /// Good when: Parameter is spatially varying but piecewise constant
        /// </remarks>
        TotalVariation,

        /// <summary>
        /// Elastic Net: Combination of L1 and L2.
        /// </summary>
        /// <remarks>
        /// Adds term: λ₁ * Σᵢ |pᵢ| + λ₂ * Σᵢ (pᵢ)²
        /// Effect: Sparse parameters with controlled magnitudes
        /// Good when: Want benefits of both L1 and L2
        /// </remarks>
        ElasticNet,

        /// <summary>
        /// Bayesian regularization using prior distributions.
        /// </summary>
        /// <remarks>
        /// Treats parameters as random variables with prior distributions.
        /// Uses maximum a posteriori (MAP) estimation.
        /// Good when: Have prior knowledge about parameter distributions
        /// </remarks>
        Bayesian
    }

    /// <summary>
    /// Configuration options for inverse problem PINN training.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    public class InverseProblemOptions<T> : AiDotNet.Models.Options.ModelOptions
    {
        /// <summary>
        /// The type of regularization to apply.
        /// </summary>
        public InverseProblemRegularization Regularization { get; set; } = InverseProblemRegularization.L2Tikhonov;

        /// <summary>
        /// Regularization strength (λ in the formulas above).
        /// </summary>
        /// <remarks>
        /// Too small: Solution may be unstable (overfitting to noise)
        /// Too large: Parameters biased toward prior/regularization
        /// Rule of thumb: Start with λ ≈ noise_level² / signal_level²
        /// </remarks>
        public T? RegularizationStrength { get; set; }

        /// <summary>
        /// Weight for the observation data loss relative to physics loss.
        /// </summary>
        /// <remarks>
        /// Higher weight: Trust observations more
        /// Lower weight: Trust physics more
        /// For noisy data, use lower weight
        /// </remarks>
        public T? DataWeight { get; set; }

        /// <summary>
        /// Whether to use separate learning rates for solution and parameters.
        /// </summary>
        /// <remarks>
        /// Parameters often need different learning rates than the neural network.
        /// Typically, parameters need smaller learning rates for stability.
        /// </remarks>
        public bool UseSeparateLearningRates { get; set; } = true;

        /// <summary>
        /// Learning rate for the unknown parameters (if separate rates are used).
        /// </summary>
        public double ParameterLearningRate { get; set; } = 0.001;

        /// <summary>
        /// Whether to log parameter estimates during training.
        /// </summary>
        public bool LogParameterHistory { get; set; } = true;

        /// <summary>
        /// Prior means for Bayesian regularization.
        /// </summary>
        public T[]? PriorMeans { get; set; }

        /// <summary>
        /// Prior standard deviations for Bayesian regularization.
        /// </summary>
        public T[]? PriorStandardDeviations { get; set; }

        /// <summary>
        /// Whether to estimate parameter uncertainty.
        /// </summary>
        public bool EstimateUncertainty { get; set; } = false;

        /// <summary>
        /// Number of samples for uncertainty estimation (if enabled).
        /// </summary>
        public int UncertaintySamples { get; set; } = 100;
    }

    /// <summary>
    /// Results from inverse problem optimization.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    public class InverseProblemResult<T>
    {
        /// <summary>
        /// The identified parameter values.
        /// </summary>
        public T[] Parameters { get; set; } = Array.Empty<T>();

        /// <summary>
        /// Names of the identified parameters.
        /// </summary>
        public string[] ParameterNames { get; set; } = Array.Empty<string>();

        /// <summary>
        /// Estimated uncertainties (standard deviations) for each parameter.
        /// </summary>
        /// <remarks>
        /// Null if uncertainty estimation was not performed.
        /// </remarks>
        public T[]? ParameterUncertainties { get; set; }

        /// <summary>
        /// Final data loss (fit to observations).
        /// </summary>
        public T DataLoss { get; set; } = default!;

        /// <summary>
        /// Final physics loss (PDE residual).
        /// </summary>
        public T PhysicsLoss { get; set; } = default!;

        /// <summary>
        /// Total loss (data + physics + regularization).
        /// </summary>
        public T TotalLoss { get; set; } = default!;

        /// <summary>
        /// History of parameter values during training.
        /// </summary>
        /// <remarks>
        /// Useful for visualizing convergence and detecting oscillations.
        /// </remarks>
        public List<T[]>? ParameterHistory { get; set; }

        /// <summary>
        /// Whether the optimization converged.
        /// </summary>
        public bool Converged { get; set; }

        /// <summary>
        /// Number of iterations until convergence.
        /// </summary>
        public int IterationsToConverge { get; set; }

        /// <summary>
        /// Correlation matrix between parameters (for uncertainty analysis).
        /// </summary>
        /// <remarks>
        /// High correlation indicates parameters are not independently identifiable.
        /// </remarks>
        public T[,]? ParameterCorrelations { get; set; }
    }
}
