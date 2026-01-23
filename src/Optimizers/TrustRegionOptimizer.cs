using AiDotNet.Tensors.Engines.DirectGpu;
using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Trust Region optimization algorithm for machine learning models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// The Trust Region optimizer is an advanced optimization technique that uses local quadratic approximations
/// of the objective function to determine the next step. It maintains a region of trust around the current
/// solution where the approximation is considered reliable.
/// 
/// <para><b>For Beginners:</b> Think of this optimizer as an explorer with a map:
/// - The "trust region" is like the area on the map the explorer trusts to be accurate.
/// - In each step, the explorer looks at this trusted area to decide where to go next.
/// - If the predictions (map) match reality well, the explorer might expand the trusted area.
/// - If the predictions are off, the explorer shrinks the trusted area and becomes more cautious.
/// 
/// This approach helps the optimizer make good decisions even in complex landscapes, balancing between
/// making progress and staying reliable.
/// </para>
/// </remarks>
public class TrustRegionOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options for configuring the Trust Region optimizer.
    /// </summary>
    /// <remarks>
    /// This field stores the configuration settings for the Trust Region optimization algorithm.
    /// It includes parameters such as initial trust region radius, expansion and contraction factors,
    /// and thresholds for determining the success of each optimization step.
    /// </remarks>
    private TrustRegionOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The current radius of the trust region.
    /// </summary>
    /// <remarks>
    /// This field represents the current size of the trust region, which is dynamically adjusted
    /// during the optimization process. The trust region is a neighborhood around the current point
    /// where the quadratic approximation of the objective function is considered reliable.
    /// </remarks>
    private T _trustRegionRadius;

    /// <summary>
    /// The current iteration count of the optimization process.
    /// </summary>
    /// <remarks>
    /// This field keeps track of the number of iterations the optimizer has performed.
    /// It's used to enforce stopping criteria based on the maximum number of iterations
    /// and can be helpful for debugging or logging purposes.
    /// </remarks>
    private int _iteration;

    /// <summary>
    /// The previous parameters, used for trust region update computation.
    /// </summary>
    private Vector<T>? _trustRegionPreviousParameters;

    /// <summary>
    /// The previous gradient, used for trust region update computation.
    /// </summary>
    private Vector<T>? _trustRegionPreviousGradient;

    /// <summary>
    /// Initializes a new instance of the TrustRegionOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">Options for configuring the Trust Region optimizer.</param>
    /// <param name="engine">The computation engine (CPU or GPU) for vectorized operations.</param>
    public TrustRegionOptimizer(
        IFullModel<T, TInput, TOutput> model,
        TrustRegionOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new TrustRegionOptimizerOptions<T, TInput, TOutput>();
        _trustRegionRadius = NumOps.Zero;

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes adaptive parameters for the Trust Region optimizer.
    /// </summary>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();

        _trustRegionRadius = NumOps.FromDouble(_options.InitialTrustRegionRadius);
        _iteration = 0;
        _trustRegionPreviousParameters = null;
        _trustRegionPreviousGradient = null;
    }

    /// <summary>
    /// Updates parameters using a simplified Cauchy-point Trust Region step.
    /// </summary>
    /// <param name="parameters">The current parameters.</param>
    /// <param name="gradient">The gradient at the current parameters.</param>
    /// <returns>The updated parameters.</returns>
    /// <remarks>
    /// This implements a simplified Trust Region step based on the Cauchy point,
    /// which uses only gradient information. The step is constrained to lie within
    /// the trust region radius. The trust region radius is adapted based on the
    /// success of previous steps.
    /// </remarks>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        _iteration++;

        // Initialize trust region radius if needed
        if (NumOps.Equals(_trustRegionRadius, NumOps.Zero))
        {
            _trustRegionRadius = NumOps.FromDouble(_options.InitialTrustRegionRadius);
        }

        // Compute gradient norm
        var gradientNorm = gradient.Norm();

        // Avoid division by zero
        if (NumOps.LessThanOrEquals(gradientNorm, NumOps.FromDouble(1e-10)))
        {
            return parameters;
        }

        // Cauchy point computation:
        // The optimal step along the steepest descent direction is:
        // alpha = min(trust_radius / ||g||, ||g||^2 / (g^T B g))
        // For gradient-only case (assuming B = I), this simplifies to:
        // alpha = min(trust_radius / ||g||, 1)
        // Then: step = -alpha * g

        var alpha = NumOps.Divide(_trustRegionRadius, gradientNorm);

        // Cap alpha at learning rate for stability
        var maxAlpha = CurrentLearningRate;
        if (NumOps.GreaterThan(alpha, maxAlpha))
        {
            alpha = maxAlpha;
        }

        // Compute the descent direction (negative gradient normalized, then scaled)
        var direction = (Vector<T>)Engine.Multiply(gradient, NumOps.Negate(NumOps.One));
        var normalizedDirection = (Vector<T>)Engine.Divide(direction, gradientNorm);

        // Step is alpha * ||g|| in the normalized direction = alpha * (-g/||g||) * ||g|| = -alpha * g
        var stepSize = NumOps.Multiply(alpha, gradientNorm);
        var step = (Vector<T>)Engine.Multiply(normalizedDirection, stepSize);

        // New parameters
        var newParameters = (Vector<T>)Engine.Add(parameters, step);

        // Adapt trust region radius based on gradient change (proxy for step success)
        if (_trustRegionPreviousGradient is not null && _trustRegionPreviousParameters is not null)
        {
            // Compute the gradient difference to estimate curvature
            var gradDiff = (Vector<T>)Engine.Subtract(gradient, _trustRegionPreviousGradient);
            var paramDiff = (Vector<T>)Engine.Subtract(parameters, _trustRegionPreviousParameters);
            var paramDiffNorm = paramDiff.Norm();

            if (NumOps.GreaterThan(paramDiffNorm, NumOps.FromDouble(1e-10)))
            {
                // If gradient is decreasing in magnitude, step was likely successful
                var prevGradNorm = _trustRegionPreviousGradient.Norm();

                if (NumOps.LessThan(gradientNorm, prevGradNorm))
                {
                    // Successful step - expand trust region
                    _trustRegionRadius = NumOps.Multiply(_trustRegionRadius, NumOps.FromDouble(_options.ExpansionFactor));
                }
                else
                {
                    // Unsuccessful step - contract trust region
                    _trustRegionRadius = NumOps.Multiply(_trustRegionRadius, NumOps.FromDouble(_options.ContractionFactor));
                }

                // Clamp trust region radius
                _trustRegionRadius = MathHelper.Clamp(
                    _trustRegionRadius,
                    NumOps.FromDouble(_options.MinTrustRegionRadius),
                    NumOps.FromDouble(_options.MaxTrustRegionRadius));
            }
        }

        // Store current state for next iteration
        _trustRegionPreviousParameters = new Vector<T>(parameters);
        _trustRegionPreviousGradient = new Vector<T>(gradient);

        return newParameters;
    }

    /// <summary>
    /// Performs the optimization process using the Trust Region algorithm.
    /// </summary>
    /// <param name="inputData">The input data for optimization.</param>
    /// <returns>The result of the optimization process.</returns>
    /// <remarks>
    /// <para><b>DataLoader Integration:</b> This method uses the DataLoader API for epoch management.
    /// Trust Region methods typically operate on the full dataset because they construct a local quadratic
    /// model of the objective function that requires accurate gradient and Hessian information. The method
    /// notifies the sampler of epoch starts using <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.NotifyEpochStart"/>
    /// for compatibility with curriculum learning and sampling strategies.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);
            _iteration++;

            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            // Use efficient Hessian computation (automatically uses IGradientComputable if available)
            var hessian = ComputeHessianEfficiently(currentSolution, inputData);

            var stepDirection = SolveSubproblem(gradient, hessian);
            var proposedSolution = MoveInDirection(currentSolution, stepDirection, NumOps.One);

            var currentStepData = EvaluateSolution(currentSolution, inputData);
            var proposedStepData = EvaluateSolution(proposedSolution, inputData);

            var actualReduction = NumOps.Subtract(currentStepData.FitnessScore, proposedStepData.FitnessScore);
            var predictedReduction = CalculatePredictedReduction(gradient, hessian, stepDirection);

            var rho = NumOps.Divide(actualReduction, predictedReduction);

            if (NumOps.GreaterThan(rho, NumOps.FromDouble(_options.AcceptanceThreshold)))
            {
                currentSolution = proposedSolution;
                currentStepData = proposedStepData;
                UpdateTrustRegionRadius(rho);
            }
            else
            {
                ShrinkTrustRegionRadius();
            }

            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(epoch, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Calculates the Hessian matrix for the current solution.
    /// </summary>
    /// <param name="currentSolution">The current solution model.</param>
    /// <param name="inputData">The input data for optimization.</param>
    /// <returns>The Hessian matrix.</returns>
    /// <remarks>
    /// This method uses finite differences to approximate the Hessian matrix.
    /// 
    /// <para><b>For Beginners:</b> The Hessian is like a map of the terrain's curvature:
    /// - It tells us how quickly the gradient changes in different directions.
    /// - This information helps predict how the function behaves around the current point.
    /// - Calculating it accurately is crucial for making good decisions in the optimization process.
    /// </para>
    /// </remarks>
    private Matrix<T> CalculateHessian(IFullModel<T, TInput, TOutput> currentSolution, OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var coefficients = currentSolution.GetParameters();
        var hessian = new Matrix<T>(coefficients.Length, coefficients.Length);
        var epsilon = NumOps.FromDouble(1e-8);

        for (int i = 0; i < coefficients.Length; i++)
        {
            for (int j = i; j < coefficients.Length; j++)
            {
                var perturbed1 = currentSolution.Clone();
                var perturbed2 = currentSolution.Clone();
                var perturbed3 = currentSolution.Clone();
                var perturbed4 = currentSolution.Clone();

                var coeffs1 = coefficients.Clone();
                var coeffs2 = coefficients.Clone();
                var coeffs3 = coefficients.Clone();
                var coeffs4 = coefficients.Clone();

                coeffs1[i] = NumOps.Add(coeffs1[i], epsilon);
                coeffs1[j] = NumOps.Add(coeffs1[j], epsilon);

                coeffs2[i] = NumOps.Add(coeffs2[i], epsilon);
                coeffs2[j] = NumOps.Subtract(coeffs2[j], epsilon);

                coeffs3[i] = NumOps.Subtract(coeffs3[i], epsilon);
                coeffs3[j] = NumOps.Add(coeffs3[j], epsilon);

                coeffs4[i] = NumOps.Subtract(coeffs4[i], epsilon);
                coeffs4[j] = NumOps.Subtract(coeffs4[j], epsilon);

                perturbed1 = perturbed1.WithParameters(coeffs1);
                perturbed2 = perturbed2.WithParameters(coeffs2);
                perturbed3 = perturbed3.WithParameters(coeffs3);
                perturbed4 = perturbed4.WithParameters(coeffs4);

                var f11 = EvaluateSolution(perturbed1, inputData).FitnessScore;
                var f12 = EvaluateSolution(perturbed2, inputData).FitnessScore;
                var f21 = EvaluateSolution(perturbed3, inputData).FitnessScore;
                var f22 = EvaluateSolution(perturbed4, inputData).FitnessScore;

                var secondDerivative = NumOps.Divide(
                    NumOps.Subtract(
                        NumOps.Add(f11, f22),
                        NumOps.Add(f12, f21)
                    ),
                    NumOps.Multiply(NumOps.FromDouble(4), NumOps.Multiply(epsilon, epsilon))
                );

                hessian[i, j] = secondDerivative;
                hessian[j, i] = secondDerivative;
            }
        }

        return hessian;
    }

    /// <summary>
    /// Moves the current solution in the specified direction with the given step size.
    /// </summary>
    /// <param name="currentSolution">The current solution model.</param>
    /// <param name="direction">The direction to move in.</param>
    /// <param name="stepSize">The size of the step to take.</param>
    /// <returns>A new solution model after moving in the specified direction.</returns>
    private IFullModel<T, TInput, TOutput> MoveInDirection(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> direction, T stepSize)
    {
        // === Vectorized Direction Move using IEngine (Phase B: US-GPU-015) ===
        // new_params = current_params + direction * stepSize

        var newModel = currentSolution.Clone();
        var currentCoefficients = newModel.GetParameters();

        var scaledDirection = (Vector<T>)Engine.Multiply(direction, stepSize);
        var newCoefficients = (Vector<T>)Engine.Add(currentCoefficients, scaledDirection);

        return newModel.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Solves the trust region subproblem using the Conjugate Gradient method.
    /// </summary>
    /// <param name="gradient">The gradient vector at the current point.</param>
    /// <param name="hessian">The Hessian matrix at the current point.</param>
    /// <returns>The solution vector to the subproblem.</returns>
    /// <remarks>
    /// This method implements the Steihaug-Toint Conjugate Gradient algorithm to solve the trust region subproblem.
    /// It finds a step direction that minimizes the quadratic model of the objective function within the trust region.
    /// 
    /// <para><b>For Beginners:</b> This method is like solving a puzzle within the trust region:
    /// - It tries to find the best direction to move, considering both the slope (gradient) and curvature (Hessian).
    /// - It uses an iterative process (Conjugate Gradient) to refine the solution.
    /// - If it hits the boundary of the trust region, it adjusts the step to stay within bounds.
    /// - The process stops when it finds a good enough solution or reaches the maximum number of iterations.
    /// </para>
    /// </remarks>
    private Vector<T> SolveSubproblem(Vector<T> gradient, Matrix<T> hessian)
    {
        // === Partially Vectorized Steihaug-Toint CG using IEngine (Phase B: US-GPU-015) ===

        var z = new Vector<T>(gradient.Length);
        var r = gradient.Clone();
        // Vectorized negation: d = -r
        var d = (Vector<T>)Engine.Multiply(r, NumOps.Negate(NumOps.One));
        var g0 = gradient.DotProduct(gradient);

        for (int i = 0; i < _options.MaxCGIterations; i++)
        {
            var Hd = hessian.Multiply(d);
            var dHd = d.DotProduct(Hd);

            if (NumOps.LessThanOrEquals(dHd, NumOps.Zero))
            {
                return ComputeBoundaryStep(z, d);
            }

            var alpha = NumOps.Divide(r.DotProduct(r), dHd);
            var zNext = z.Add(d.Multiply(alpha));

            if (NumOps.GreaterThan(zNext.Norm(), _trustRegionRadius))
            {
                return ComputeBoundaryStep(z, d);
            }

            z = zNext;
            var rNext = r.Add(Hd.Multiply(alpha));
            var beta = NumOps.Divide(rNext.DotProduct(rNext), r.DotProduct(r));

            // Vectorized negation: dNext = -rNext
            var dNext = (Vector<T>)Engine.Multiply(rNext, NumOps.Negate(NumOps.One));
            d = dNext.Add(d.Multiply(beta));

            r = rNext;

            if (NumOps.LessThan(r.Norm(), NumOps.Multiply(NumOps.FromDouble(_options.CGTolerance), g0)))
            {
                break;
            }
        }

        return z;
    }

    /// <summary>
    /// Computes a step that lies on the boundary of the trust region.
    /// </summary>
    /// <param name="z">The current interior point.</param>
    /// <param name="d">The direction vector.</param>
    /// <returns>A vector representing a step to the trust region boundary.</returns>
    /// <remarks>
    /// This method is called when the Conjugate Gradient method proposes a step that would exit the trust region.
    /// It calculates the intersection of the proposed step with the trust region boundary.
    /// 
    /// <para><b>For Beginners:</b> Think of this as finding where a line (your step) intersects a sphere (the trust region):
    /// - If the proposed step goes outside the trust region, we need to stop at the boundary.
    /// - This method calculates exactly where that intersection occurs.
    /// - It ensures we take the largest possible step while staying within our "trusted" area.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeBoundaryStep(Vector<T> z, Vector<T> d)
    {
        var a = d.DotProduct(d);
        var b = NumOps.Multiply(NumOps.FromDouble(2), z.DotProduct(d));
        var c = NumOps.Subtract(z.DotProduct(z), NumOps.Multiply(_trustRegionRadius, _trustRegionRadius));
        var tau = SolveQuadratic(a, b, c);

        return z.Add(d.Multiply(tau));
    }

    /// <summary>
    /// Solves a quadratic equation of the form ax^2 + bx + c = 0.
    /// </summary>
    /// <param name="a">The coefficient of x^2.</param>
    /// <param name="b">The coefficient of x.</param>
    /// <param name="c">The constant term.</param>
    /// <returns>The positive root of the quadratic equation.</returns>
    /// <remarks>
    /// This method is used in computing the boundary step. It solves the quadratic equation that arises
    /// when finding the intersection of a line with a sphere (in this case, the trust region boundary).
    /// 
    /// <para><b>For Beginners:</b> This is like solving a basic algebra problem:
    /// - We're finding where a curved line (parabola) crosses zero.
    /// - In our case, this helps find where our step meets the edge of the trust region.
    /// - We always return the positive solution, as we're interested in moving forward, not backward.
    /// </para>
    /// </remarks>
    private T SolveQuadratic(T a, T b, T c)
    {
        var discriminant = NumOps.Subtract(NumOps.Multiply(b, b), NumOps.Multiply(NumOps.Multiply(NumOps.FromDouble(4), a), c));
        var sqrtDiscriminant = NumOps.Sqrt(discriminant);
        var tau1 = NumOps.Divide(NumOps.Add(NumOps.Negate(b), sqrtDiscriminant), NumOps.Multiply(NumOps.FromDouble(2), a));
        var tau2 = NumOps.Divide(NumOps.Subtract(NumOps.Negate(b), sqrtDiscriminant), NumOps.Multiply(NumOps.FromDouble(2), a));

        return NumOps.GreaterThan(tau1, NumOps.Zero) ? tau1 : tau2;
    }

    /// <summary>
    /// Calculates the predicted reduction in the objective function for a given step.
    /// </summary>
    /// <param name="gradient">The gradient at the current point.</param>
    /// <param name="hessian">The Hessian matrix at the current point.</param>
    /// <param name="stepDirection">The proposed step direction.</param>
    /// <returns>The predicted reduction in the objective function.</returns>
    /// <remarks>
    /// This method computes how much the quadratic model predicts the objective function will decrease
    /// if we take the proposed step. It's used to evaluate the quality of the proposed step.
    /// 
    /// <para><b>For Beginners:</b> This is like estimating how much you'll save on a shopping trip:
    /// - The gradient tells you the initial rate of savings.
    /// - The Hessian helps adjust that estimate based on how the savings rate might change.
    /// - Combining these gives you a prediction of your total savings (or in our case, improvement in the function).
    /// </para>
    /// </remarks>
    private T CalculatePredictedReduction(Vector<T> gradient, Matrix<T> hessian, Vector<T> stepDirection)
    {
        var linearTerm = gradient.DotProduct(stepDirection);
        var quadraticTerm = stepDirection.DotProduct(hessian.Multiply(stepDirection));

        return NumOps.Add(linearTerm, NumOps.Multiply(NumOps.FromDouble(0.5), quadraticTerm));
    }

    /// <summary>
    /// Updates the trust region radius based on the success of the last step.
    /// </summary>
    /// <param name="rho">The ratio of actual improvement to predicted improvement.</param>
    /// <remarks>
    /// This method adjusts the size of the trust region based on how well the quadratic model predicted
    /// the actual change in the objective function. If the prediction was very accurate, the trust region
    /// may be expanded. If it was poor, the trust region is contracted.
    /// 
    /// <para><b>For Beginners:</b> This is like adjusting how much you trust your map:
    /// - If your map (model) accurately predicted the terrain, you might explore a larger area next time.
    /// - If the map wasn't very accurate, you'd be more cautious and explore a smaller area.
    /// - The method ensures the trust region stays within reasonable bounds, neither too large nor too small.
    /// </para>
    /// </remarks>
    private void UpdateTrustRegionRadius(T rho)
    {
        if (NumOps.GreaterThan(rho, NumOps.FromDouble(_options.VerySuccessfulThreshold)))
        {
            _trustRegionRadius = NumOps.Multiply(_trustRegionRadius, NumOps.FromDouble(_options.ExpansionFactor));
        }
        else if (NumOps.LessThan(rho, NumOps.FromDouble(_options.UnsuccessfulThreshold)))
        {
            _trustRegionRadius = NumOps.Multiply(_trustRegionRadius, NumOps.FromDouble(_options.ContractionFactor));
        }

        _trustRegionRadius = MathHelper.Clamp(_trustRegionRadius, NumOps.FromDouble(_options.MinTrustRegionRadius), NumOps.FromDouble(_options.MaxTrustRegionRadius));
    }

    /// <summary>
    /// Reduces the size of the trust region.
    /// </summary>
    /// <remarks>
    /// This method is called when a step is rejected, indicating that the current trust region may be too large.
    /// It contracts the trust region to promote more conservative steps in subsequent iterations.
    /// 
    /// <para><b>For Beginners:</b> This is like becoming more cautious after a misstep:
    /// - If your last move wasn't good, you'd naturally want to take smaller, more careful steps.
    /// - This method does exactly that - it makes the "trusted" area smaller.
    /// - It ensures that even after shrinking, the trust region isn't too small to make progress.
    /// </para>
    /// </remarks>
    private void ShrinkTrustRegionRadius()
    {
        _trustRegionRadius = NumOps.Multiply(_trustRegionRadius, NumOps.FromDouble(_options.ContractionFactor));
        _trustRegionRadius = MathHelper.Clamp(_trustRegionRadius, NumOps.FromDouble(_options.MinTrustRegionRadius), NumOps.FromDouble(_options.MaxTrustRegionRadius));
    }

    /// <summary>
    /// Updates adaptive parameters based on the current and previous optimization steps.
    /// </summary>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    /// <remarks>
    /// This method adjusts the trust region radius based on the improvement in the objective function.
    /// It's part of the adaptive behavior of the trust region algorithm, allowing it to respond to the
    /// characteristics of the objective function landscape.
    /// 
    /// <para><b>For Beginners:</b> This is like learning from your recent experiences:
    /// - If your last step improved things, you might become a bit more adventurous.
    /// - If things got worse, you'd become more cautious.
    /// - This method automatically adjusts how boldly or cautiously the algorithm explores,
    ///   based on whether recent steps have been successful or not.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveTrustRegionRadius)
        {
            var improvement = NumOps.Subtract(currentStepData.FitnessScore, previousStepData.FitnessScore);
            var adaptationRate = NumOps.FromDouble(_options.AdaptationRate);

            if (NumOps.GreaterThan(improvement, NumOps.Zero))
            {
                _trustRegionRadius = NumOps.Multiply(_trustRegionRadius, NumOps.Add(NumOps.One, adaptationRate));
            }
            else
            {
                _trustRegionRadius = NumOps.Multiply(_trustRegionRadius, NumOps.Subtract(NumOps.One, adaptationRate));
            }

            _trustRegionRadius = MathHelper.Clamp(_trustRegionRadius, NumOps.FromDouble(_options.MinTrustRegionRadius), NumOps.FromDouble(_options.MaxTrustRegionRadius));
        }
    }

    /// <summary>
    /// Updates the optimizer options.
    /// </summary>
    /// <param name="options">The new options to be set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of type TrustRegionOptimizerOptions.</exception>
    /// <remarks>
    /// This method allows updating the optimizer's configuration during runtime. It ensures that only
    /// the correct type of options (TrustRegionOptimizerOptions) can be set for this optimizer.
    /// 
    /// <para><b>For Beginners:</b> This is like changing the settings on your GPS mid-journey:
    /// - You can adjust how the optimizer behaves without starting over.
    /// - It checks to make sure you're using the right kind of settings for this specific optimizer.
    /// - If you try to use the wrong type of settings, it will let you know with an error.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is TrustRegionOptimizerOptions<T, TInput, TOutput> trustRegionOptions)
        {
            _options = trustRegionOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected TrustRegionOptimizerOptions.");
        }
    }

    /// <summary>
    /// Retrieves the current options of the optimizer.
    /// </summary>
    /// <returns>The current TrustRegionOptimizerOptions.</returns>
    /// <remarks>
    /// This method provides access to the current configuration of the Trust Region optimizer.
    /// It allows inspection of the optimizer's settings without modifying them.
    /// 
    /// <para><b>For Beginners:</b> This is like checking your current GPS settings:
    /// - You can see how the optimizer is currently configured.
    /// - This is useful if you want to review or log the current settings.
    /// - It doesn't change anything; it just lets you look at the current setup.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Updates parameters using GPU-accelerated Trust Region method.
    /// </summary>
    public override void UpdateParametersGpu(IGpuBuffer parameters, IGpuBuffer gradients, int parameterCount, IDirectGpuBackend backend)
    {
        // Trust Region requires complex Hessian computations - use CPU fallback for now
        // TODO: Implement full GPU Trust Region with Hessian approximation
        throw new NotSupportedException("Trust Region GPU update requires Hessian computation - use CPU optimizer or different method");
    }

    /// <summary>
    /// Serializes the current state of the optimizer into a byte array.
    /// </summary>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    /// <remarks>
    /// This method saves the current state of the optimizer, including its base class state and
    /// specific Trust Region optimizer properties. This allows the optimizer's state to be stored
    /// or transmitted and later reconstructed.
    /// 
    /// <para><b>For Beginners:</b> This is like taking a snapshot of the optimizer:
    /// - It captures all the important information about the optimizer's current state.
    /// - This snapshot can be saved or sent somewhere else.
    /// - Later, you can use this snapshot to recreate the optimizer exactly as it was.
    /// - It's useful for things like saving progress, or moving the optimization process to a different machine.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            string optionsJson = JsonConvert.SerializeObject(_options);
            writer.Write(optionsJson);

            writer.Write(_iteration);
            writer.Write(Convert.ToDouble(_trustRegionRadius));

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes the optimizer's state from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of optimizer options fails.</exception>
    /// <remarks>
    /// This method reconstructs the optimizer's state from a serialized byte array. It restores both
    /// the base class state and the specific properties of the Trust Region optimizer.
    /// 
    /// <para><b>For Beginners:</b> This is like reconstructing the optimizer from a snapshot:
    /// - It takes the snapshot (byte array) created by the Serialize method.
    /// - From this snapshot, it rebuilds the optimizer to the exact state it was in when serialized.
    /// - This is useful for resuming a paused optimization process or moving it to a different machine.
    /// - If there's a problem reading the optimizer's settings, it will raise an error to let you know.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<TrustRegionOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _iteration = reader.ReadInt32();
            _trustRegionRadius = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}
