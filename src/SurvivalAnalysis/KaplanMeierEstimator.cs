using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.SurvivalAnalysis;

/// <summary>
/// Implements the Kaplan-Meier estimator for non-parametric survival analysis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The Kaplan-Meier estimator is a non-parametric method for estimating the survival function
/// from lifetime data. It doesn't use covariates - it estimates a single survival curve for
/// all subjects in the dataset.
/// </para>
/// <para>
/// <b>For Beginners:</b> Kaplan-Meier is the simplest and most widely used survival method.
/// It creates a "staircase" survival curve that shows the probability of survival over time.
///
/// How it works:
/// 1. Sort all observation times
/// 2. At each time point where an event occurs:
///    - Count how many subjects are "at risk" (still in the study)
///    - Count how many had the event
///    - Survival probability = (at risk - events) / at risk
/// 3. Cumulative survival = product of all these probabilities up to time t
///
/// Example:
/// - Time 0: 100 patients, all alive → S(0) = 1.0
/// - Time 1: 100 at risk, 10 die → S(1) = 1.0 × (90/100) = 0.90
/// - Time 2: 90 at risk (some left the study), 5 die → S(2) = 0.90 × (85/90) = 0.85
///
/// Key features:
/// - No assumptions about the shape of survival (non-parametric)
/// - Handles censoring naturally
/// - Does NOT use patient features - same curve for everyone
///
/// For comparing groups or using features, use Cox Proportional Hazards instead.
///
/// References:
/// - Kaplan &amp; Meier (1958). "Nonparametric Estimation from Incomplete Observations"
/// </para>
/// </remarks>
public class KaplanMeierEstimator<T> : SurvivalModelBase<T>
{
    /// <summary>
    /// Stores the survival probability at each event time.
    /// </summary>
    private Vector<T>? _survivalProbabilities;

    /// <summary>
    /// Stores the number at risk at each event time.
    /// </summary>
    private Vector<int>? _numberAtRisk;

    /// <summary>
    /// Stores the number of events at each event time.
    /// </summary>
    private Vector<int>? _numberEvents;

    /// <summary>
    /// Gets the model type.
    /// </summary>
    public override ModelType GetModelType() => ModelType.KaplanMeierEstimator;

    /// <summary>
    /// Gets whether JIT compilation is supported.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Kaplan-Meier supports JIT compilation by embedding the survival
    /// curve as a computation graph. The step function is represented using piecewise
    /// evaluation with comparisons and weighted sums, which can be compiled to fast
    /// native code for accelerated inference.
    ///
    /// The computation graph embeds:
    /// 1. Event times as constants
    /// 2. Survival probabilities as constants
    /// 3. Comparison operations to find which time bin the input falls into
    /// 4. Weighted sum to select the correct survival probability
    ///
    /// This approach allows the JIT compiler to optimize the lookup operations
    /// for batch evaluation of survival probabilities.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Initializes a new instance of the KaplanMeierEstimator class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Kaplan-Meier has no parameters to set - it's a non-parametric
    /// method that estimates the survival curve directly from the data.
    ///
    /// Usage:
    /// <code>
    /// var km = new KaplanMeierEstimator&lt;double&gt;();
    /// km.FitSurvival(features, times, events);
    /// var survivalProbs = km.GetBaselineSurvival(timePoints);
    /// </code>
    /// </para>
    /// </remarks>
    public KaplanMeierEstimator()
    {
    }

    /// <summary>
    /// Fits the Kaplan-Meier estimator to the survival data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes the Kaplan-Meier survival curve using the
    /// product-limit estimator. The formula at each event time is:
    ///
    /// S(t) = S(t_previous) × (n_at_risk - n_events) / n_at_risk
    ///
    /// Note: Kaplan-Meier ignores the feature matrix X - it estimates a single curve
    /// for all subjects. If you want to use features, use CoxProportionalHazards.
    /// </para>
    /// </remarks>
    protected override void FitSurvivalCore(Matrix<T> x, Vector<T> times, Vector<int> events)
    {
        // Get unique event times
        TrainedEventTimes = GetUniqueEventTimes(times, events);

        if (TrainedEventTimes.Length == 0)
        {
            // No events - survival is 1.0 everywhere
            TrainedEventTimes = new Vector<T>(1);
            TrainedEventTimes[0] = NumOps.FromDouble(1.0);
            _survivalProbabilities = new Vector<T>(1);
            _survivalProbabilities[0] = NumOps.One;
            BaselineSurvivalFunction = _survivalProbabilities;
            _numberAtRisk = new Vector<int>(1) { [0] = times.Length };
            _numberEvents = new Vector<int>(1) { [0] = 0 };
            return;
        }

        int numTimes = TrainedEventTimes.Length;
        _survivalProbabilities = new Vector<T>(numTimes);
        _numberAtRisk = new Vector<int>(numTimes);
        _numberEvents = new Vector<int>(numTimes);

        T cumulativeSurvival = NumOps.One;

        for (int t = 0; t < numTimes; t++)
        {
            T eventTime = TrainedEventTimes[t];
            double eventTimeDouble = NumOps.ToDouble(eventTime);

            // Count number at risk and number of events at this time
            int atRisk = 0;
            int numEventsAtTime = 0;

            for (int i = 0; i < times.Length; i++)
            {
                double ti = NumOps.ToDouble(times[i]);

                // At risk if observation time >= event time
                if (ti >= eventTimeDouble)
                {
                    atRisk++;
                }

                // Event at this time if event occurred and time matches
                if (events[i] == 1 && Math.Abs(ti - eventTimeDouble) < 1e-10)
                {
                    numEventsAtTime++;
                }
            }

            _numberAtRisk[t] = atRisk;
            _numberEvents[t] = numEventsAtTime;

            // Kaplan-Meier product-limit estimator
            if (atRisk > 0)
            {
                T survivalFraction = NumOps.FromDouble((double)(atRisk - numEventsAtTime) / atRisk);
                cumulativeSurvival = NumOps.Multiply(cumulativeSurvival, survivalFraction);
            }

            _survivalProbabilities[t] = cumulativeSurvival;
        }

        BaselineSurvivalFunction = _survivalProbabilities;
    }

    /// <summary>
    /// Predicts survival probabilities at specified time points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Kaplan-Meier returns the same survival curve for all subjects
    /// (it ignores the features). The prediction interpolates between the observed event times.
    /// </para>
    /// </remarks>
    public override Matrix<T> PredictSurvivalProbability(Matrix<T> x, Vector<T> times)
    {
        EnsureFitted();

        var result = new Matrix<T>(x.Rows, times.Length);
        var baseline = GetBaselineSurvival(times);

        // Kaplan-Meier gives same survival for all subjects
        for (int i = 0; i < x.Rows; i++)
        {
            for (int t = 0; t < times.Length; t++)
            {
                result[i, t] = baseline[t];
            }
        }

        return result;
    }

    /// <summary>
    /// Predicts hazard ratios (all 1.0 for Kaplan-Meier since it doesn't use covariates).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Kaplan-Meier doesn't estimate covariate effects, so all subjects
    /// have a hazard ratio of 1.0 (same as baseline). For hazard ratios based on features,
    /// use CoxProportionalHazards.
    /// </para>
    /// </remarks>
    public override Vector<T> PredictHazardRatio(Matrix<T> x)
    {
        EnsureFitted();

        // Kaplan-Meier doesn't estimate covariate effects
        var result = new Vector<T>(x.Rows);
        for (int i = 0; i < x.Rows; i++)
        {
            result[i] = NumOps.One;
        }

        return result;
    }

    /// <summary>
    /// Gets the baseline survival function at specified time points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns the Kaplan-Meier survival curve evaluated at the
    /// requested time points. The curve is a step function - survival probability only changes
    /// at actual event times.
    /// </para>
    /// </remarks>
    public override Vector<T> GetBaselineSurvival(Vector<T> times)
    {
        EnsureFitted();

        if (TrainedEventTimes is null || _survivalProbabilities is null)
        {
            throw new InvalidOperationException("Model has no survival function stored.");
        }

        var result = new Vector<T>(times.Length);

        for (int i = 0; i < times.Length; i++)
        {
            double t = NumOps.ToDouble(times[i]);

            // Find survival at this time (step function - use last event time <= t)
            T survival = NumOps.One;
            for (int j = 0; j < TrainedEventTimes.Length; j++)
            {
                if (NumOps.ToDouble(TrainedEventTimes[j]) <= t)
                {
                    survival = _survivalProbabilities[j];
                }
                else
                {
                    break;
                }
            }

            result[i] = survival;
        }

        return result;
    }

    /// <summary>
    /// Standard prediction - returns survival probability at a reference time.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> x)
    {
        EnsureFitted();

        // Return survival at median observed time
        if (TrainedEventTimes is null || TrainedEventTimes.Length == 0)
        {
            var ones = new Vector<T>(x.Rows);
            for (int i = 0; i < x.Rows; i++)
            {
                ones[i] = NumOps.One;
            }
            return ones;
        }

        int medianIdx = TrainedEventTimes.Length / 2;
        var time = new Vector<T>(1) { [0] = TrainedEventTimes[medianIdx] };
        var probs = PredictSurvivalProbability(x, time);

        var result = new Vector<T>(x.Rows);
        for (int i = 0; i < x.Rows; i++)
        {
            result[i] = probs[i, 0];
        }

        return result;
    }

    /// <summary>
    /// Gets the number at risk at each event time.
    /// </summary>
    /// <returns>Vector of counts at risk.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> "At risk" means the subject is still being observed and hasn't
    /// had the event yet. This is useful for understanding the reliability of survival estimates -
    /// estimates based on more subjects are more reliable.
    /// </para>
    /// </remarks>
    public Vector<int>? GetNumberAtRisk() => _numberAtRisk;

    /// <summary>
    /// Gets the number of events at each event time.
    /// </summary>
    /// <returns>Vector of event counts.</returns>
    public Vector<int>? GetNumberEvents() => _numberEvents;

    /// <summary>
    /// Gets the event times used in the survival curve.
    /// </summary>
    /// <returns>Vector of unique event times.</returns>
    public Vector<T>? GetEventTimes() => TrainedEventTimes;

    /// <summary>
    /// Gets the survival probability at each event time.
    /// </summary>
    /// <returns>Vector of survival probabilities.</returns>
    public Vector<T>? GetSurvivalProbabilities() => _survivalProbabilities;

    #region JIT Compilation Support

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the survival probability prediction.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Kaplan-Meier survival curve is a step function that can be
    /// JIT compiled using a piecewise evaluation approach. The computation graph:
    ///
    /// 1. Event times t₁, t₂, ..., tₙ (constants)
    /// 2. Survival probabilities S₁, S₂, ..., Sₙ (constants)
    /// 3. For input time t, computes indicators: Iᵢ = sigmoid(steepness × (t - tᵢ))
    /// 4. Combines: S(t) = S₁×(1-I₁) + S₂×(I₁-I₂) + ... + Sₙ×Iₙ₋₁
    ///
    /// The steep sigmoid approximates the step function while remaining differentiable.
    /// For exact evaluation (non-differentiable), use the standard Predict method.
    ///
    /// This approach enables:
    /// - Batch evaluation of survival probabilities
    /// - JIT optimization of the lookup operations
    /// - GPU acceleration for large datasets
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (!IsFitted || TrainedEventTimes is null || _survivalProbabilities is null)
        {
            throw new InvalidOperationException(
                "Model must be fitted before exporting computation graph.");
        }

        if (inputNodes is null)
        {
            throw new ArgumentNullException(nameof(inputNodes));
        }

        int numEventTimes = TrainedEventTimes.Length;

        // Create input placeholder for query time: [batchSize, 1]
        var inputTensor = new Tensor<T>(new int[] { 1, 1 });
        var inputNode = TensorOperations<T>.Variable(inputTensor, "query_time");
        inputNodes.Add(inputNode);

        // Create constant tensor for event times: [numEventTimes, 1]
        var eventTimesTensor = new Tensor<T>(new int[] { numEventTimes, 1 });
        for (int i = 0; i < numEventTimes; i++)
        {
            eventTimesTensor[i, 0] = TrainedEventTimes[i];
        }
        var eventTimesNode = TensorOperations<T>.Constant(eventTimesTensor, "event_times");
        inputNodes.Add(eventTimesNode);

        // Create constant tensor for survival probabilities: [numEventTimes, 1]
        var survivalTensor = new Tensor<T>(new int[] { numEventTimes, 1 });
        for (int i = 0; i < numEventTimes; i++)
        {
            survivalTensor[i, 0] = _survivalProbabilities[i];
        }
        var survivalNode = TensorOperations<T>.Constant(survivalTensor, "survival_probs");
        inputNodes.Add(survivalNode);

        // For JIT compilation of step function lookup:
        // We use a differentiable approximation with steep sigmoid:
        // indicator_i = sigmoid(steepness × (time - event_time_i))
        //
        // The survival at time t is the survival at the last event time <= t.
        // This is computed as: S(t) = Σ weight_i × survival_i
        // where weight_i represents the probability that time falls in interval [t_i, t_{i+1})
        //
        // For exact step function: weight_i = I(t >= t_i) × I(t < t_{i+1})
        // For differentiable: weight_i = sigmoid(steepness × (t - t_i)) × (1 - sigmoid(steepness × (t - t_{i+1})))

        // Steepness constant for sigmoid approximation (higher = closer to step function)
        double steepness = 100.0;
        var steepnessTensor = new Tensor<T>(new int[] { 1, 1 });
        steepnessTensor[0, 0] = NumOps.FromDouble(steepness);
        var steepnessNode = TensorOperations<T>.Constant(steepnessTensor, "steepness");
        inputNodes.Add(steepnessNode);

        // Compute indicators for each event time: sigmoid(steepness × (time - event_time_i))
        // This represents I(time >= event_time_i) in a differentiable way
        //
        // For Kaplan-Meier step function:
        // S(t) = S_k where k = max{i : event_time_i <= t}
        //
        // This can be computed as:
        // S(t) = S_0 × (1 - I_0) + S_1 × (I_0 - I_1) + S_2 × (I_1 - I_2) + ... + S_n × I_{n-1}
        //      = S_0 - S_0×I_0 + S_1×I_0 - S_1×I_1 + S_2×I_1 - ... + S_n×I_{n-1}
        //      = S_0 + Σ (S_{i+1} - S_i) × I_i

        // Initial survival before any event is 1.0
        var oneTensor = new Tensor<T>(new int[] { 1, 1 });
        oneTensor[0, 0] = NumOps.One;
        var oneNode = TensorOperations<T>.Constant(oneTensor, "one");

        // Start with survival = 1.0 (before any events)
        var outputNode = oneNode;

        // Build the step function as sum of weighted differences
        // For each event time i: add (S_i - S_{i-1}) × sigmoid(steepness × (t - t_i))
        for (int i = 0; i < numEventTimes; i++)
        {
            // Extract event time i
            var eventTimeSingleTensor = new Tensor<T>(new int[] { 1, 1 });
            eventTimeSingleTensor[0, 0] = TrainedEventTimes[i];
            var eventTimeNode = TensorOperations<T>.Constant(eventTimeSingleTensor, $"event_time_{i}");

            // Compute (time - event_time_i)
            var diffNode = TensorOperations<T>.Subtract(inputNode, eventTimeNode);

            // Multiply by steepness
            var scaledDiffNode = TensorOperations<T>.ElementwiseMultiply(diffNode, steepnessNode);

            // Apply sigmoid: indicator_i = sigmoid(steepness × (time - event_time_i))
            var indicatorNode = TensorOperations<T>.Sigmoid(scaledDiffNode);

            // Compute survival difference: delta_i = S_i - S_{i-1}
            // For i=0, S_{-1} = 1.0
            double prevSurvival = i == 0 ? 1.0 : NumOps.ToDouble(_survivalProbabilities[i - 1]);
            double currSurvival = NumOps.ToDouble(_survivalProbabilities[i]);
            double survivalDiff = currSurvival - prevSurvival;

            // Create delta tensor
            var deltaTensor = new Tensor<T>(new int[] { 1, 1 });
            deltaTensor[0, 0] = NumOps.FromDouble(survivalDiff);
            var deltaNode = TensorOperations<T>.Constant(deltaTensor, $"survival_diff_{i}");

            // Compute contribution: delta_i × indicator_i
            var contributionNode = TensorOperations<T>.ElementwiseMultiply(deltaNode, indicatorNode);

            // Add to running sum: output = output + contribution
            outputNode = TensorOperations<T>.Add(outputNode, contributionNode);
        }

        outputNode.Name = "survival_probability";
        return outputNode;
    }

    #endregion

    #region IFullModel Implementation

    /// <summary>
    /// Gets all model parameters as a single vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Kaplan-Meier is non-parametric, meaning it doesn't have
    /// traditional parameters like coefficients. This returns the survival probabilities
    /// at each event time, which represent the "state" of the fitted model.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        if (_survivalProbabilities is null)
        {
            return new Vector<T>(0);
        }

        return _survivalProbabilities;
    }

    /// <summary>
    /// Sets the parameters for this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For Kaplan-Meier, this sets the survival probabilities.
    /// This is mainly used during deserialization to restore a fitted model.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length > 0)
        {
            _survivalProbabilities = parameters;
            BaselineSurvivalFunction = parameters;
        }
    }

    /// <summary>
    /// Creates a new instance of the model with specified parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a copy of the Kaplan-Meier estimator
    /// with the given survival probabilities. Useful for model cloning.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = new KaplanMeierEstimator<T>();
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <summary>
    /// Creates a new instance of the same type.
    /// </summary>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new KaplanMeierEstimator<T>();
    }

    #endregion
}
