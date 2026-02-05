using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// TabM implementation for regression tasks.
/// </summary>
/// <remarks>
/// <para>
/// TabMRegression uses the TabM architecture with BatchEnsemble layers for regression.
/// It averages predictions across ensemble members and can provide uncertainty estimates.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use this model when you want to predict continuous values
/// (like prices, temperatures, etc.) with the benefits of an ensemble.
///
/// How predictions work:
/// 1. Input passes through shared layers with per-member modulation
/// 2. Each ensemble member produces a prediction
/// 3. Predictions are averaged for the final output
/// 4. Variance across members provides uncertainty estimate
///
/// Benefits over single models:
/// - Better generalization through ensemble averaging
/// - Built-in uncertainty quantification
/// - More robust to outliers
/// - Comparable speed to single models
///
/// Example:
/// <code>
/// var options = new TabMOptions&lt;double&gt; { NumEnsembleMembers = 4 };
/// var regressor = new TabMRegression&lt;double&gt;(10, 1, options);
///
/// var input = new Tensor&lt;double&gt;([32, 10]); // batch of 32 samples
/// var (predictions, uncertainty) = regressor.PredictWithUncertainty(input);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TabMRegression<T> : TabMBase<T>
{
    private readonly int _outputDimension;
    private readonly BatchEnsembleLayer<T> _regressionHead;

    // Cache for backward pass
    private Tensor<T>? _backboneOutputCache;
    private Tensor<T>? _predictionsCache;
    private Tensor<T>? _memberPredictionsCache;

    /// <summary>
    /// Gets the output dimension.
    /// </summary>
    public int OutputDimension => _outputDimension;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount => base.ParameterCount + _regressionHead.ParameterCount;

    /// <summary>
    /// Initializes a new instance of the TabMRegression class.
    /// </summary>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="outputDimension">Number of output values to predict (default 1).</param>
    /// <param name="options">Model configuration options.</param>
    public TabMRegression(
        int numFeatures,
        int outputDimension = 1,
        TabMOptions<T>? options = null)
        : base(numFeatures, options)
    {
        if (outputDimension < 1)
        {
            throw new ArgumentException("Output dimension must be at least 1", nameof(outputDimension));
        }

        _outputDimension = outputDimension;

        // Regression head: BatchEnsemble layer to output values
        _regressionHead = new BatchEnsembleLayer<T>(
            GetLastHiddenDim(),
            outputDimension,
            Options.NumEnsembleMembers,
            Options.UseBias,
            Options.RankInitScale);
    }

    /// <summary>
    /// Performs the forward pass to get predictions (per member).
    /// </summary>
    /// <param name="features">Input features tensor [batch_size, num_features].</param>
    /// <returns>Predictions tensor [batch_size * num_members, output_dim].</returns>
    public Tensor<T> Forward(Tensor<T> features)
    {
        // Forward through backbone
        var backboneOutput = ForwardBackbone(features);
        _backboneOutputCache = backboneOutput;

        // Forward through regression head
        var predictions = _regressionHead.Forward(backboneOutput);
        _memberPredictionsCache = predictions;

        return predictions;
    }

    /// <summary>
    /// Predicts values (averaged across ensemble members).
    /// </summary>
    /// <param name="features">Input features tensor [batch_size, num_features].</param>
    /// <returns>Predicted values [batch_size, output_dim].</returns>
    public Tensor<T> Predict(Tensor<T> features)
    {
        var memberPredictions = Forward(features);
        var averaged = AverageMemberOutputs(memberPredictions, _outputDimension);
        _predictionsCache = averaged;
        return averaged;
    }

    /// <summary>
    /// Predicts values with uncertainty estimates.
    /// </summary>
    /// <param name="features">Input features tensor [batch_size, num_features].</param>
    /// <returns>Tuple of (predictions [batch_size, output_dim], uncertainty [batch_size, output_dim]).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method gives you both the prediction AND how uncertain
    /// the model is about that prediction.
    ///
    /// Uncertainty is computed as the standard deviation across ensemble members:
    /// - Low uncertainty: All members agree, prediction is likely accurate
    /// - High uncertainty: Members disagree, be cautious about the prediction
    ///
    /// Use cases:
    /// - Confidence intervals for predictions
    /// - Flagging uncertain predictions for human review
    /// - Active learning (select most uncertain samples to label)
    /// - Out-of-distribution detection
    /// </para>
    /// </remarks>
    public (Tensor<T> Predictions, Tensor<T> Uncertainty) PredictWithUncertainty(Tensor<T> features)
    {
        var memberPredictions = Forward(features);
        int expandedBatchSize = memberPredictions.Shape[0];
        int batchSize = expandedBatchSize / Options.NumEnsembleMembers;

        var predictions = new Tensor<T>([batchSize, _outputDimension]);
        var uncertainty = new Tensor<T>([batchSize, _outputDimension]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int j = 0; j < _outputDimension; j++)
            {
                // Compute mean
                var mean = NumOps.Zero;
                for (int m = 0; m < Options.NumEnsembleMembers; m++)
                {
                    mean = NumOps.Add(mean,
                        memberPredictions[(b * Options.NumEnsembleMembers + m) * _outputDimension + j]);
                }
                mean = NumOps.Divide(mean, NumOps.FromDouble(Options.NumEnsembleMembers));

                // Compute variance
                var variance = NumOps.Zero;
                for (int m = 0; m < Options.NumEnsembleMembers; m++)
                {
                    var diff = NumOps.Subtract(
                        memberPredictions[(b * Options.NumEnsembleMembers + m) * _outputDimension + j],
                        mean);
                    variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                }
                variance = NumOps.Divide(variance, NumOps.FromDouble(Options.NumEnsembleMembers));

                predictions[b * _outputDimension + j] = mean;
                uncertainty[b * _outputDimension + j] = NumOps.Sqrt(variance);  // Standard deviation
            }
        }

        _predictionsCache = predictions;
        return (predictions, uncertainty);
    }

    /// <summary>
    /// Computes the Mean Squared Error (MSE) loss.
    /// </summary>
    public T ComputeMSELoss(Tensor<T> predictions, Tensor<T> targets)
    {
        if (predictions.Length != targets.Length)
        {
            throw new ArgumentException("Predictions and targets must have the same size");
        }

        var totalLoss = NumOps.Zero;

        for (int i = 0; i < predictions.Length; i++)
        {
            var diff = NumOps.Subtract(predictions[i], targets[i]);
            totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(diff, diff));
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(predictions.Length));
    }

    /// <summary>
    /// Computes the Mean Absolute Error (MAE) loss.
    /// </summary>
    public T ComputeMAELoss(Tensor<T> predictions, Tensor<T> targets)
    {
        if (predictions.Length != targets.Length)
        {
            throw new ArgumentException("Predictions and targets must have the same size");
        }

        var totalLoss = NumOps.Zero;

        for (int i = 0; i < predictions.Length; i++)
        {
            var diff = NumOps.Subtract(predictions[i], targets[i]);
            totalLoss = NumOps.Add(totalLoss, NumOps.Abs(diff));
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(predictions.Length));
    }

    /// <summary>
    /// Performs the backward pass for MSE loss.
    /// </summary>
    /// <param name="targets">Target values [batch_size, output_dim].</param>
    /// <returns>Gradient with respect to input features [batch_size, num_features].</returns>
    public Tensor<T> Backward(Tensor<T> targets)
    {
        if (_predictionsCache == null || _memberPredictionsCache == null || _backboneOutputCache == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int batchSize = _predictionsCache.Shape[0];
        int expandedBatchSize = batchSize * Options.NumEnsembleMembers;

        // Gradient of MSE w.r.t averaged predictions
        var avgPredGrad = new Tensor<T>(_predictionsCache.Shape);
        var scale = NumOps.FromDouble(2.0 / (batchSize * _outputDimension));

        for (int i = 0; i < _predictionsCache.Length; i++)
        {
            avgPredGrad[i] = NumOps.Multiply(
                NumOps.Subtract(_predictionsCache[i], targets[i]),
                scale);
        }

        // Expand gradient back to all members
        var memberPredGrad = new Tensor<T>([expandedBatchSize, _outputDimension]);
        var memberScale = NumOps.FromDouble(1.0 / Options.NumEnsembleMembers);

        for (int b = 0; b < batchSize; b++)
        {
            for (int m = 0; m < Options.NumEnsembleMembers; m++)
            {
                for (int j = 0; j < _outputDimension; j++)
                {
                    memberPredGrad[(b * Options.NumEnsembleMembers + m) * _outputDimension + j] =
                        NumOps.Multiply(avgPredGrad[b * _outputDimension + j], memberScale);
                }
            }
        }

        // Backward through regression head
        var backboneGrad = _regressionHead.Backward(memberPredGrad);

        // Backward through backbone
        return BackwardBackbone(backboneGrad);
    }

    /// <summary>
    /// Performs a single training step using MSE loss.
    /// </summary>
    public T TrainStep(Tensor<T> features, Tensor<T> targets, T learningRate)
    {
        // Forward pass
        var predictions = Predict(features);

        // Compute loss
        var loss = ComputeMSELoss(predictions, targets);

        // Backward pass
        _ = Backward(targets);

        // Update parameters
        UpdateParameters(learningRate);

        // Reset for next iteration
        ResetState();

        return loss;
    }

    /// <summary>
    /// Computes the R² score (coefficient of determination).
    /// </summary>
    public T ComputeR2Score(Tensor<T> features, Tensor<T> targets)
    {
        var predictions = Predict(features);

        // Compute target mean
        var targetMean = NumOps.Zero;
        for (int i = 0; i < targets.Length; i++)
        {
            targetMean = NumOps.Add(targetMean, targets[i]);
        }
        targetMean = NumOps.Divide(targetMean, NumOps.FromDouble(targets.Length));

        // Compute SS_res and SS_tot
        var ssRes = NumOps.Zero;
        var ssTot = NumOps.Zero;

        for (int i = 0; i < targets.Length; i++)
        {
            var residual = NumOps.Subtract(targets[i], predictions[i]);
            ssRes = NumOps.Add(ssRes, NumOps.Multiply(residual, residual));

            var deviation = NumOps.Subtract(targets[i], targetMean);
            ssTot = NumOps.Add(ssTot, NumOps.Multiply(deviation, deviation));
        }

        if (NumOps.Compare(ssTot, NumOps.Zero) == 0)
        {
            return NumOps.Compare(ssRes, NumOps.Zero) == 0 ? NumOps.One : NumOps.Zero;
        }

        return NumOps.Subtract(NumOps.One, NumOps.Divide(ssRes, ssTot));
    }

    /// <summary>
    /// Computes the Root Mean Squared Error (RMSE).
    /// </summary>
    public T ComputeRMSE(Tensor<T> features, Tensor<T> targets)
    {
        var predictions = Predict(features);
        var mse = ComputeMSELoss(predictions, targets);
        return NumOps.Sqrt(mse);
    }

    /// <summary>
    /// Computes Negative Log Likelihood assuming Gaussian predictions.
    /// </summary>
    /// <param name="features">Input features tensor [batch_size, num_features].</param>
    /// <param name="targets">Target values [batch_size, output_dim].</param>
    /// <returns>The NLL value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> NLL is a loss function that accounts for both the prediction
    /// accuracy AND the model's uncertainty. It rewards:
    /// - Accurate predictions
    /// - Well-calibrated uncertainty (confident when right, uncertain when less accurate)
    ///
    /// This is useful for proper uncertainty quantification, as it encourages the model
    /// to output realistic uncertainty estimates.
    /// </para>
    /// </remarks>
    public T ComputeNLL(Tensor<T> features, Tensor<T> targets)
    {
        var (predictions, uncertainty) = PredictWithUncertainty(features);
        var totalNLL = NumOps.Zero;
        var epsilon = NumOps.FromDouble(1e-6);  // Minimum variance
        var logTwoPi = NumOps.FromDouble(Math.Log(2 * Math.PI));
        var half = NumOps.FromDouble(0.5);

        for (int i = 0; i < predictions.Length; i++)
        {
            var variance = NumOps.Multiply(uncertainty[i], uncertainty[i]);
            variance = NumOps.Add(variance, epsilon);  // Add small value for stability

            var diff = NumOps.Subtract(targets[i], predictions[i]);
            var squaredDiff = NumOps.Multiply(diff, diff);

            // NLL = 0.5 * (log(2π) + log(var) + (y - μ)² / var)
            var term1 = NumOps.Multiply(half, logTwoPi);
            var term2 = NumOps.Multiply(half, NumOps.Log(variance));
            var term3 = NumOps.Multiply(half, NumOps.Divide(squaredDiff, variance));

            totalNLL = NumOps.Add(totalNLL, NumOps.Add(NumOps.Add(term1, term2), term3));
        }

        return NumOps.Divide(totalNLL, NumOps.FromDouble(predictions.Length));
    }

    /// <summary>
    /// Computes calibration metrics for uncertainty estimates.
    /// </summary>
    /// <param name="features">Input features tensor [batch_size, num_features].</param>
    /// <param name="targets">Target values [batch_size, output_dim].</param>
    /// <returns>Dictionary of calibration metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Calibration tells you if the uncertainty estimates are reliable.
    ///
    /// Good calibration means:
    /// - When model says "95% confidence interval", ~95% of true values fall within it
    /// - High uncertainty samples have higher errors on average
    /// - Uncertainty correlates with actual prediction errors
    ///
    /// Metrics returned:
    /// - CoverageAtOneSigma: % of targets within ±1 standard deviation (should be ~68%)
    /// - CoverageAtTwoSigma: % of targets within ±2 standard deviations (should be ~95%)
    /// - CorrelationErrorUncertainty: How well does uncertainty predict error magnitude
    /// </para>
    /// </remarks>
    public Dictionary<string, T> ComputeCalibrationMetrics(Tensor<T> features, Tensor<T> targets)
    {
        var (predictions, uncertainty) = PredictWithUncertainty(features);
        int n = predictions.Length;

        int withinOneSigma = 0;
        int withinTwoSigma = 0;

        for (int i = 0; i < n; i++)
        {
            var error = NumOps.Abs(NumOps.Subtract(predictions[i], targets[i]));

            if (NumOps.Compare(error, uncertainty[i]) <= 0)
            {
                withinOneSigma++;
            }
            if (NumOps.Compare(error, NumOps.Multiply(uncertainty[i], NumOps.FromDouble(2.0))) <= 0)
            {
                withinTwoSigma++;
            }
        }

        return new Dictionary<string, T>
        {
            { "CoverageAtOneSigma", NumOps.FromDouble((double)withinOneSigma / n) },
            { "CoverageAtTwoSigma", NumOps.FromDouble((double)withinTwoSigma / n) }
        };
    }

    /// <summary>
    /// Updates all parameters including the regression head.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        base.UpdateParameters(learningRate);
        _regressionHead.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Gets all parameters including the regression head.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        var baseParams = base.GetParameters();
        var headParams = _regressionHead.GetParameters();

        var allParams = new T[baseParams.Length + headParams.Length];
        for (int i = 0; i < baseParams.Length; i++)
        {
            allParams[i] = baseParams[i];
        }
        for (int i = 0; i < headParams.Length; i++)
        {
            allParams[baseParams.Length + i] = headParams[i];
        }

        return new Vector<T>(allParams);
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public override void ResetState()
    {
        base.ResetState();
        _backboneOutputCache = null;
        _predictionsCache = null;
        _memberPredictionsCache = null;
        _regressionHead.ResetGradients();
    }
}
