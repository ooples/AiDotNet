using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// FT-Transformer implementation for regression tasks.
/// </summary>
/// <remarks>
/// <para>
/// FTTransformerRegression applies the FT-Transformer architecture to regression problems.
/// It uses the [CLS] token output with a linear regression head to predict continuous values.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use this model when you want to predict continuous numbers
/// (like house prices, temperatures, stock returns, etc.).
///
/// How it works:
/// 1. Features are tokenized and processed by transformer layers
/// 2. The [CLS] token captures information from all features
/// 3. A linear layer maps the [CLS] representation to the output value(s)
///
/// Example:
/// <code>
/// // Create regressor for 10 features predicting 1 output
/// var options = new FTTransformerOptions&lt;double&gt; { EmbeddingDimension = 128, NumLayers = 2 };
/// var regressor = new FTTransformerRegression&lt;double&gt;(10, 1, options);
///
/// // Forward pass
/// var input = new Tensor&lt;double&gt;([32, 10]); // batch of 32 samples, 10 features
/// var predictions = regressor.Predict(input);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FTTransformerRegression<T> : FTTransformerBase<T>
{
    private readonly int _outputDimension;
    private readonly FullyConnectedLayer<T> _regressionHead;

    // Cache for backward pass
    private Tensor<T>? _clsOutputCache;
    private Tensor<T>? _predictionsCache;

    /// <summary>
    /// Gets the output dimension (number of target values to predict).
    /// </summary>
    public int OutputDimension => _outputDimension;

    /// <summary>
    /// Gets the total number of trainable parameters including the regression head.
    /// </summary>
    public override int ParameterCount => base.ParameterCount + _regressionHead.ParameterCount;

    /// <summary>
    /// Initializes a new instance of the FTTransformerRegression class.
    /// </summary>
    /// <param name="numNumericalFeatures">Number of numerical input features.</param>
    /// <param name="outputDimension">Number of output values to predict (default 1).</param>
    /// <param name="options">Model configuration options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creating a regressor:
    /// - numNumericalFeatures: How many number columns in your input data
    /// - outputDimension: How many values to predict (usually 1 for single-target regression)
    /// - options: Model configuration (see FTTransformerOptions for details)
    ///
    /// For multi-target regression, set outputDimension to the number of targets.
    /// </para>
    /// </remarks>
    public FTTransformerRegression(
        int numNumericalFeatures,
        int outputDimension = 1,
        FTTransformerOptions<T>? options = null)
        : base(numNumericalFeatures, options)
    {
        if (outputDimension < 1)
        {
            throw new ArgumentException("Output dimension must be at least 1", nameof(outputDimension));
        }

        _outputDimension = outputDimension;

        // Regression head: Linear layer from embedding dimension to output dimension
        _regressionHead = new FullyConnectedLayer<T>(
            Options.EmbeddingDimension,
            outputDimension,
            (IActivationFunction<T>?)null);  // No activation for regression
    }

    /// <summary>
    /// Performs the forward pass to get predictions.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <param name="categoricalIndices">Categorical feature indices [batch_size, num_categorical] or null.</param>
    /// <returns>Predicted values tensor [batch_size, output_dim].</returns>
    public Tensor<T> Forward(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        // Get [CLS] representation from backbone
        var clsOutput = ForwardBackbone(numericalFeatures, categoricalIndices);
        _clsOutputCache = clsOutput;

        // Apply regression head
        var predictions = _regressionHead.Forward(clsOutput);
        _predictionsCache = predictions;

        return predictions;
    }

    /// <summary>
    /// Performs the forward pass with numerical features only.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <returns>Predicted values tensor [batch_size, output_dim].</returns>
    public Tensor<T> Forward(Tensor<T> numericalFeatures)
    {
        return Forward(numericalFeatures, null);
    }

    /// <summary>
    /// Alias for Forward - makes predictions on the input data.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <param name="categoricalIndices">Categorical feature indices [batch_size, num_categorical] or null.</param>
    /// <returns>Predicted values tensor [batch_size, output_dim].</returns>
    public Tensor<T> Predict(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        return Forward(numericalFeatures, categoricalIndices);
    }

    /// <summary>
    /// Alias for Forward with numerical features only.
    /// </summary>
    public Tensor<T> Predict(Tensor<T> numericalFeatures)
    {
        return Forward(numericalFeatures, null);
    }

    /// <summary>
    /// Computes the Mean Squared Error (MSE) loss.
    /// </summary>
    /// <param name="predictions">Predicted values [batch_size, output_dim].</param>
    /// <param name="targets">Target values [batch_size, output_dim].</param>
    /// <returns>The average MSE loss.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MSE measures the average squared difference between predictions and targets:
    /// - Lower MSE = better predictions
    /// - MSE = 0 means perfect predictions
    /// - Penalizes large errors more than small ones
    ///
    /// Formula: MSE = (1/n) * Σ(prediction - target)²
    /// </para>
    /// </remarks>
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
            var squaredError = NumOps.Multiply(diff, diff);
            totalLoss = NumOps.Add(totalLoss, squaredError);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(predictions.Length));
    }

    /// <summary>
    /// Computes the Mean Absolute Error (MAE) loss.
    /// </summary>
    /// <param name="predictions">Predicted values [batch_size, output_dim].</param>
    /// <param name="targets">Target values [batch_size, output_dim].</param>
    /// <returns>The average MAE loss.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MAE measures the average absolute difference between predictions and targets:
    /// - Lower MAE = better predictions
    /// - MAE = 0 means perfect predictions
    /// - Less sensitive to outliers than MSE
    ///
    /// Formula: MAE = (1/n) * Σ|prediction - target|
    /// </para>
    /// </remarks>
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
            var absError = NumOps.Abs(diff);
            totalLoss = NumOps.Add(totalLoss, absError);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(predictions.Length));
    }

    /// <summary>
    /// Performs the backward pass for MSE loss.
    /// </summary>
    /// <param name="targets">Target values [batch_size, output_dim].</param>
    /// <returns>Gradient with respect to numerical input [batch_size, num_numerical].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The backward pass computes how to adjust the model weights
    /// to reduce the loss. For MSE, the gradient is:
    ///
    /// d(MSE)/d(prediction) = 2 * (prediction - target) / n
    /// </para>
    /// </remarks>
    public Tensor<T> Backward(Tensor<T> targets)
    {
        if (_predictionsCache == null || _clsOutputCache == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int batchSize = _predictionsCache.Shape[0];
        int outputDim = _predictionsCache.Shape.Length > 1 ? _predictionsCache.Shape[1] : 1;

        // Gradient of MSE: 2 * (predictions - targets) / n
        var predictionGrad = new Tensor<T>(_predictionsCache.Shape);
        var scale = NumOps.FromDouble(2.0 / (batchSize * outputDim));

        for (int i = 0; i < _predictionsCache.Length; i++)
        {
            var diff = NumOps.Subtract(_predictionsCache[i], targets[i]);
            predictionGrad[i] = NumOps.Multiply(diff, scale);
        }

        // Backward through regression head
        var clsGrad = _regressionHead.Backward(predictionGrad);

        // Backward through backbone
        return BackwardBackbone(clsGrad);
    }

    /// <summary>
    /// Performs a single training step using MSE loss.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <param name="targets">Target values [batch_size, output_dim].</param>
    /// <param name="learningRate">The learning rate.</param>
    /// <param name="categoricalIndices">Categorical feature indices [batch_size, num_categorical] or null.</param>
    /// <returns>The training MSE loss for this step.</returns>
    public T TrainStep(
        Tensor<T> numericalFeatures,
        Tensor<T> targets,
        T learningRate,
        Matrix<int>? categoricalIndices = null)
    {
        // Forward pass
        var predictions = Forward(numericalFeatures, categoricalIndices);

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
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <param name="targets">Target values [batch_size, output_dim].</param>
    /// <param name="categoricalIndices">Categorical feature indices [batch_size, num_categorical] or null.</param>
    /// <returns>The R² score (1.0 = perfect predictions, 0.0 = same as predicting mean).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> R² tells you how much of the variance in the target is explained by the model:
    /// - R² = 1.0: Perfect predictions
    /// - R² = 0.0: Model is as good as just predicting the mean
    /// - R² &lt; 0: Model is worse than predicting the mean
    ///
    /// Formula: R² = 1 - (SS_res / SS_tot)
    /// where SS_res = Σ(y - ŷ)² and SS_tot = Σ(y - ȳ)²
    /// </para>
    /// </remarks>
    public T ComputeR2Score(Tensor<T> numericalFeatures, Tensor<T> targets, Matrix<int>? categoricalIndices = null)
    {
        var predictions = Predict(numericalFeatures, categoricalIndices);

        // Compute target mean
        var targetMean = NumOps.Zero;
        for (int i = 0; i < targets.Length; i++)
        {
            targetMean = NumOps.Add(targetMean, targets[i]);
        }
        targetMean = NumOps.Divide(targetMean, NumOps.FromDouble(targets.Length));

        // Compute SS_res = Σ(y - ŷ)² and SS_tot = Σ(y - ȳ)²
        var ssRes = NumOps.Zero;
        var ssTot = NumOps.Zero;

        for (int i = 0; i < targets.Length; i++)
        {
            var residual = NumOps.Subtract(targets[i], predictions[i]);
            ssRes = NumOps.Add(ssRes, NumOps.Multiply(residual, residual));

            var deviation = NumOps.Subtract(targets[i], targetMean);
            ssTot = NumOps.Add(ssTot, NumOps.Multiply(deviation, deviation));
        }

        // R² = 1 - (SS_res / SS_tot)
        // Handle case where SS_tot is zero (all targets are the same)
        if (NumOps.Compare(ssTot, NumOps.Zero) == 0)
        {
            return NumOps.Compare(ssRes, NumOps.Zero) == 0 ? NumOps.One : NumOps.Zero;
        }

        return NumOps.Subtract(NumOps.One, NumOps.Divide(ssRes, ssTot));
    }

    /// <summary>
    /// Computes the Root Mean Squared Error (RMSE).
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <param name="targets">Target values [batch_size, output_dim].</param>
    /// <param name="categoricalIndices">Categorical feature indices [batch_size, num_categorical] or null.</param>
    /// <returns>The RMSE value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> RMSE is the square root of MSE:
    /// - Same units as the target variable
    /// - Easier to interpret than MSE
    /// - Lower RMSE = better predictions
    ///
    /// Formula: RMSE = √MSE
    /// </para>
    /// </remarks>
    public T ComputeRMSE(Tensor<T> numericalFeatures, Tensor<T> targets, Matrix<int>? categoricalIndices = null)
    {
        var predictions = Predict(numericalFeatures, categoricalIndices);
        var mse = ComputeMSELoss(predictions, targets);
        return NumOps.Sqrt(mse);
    }

    /// <summary>
    /// Computes the Mean Absolute Percentage Error (MAPE).
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <param name="targets">Target values [batch_size, output_dim]. Should not contain zeros.</param>
    /// <param name="categoricalIndices">Categorical feature indices [batch_size, num_categorical] or null.</param>
    /// <returns>The MAPE value (0.1 = 10% error).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MAPE expresses error as a percentage of the actual values:
    /// - MAPE = 0.1 means predictions are off by 10% on average
    /// - Useful when you care about relative errors
    /// - Warning: undefined when target = 0
    ///
    /// Formula: MAPE = (1/n) * Σ|prediction - target| / |target|
    /// </para>
    /// </remarks>
    public T ComputeMAPE(Tensor<T> numericalFeatures, Tensor<T> targets, Matrix<int>? categoricalIndices = null)
    {
        var predictions = Predict(numericalFeatures, categoricalIndices);
        var totalAPE = NumOps.Zero;
        int count = 0;
        var epsilon = NumOps.FromDouble(1e-10);  // Avoid division by zero

        for (int i = 0; i < targets.Length; i++)
        {
            var targetAbs = NumOps.Abs(targets[i]);
            // Skip if target is essentially zero
            if (NumOps.Compare(targetAbs, epsilon) > 0)
            {
                var absError = NumOps.Abs(NumOps.Subtract(predictions[i], targets[i]));
                var percentageError = NumOps.Divide(absError, targetAbs);
                totalAPE = NumOps.Add(totalAPE, percentageError);
                count++;
            }
        }

        if (count == 0)
        {
            return NumOps.Zero;  // All targets were zero
        }

        return NumOps.Divide(totalAPE, NumOps.FromDouble(count));
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
    /// Sets all parameters including the regression head.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        int baseCount = base.ParameterCount - _regressionHead.ParameterCount;
        var baseParams = new Vector<T>(baseCount);
        for (int i = 0; i < baseCount; i++)
        {
            baseParams[i] = parameters[i];
        }
        base.SetParameters(baseParams);

        int headCount = _regressionHead.ParameterCount;
        var headParams = new Vector<T>(headCount);
        for (int i = 0; i < headCount; i++)
        {
            headParams[i] = parameters[baseCount + i];
        }
        _regressionHead.SetParameters(headParams);
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public override void ResetState()
    {
        base.ResetState();
        _clsOutputCache = null;
        _predictionsCache = null;
        _regressionHead.ResetState();
    }
}
