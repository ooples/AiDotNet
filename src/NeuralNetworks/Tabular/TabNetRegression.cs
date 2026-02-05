using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// TabNet implementation for regression tasks.
/// </summary>
/// <remarks>
/// <para>
/// TabNetRegression extends the TabNet architecture for predicting continuous values.
/// It uses the same attention-based feature selection mechanism but with an output
/// layer suitable for regression.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use TabNetRegression when you want to predict a number
/// (continuous value) from tabular data.
///
/// Example use cases:
/// - Predicting house prices from features like square footage, bedrooms, location
/// - Forecasting sales based on historical data and market indicators
/// - Estimating customer lifetime value from demographic and behavioral data
///
/// Key features:
/// - **Automatic Feature Selection**: Learns which columns in your data matter most
/// - **Interpretability**: You can see exactly which features the model used
/// - **No Feature Engineering**: Often works well without manual feature preprocessing
/// - **Competitive Performance**: Matches or beats gradient boosting methods
///
/// Basic usage:
/// <code>
/// var options = new TabNetOptions&lt;double&gt;
/// {
///     NumDecisionSteps = 5,
///     FeatureDimension = 64
/// };
/// var model = new TabNetRegression&lt;double&gt;(inputFeatures: 10, options: options);
///
/// // Training
/// var prediction = model.Forward(inputBatch);
/// var loss = ComputeMSELoss(prediction, targets);
/// model.Backward(lossGradient);
/// model.UpdateParameters(learningRate);
///
/// // Get feature importance
/// var importance = model.GetFeatureImportance();
/// </code>
/// </para>
/// <para>
/// Reference: "TabNet: Attentive Interpretable Tabular Learning" (Arik &amp; Pfister, AAAI 2021)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TabNetRegression<T> : TabNetBase<T>
{
    /// <summary>
    /// Initializes a new instance of the TabNetRegression class for single-output regression.
    /// </summary>
    /// <param name="inputFeatures">Number of input features (columns in your data).</param>
    /// <param name="options">TabNet configuration options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a TabNet model that predicts a single value.
    ///
    /// For example, predicting house price:
    /// <code>
    /// // 10 features: bedrooms, bathrooms, sqft, etc.
    /// var model = new TabNetRegression&lt;double&gt;(10);
    /// </code>
    /// </para>
    /// </remarks>
    public TabNetRegression(int inputFeatures, TabNetOptions<T>? options = null)
        : base(inputFeatures, outputDim: 1, options)
    {
    }

    /// <summary>
    /// Initializes a new instance of the TabNetRegression class for multi-output regression.
    /// </summary>
    /// <param name="inputFeatures">Number of input features.</param>
    /// <param name="outputTargets">Number of values to predict.</param>
    /// <param name="options">TabNet configuration options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you need to predict multiple values at once.
    ///
    /// For example, predicting both price and days-on-market for a house:
    /// <code>
    /// // 10 input features, 2 outputs (price, days)
    /// var model = new TabNetRegression&lt;double&gt;(10, 2);
    /// </code>
    ///
    /// Multi-output regression can be more efficient than training separate models
    /// because the features are shared across all predictions.
    /// </para>
    /// </remarks>
    public TabNetRegression(int inputFeatures, int outputTargets, TabNetOptions<T>? options = null)
        : base(inputFeatures, outputTargets, options)
    {
    }

    /// <summary>
    /// Performs prediction for regression.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch_size, input_features].</param>
    /// <returns>Predictions of shape [batch_size, output_targets].</returns>
    /// <remarks>
    /// <para>
    /// For regression, the output is returned directly without any activation function.
    /// This allows the model to predict any real-valued number.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method takes your input data and returns predictions.
    ///
    /// Example:
    /// <code>
    /// // Input: batch of 32 samples, each with 10 features
    /// var input = new Tensor&lt;double&gt;(new[] { 32, 10 });
    /// // ... fill input with your data ...
    ///
    /// // Output: 32 predictions (one per sample)
    /// var predictions = model.Predict(input);
    /// </code>
    /// </para>
    /// </remarks>
    public Tensor<T> Predict(Tensor<T> input)
    {
        return Forward(input);
    }

    /// <summary>
    /// Computes the Mean Squared Error (MSE) loss.
    /// </summary>
    /// <param name="predictions">Model predictions.</param>
    /// <param name="targets">Ground truth values.</param>
    /// <returns>The MSE loss value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MSE measures how far off your predictions are from the true values.
    ///
    /// It works by:
    /// 1. Computing the difference between each prediction and target
    /// 2. Squaring each difference (so negative errors count the same as positive)
    /// 3. Averaging all the squared differences
    ///
    /// Lower MSE = better predictions. MSE = 0 means perfect predictions.
    ///
    /// MSE penalizes large errors more than small ones because of the squaring.
    /// </para>
    /// </remarks>
    public T ComputeMSELoss(Tensor<T> predictions, Tensor<T> targets)
    {
        var sumSquaredError = NumOps.Zero;
        int count = predictions.Length;

        for (int i = 0; i < count; i++)
        {
            var diff = NumOps.Subtract(predictions[i], targets[i]);
            var squaredDiff = NumOps.Multiply(diff, diff);
            sumSquaredError = NumOps.Add(sumSquaredError, squaredDiff);
        }

        return NumOps.Divide(sumSquaredError, NumOps.FromDouble(count));
    }

    /// <summary>
    /// Computes the gradient of MSE loss for backpropagation.
    /// </summary>
    /// <param name="predictions">Model predictions.</param>
    /// <param name="targets">Ground truth values.</param>
    /// <returns>Gradient tensor of shape [batch_size, output_targets].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes how to adjust the model's output to reduce the loss.
    ///
    /// The gradient of MSE is: 2 * (prediction - target) / n
    ///
    /// This tells the model:
    /// - If prediction > target: decrease the prediction (positive gradient)
    /// - If prediction &lt; target: increase the prediction (negative gradient)
    /// - The magnitude tells how big the adjustment should be
    /// </para>
    /// </remarks>
    public Tensor<T> ComputeMSEGradient(Tensor<T> predictions, Tensor<T> targets)
    {
        var gradient = new Tensor<T>(predictions.Shape);
        var scale = NumOps.FromDouble(2.0 / predictions.Length);

        for (int i = 0; i < predictions.Length; i++)
        {
            var diff = NumOps.Subtract(predictions[i], targets[i]);
            gradient[i] = NumOps.Multiply(diff, scale);
        }

        return gradient;
    }

    /// <summary>
    /// Computes the Mean Absolute Error (MAE) loss.
    /// </summary>
    /// <param name="predictions">Model predictions.</param>
    /// <param name="targets">Ground truth values.</param>
    /// <returns>The MAE loss value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MAE is another way to measure prediction error.
    ///
    /// Unlike MSE which squares errors, MAE uses absolute values:
    /// 1. Compute the difference between each prediction and target
    /// 2. Take the absolute value of each difference
    /// 3. Average all the absolute differences
    ///
    /// MAE is more robust to outliers than MSE because it doesn't square errors.
    /// If you have a few extreme values in your data, MAE might be a better choice.
    /// </para>
    /// </remarks>
    public T ComputeMAELoss(Tensor<T> predictions, Tensor<T> targets)
    {
        var sumAbsoluteError = NumOps.Zero;
        int count = predictions.Length;

        for (int i = 0; i < count; i++)
        {
            var diff = NumOps.Subtract(predictions[i], targets[i]);
            var absDiff = NumOps.Abs(diff);
            sumAbsoluteError = NumOps.Add(sumAbsoluteError, absDiff);
        }

        return NumOps.Divide(sumAbsoluteError, NumOps.FromDouble(count));
    }

    /// <summary>
    /// Computes the total loss including sparsity regularization.
    /// </summary>
    /// <param name="predictions">Model predictions.</param>
    /// <param name="targets">Ground truth values.</param>
    /// <returns>Total loss (MSE + sparsity regularization).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This combines the prediction error with a regularization term.
    ///
    /// The total loss has two parts:
    /// 1. **MSE**: How accurate are the predictions?
    /// 2. **Sparsity Loss**: How focused is the feature selection?
    ///
    /// The sparsity coefficient in TabNetOptions controls the balance between these.
    /// Higher sparsity coefficient = more pressure to use fewer features.
    /// </para>
    /// </remarks>
    public T ComputeTotalLoss(Tensor<T> predictions, Tensor<T> targets)
    {
        var mseLoss = ComputeMSELoss(predictions, targets);
        var sparsityLoss = ComputeSparsityLoss();
        return NumOps.Add(mseLoss, sparsityLoss);
    }

    /// <summary>
    /// Performs a single training step.
    /// </summary>
    /// <param name="input">Input batch.</param>
    /// <param name="targets">Target values.</param>
    /// <param name="learningRate">Learning rate for parameter updates.</param>
    /// <returns>The total loss for this batch.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a convenience method that does one complete training iteration:
    ///
    /// 1. Forward pass: compute predictions
    /// 2. Compute loss: how wrong are the predictions?
    /// 3. Backward pass: compute gradients
    /// 4. Update parameters: adjust the model based on gradients
    ///
    /// Call this repeatedly with different batches to train the model.
    /// </para>
    /// </remarks>
    public T TrainStep(Tensor<T> input, Tensor<T> targets, T learningRate)
    {
        // Forward pass
        var predictions = Forward(input);

        // Compute loss
        var loss = ComputeTotalLoss(predictions, targets);

        // Compute gradient
        var lossGradient = ComputeMSEGradient(predictions, targets);

        // Backward pass
        Backward(lossGradient);

        // Update parameters
        UpdateParameters(learningRate);

        return loss;
    }

    /// <summary>
    /// Computes the R² (coefficient of determination) score.
    /// </summary>
    /// <param name="predictions">Model predictions.</param>
    /// <param name="targets">Ground truth values.</param>
    /// <returns>R² score between -∞ and 1 (1 is perfect).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> R² tells you how much of the variation in the data your model explains.
    ///
    /// Interpretation:
    /// - R² = 1.0: Perfect predictions
    /// - R² = 0.0: Model predicts the mean for everything (no better than guessing)
    /// - R² &lt; 0: Model is worse than just predicting the mean
    ///
    /// For example, R² = 0.8 means your model explains 80% of the variation in the data.
    /// The remaining 20% is unexplained (noise, missing features, etc.).
    /// </para>
    /// </remarks>
    public T ComputeR2Score(Tensor<T> predictions, Tensor<T> targets)
    {
        // Compute mean of targets
        var targetMean = NumOps.Zero;
        for (int i = 0; i < targets.Length; i++)
        {
            targetMean = NumOps.Add(targetMean, targets[i]);
        }
        targetMean = NumOps.Divide(targetMean, NumOps.FromDouble(targets.Length));

        // Compute SS_res (residual sum of squares)
        var ssRes = NumOps.Zero;
        for (int i = 0; i < predictions.Length; i++)
        {
            var diff = NumOps.Subtract(predictions[i], targets[i]);
            ssRes = NumOps.Add(ssRes, NumOps.Multiply(diff, diff));
        }

        // Compute SS_tot (total sum of squares)
        var ssTot = NumOps.Zero;
        for (int i = 0; i < targets.Length; i++)
        {
            var diff = NumOps.Subtract(targets[i], targetMean);
            ssTot = NumOps.Add(ssTot, NumOps.Multiply(diff, diff));
        }

        // R² = 1 - SS_res / SS_tot
        var ratio = NumOps.Divide(ssRes, ssTot);
        return NumOps.Subtract(NumOps.One, ratio);
    }
}
