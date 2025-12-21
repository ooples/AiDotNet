namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the ARMA (AutoRegressive Moving Average) time series forecasting model.
/// </summary>
/// <typeparam name="T">The data type of the time series values.</typeparam>
/// <remarks>
/// <para>
/// ARMA is a statistical model used for analyzing and forecasting time series data. It combines two components:
/// AutoRegressive (AR) and Moving Average (MA) to model stationary time series data.
/// </para>
/// <para><b>For Beginners:</b> ARMA is a method for predicting future values in a time series (data collected 
/// over time, like daily stock prices or monthly sales figures). It works by combining two techniques: 
/// looking at how past values influence future ones (AutoRegressive) and accounting for the impact of past 
/// prediction errors (Moving Average). Think of it like predicting tomorrow's weather by considering both 
/// today's weather (AR component) and how accurate previous forecasts have been (MA component). Unlike ARIMA, 
/// ARMA assumes your data doesn't have strong upward or downward trends that need to be removed first.</para>
/// </remarks>
public class ARMAOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the order of the AutoRegressive (AR) component.
    /// </summary>
    /// <value>The AR order, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The AR order specifies how many previous time steps are used to predict the current value.
    /// For example, AROrder=1 means the model uses only the immediately preceding value.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how many past data points the model looks at to make predictions.
    /// With the default value of 1, the model only considers what happened in the previous time period. If you set it to 2,
    /// it would look at the last two time periods, and so on. Think of it like predicting tomorrow's sales based on
    /// today's sales (AROrder=1) versus considering both today and yesterday (AROrder=2). Higher values can capture more complex
    /// patterns but might make the model unnecessarily complicated for simple data.</para>
    /// </remarks>
    public int AROrder { get; set; } = 1;

    /// <summary>
    /// Gets or sets the order of the Moving Average (MA) component.
    /// </summary>
    /// <value>The MA order, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// The MA order specifies how many previous forecast errors are used in the model.
    /// For example, MAOrder=1 means the model incorporates the error from the previous prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how many past prediction errors the model considers.
    /// With the default value of 1, the model adjusts based on how wrong its most recent prediction was.
    /// Think of it like a weather forecaster who not only looks at yesterday's weather but also considers how wrong
    /// their previous forecast was, learning from their mistakes. If yesterday's prediction was too high, they might
    /// adjust today's prediction downward a bit to compensate.</para>
    /// </remarks>
    public int MAOrder { get; set; } = 1;

    /// <summary>
    /// Gets or sets the learning rate for the optimization algorithm.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.01.</value>
    /// <remarks>
    /// <para>
    /// The learning rate controls how quickly the model parameters are updated during training.
    /// A smaller value leads to slower but potentially more precise convergence.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the model adjusts its predictions based on errors.
    /// The default value of 0.01 means the model makes small, cautious adjustments. Think of it like turning a steering
    /// wheel - a small learning rate makes tiny adjustments (good for fine-tuning), while a larger value makes bigger
    /// adjustments (which might help learn faster but could overshoot the optimal solution). For most cases, the default
    /// small value works well because it helps the model find more accurate predictions, even if it takes a bit longer.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the maximum number of iterations for the optimization algorithm.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// This parameter limits how long the model will train before stopping, preventing excessive computation time.
    /// The algorithm will stop either when it converges or when it reaches this number of iterations.
    /// </para>
    /// <para><b>For Beginners:</b> This is simply a safety limit on how long the model will try to improve itself.
    /// The default value of 1000 means the model will make at most 1000 attempts to refine its predictions before stopping.
    /// Think of it like telling someone they can have up to 1000 tries to solve a puzzle - they might solve it sooner,
    /// but they won't keep trying forever if they're struggling to make progress.</para>
    /// </remarks>
    public int MaxIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the convergence tolerance for the optimization algorithm.
    /// </summary>
    /// <value>The convergence tolerance, defaulting to 0.000001 (1e-6).</value>
    /// <remarks>
    /// <para>
    /// The algorithm stops when the improvement between iterations is smaller than this tolerance value,
    /// indicating that the model has converged to a solution.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how precise the model needs to be before it stops training.
    /// The default value of 0.000001 is very small, meaning the model will keep improving until changes become extremely tiny.
    /// Think of it like painting a wall - you might stop when you can barely see any difference between coats of paint.
    /// A smaller tolerance means more precision but might take longer to achieve. For most applications, the default value
    /// provides a good balance between accuracy and training time.</para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-6;
}
