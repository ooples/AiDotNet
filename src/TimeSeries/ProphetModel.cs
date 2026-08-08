using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Helpers;

namespace AiDotNet.TimeSeries;

/// <summary>
/// Represents a Prophet model for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The Prophet model is a procedure for forecasting time series data based on an additive model 
/// where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.
/// </para>
/// <para><b>For Beginners:</b> Think of the Prophet model as a smart crystal ball for predicting future values 
/// in a series of data points over time. It's like predicting weather, but for any kind of data that changes 
/// over time, such as sales, website traffic, or stock prices. The model looks at past patterns, including 
/// seasonal changes (like how sales might increase during holidays) and overall trends, to make educated 
/// guesses about future values.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Prophet-style model for time series with trend and seasonality
/// var options = new ProphetOptions&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;();
/// var prophet = new ProphetModel&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(options);
/// prophet.Train(dateFeatures, values);
/// Vector&lt;double&gt; forecast = prophet.Predict(futureDateFeatures);
/// </code>
/// </example>
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Forecasting at Scale", "https://doi.org/10.1080/00031305.2017.1380080", Year = 2018, Authors = "Sean J. Taylor, Benjamin Letham")]
public class ProphetModel<T, TInput, TOutput> : TimeSeriesModelBase<T>
{
    /// <summary>
    /// Stores the configuration options for the Prophet model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds all user-configurable settings for the model, including seasonal periods,
    /// holidays, changepoint settings, regressor configurations, and optimization parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the model's instruction manual. It contains
    /// all the settings that control how the model works, such as which seasonal patterns to look for,
    /// which holidays to consider, and how the model should learn from your data.
    /// </para>
    /// </remarks>
    private ProphetOptions<T, TInput, TOutput> _prophetOptions;

    /// <summary>
    /// Gets whether parameter optimization succeeded during the most recent training run.
    /// </summary>
    /// <remarks>
    /// When <see cref="ProphetOptions{T, TInput, TOutput}.OptimizeParameters"/> is disabled or optimization fails,
    /// this value is <see langword="false"/>.
    /// </remarks>
    public bool IsOptimized { get; private set; }

    /// <summary>
    /// Represents the overall trend component of the time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The trend component captures the long-term increase or decrease in the time series data.
    /// It's one of the fundamental components of the Prophet decomposition.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the main direction your data is heading in over time.
    /// For example, if you're tracking sales over many years, this captures whether sales are generally
    /// increasing, decreasing, or staying flat over the long term, ignoring seasonal ups and downs.
    /// </para>
    /// <para>
    /// The trend follows the paper's piecewise-linear growth model
    /// <c>g(t) = (k + a(t)·delta)·t + (m + a(t)·gamma)</c> with <c>gamma_j = -s_j·delta_j</c>, which is
    /// equivalent to the continuous hinge form <c>g(t) = m + k·t + Σ_j delta_j·max(0, t - s_j)</c>.
    /// </para>
    /// </remarks>
    private T _k;

    /// <summary>The trend offset (intercept) <c>m</c> in <c>g(t) = m + k·t + Σ_j delta_j·max(0, t - s_j)</c>.</summary>
    private T _m;

    /// <summary>
    /// The per-changepoint rate adjustments <c>delta</c>. Each entry is the change in slope applied from the
    /// corresponding changepoint time onward, giving the trend its piecewise-linear (time-varying) shape.
    /// </summary>
    private Vector<T> _delta;

    /// <summary>
    /// The changepoint locations <c>s_j</c> (in the same units as the time feature). These are fixed during
    /// fitting (placed uniformly over the first 80% of the training range, or taken from the options) and the
    /// magnitude of each trend change is learned into <see cref="_delta"/>.
    /// </summary>
    private Vector<T> _changepointTimes;

    /// <summary>
    /// The seasonal periods actually used by the fit (in time-feature units), captured at training time so
    /// prediction and serialization index <see cref="_seasonalComponents"/> in exactly the same order.
    /// </summary>
    private double[] _effectiveSeasonalPeriods;

    /// <summary>
    /// Stores the coefficients for all seasonal components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains coefficients for the Fourier series representation of all seasonal components.
    /// Each seasonal period (yearly, monthly, weekly, etc.) is represented by multiple sine and cosine terms.
    /// </para>
    /// <para><b>For Beginners:</b> This holds information about repeating patterns in your data.
    /// For example, ice cream sales might go up every summer and down every winter, or website traffic
    /// might be higher on weekends than weekdays. These coefficients help the model capture these
    /// regular patterns at different time scales (daily, weekly, yearly, etc.).
    /// </para>
    /// </remarks>
    private Vector<T> _seasonalComponents;

    /// <summary>
    /// Stores the effect of each holiday on the time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains coefficients representing how much each defined holiday affects the
    /// time series values. Each element corresponds to a holiday defined in ProphetOptions.Holidays.
    /// </para>
    /// <para><b>For Beginners:</b> This tracks how special days like holidays affect your data.
    /// For example, retail sales might spike before Christmas or drop on certain holidays.
    /// Each number in this list tells the model how much a specific holiday tends to
    /// increase or decrease the values in your data.
    /// </para>
    /// </remarks>
    private Vector<T> _holidayComponents;

    /// <summary>
    /// Stores the coefficients for additional regressor variables.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains coefficients for external regressor variables that may influence
    /// the time series but are not part of the core seasonal or trend components.
    /// </para>
    /// <para><b>For Beginners:</b> This holds information about how external factors affect your data.
    /// For example, if you're predicting ice cream sales, temperature might be a regressor because
    /// hot weather leads to more ice cream sales. These numbers tell the model exactly how much
    /// each external factor influences your data.
    /// </para>
    /// </remarks>
    private Vector<T> _regressors;

    /// <summary>
    /// The anomaly detection threshold computed during training.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This value determines when a prediction error is large enough
    /// to be considered an anomaly. It's computed from the training data residuals.
    /// </para>
    /// </remarks>
    private T _anomalyThreshold;

    /// <summary>
    /// The standard deviation of residuals computed during training.
    /// </summary>
    private T _residualStdDev;

    /// <summary>
    /// The mean of residuals computed during training.
    /// </summary>
    private T _residualMean;

    /// <summary>
    /// Initializes a new instance of the <see cref="ProphetModel{T}"/> class.
    /// </summary>
    /// <param name="options">The options for configuring the Prophet model. If null, default options are used.</param>
    /// <remarks>
    /// <para>
    /// This constructor sets up the Prophet model with the specified options or default options if none are provided.
    /// It initializes all the components of the model, including trend, seasonal components, holiday effects, 
    /// changepoints, and regressors.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting up your prediction tool. You can customize how it works 
    /// by providing options, or let it use default settings. The model prepares itself to handle different aspects 
    /// of your data, such as overall direction (trend), repeating patterns (seasonal components), special events 
    /// (holidays), sudden changes (changepoints), and other factors that might affect your predictions (regressors).
    /// </para>
    /// </remarks>
    public ProphetModel(ProphetOptions<T, TInput, TOutput>? options = null)
        : base(options ?? new ProphetOptions<T, TInput, TOutput>())
    {
        _prophetOptions = options ?? new ProphetOptions<T, TInput, TOutput>();

        // Initialize model components (piecewise-linear trend g(t) = m + k*t + sum_j delta_j*max(0, t - s_j))
        _m = NumOps.FromDouble(_prophetOptions.InitialTrendValue);
        _k = NumOps.Zero;
        _delta = new Vector<T>(0);
        _changepointTimes = new Vector<T>(0);
        _effectiveSeasonalPeriods = Array.Empty<double>();
        _seasonalComponents = new Vector<T>(0);
        _holidayComponents = new Vector<T>(_prophetOptions.Holidays.Count);
        _regressors = new Vector<T>(Math.Max(0, _prophetOptions.RegressorCount));

        // Initialize anomaly detection components
        _anomalyThreshold = NumOps.Zero;
        _residualStdDev = NumOps.Zero;
        _residualMean = NumOps.Zero;
    }

    /// <summary>
    /// Fits every component of the decomposable Prophet model to the training data by (ridge-regularized)
    /// least squares / MAP estimation.
    /// </summary>
    /// <param name="x">The input matrix. Column 0 is the time feature; columns 1.. are optional regressors.</param>
    /// <param name="y">The observed target values.</param>
    /// <remarks>
    /// <para>
    /// The additive Prophet model is <c>y(t) = g(t) + s(t) + h(t) + epsilon</c> (Taylor &amp; Letham 2018):
    /// </para>
    /// <list type="bullet">
    /// <item><description>
    /// <b>Trend g(t)</b> — piecewise-linear growth with changepoints:
    /// <c>g(t) = (k + a(t)·delta)·t + (m + a(t)·gamma)</c> with <c>gamma_j = -s_j·delta_j</c> (which keeps
    /// g continuous). This is equivalent to the hinge form <c>g(t) = m + k·t + Σ_j delta_j·max(0, t - s_j)</c>,
    /// which is linear in the parameters <c>[m, k, delta_1..delta_S]</c>.
    /// </description></item>
    /// <item><description>
    /// <b>Seasonality s(t)</b> — a Fourier series per period P:
    /// <c>Σ_{n=1}^{N} (a_n·cos(2πnt/P) + b_n·sin(2πnt/P))</c>, linear in the coefficients <c>a_n, b_n</c>.
    /// </description></item>
    /// <item><description><b>Holidays h(t)</b> and <b>extra regressors</b> — additive indicator/linear terms.</description></item>
    /// </list>
    /// <para>
    /// Because every component is linear in its parameters, the whole model is a single linear regression over
    /// the design matrix <c>[1, t, max(0,t-s_j) …, cos/sin … , holidays … , regressors …]</c>. We solve the
    /// ridge (Tikhonov) normal equations, leaving the offset <c>m</c> and base rate <c>k</c> unregularized (so
    /// the fit is exactly translation- and scale-equivariant in the targets) while shrinking the changepoint,
    /// seasonal, holiday and regressor coefficients toward zero — the least-squares analogue of Prophet's
    /// Laplace/Normal priors, governed by the corresponding <c>*PriorScale</c> options.
    /// </para>
    /// <para><b>For Beginners:</b> this is the step where the model actually learns from your data. It writes
    /// the forecast as a sum of simple building blocks (a sloped line that can bend at a few points, repeating
    /// seasonal waves, holiday bumps) and then solves one big "line of best fit" that picks the best strength
    /// for every block at once.</para>
    /// </remarks>
    private void FitLeastSquares(Matrix<T> x, Vector<T> y)
    {
        int n = y.Length;

        // --- Time range (used only to place changepoints) ---
        double tMin = double.MaxValue, tMax = double.MinValue;
        for (int i = 0; i < n; i++)
        {
            double ti = Convert.ToDouble(x[i, 0]);
            if (ti < tMin) tMin = ti;
            if (ti > tMax) tMax = ti;
        }
        if (!(tMax > tMin)) tMax = tMin + 1.0;

        // --- Fixed structure of the model (changepoint locations, seasonal periods) ---
        _changepointTimes = BuildChangepointTimes(n, tMin, tMax);
        int changepointCount = _changepointTimes.Length;

        _effectiveSeasonalPeriods = ComputeEffectiveSeasonalPeriods(n);
        int order = Math.Max(1, _prophetOptions.FourierOrder);
        int[] harmonicsPerPeriod = new int[_effectiveSeasonalPeriods.Length];
        int seasonalLen = 0;
        for (int pi = 0; pi < _effectiveSeasonalPeriods.Length; pi++)
        {
            harmonicsPerPeriod[pi] = Math.Min(order, Math.Max(1, (int)Math.Floor(_effectiveSeasonalPeriods[pi] / 2.0)));
            seasonalLen += 2 * harmonicsPerPeriod[pi];
        }

        int holidayCount = _prophetOptions.Holidays?.Count ?? 0;
        int regressorCount = Math.Max(0, _prophetOptions.RegressorCount);

        // --- Column layout: [m | k | delta(S) | seasonal | holidays | regressors] ---
        int p = 2 + changepointCount + seasonalLen + holidayCount + regressorCount;
        var design = new Matrix<T>(n, p);
        var ridge = new double[p];

        // Ridge weights: larger prior scale => more flexibility => less shrinkage (Prophet semantics).
        // m and k stay unregularized so translation/scaling equivariance in the targets is exact.
        double changepointRidge = 1.0 / Math.Max(1e-8, _prophetOptions.ChangePointPriorScale);
        double seasonalRidge = 1.0 / Math.Max(1e-8, _prophetOptions.SeasonalityPriorScale);
        double holidayRidge = 1.0 / Math.Max(1e-8, _prophetOptions.HolidayPriorScale);
        const double regressorRidge = 1e-6; // stabilizes a constant/collinear regressor column

        for (int i = 0; i < n; i++)
        {
            T ti = x[i, 0];
            int col = 0;

            // Offset m and base rate k (unregularized).
            design[i, col] = NumOps.One; ridge[col] = 0.0; col++;
            design[i, col] = ti; ridge[col] = 0.0; col++;

            // Changepoint hinges: max(0, t - s_j) — turns k*t into a piecewise-linear trend.
            for (int j = 0; j < changepointCount; j++)
            {
                T diff = NumOps.Subtract(ti, _changepointTimes[j]);
                design[i, col] = NumOps.GreaterThan(diff, NumOps.Zero) ? diff : NumOps.Zero;
                ridge[col] = changepointRidge;
                col++;
            }

            // Fourier seasonality: cos then sin for each harmonic of each period.
            for (int pi = 0; pi < _effectiveSeasonalPeriods.Length; pi++)
            {
                double period = _effectiveSeasonalPeriods[pi];
                for (int h = 1; h <= harmonicsPerPeriod[pi]; h++)
                {
                    T angle = NumOps.Multiply(NumOps.FromDouble(2.0 * Math.PI * h / period), ti);
                    design[i, col] = MathHelper.Cos(angle); ridge[col] = seasonalRidge; col++;
                    design[i, col] = MathHelper.Sin(angle); ridge[col] = seasonalRidge; col++;
                }
            }

            // Holiday indicators.
            for (int hc = 0; hc < holidayCount; hc++)
            {
                design[i, col] = IsHoliday(ti, hc) ? NumOps.One : NumOps.Zero;
                ridge[col] = holidayRidge;
                col++;
            }

            // Extra regressors (columns 1.. of the input).
            for (int r = 0; r < regressorCount; r++)
            {
                design[i, col] = (1 + r) < x.Columns ? x[i, 1 + r] : NumOps.Zero;
                ridge[col] = regressorRidge;
                col++;
            }
        }

        // Solve (DᵀD + Λ) β = Dᵀy.
        Matrix<T> normal = design.Transpose().Multiply(design);
        for (int d = 0; d < p; d++)
        {
            normal[d, d] = NumOps.Add(normal[d, d], NumOps.FromDouble(ridge[d]));
        }
        Vector<T> rhs = design.Transpose().Multiply(y);

        Vector<T> beta;
        try
        {
            // Cholesky is fast and stable for the (regularized, PD) normal matrix.
            beta = new CholeskyDecomposition<T>(normal).Solve(rhs);
        }
        catch (Exception)
        {
            // Fall back to SVD if the matrix is ill-conditioned.
            beta = new SvdDecomposition<T>(normal).Solve(rhs);
        }

        // Unpack the solution back into the model components.
        int idx = 0;
        _m = beta[idx++];
        _k = beta[idx++];

        _delta = new Vector<T>(changepointCount);
        for (int j = 0; j < changepointCount; j++) _delta[j] = beta[idx++];

        _seasonalComponents = new Vector<T>(seasonalLen);
        for (int s = 0; s < seasonalLen; s++) _seasonalComponents[s] = beta[idx++];

        _holidayComponents = new Vector<T>(holidayCount);
        for (int hc = 0; hc < holidayCount; hc++) _holidayComponents[hc] = beta[idx++];

        _regressors = new Vector<T>(regressorCount);
        for (int r = 0; r < regressorCount; r++) _regressors[r] = beta[idx++];
    }

    /// <summary>
    /// Builds the fixed changepoint locations <c>s_j</c>. Uses the user-supplied changepoints when provided,
    /// otherwise places up to 25 of them uniformly over the first 80% of the training time range (the Prophet
    /// default), so the model can bend its trend but not overfit the very end of the series.
    /// </summary>
    private Vector<T> BuildChangepointTimes(int n, double tMin, double tMax)
    {
        var configured = _prophetOptions.Changepoints;
        if (configured != null && configured.Count > 0)
        {
            var explicitCps = new Vector<T>(configured.Count);
            for (int i = 0; i < configured.Count; i++) explicitCps[i] = configured[i];
            return explicitCps;
        }

        int count = Math.Min(25, Math.Max(0, n - 2));
        if (count <= 0) return new Vector<T>(0);

        double span = 0.8 * (tMax - tMin);
        var times = new Vector<T>(count);
        for (int j = 1; j <= count; j++)
        {
            times[j - 1] = NumOps.FromDouble(tMin + span * ((double)j / count));
        }
        return times;
    }

    /// <summary>
    /// Determines which seasonal periods to model. Uses the explicit <see cref="ProphetOptions{T, TInput, TOutput}.SeasonalPeriods"/>
    /// when supplied; otherwise falls back to the enabled standard seasonalities (weekly/yearly/daily), keeping
    /// only periods the training window actually covers (at least two full cycles) so seasonality is identifiable.
    /// </summary>
    private double[] ComputeEffectiveSeasonalPeriods(int n)
    {
        var periods = new List<double>();
        var configured = _prophetOptions.SeasonalPeriods;
        if (configured != null && configured.Count > 0)
        {
            foreach (int period in configured)
            {
                if (period >= 2 && period <= n) periods.Add(period);
            }
        }
        else
        {
            if (_prophetOptions.WeeklySeasonality && 2 * 7 <= n) periods.Add(7.0);
            if (_prophetOptions.YearlySeasonality && 2 * 365.25 <= n) periods.Add(365.25);
            if (_prophetOptions.DailySeasonality && 2 * 24 <= n) periods.Add(24.0);
        }
        return periods.ToArray();
    }

    /// <summary>Returns whether the time value <paramref name="t"/> falls on the holiday at index <paramref name="holidayIndex"/>.</summary>
    private bool IsHoliday(T t, int holidayIndex)
    {
        try
        {
            DateTime date = DateTime.FromOADate(Convert.ToDouble(t));
            return date.Date == _prophetOptions.Holidays[holidayIndex].Date;
        }
        catch (Exception)
        {
            return false;
        }
    }

    /// <summary>
    /// Evaluates the piecewise-linear trend <c>g(t) = m + k·t + Σ_j delta_j·max(0, t - s_j)</c> at a single time.
    /// </summary>
    private T GetTrendComponent(T t)
    {
        T trend = NumOps.Add(_m, NumOps.Multiply(_k, t));
        int count = Math.Min(_delta.Length, _changepointTimes.Length);
        for (int j = 0; j < count; j++)
        {
            T diff = NumOps.Subtract(t, _changepointTimes[j]);
            if (NumOps.GreaterThan(diff, NumOps.Zero))
            {
                trend = NumOps.Add(trend, NumOps.Multiply(_delta[j], diff));
            }
        }
        return trend;
    }

    /// <summary>
    /// Optimizes the model parameters using the specified or default optimizer.
    /// </summary>
    /// <param name="x">The input matrix.</param>
    /// <param name="y">The output vector.</param>
    /// <remarks>
    /// <para>
    /// This method performs the main optimization of the model parameters. It initializes the parameters,
    /// prepares the optimization input data, runs the optimization process, and updates the model parameters
    /// with the optimized values.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the model really learns from the data. It starts with initial
    /// guesses for all its parameters, then uses an optimization algorithm to adjust these parameters to better
    /// fit the data. It's like fine-tuning all the knobs of the model to make its predictions as accurate as possible.
    /// </para>
    /// </remarks>
    private void OptimizeParameters(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int p = x.Columns;
        int regressorCount = Math.Max(0, _prophetOptions.RegressorCount);

        if (regressorCount > 0 && regressorCount > Math.Max(0, p - 1))
        {
            throw new ArgumentException(
                $"RegressorCount ({regressorCount}) exceeds available input columns ({p}). The input matrix must include time plus regressors.",
                nameof(x));
        }

        if (_regressors == null || _regressors.Length != regressorCount)
        {
            _regressors = new Vector<T>(regressorCount);
        }

        // Use the user-defined optimizer if provided, otherwise use LFGSOptimizer as default
        var optimizer = _prophetOptions.Optimizer ?? new LBFGSOptimizer<T, Matrix<T>, Vector<T>>(this);

        // Prepare the optimization input data
        var inputData = new OptimizationInputData<T, Matrix<T>, Vector<T>>()
        {
            XTrain = x,
            YTrain = y,
            XValidation = x,
            YValidation = y,
            XTest = x,
            YTest = y
        };

        // Run optimization
        var result = optimizer.Optimize(inputData);

        var optimizedParameters = (result.BestSolution as IParameterizable<T, Matrix<T>, Vector<T>>)?.GetParameters();
        if (optimizedParameters != null && optimizedParameters.Length > 0)
        {
            ApplyParameters(optimizedParameters);
        }
    }

    /// <summary>
    /// Predicts output values for the given input matrix.
    /// </summary>
    /// <param name="input">The input matrix to make predictions for.</param>
    /// <returns>A vector of predicted values.</returns>
    /// <remarks>
    /// <para>
    /// This method generates predictions for each row in the input matrix by calling PredictSingle for each row.
    /// </para>
    /// <para><b>For Beginners:</b> This method takes a bunch of input data and produces predictions for each piece
    /// of that data. It's like running many individual predictions all at once.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        int n = input.Rows;
        Vector<T> predictions = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            predictions[i] = PredictSingle(input.GetRow(i));
        }

        return predictions;
    }

    /// <summary>
    /// Predicts a single output value for the given input vector.
    /// </summary>
    /// <param name="x">The input vector to make a prediction for.</param>
    /// <returns>The predicted value.</returns>
    /// <remarks>
    /// <para>
    /// This method combines all components of the Prophet model (trend, seasonal, holiday, changepoint, and regressor effects)
    /// to produce a single prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes a prediction for a single point in time. It adds up all the
    /// different effects our model has learned (like overall trends, seasonal patterns, holiday effects, and so on)
    /// to come up with a final prediction.
    /// </para>
    /// </remarks>
    private T PredictSingleInternal(Vector<T> x)
    {
        // y(t) = g(t) + s(t) + h(t) + regressors. The trend g(t) already includes the changepoint
        // (piecewise-linear) effect, so there is no separate changepoint term to add here.
        T prediction = GetTrendComponent(x[0]);
        prediction = NumOps.Add(prediction, GetSeasonalComponent(x));
        prediction = NumOps.Add(prediction, GetHolidayComponent(x));
        prediction = NumOps.Add(prediction, GetRegressorEffect(x));

        return prediction;
    }

    /// <summary>
    /// Calculates the seasonal component of the time series for a given input vector.
    /// </summary>
    /// <param name="x">The input vector containing the time index.</param>
    /// <returns>The calculated seasonal component.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the seasonal effect using Fourier series for each seasonal period defined in the model options.
    /// It combines sine and cosine terms to represent cyclical patterns in the data.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how much the prediction should be adjusted based on seasonal patterns.
    /// For example, it might account for weekly or yearly cycles in your data. It uses mathematical functions (sine and cosine)
    /// to create smooth, repeating patterns that match the seasonality in your data.
    /// </para>
    /// </remarks>
    private T GetSeasonalComponent(Vector<T> x)
    {
        T seasonal = NumOps.Zero;
        if (_effectiveSeasonalPeriods == null || _seasonalComponents == null || _seasonalComponents.Length == 0)
        {
            return seasonal;
        }

        T t = x[0]; // The time value is the first element of x.
        int order = Math.Max(1, _prophetOptions.FourierOrder);
        int idx = 0;

        for (int pi = 0; pi < _effectiveSeasonalPeriods.Length; pi++)
        {
            double period = _effectiveSeasonalPeriods[pi];
            int harmonics = Math.Min(order, Math.Max(1, (int)Math.Floor(period / 2.0)));
            for (int h = 1; h <= harmonics; h++)
            {
                if (idx + 1 >= _seasonalComponents.Length) return seasonal;
                T angle = NumOps.Multiply(NumOps.FromDouble(2.0 * Math.PI * h / period), t);
                seasonal = NumOps.Add(seasonal, NumOps.Multiply(_seasonalComponents[idx], MathHelper.Cos(angle)));
                idx++;
                seasonal = NumOps.Add(seasonal, NumOps.Multiply(_seasonalComponents[idx], MathHelper.Sin(angle)));
                idx++;
            }
        }

        return seasonal;
    }

    /// <summary>
    /// Calculates the holiday component of the time series for a given input vector.
    /// </summary>
    /// <param name="x">The input vector containing the date information.</param>
    /// <returns>The calculated holiday component.</returns>
    /// <remarks>
    /// <para>
    /// This method checks if the current date is a holiday and returns the corresponding holiday effect.
    /// It assumes that only one holiday can occur on a given day.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the prediction based on whether the current date is a holiday.
    /// Holidays often have unique patterns that differ from regular days, so this helps the model account for those special days.
    /// </para>
    /// </remarks>
    private T GetHolidayComponent(Vector<T> x)
    {
        T _holidayComponent = NumOps.Zero;
        DateTime _currentDate = DateTime.FromOADate(Convert.ToDouble(x[0])); // Assume the date is the first element of x

        for (int _i = 0; _i < _prophetOptions.Holidays.Count; _i++)
        {
            if (_currentDate.Date == _prophetOptions.Holidays[_i].Date)
            {
                _holidayComponent = NumOps.Add(_holidayComponent, _holidayComponents[_i]);
                break; // Assume only one holiday per day
            }
        }

        return _holidayComponent;
    }

    /// <summary>
    /// Calculates the regressor effect of the time series for a given input vector.
    /// </summary>
    /// <param name="x">The input vector containing the regressor values.</param>
    /// <returns>The calculated regressor effect.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the combined effect of all regressors on the prediction.
    /// Regressors are external factors that can influence the time series.
    /// </para>
    /// <para><b>For Beginners:</b> Regressors are additional pieces of information that can help explain changes in your data.
    /// For example, if you're predicting ice cream sales, temperature could be a regressor because it affects how much ice cream people buy.
    /// This method calculates how all these extra factors combine to influence the prediction.
    /// </para>
    /// </remarks>
    private T GetRegressorEffect(Vector<T> x)
    {
        T _regressorEffect = NumOps.Zero;

        for (int _i = 0; _i < _regressors.Length; _i++)
        {
            // Assume regressors start from the second element of x
            _regressorEffect = NumOps.Add(_regressorEffect, NumOps.Multiply(_regressors[_i], x[_i + 1]));
        }

        return _regressorEffect;
    }

    /// <summary>
    /// Calculates the total size of the model's state vector.
    /// </summary>
    /// <returns>The size of the state vector.</returns>
    /// <remarks>
    /// <para>
    /// This method determines the total number of parameters in the model, including trend, seasonal components,
    /// holiday components, changepoint, and regressors.
    /// </para>
    /// <para><b>For Beginners:</b> This method counts how many different pieces of information the model is using to make predictions.
    /// It's like counting all the ingredients in a recipe. This count helps the model know how much information it needs to keep track of.
    /// </para>
    /// </remarks>
    private int GetStateSize()
    {
        // [m, k] + delta(changepoints) + seasonal + holiday + regressors
        return 2 + _delta.Length + _seasonalComponents.Length + _holidayComponents.Length + _regressors.Length;
    }

    /// <summary>
    /// Retrieves the current state of the model as a vector.
    /// </summary>
    /// <returns>A vector representing the current state of the model.</returns>
    /// <remarks>
    /// <para>
    /// This method collects all current parameter values into a single vector, including trend, seasonal components,
    /// holiday components, changepoint, and regressors.
    /// </para>
    /// <para><b>For Beginners:</b> This method gathers all the current settings of the model into one list.
    /// It's like taking a snapshot of the model at a specific moment. This is useful for saving the model's state
    /// or for certain types of calculations that need all the model's information at once.
    /// </para>
    /// </remarks>
    private Vector<T> GetCurrentState()
    {
        int _stateSize = GetStateSize();
        Vector<T> _currentState = new Vector<T>(_stateSize);
        int _index = 0;

        _currentState[_index++] = _m;
        _currentState[_index++] = _k;
        for (int _i = 0; _i < _delta.Length; _i++)
        {
            _currentState[_index++] = _delta[_i];
        }
        for (int _i = 0; _i < _seasonalComponents.Length; _i++)
        {
            _currentState[_index++] = _seasonalComponents[_i];
        }
        for (int _i = 0; _i < _holidayComponents.Length; _i++)
        {
            _currentState[_index++] = _holidayComponents[_i];
        }
        for (int _i = 0; _i < _regressors.Length; _i++)
        {
            _currentState[_index++] = _regressors[_i];
        }

        return _currentState;
    }

    /// <summary>
    /// Evaluates the performance of the Prophet model on a test dataset.
    /// </summary>
    /// <param name="xTest">The input matrix containing the test features.</param>
    /// <param name="yTest">The vector containing the true target values for the test set.</param>
    /// <returns>A dictionary containing various performance metrics.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates several common regression metrics to assess the model's performance:
    /// - Mean Absolute Error (MAE): Measures the average magnitude of errors in the predictions.
    /// - Mean Squared Error (MSE): Measures the average squared difference between predictions and actual values.
    /// - Root Mean Squared Error (RMSE): The square root of MSE, providing a metric in the same units as the target variable.
    /// - R-squared (R2): Represents the proportion of variance in the dependent variable that is predictable from the independent variable(s).
    /// </para>
    /// <para><b>For Beginners:</b> This method helps us understand how well our model is performing. 
    /// It compares the model's predictions to the actual values we know are correct, and calculates 
    /// several numbers that tell us how close our predictions are:
    /// - MAE: On average, how far off are our predictions? (in the same units as our data)
    /// - MSE: Similar to MAE, but punishes big mistakes more (in squared units)
    /// - RMSE: Like MSE, but back in the original units of our data
    /// - R2: How much of the changes in our data does our model explain? (from 0 to 1, where 1 is perfect)
    /// 
    /// These numbers help us decide if our model is good enough or if we need to improve it.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest)
    {
        Vector<T> predictions = Predict(xTest);
        Dictionary<string, T> metrics = new Dictionary<string, T>();

        // Mean Absolute Error (MAE)
        metrics["MAE"] = StatisticsHelper<T>.CalculateMeanAbsoluteError(yTest, predictions);

        // Mean Squared Error (MSE)
        metrics["MSE"] = StatisticsHelper<T>.CalculateMeanSquaredError(yTest, predictions);

        // Root Mean Squared Error (RMSE)
        metrics["RMSE"] = StatisticsHelper<T>.CalculateRootMeanSquaredError(yTest, predictions);

        // R-squared (R2)
        metrics["R2"] = StatisticsHelper<T>.CalculateR2(yTest, predictions);

        return metrics;
    }

    /// <summary>
    /// Serializes the core components of the Prophet model.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the serialized data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the model's parameters and options to a binary stream. It includes the trend,
    /// seasonal components, holiday components, changepoint, regressors, and various model options.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves all the important parts of our model to a file.
    /// It's like taking a snapshot of the model's current state, including all the patterns it has learned.
    /// This allows us to save our trained model and use it later without having to retrain it.
    /// </para>
    /// </remarks>
    protected override void SerializeCore(BinaryWriter writer)
    {
        // Piecewise-linear trend: offset m, base rate k, changepoint rate-adjustments (delta) and their fixed locations.
        writer.Write(Convert.ToDouble(_m));
        writer.Write(Convert.ToDouble(_k));
        writer.Write(_delta.Length);
        for (int i = 0; i < _delta.Length; i++)
        {
            writer.Write(Convert.ToDouble(_delta[i]));
        }
        writer.Write(_changepointTimes.Length);
        for (int i = 0; i < _changepointTimes.Length; i++)
        {
            writer.Write(Convert.ToDouble(_changepointTimes[i]));
        }

        // Seasonal periods actually used (so prediction indexes the Fourier coefficients identically) + Fourier order.
        writer.Write(_effectiveSeasonalPeriods.Length);
        for (int i = 0; i < _effectiveSeasonalPeriods.Length; i++)
        {
            writer.Write(_effectiveSeasonalPeriods[i]);
        }
        writer.Write(_prophetOptions.FourierOrder);

        // Seasonal Fourier coefficients.
        writer.Write(_seasonalComponents.Length);
        for (int i = 0; i < _seasonalComponents.Length; i++)
        {
            writer.Write(Convert.ToDouble(_seasonalComponents[i]));
        }

        // Holiday and regressor coefficients.
        writer.Write(_holidayComponents.Length);
        for (int i = 0; i < _holidayComponents.Length; i++)
        {
            writer.Write(Convert.ToDouble(_holidayComponents[i]));
        }
        writer.Write(_regressors.Length);
        for (int i = 0; i < _regressors.Length; i++)
        {
            writer.Write(Convert.ToDouble(_regressors[i]));
        }

        // Write options
        writer.Write(_prophetOptions.SeasonalPeriods.Count);
        foreach (var period in _prophetOptions.SeasonalPeriods)
        {
            writer.Write(period);
        }
        writer.Write(_prophetOptions.Holidays.Count);
        foreach (var holiday in _prophetOptions.Holidays)
        {
            writer.Write(holiday.Ticks);
        }
        writer.Write(_prophetOptions.RegressorCount);
    }

    /// <summary>
    /// Deserializes the core components of the Prophet model.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the serialized data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the model's parameters and options from a binary stream. It reconstructs
    /// the trend, seasonal components, holiday components, changepoint, regressors, and various model options.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved model from a file.
    /// It's like restoring a snapshot of the model's state, including all the patterns it had learned.
    /// This allows us to use a trained model without having to retrain it every time we want to use it.
    /// </para>
    /// </remarks>
    protected override void DeserializeCore(BinaryReader reader)
    {
        // Piecewise-linear trend.
        _m = NumOps.FromDouble(reader.ReadDouble());
        _k = NumOps.FromDouble(reader.ReadDouble());
        int deltaLength = reader.ReadInt32();
        _delta = new Vector<T>(deltaLength);
        for (int i = 0; i < deltaLength; i++)
        {
            _delta[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        int changepointLength = reader.ReadInt32();
        _changepointTimes = new Vector<T>(changepointLength);
        for (int i = 0; i < changepointLength; i++)
        {
            _changepointTimes[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Effective seasonal periods + Fourier order.
        int effectivePeriodCount = reader.ReadInt32();
        _effectiveSeasonalPeriods = new double[effectivePeriodCount];
        for (int i = 0; i < effectivePeriodCount; i++)
        {
            _effectiveSeasonalPeriods[i] = reader.ReadDouble();
        }
        int fourierOrder = reader.ReadInt32();

        // Seasonal Fourier coefficients.
        int seasonalLength = reader.ReadInt32();
        _seasonalComponents = new Vector<T>(seasonalLength);
        for (int i = 0; i < seasonalLength; i++)
        {
            _seasonalComponents[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Holiday and regressor coefficients.
        int holidayLength = reader.ReadInt32();
        _holidayComponents = new Vector<T>(holidayLength);
        for (int i = 0; i < holidayLength; i++)
        {
            _holidayComponents[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        int regressorLength = reader.ReadInt32();
        _regressors = new Vector<T>(regressorLength);
        for (int i = 0; i < regressorLength; i++)
        {
            _regressors[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Read options
        _prophetOptions = new ProphetOptions<T, TInput, TOutput>
        {
            FourierOrder = fourierOrder
        };
        int seasonalPeriodsCount = reader.ReadInt32();
        for (int i = 0; i < seasonalPeriodsCount; i++)
        {
            _prophetOptions.SeasonalPeriods.Add(reader.ReadInt32());
        }
        int holidaysCount = reader.ReadInt32();
        for (int i = 0; i < holidaysCount; i++)
        {
            _prophetOptions.Holidays.Add(new DateTime(reader.ReadInt64()));
        }
        _prophetOptions.RegressorCount = reader.ReadInt32();
    }

    /// <summary>
    /// Core implementation of the training logic for the Prophet model.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target vector containing the values to be predicted.</param>
    /// <remarks>
    /// <para>
    /// This protected method contains the actual implementation of the training process.
    /// It's called by the public Train method and handles the core training logic.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the real learning happens for the model.
    /// The method carefully validates your data, then initializes the model components
    /// (like trend and seasonal patterns), and finally optimizes all the parameters to
    /// best fit your data. If anything goes wrong during this process, it provides clear
    /// error messages to help you understand and fix the issue.
    /// </para>
    /// <para>
    /// If <see cref="ProphetOptions{T, TInput, TOutput}.OptimizeParameters"/> is enabled, the model will attempt
    /// parameter optimization. If optimization fails, training continues using the initial parameter estimates,
    /// a warning is emitted via <see cref="System.Diagnostics.Trace"/>, and <see cref="IsOptimized"/> remains
    /// <see langword="false"/>.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Matrix<T> x, Vector<T> y)
    {
        // Validate inputs
        if (x == null)
        {
            throw new ArgumentNullException(nameof(x), "Input matrix cannot be null");
        }

        if (y == null)
        {
            throw new ArgumentNullException(nameof(y), "Target vector cannot be null");
        }

        if (x.Rows != y.Length)
        {
            throw new ArgumentException($"Input matrix rows ({x.Rows}) must match target vector length ({y.Length})");
        }

        if (x.Rows == 0)
        {
            throw new ArgumentException("Cannot train on empty dataset");
        }

        // Fit every component (piecewise-linear trend, Fourier seasonality, holidays, regressors)
        // to the (t, y) training pairs by ridge-regularized least squares / MAP estimation.
        FitLeastSquares(x, y);

        int n = y.Length;
        Matrix<T> states = new Matrix<T>(n, GetStateSize());

        SyncModelParametersFromState();

        IsOptimized = false;
        if (_prophetOptions.OptimizeParameters)
        {
            try
            {
                OptimizeParameters(x, y);
                IsOptimized = true;
            }
            catch (InvalidOperationException ex)
            {
                System.Diagnostics.Trace.TraceWarning($"[ProphetModel] Parameter optimization failed; using initial estimates. {ex}");
            }
            catch (ArgumentException ex)
            {
                System.Diagnostics.Trace.TraceWarning($"[ProphetModel] Parameter optimization failed; using initial estimates. {ex}");
            }
            catch (ArithmeticException ex)
            {
                System.Diagnostics.Trace.TraceWarning($"[ProphetModel] Parameter optimization failed; using initial estimates. {ex}");
            }
        }

        // Store final state for future reference
        states.SetRow(n - 1, GetCurrentState());

        // Compute residual statistics if anomaly detection or prediction intervals are enabled
        // _residualStdDev is needed for both features
        if (_prophetOptions.EnableAnomalyDetection || _prophetOptions.ComputePredictionIntervals)
        {
            ComputeAnomalyThresholdFromTraining(x, y);
        }
    }

    /// <summary>
    /// Computes the anomaly detection threshold from training data residuals.
    /// </summary>
    /// <param name="x">The training input matrix.</param>
    /// <param name="y">The training target vector.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates what counts as "normal" variation in your data
    /// by looking at how far off the model's predictions were during training. The threshold is computed
    /// so that only unusually large prediction errors get flagged as anomalies.
    /// </para>
    /// </remarks>
    private void ComputeAnomalyThresholdFromTraining(Matrix<T> x, Vector<T> y)
    {
        // Compute predictions for training data
        Vector<T> predictions = Predict(x);

        // Compute residuals (absolute errors)
        List<T> absResiduals = new List<T>();
        for (int i = 0; i < y.Length; i++)
        {
            T residual = NumOps.Abs(NumOps.Subtract(y[i], predictions[i]));
            absResiduals.Add(residual);
        }

        if (absResiduals.Count == 0)
        {
            _residualMean = NumOps.Zero;
            _residualStdDev = NumOps.One;
            _anomalyThreshold = NumOps.FromDouble(_prophetOptions.AnomalyThresholdSigma);
            return;
        }

        // Compute mean of absolute residuals
        T sum = NumOps.Zero;
        for (int i = 0; i < absResiduals.Count; i++)
        {
            sum = NumOps.Add(sum, absResiduals[i]);
        }
        _residualMean = NumOps.Divide(sum, NumOps.FromDouble(absResiduals.Count));

        // Compute standard deviation of absolute residuals
        T sumSquaredDiff = NumOps.Zero;
        for (int i = 0; i < absResiduals.Count; i++)
        {
            T diff = NumOps.Subtract(absResiduals[i], _residualMean);
            sumSquaredDiff = NumOps.Add(sumSquaredDiff, NumOps.Multiply(diff, diff));
        }
        _residualStdDev = NumOps.Sqrt(NumOps.Divide(sumSquaredDiff, NumOps.FromDouble(absResiduals.Count)));

        // If std dev is zero (all residuals are the same), use a small default
        if (NumOps.Equals(_residualStdDev, NumOps.Zero))
        {
            _residualStdDev = NumOps.FromDouble(0.001);
        }

        // Threshold = mean + (sigma * stddev)
        _anomalyThreshold = NumOps.Add(
            _residualMean,
            NumOps.Multiply(NumOps.FromDouble(_prophetOptions.AnomalyThresholdSigma), _residualStdDev)
        );
    }

    private void SyncModelParametersFromState()
    {
        base.ApplyParameters(GetCurrentState());
    }

    protected override void ApplyParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters), "Parameters vector cannot be null.");
        }

        int expectedLength = GetStateSize();
        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException($"Expected {expectedLength} parameters, but got {parameters.Length}.", nameof(parameters));
        }

        // Layout must match GetCurrentState: [m, k, delta(changepoints), seasonal, holiday, regressors].
        int index = 0;
        _m = parameters[index++];
        _k = parameters[index++];

        for (int i = 0; i < _delta.Length; i++)
        {
            _delta[i] = parameters[index++];
        }

        for (int i = 0; i < _seasonalComponents.Length; i++)
        {
            _seasonalComponents[i] = parameters[index++];
        }

        for (int i = 0; i < _holidayComponents.Length; i++)
        {
            _holidayComponents[i] = parameters[index++];
        }

        for (int i = 0; i < _regressors.Length; i++)
        {
            _regressors[i] = parameters[index++];
        }

        base.ApplyParameters(parameters);
    }

    public override void SetParameters(Vector<T> parameters)
    {
        ApplyParameters(parameters);
    }

    /// <summary>
    /// Predicts a single value based on the input vector.
    /// </summary>
    /// <param name="input">The input vector containing time and other regressor values.</param>
    /// <returns>The predicted value for the given input.</returns>
    /// <remarks>
    /// <para>
    /// This method generates a prediction for a single input vector by combining the effects
    /// of trend, seasonality, holidays, changepoints, and regressors as learned during training.
    /// </para>
    /// <para><b>For Beginners:</b> This is the method that makes a prediction for a single point in time.
    /// It takes all the patterns the model has learned (trends, seasons, holiday effects, etc.)
    /// and combines them to predict what value we should expect at the given time point.
    /// 
    /// The input vector should contain:
    /// - Time information (typically the first element)
    /// - Any additional regressor values that the model was trained with
    /// </para>
    /// </remarks>
    public override T PredictSingle(Vector<T> input)
    {
        // Validate the input
        if (input == null)
        {
            throw new ArgumentNullException(nameof(input), "Input vector cannot be null");
        }

        // Check if model has been trained
        if (!IsTrained)
        {
            throw new InvalidOperationException("Model must be trained before making predictions");
        }

        // Check if input has the right dimensions
        int expectedLength = 1 + _prophetOptions.RegressorCount; // Time plus regressors
        if (input.Length < expectedLength)
        {
            throw new ArgumentException($"Input vector length ({input.Length}) is too short. Expected at least {expectedLength} elements.");
        }

        // Use the private implementation to make the actual prediction
        T prediction = PredictSingleInternal(input);

        // Apply any post-processing logic if needed
        if (_prophetOptions.ApplyTransformation)
        {
            prediction = _prophetOptions.TransformPrediction(prediction);
        }

        return prediction;
    }

    /// <summary>
    /// Gets metadata about the model, including its type, parameters, and configuration.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method provides detailed information about the model's configuration, learned parameters,
    /// and operational statistics. It's useful for model versioning, comparison, and documentation.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a detailed report card about the model.
    /// It includes information about how the model was set up and what it learned during training.
    /// This is useful for:
    /// - Keeping track of different models you've created
    /// - Comparing models to see which one works better
    /// - Documenting what your model does for other people to understand
    /// - Saving the model's configuration for future reference
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            AdditionalInfo = []
        };

        // Add basic information
        metadata.AdditionalInfo["ModelName"] = "Prophet Time Series Model";
        metadata.AdditionalInfo["Version"] = "1.0";

        // Add trend information (piecewise-linear: offset m, base rate k)
        metadata.AdditionalInfo["TrendOffset"] = Convert.ToDouble(_m);
        metadata.AdditionalInfo["TrendRate"] = Convert.ToDouble(_k);

        // Add seasonal information
        metadata.AdditionalInfo["SeasonalPeriodCount"] = _prophetOptions.SeasonalPeriods.Count;
        for (int i = 0; i < _prophetOptions.SeasonalPeriods.Count; i++)
        {
            metadata.AdditionalInfo[$"SeasonalPeriod_{i}"] = _prophetOptions.SeasonalPeriods[i];
        }
        metadata.AdditionalInfo["FourierOrder"] = _prophetOptions.FourierOrder;
        metadata.AdditionalInfo["SeasonalComponentsCount"] = _seasonalComponents.Length;

        // Add holiday information
        metadata.AdditionalInfo["HolidayCount"] = _prophetOptions.Holidays.Count;
        metadata.AdditionalInfo["HolidayComponentsCount"] = _holidayComponents.Length;

        // Add changepoint information (number of trend rate-adjustments fitted)
        metadata.AdditionalInfo["ChangepointCount"] = _delta.Length;

        // Add regressor information
        metadata.AdditionalInfo["RegressorCount"] = _prophetOptions.RegressorCount;
        if (_prophetOptions.RegressorCount > 0 && _regressors != null)
        {
            for (int i = 0; i < _prophetOptions.RegressorCount && i < _regressors.Length; i++)
            {
                metadata.AdditionalInfo[$"RegressorCoefficient_{i}"] = Convert.ToDouble(_regressors[i]);
            }
        }

        // Include optimizer information if available
        if (_prophetOptions.Optimizer != null)
        {
            metadata.AdditionalInfo["OptimizerType"] = _prophetOptions.Optimizer.GetType().Name;
        }

        // Add model state size
        metadata.AdditionalInfo["StateSize"] = GetStateSize();

        // Include serialized model data
        metadata.ModelData = SerializeForMetadata();

        return metadata;
    }

    /// <summary>
    /// Creates a new instance of the Prophet model with the same options.
    /// </summary>
    /// <returns>A new instance of the Prophet model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a fresh copy of the model with the same configuration options
    /// but without any trained parameters. It's used for creating new models with similar configurations.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes a fresh copy of the model with the same settings,
    /// but without any of the learned information. It's like creating a new recipe book with the same 
    /// instructions but no completed recipes yet. This is useful when you want to:
    /// - Train the same type of model on different data
    /// - Create multiple similar models for comparison
    /// - Start over with the same configuration but different training
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateInstance()
    {
        // Create a deep copy of options to avoid reference issues
        var newOptions = new ProphetOptions<T, TInput, TOutput>
        {
            // Basic settings
            InitialTrendValue = _prophetOptions.InitialTrendValue,
            InitialChangepointValue = _prophetOptions.InitialChangepointValue,
            FourierOrder = _prophetOptions.FourierOrder,
            RegressorCount = _prophetOptions.RegressorCount,
            ApplyTransformation = _prophetOptions.ApplyTransformation,

            // Copy the optimizer if possible
            Optimizer = _prophetOptions.Optimizer
        };

        // Deep copy seasonal periods
        newOptions.SeasonalPeriods = [.. _prophetOptions.SeasonalPeriods];

        // Deep copy holidays
        newOptions.Holidays = [.. _prophetOptions.Holidays];

        // Deep copy changepoints
        newOptions.Changepoints = [.. _prophetOptions.Changepoints];

        // Create a new instance with the copied options
        return new ProphetModel<T, TInput, TOutput>(newOptions);
    }

    /// <summary>
    /// Computes the average seasonal effect for JIT approximation.
    /// </summary>
    private T ComputeAverageSeasonalEffect()
    {
        T avgEffect = NumOps.Zero;
        int fourierTerms = _prophetOptions.FourierOrder * 2;

        // Compute average over all Fourier terms
        for (int j = 0; j < Math.Min(fourierTerms, _seasonalComponents.Length); j++)
        {
            // Average contribution of sin/cos terms is approximately 0.5 * coefficient
            avgEffect = NumOps.Add(avgEffect, NumOps.Multiply(_seasonalComponents[j], NumOps.FromDouble(0.5)));
        }

        return avgEffect;
    }

    /// <summary>
    /// Computes the average holiday effect for JIT approximation.
    /// </summary>
    private T ComputeAverageHolidayEffect()
    {
        if (_holidayComponents == null || _holidayComponents.Length == 0)
            return NumOps.Zero;

        T sum = NumOps.Zero;
        for (int i = 0; i < _holidayComponents.Length; i++)
        {
            sum = NumOps.Add(sum, _holidayComponents[i]);
        }

        // Average holiday effect weighted by probability of holiday
        // Assumes holidays are relatively rare (approx 10-15 days per year)
        T holidayProbability = NumOps.FromDouble(15.0 / 365.0);
        return NumOps.Multiply(NumOps.Divide(sum, NumOps.FromDouble(_holidayComponents.Length)), holidayProbability);
    }

    /// <summary>
    /// Computes the average changepoint effect for JIT approximation.
    /// </summary>
    private T ComputeAverageChangepointEffect()
    {
        // For JIT, approximate the cumulative changepoint (piecewise-linear) effect at an average time
        // as half the sum of the fitted per-changepoint rate adjustments.
        if (_delta == null || _delta.Length == 0)
            return NumOps.Zero;

        T sum = NumOps.Zero;
        for (int i = 0; i < _delta.Length; i++)
        {
            sum = NumOps.Add(sum, _delta[i]);
        }

        return NumOps.Multiply(sum, NumOps.FromDouble(0.5));
    }

    /// <summary>
    /// Detects anomalies in a time series by comparing predictions to actual values.
    /// </summary>
    /// <param name="x">The input matrix containing time indices and any regressors.</param>
    /// <param name="y">The actual time series values.</param>
    /// <returns>A boolean array where true indicates an anomaly at that position.</returns>
    /// <remarks>
    /// <para>
    /// This method uses the Prophet model to predict each point in the time series based on
    /// trend, seasonality, holidays, and regressors, then flags points where the prediction error
    /// exceeds the anomaly threshold computed during training.
    /// </para>
    /// <para><b>For Beginners:</b> This method identifies unusual data points by comparing
    /// what actually happened to what the model predicted should happen, accounting for trends,
    /// seasons, and holidays.
    ///
    /// Prophet is particularly good at detecting contextual anomalies - values that are unusual
    /// given the current context. For example:
    /// - A high sales value in July that would be normal in December might be flagged
    /// - A normal weekday value occurring on a holiday might be flagged
    ///
    /// The result tells you which points are unusual enough to warrant investigation.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the model hasn't been trained or anomaly detection wasn't enabled.
    /// </exception>
    public bool[] DetectAnomalies(Matrix<T> x, Vector<T> y)
    {
        if (!IsTrained)
        {
            throw new InvalidOperationException("Model must be trained before detecting anomalies.");
        }

        if (!_prophetOptions.EnableAnomalyDetection)
        {
            throw new InvalidOperationException("Anomaly detection was not enabled during training. " +
                "Set EnableAnomalyDetection = true in ProphetOptions and retrain the model.");
        }

        var scores = ComputeAnomalyScores(x, y);
        bool[] anomalies = new bool[scores.Length];

        for (int i = 0; i < scores.Length; i++)
        {
            anomalies[i] = NumOps.GreaterThan(scores[i], _anomalyThreshold);
        }

        return anomalies;
    }

    /// <summary>
    /// Computes anomaly scores for each point in a time series.
    /// </summary>
    /// <param name="x">The input matrix containing time indices and any regressors.</param>
    /// <param name="y">The actual time series values.</param>
    /// <returns>A vector of anomaly scores (absolute prediction errors) for each point.</returns>
    /// <remarks>
    /// <para>
    /// The anomaly score is the absolute difference between the actual value and the predicted value.
    /// The predicted value accounts for trend, seasonality, holidays, and regressor effects,
    /// so the score represents how much the actual value deviates from the contextual expectation.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you exactly how unusual each point is,
    /// considering all the patterns Prophet has learned:
    /// - A score of 0 means the value matched the prediction perfectly
    /// - Higher scores mean the value was more unexpected
    ///
    /// Unlike simple anomaly detection that just looks at absolute values, Prophet considers:
    /// - Is this value unusual for this time of year?
    /// - Is this value unusual for this day of the week?
    /// - Is this value unusual given the overall trend?
    /// - Is this value unusual for a holiday/non-holiday?
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when the model hasn't been trained.</exception>
    public Vector<T> ComputeAnomalyScores(Matrix<T> x, Vector<T> y)
    {
        if (!IsTrained)
        {
            throw new InvalidOperationException("Model must be trained before computing anomaly scores.");
        }

        if (x.Rows != y.Length)
        {
            throw new ArgumentException($"Input rows ({x.Rows}) must match target length ({y.Length}).");
        }

        // Get predictions for all points
        Vector<T> predictions = Predict(x);
        Vector<T> scores = new Vector<T>(y.Length);

        for (int i = 0; i < y.Length; i++)
        {
            scores[i] = NumOps.Abs(NumOps.Subtract(y[i], predictions[i]));
        }

        return scores;
    }

    /// <summary>
    /// Detects anomalies and returns detailed information about each detected anomaly.
    /// </summary>
    /// <param name="x">The input matrix containing time indices and any regressors.</param>
    /// <param name="y">The actual time series values.</param>
    /// <returns>A list of tuples containing (index, actual value, predicted value, score) for each anomaly.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method not only tells you which points are anomalies,
    /// but also provides context to help you understand why:
    /// - Index: The position of the anomaly in the time series
    /// - Actual: What the value actually was
    /// - Predicted: What Prophet expected based on trends, seasons, and holidays
    /// - Score: How far off the prediction was
    ///
    /// The difference between Actual and Predicted tells you the direction of the anomaly:
    /// - Actual > Predicted: Unexpectedly high value
    /// - Actual < Predicted: Unexpectedly low value
    /// </para>
    /// </remarks>
    public List<(int Index, T Actual, T Predicted, T Score)> DetectAnomaliesDetailed(Matrix<T> x, Vector<T> y)
    {
        if (!IsTrained)
        {
            throw new InvalidOperationException("Model must be trained before detecting anomalies.");
        }

        if (!_prophetOptions.EnableAnomalyDetection)
        {
            throw new InvalidOperationException("Anomaly detection was not enabled during training.");
        }

        Vector<T> predictions = Predict(x);
        var anomalies = new List<(int Index, T Actual, T Predicted, T Score)>();

        for (int i = 0; i < y.Length; i++)
        {
            T prediction = predictions[i];
            T actual = y[i];
            T score = NumOps.Abs(NumOps.Subtract(actual, prediction));

            if (NumOps.GreaterThan(score, _anomalyThreshold))
            {
                anomalies.Add((i, actual, prediction, score));
            }
        }

        return anomalies;
    }

    /// <summary>
    /// Gets the current anomaly detection threshold.
    /// </summary>
    /// <returns>The anomaly threshold computed during training.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you the cutoff value used to decide whether
    /// a prediction error is large enough to be an anomaly. Values with scores above this
    /// threshold are considered anomalies.
    /// </para>
    /// </remarks>
    public T GetAnomalyThreshold()
    {
        if (!_prophetOptions.EnableAnomalyDetection)
        {
            throw new InvalidOperationException("Anomaly detection was not enabled during training.");
        }
        return _anomalyThreshold;
    }

    /// <summary>
    /// Sets a custom anomaly detection threshold.
    /// </summary>
    /// <param name="threshold">The new threshold value.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> If the automatic threshold is flagging too many or too few
    /// anomalies, you can set your own:
    /// - Higher threshold = fewer anomalies detected (only extreme values)
    /// - Lower threshold = more anomalies detected (more sensitive)
    ///
    /// Tip: Look at the scores from ComputeAnomalyScores to decide on a good threshold
    /// for your specific use case.
    /// </para>
    /// </remarks>
    public void SetAnomalyThreshold(T threshold)
    {
        _anomalyThreshold = threshold;
    }

    /// <summary>
    /// Computes prediction intervals for future forecasts.
    /// </summary>
    /// <param name="x">The input matrix for forecasting.</param>
    /// <returns>A tuple containing (predictions, lower bounds, upper bounds).</returns>
    /// <remarks>
    /// <para>
    /// Prediction intervals provide uncertainty bounds around the point predictions.
    /// Points outside these intervals are likely anomalies. The width of the interval
    /// is controlled by the PredictionIntervalWidth option.
    /// </para>
    /// <para><b>For Beginners:</b> Instead of just getting a single predicted value,
    /// this method gives you a range where the actual value is likely to fall:
    /// - Lower bound: The low end of the expected range
    /// - Upper bound: The high end of the expected range
    ///
    /// If an actual value falls outside this range, it's probably an anomaly.
    /// For example, if the predicted range for sales is $800-$1200 and actual sales were $2000,
    /// that's clearly unusual.
    /// </para>
    /// </remarks>
    public (Vector<T> Predictions, Vector<T> LowerBound, Vector<T> UpperBound) PredictWithIntervals(Matrix<T> x)
    {
        if (!IsTrained)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        // Check that residual statistics were computed during training
        if (NumOps.Equals(_residualStdDev, NumOps.Zero))
        {
            throw new InvalidOperationException(
                "Prediction intervals require residual statistics. " +
                "Enable ComputePredictionIntervals or EnableAnomalyDetection in ProphetOptions before training.");
        }

        Vector<T> predictions = Predict(x);
        Vector<T> lowerBound = new Vector<T>(predictions.Length);
        Vector<T> upperBound = new Vector<T>(predictions.Length);

        // Calculate z-score from confidence level using inverse normal CDF
        // For confidence level c, z = Φ^(-1)((1 + c) / 2)
        // Examples: 95% → 1.96, 90% → 1.645, 80% → 1.28
        T probability = NumOps.Divide(
            NumOps.Add(NumOps.One, NumOps.FromDouble(_prophetOptions.PredictionIntervalWidth)),
            NumOps.FromDouble(2.0));
        T zScore = StatisticsHelper<T>.CalculateInverseNormalCDF(probability);

        T halfWidth = NumOps.Multiply(zScore, _residualStdDev);

        for (int i = 0; i < predictions.Length; i++)
        {
            lowerBound[i] = NumOps.Subtract(predictions[i], halfWidth);
            upperBound[i] = NumOps.Add(predictions[i], halfWidth);
        }

        return (predictions, lowerBound, upperBound);
    }
}
