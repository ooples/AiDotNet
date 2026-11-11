using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

/// <summary>
/// Implements the SEATS (Seasonal Extraction in ARIMA Time Series) decomposition method for time series data.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SEATS is an advanced method for breaking down time series data into different components:
/// trend (long-term direction), seasonal patterns (regular fluctuations), and irregular components (random noise).
/// It uses statistical models to identify these patterns in your data.
/// </para>
/// <para>
/// This implementation supports multiple algorithm variants: Standard, Canonical, and Burman.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Vector<>), typeof(Vector<>))]
    [ResearchPaper("SEATS: Signal Extraction in ARIMA Time Series", "https://www.bde.es/f/webbde/SES/Secciones/Publicaciones/PublicacionesSeriadas/DocumentosTrabajo/01/Fich/dt0109e.pdf")]
public class SEATSDecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly SARIMAOptions<T> _sarimaOptions;
    private readonly SEATSAlgorithmType _algorithm;

    /// <summary>
    /// Initializes a new instance of the <see cref="SEATSDecomposition{T}"/> class.
    /// </summary>
    /// <param name="timeSeries">The time series data to decompose.</param>
    /// <param name="sarimaOptions">Options for the SARIMA model. If null, default options will be used.</param>
    /// <param name="algorithm">The algorithm variant to use for decomposition.</param>
    /// <param name="forecastHorizon">Kept for backward compatibility; this parameter is not used.</param>
    /// <remarks>
    /// <b>For Beginners:</b> When creating a SEATS decomposition:
    /// - timeSeries: Your data points arranged in time order
    /// - sarimaOptions: Settings for the statistical model (you can use default settings)
    /// - algorithm: Which calculation method to use (Standard is a good default)
    /// </remarks>
    public SEATSDecomposition(Vector<T> timeSeries, SARIMAOptions<T>? sarimaOptions = null, SEATSAlgorithmType algorithm = SEATSAlgorithmType.Standard, int forecastHorizon = 10)
        : base(timeSeries)
    {
        _ = forecastHorizon; // kept for API compatibility
        _sarimaOptions = sarimaOptions ?? new();
        _algorithm = algorithm;
        Decompose();
    }

    /// <summary>
    /// Performs the time series decomposition based on the selected algorithm.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm is specified.</exception>
    protected override void Decompose()
    {
        switch (_algorithm)
        {
            case SEATSAlgorithmType.Standard:
                DecomposeStandard();
                break;
            case SEATSAlgorithmType.Canonical:
                DecomposeCanonical();
                break;
            case SEATSAlgorithmType.Burman:
                DecomposeBurman();
                break;
            default:
                throw new ArgumentException("Unsupported SEATS algorithm");
        }
    }

    /// <summary>
    /// Performs standard SEATS decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method fits a statistical model to your data and then
    /// separates it into trend (long-term direction), seasonal patterns, and random noise.
    /// </remarks>
    private void DecomposeStandard()
    {
        // Step 1: Fit SARIMA model
        var sarimaModel = new SARIMAModel<T>(_sarimaOptions);
        sarimaModel.Train(new Matrix<T>(TimeSeries.Length, 1), TimeSeries);

        // Step 2: Extract components
        Vector<T> trend = ExtractTrendComponent(sarimaModel);
        Vector<T> seasonal = ExtractSeasonalComponent(sarimaModel);
        Vector<T> irregular = ExtractIrregularComponent(trend, seasonal);

        // Add components
        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Irregular, irregular);
    }

    /// <summary>
    /// Performs canonical SEATS decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The canonical method is a specific approach that ensures the components
    /// are as independent from each other as possible. This means the trend, seasonal, and
    /// irregular components have minimal correlation with each other.
    /// </remarks>
    private void DecomposeCanonical()
    {
        // Step 1: Fit SARIMA model
        var sarimaModel = new SARIMAModel<T>(_sarimaOptions);
        sarimaModel.Train(new Matrix<T>(TimeSeries.Length, 1), TimeSeries);

        // Step 2: Extract components using canonical decomposition
        Vector<T> trend = ExtractCanonicalTrendComponent(sarimaModel);
        Vector<T> seasonal = ExtractCanonicalSeasonalComponent(sarimaModel);
        Vector<T> irregular = ExtractCanonicalIrregularComponent(sarimaModel, trend, seasonal);

        // Add components
        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Irregular, irregular);
    }

    /// <summary>
    /// Performs Burman's variant of SEATS decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Burman's method is an alternative approach to decomposition that uses
    /// special filters designed to better separate the components in certain types of data.
    /// It can be more effective for some complex time series.
    /// </remarks>
    private void DecomposeBurman()
    {
        // Step 1: Fit SARIMA model
        var sarimaModel = new SARIMAModel<T>(_sarimaOptions);
        sarimaModel.Train(new Matrix<T>(TimeSeries.Length, 1), TimeSeries);

        // Step 2: Extract components using Burman's method
        Vector<T> trend = ExtractBurmanTrendComponent(sarimaModel);
        Vector<T> seasonal = ExtractBurmanSeasonalComponent(sarimaModel);
        Vector<T> irregular = ExtractBurmanIrregularComponent(sarimaModel, trend, seasonal);

        // Add components
        AddComponent(DecompositionComponentType.Trend, trend);
        AddComponent(DecompositionComponentType.Seasonal, seasonal);
        AddComponent(DecompositionComponentType.Irregular, irregular);
    }

    /// <summary>
    /// Extracts the trend component using canonical decomposition.
    /// </summary>
    /// <param name="model">The fitted SARIMA model.</param>
    /// <returns>The trend component of the time series.</returns>
    private Vector<T> ExtractCanonicalTrendComponent(SARIMAModel<T> model)
    {
        // Extract trend component using canonical decomposition
        var trendFilter = DesignCanonicalTrendFilter(model);
        return ApplyFilter(TimeSeries, trendFilter);
    }

    /// <summary>
    /// Extracts the seasonal component using canonical decomposition.
    /// </summary>
    /// <param name="model">The fitted SARIMA model.</param>
    /// <returns>The seasonal component of the time series.</returns>
    private Vector<T> ExtractCanonicalSeasonalComponent(SARIMAModel<T> model)
    {
        // Extract seasonal component using canonical decomposition
        var seasonalFilter = DesignCanonicalSeasonalFilter(model);
        return ApplyFilter(TimeSeries, seasonalFilter);
    }

    /// <summary>
    /// Extracts the irregular component as the residual after removing trend and seasonal components.
    /// </summary>
    /// <param name="model">The fitted SARIMA model.</param>
    /// <param name="trend">The extracted trend component.</param>
    /// <param name="seasonal">The extracted seasonal component.</param>
    /// <returns>The irregular component of the time series.</returns>
    private Vector<T> ExtractCanonicalIrregularComponent(SARIMAModel<T> model, Vector<T> trend, Vector<T> seasonal)
    {
        // Calculate irregular component as the residual
        return TimeSeries.Subtract(trend).Subtract(seasonal);
    }

    /// <summary>
    /// Extracts the trend component using Burman's method.
    /// </summary>
    /// <param name="model">The fitted SARIMA model.</param>
    /// <returns>The trend component of the time series.</returns>
    private Vector<T> ExtractBurmanTrendComponent(SARIMAModel<T> model)
    {
        // Extract trend component using Burman's method
        var trendFilter = DesignBurmanTrendFilter(model);
        return ApplyFilter(TimeSeries, trendFilter);
    }

    /// <summary>
    /// Extracts the seasonal component using Burman's method.
    /// </summary>
    /// <param name="model">The fitted SARIMA model.</param>
    /// <returns>The seasonal component of the time series.</returns>
    private Vector<T> ExtractBurmanSeasonalComponent(SARIMAModel<T> model)
    {
        // Extract seasonal component using Burman's method
        var seasonalFilter = DesignBurmanSeasonalFilter(model);
        return ApplyFilter(TimeSeries, seasonalFilter);
    }

    /// <summary>
    /// Extracts the irregular component as the residual after removing trend and seasonal components.
    /// </summary>
    /// <param name="model">The fitted SARIMA model.</param>
    /// <param name="trend">The extracted trend component.</param>
    /// <param name="seasonal">The extracted seasonal component.</param>
    /// <returns>The irregular component of the time series.</returns>
    private Vector<T> ExtractBurmanIrregularComponent(SARIMAModel<T> model, Vector<T> trend, Vector<T> seasonal)
    {
        // Calculate irregular component as the residual
        return TimeSeries.Subtract(trend).Subtract(seasonal);
    }

    /// <summary>
    /// Designs a canonical trend filter based on the provided SARIMA model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A canonical trend filter helps identify the long-term direction or pattern in your data
    /// by removing short-term fluctuations. It's like looking at the general path of a hiking trail
    /// while ignoring small ups and downs along the way.
    /// </para>
    /// </remarks>
    /// <param name="model">The SARIMA model containing parameters for filter design.</param>
    /// <returns>A vector representing the designed filter coefficients.</returns>
    private Vector<T> DesignCanonicalTrendFilter(SARIMAModel<T> model)
    {
        var arParams = model.GetARParameters();
        var maParams = model.GetMAParameters();
        int p = arParams.Length;
        int q = maParams.Length;

        // Calculate the spectral density at frequency zero
        T spectralDensityAtZero = CalculateSpectralDensityAtZero(arParams, maParams);

        // Design the canonical trend filter: (ar[k] - ma[k]) / S(0), both zero padded to the filter length
        int filterLength = Math.Max(p, q) + 1;
        var difference = Engine.TensorSubtract(PadTo(arParams, filterLength), PadTo(maParams, filterLength));
        return Engine.TensorMultiplyScalar(difference, NumOps.Divide(NumOps.One, spectralDensityAtZero)).ToVector();
    }

    /// <summary>The coefficients as a tensor, zero padded at the end to <paramref name="length"/>.</summary>
    private Tensor<T> PadTo(Vector<T> coefficients, int length)
    {
        if (coefficients.Length == 0)
        {
            return new Tensor<T>(new[] { length });
        }

        return Engine.TensorConstantPad(Tensor<T>.FromVector(coefficients), new[] { 0, length - coefficients.Length }, NumOps.Zero);
    }

    /// <summary>cos(k * x) for every k in <paramref name="lags"/> (rows) and x in <paramref name="angles"/> (columns).</summary>
    private Tensor<T> CosineTable(Tensor<T> lags, Tensor<T> angles)
    {
        int rows = lags.Length;
        int columns = angles.Length;
        var products = Engine.TensorMultiply(
            Engine.TensorBroadcastTo(lags.Reshape(new[] { rows, 1 }), new[] { rows, columns }),
            Engine.TensorBroadcastTo(angles.Reshape(new[] { 1, columns }), new[] { rows, columns }));
        return Engine.TensorCos(products);
    }

    /// <summary>2 pi j / s for j in [0, s).</summary>
    private Tensor<T> SeasonalFrequencies(int s)
    {
        var scaled = Engine.TensorMultiplyScalar(Engine.TensorArange<T>(0, s, 1), NumOps.FromDouble(2 * Math.PI));
        return Engine.TensorDivide(scaled, Engine.TensorFullLike(scaled, NumOps.FromDouble(s)));
    }

    /// <summary>
    /// Designs a canonical seasonal filter based on the provided SARIMA model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A canonical seasonal filter helps identify repeating patterns in your data
    /// that occur at regular intervals (like daily, weekly, or yearly patterns). It's like identifying
    /// that ice cream sales always go up in summer and down in winter, year after year.
    /// </para>
    /// </remarks>
    /// <param name="model">The SARIMA model containing parameters for filter design.</param>
    /// <returns>A vector representing the designed filter coefficients.</returns>
    private Vector<T> DesignCanonicalSeasonalFilter(SARIMAModel<T> model)
    {
        var sarParams = model.GetSeasonalARParameters();
        var smaParams = model.GetSeasonalMAParameters();
        int P = sarParams.Length;
        int Q = smaParams.Length;
        int s = model.GetSeasonalPeriod();

        // Design the canonical seasonal filter
        int filterLength = s * Math.Max(P, Q) + 1;
        T normalizer = NumOps.Divide(NumOps.One, NumOps.FromDouble(s - 1));
        if (s < 2)
        {
            // No seasonal frequencies: an empty sum scaled by 1 / (s - 1).
            return Engine.TensorMultiplyScalar(new Tensor<T>(new[] { filterLength }), normalizer).ToVector();
        }

        // Calculate the spectral density at seasonal frequencies
        Tensor<T> spectralDensities = CalculateSpectralDensitiesAtSeasonalFrequencies(sarParams, smaParams, s);

        // coeff[k] = (1 / (s - 1)) * sum_{j=1}^{s-1} cos(k * 2 pi j / s) / S(2 pi j / s)
        var frequencies = Engine.TensorSlice(SeasonalFrequencies(s), new[] { 1 }, new[] { s - 1 });
        var inverseDensities = Engine.TensorReciprocal(Engine.TensorSlice(spectralDensities, new[] { 1 }, new[] { s - 1 }));
        var cosines = CosineTable(Engine.TensorArange<T>(0, filterLength, 1), frequencies);
        var sums = Engine.TensorMatMul(cosines, inverseDensities.Reshape(new[] { s - 1, 1 })).Reshape(new[] { filterLength });
        return Engine.TensorMultiplyScalar(sums, normalizer).ToVector();
    }

    /// <summary>
    /// Designs a Burman trend filter based on the provided SARIMA model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Burman trend filter is an advanced technique to extract the underlying trend
    /// from your time series data. It's particularly good at handling complex data where simpler methods
    /// might fail. Think of it as a sophisticated camera lens that can focus on just the main subject
    /// while blurring out all the background noise.
    /// </para>
    /// </remarks>
    /// <param name="model">The SARIMA model containing parameters for filter design.</param>
    /// <returns>A vector representing the designed filter coefficients.</returns>
    private Vector<T> DesignBurmanTrendFilter(SARIMAModel<T> model)
    {
        var arParams = model.GetARParameters();
        var maParams = model.GetMAParameters();
        int p = arParams.Length;
        int q = maParams.Length;

        // Design the Burman trend filter:
        // coeff[k] = (1 / pi) * integral_0^pi cos(k f) / S(f) df for k in [-L/2, L/2],
        // with the integral as a 1000-step left Riemann sum.
        int filterLength = Math.Max(p, q) * 2 + 1;
        int halfLength = filterLength / 2;
        const int steps = 1000;
        T stepSize = NumOps.Divide(NumOps.FromDouble(Math.PI), NumOps.FromDouble(steps));
        var grid = Engine.TensorMultiplyScalar(Engine.TensorArange<T>(0, steps, 1), stepSize);
        var inverseDensity = Engine.TensorReciprocal(CalculateSpectralDensity(grid, arParams, maParams));

        var lags = Engine.TensorArange<T>(-halfLength, halfLength + 1, 1);
        var sums = Engine.TensorMatMul(CosineTable(lags, grid), inverseDensity.Reshape(new[] { steps, 1 })).Reshape(new[] { filterLength });
        var integrals = Engine.TensorMultiplyScalar(sums, stepSize);
        return Engine.TensorDivide(integrals, Engine.TensorFullLike(integrals, NumOps.FromDouble(Math.PI))).ToVector();
    }

    /// <summary>
    /// Calculates the spectral density at frequency zero for the given AR and MA parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Spectral density is a way to measure how much of each frequency component
    /// contributes to your data. At frequency zero, we're looking at the very long-term component
    /// (like the overall average level). This helps us understand the strength of the trend in your data.
    /// </para>
    /// </remarks>
    /// <param name="arParams">The autoregressive parameters.</param>
    /// <param name="maParams">The moving average parameters.</param>
    /// <returns>The spectral density value at frequency zero.</returns>
    private T CalculateSpectralDensityAtZero(Vector<T> arParams, Vector<T> maParams)
    {
        T arSum = arParams.Length > 0
            ? NumOps.Subtract(NumOps.One, Engine.TensorSum(Tensor<T>.FromVector(arParams)))
            : NumOps.One;
        T maSum = maParams.Length > 0
            ? NumOps.Add(NumOps.One, Engine.TensorSum(Tensor<T>.FromVector(maParams)))
            : NumOps.One;

        return NumOps.Divide(NumOps.Multiply(maSum, maSum), NumOps.Multiply(arSum, arSum));
    }

    /// <summary>
    /// Calculates the spectral densities at seasonal frequencies for the given seasonal AR and MA parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how strong each seasonal pattern is in your data.
    /// For example, if you have monthly data, it measures the strength of patterns that repeat
    /// every month, every two months, etc. This helps identify which seasonal patterns are most important.
    /// </para>
    /// </remarks>
    /// <param name="sarParams">The seasonal autoregressive parameters.</param>
    /// <param name="smaParams">The seasonal moving average parameters.</param>
    /// <param name="s">The seasonal period.</param>
    /// <returns>An array of spectral density values at seasonal frequencies.</returns>
    private Tensor<T> CalculateSpectralDensitiesAtSeasonalFrequencies(Vector<T> sarParams, Vector<T> smaParams, int s)
    {
        if (s <= 0)
        {
            return new Tensor<T>(new[] { 0 });
        }

        return CalculateSpectralDensity(SeasonalFrequencies(s), sarParams, smaParams);
    }

    /// <summary>
    /// Calculates the spectral density at a given frequency for the provided AR and MA parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method measures how much a specific pattern (at a particular frequency)
    /// contributes to your data. Think of it like analyzing the ingredients in a recipe - 
    /// this tells you how much of each ingredient (frequency) is present in your data "dish".
    /// </para>
    /// </remarks>
    /// <param name="freqs">The frequencies at which to calculate the spectral density.</param>
    /// <param name="arParams">The autoregressive parameters.</param>
    /// <param name="maParams">The moving average parameters.</param>
    /// <returns>The spectral density at each of the specified frequencies.</returns>
    private Tensor<T> CalculateSpectralDensity(Tensor<T> freqs, Vector<T> arParams, Vector<T> maParams)
    {
        return SpectralDensityRatio(
            freqs, arParams, maParams,
            HarmonicMultipliers(arParams.Length, 0, 1), HarmonicMultipliers(maParams.Length, 0, 1));
    }

    /// <summary>(start + i) * scale for i in [0, count).</summary>
    private Tensor<T> HarmonicMultipliers(int count, int start, int scale)
    {
        if (count == 0)
        {
            return new Tensor<T>(new[] { 0 });
        }

        return Engine.TensorMultiplyScalar(Engine.TensorArange<T>(start, start + count, 1), NumOps.FromDouble(scale));
    }

    /// <summary>
    /// (2 * ma^2) / (2 * ar^2) with ar = 1 - sum a_i cos(m_i f) + sum a_i sin(m_i f) and
    /// ma = 1 + sum b_i cos(m_i f) - sum b_i sin(m_i f), for every frequency f in <paramref name="freqs"/>.
    /// </summary>
    private Tensor<T> SpectralDensityRatio(Tensor<T> freqs, Vector<T> arParams, Vector<T> maParams, Tensor<T> arMultipliers, Tensor<T> maMultipliers)
    {
        int count = freqs.Length;
        var arTerm = Engine.TensorOnesLike(freqs);
        var maTerm = Engine.TensorOnesLike(freqs);

        if (arParams.Length > 0)
        {
            var (cosSum, sinSum) = HarmonicSums(freqs, arMultipliers, arParams);
            arTerm = Engine.TensorAdd(Engine.TensorSubtract(arTerm, cosSum), sinSum);
        }

        if (maParams.Length > 0)
        {
            var (cosSum, sinSum) = HarmonicSums(freqs, maMultipliers, maParams);
            maTerm = Engine.TensorSubtract(Engine.TensorAdd(maTerm, cosSum), sinSum);
        }

        var arSquared = Engine.TensorMultiplyScalar(Engine.TensorMultiply(arTerm, arTerm), NumOps.FromDouble(2));
        var maSquared = Engine.TensorMultiplyScalar(Engine.TensorMultiply(maTerm, maTerm), NumOps.FromDouble(2));
        return Engine.TensorMultiply(maSquared, Engine.TensorReciprocal(arSquared)).Reshape(new[] { count });
    }

    /// <summary>sum_i c_i cos(m_i f) and sum_i c_i sin(m_i f) for every frequency f.</summary>
    private (Tensor<T> Cos, Tensor<T> Sin) HarmonicSums(Tensor<T> freqs, Tensor<T> multipliers, Vector<T> coefficients)
    {
        int count = freqs.Length;
        int order = coefficients.Length;
        var angles = Engine.TensorMultiply(
            Engine.TensorBroadcastTo(freqs.Reshape(new[] { count, 1 }), new[] { count, order }),
            Engine.TensorBroadcastTo(multipliers.Reshape(new[] { 1, order }), new[] { count, order }));
        var c = Tensor<T>.FromVector(coefficients, new[] { order, 1 });
        var cosSum = Engine.TensorMatMul(Engine.TensorCos(angles), c).Reshape(new[] { count });
        var sinSum = Engine.TensorMatMul(Engine.TensorSin(angles), c).Reshape(new[] { count });
        return (cosSum, sinSum);
    }

    /// <summary>
    /// Designs a Burman seasonal filter based on the provided SARIMA model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A Burman seasonal filter helps identify and extract repeating patterns
    /// in your data (like weekly, monthly, or yearly cycles). It's like having a special lens
    /// that only shows you the seasonal patterns while filtering out other components.
    /// </para>
    /// </remarks>
    /// <param name="model">The SARIMA model containing parameters for filter design.</param>
    /// <returns>A vector representing the designed seasonal filter coefficients.</returns>
    private Vector<T> DesignBurmanSeasonalFilter(SARIMAModel<T> model)
    {
        var _sarParams = model.GetSeasonalARParameters();
        var _smaParams = model.GetSeasonalMAParameters();
        int _seasonalPeriod = model.GetSeasonalPeriod();
        int _seasonalArOrder = _sarParams.Length;
        int _seasonalMaOrder = _smaParams.Length;

        // Design the Burman seasonal filter:
        // coeff[k] = (1 / 2pi) * integral_0^2pi cos(k f) / S_s(f) df, as a 1000-step left Riemann sum.
        int _filterLength = _seasonalPeriod * (Math.Max(_seasonalArOrder, _seasonalMaOrder) + 1);
        const int steps = 1000;
        T stepSize = NumOps.Divide(NumOps.FromDouble(2 * Math.PI), NumOps.FromDouble(steps));
        var grid = Engine.TensorMultiplyScalar(Engine.TensorArange<T>(0, steps, 1), stepSize);
        var inverseDensity = Engine.TensorReciprocal(
            CalculateSeasonalSpectralDensity(grid, _sarParams, _smaParams, _seasonalPeriod));

        var lags = Engine.TensorArange<T>(0, _filterLength, 1);
        var sums = Engine.TensorMatMul(CosineTable(lags, grid), inverseDensity.Reshape(new[] { steps, 1 })).Reshape(new[] { _filterLength });
        var integrals = Engine.TensorMultiplyScalar(sums, stepSize);
        var coefficients = Engine.TensorDivide(integrals, Engine.TensorFullLike(integrals, NumOps.FromDouble(2 * Math.PI)));

        // Normalize the filter by the sum of absolute coefficients
        T _sum = Engine.TensorSum(Engine.TensorAbs(coefficients));
        return Engine.TensorMultiply(coefficients, Engine.TensorReciprocal(Engine.TensorFullLike(coefficients, _sum))).ToVector();
    }

    /// <summary>
    /// Calculates the spectral density at a given frequency for seasonal components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Spectral density for seasonal components measures how strongly
    /// each seasonal pattern (like weekly or monthly cycles) appears in your data.
    /// It's like analyzing which musical notes are loudest in a song, but for time patterns
    /// instead of sound.
    /// </para>
    /// </remarks>
    /// <param name="freqs">The frequencies at which to calculate the spectral density.</param>
    /// <param name="sarParams">The seasonal autoregressive parameters.</param>
    /// <param name="smaParams">The seasonal moving average parameters.</param>
    /// <param name="s">The seasonal period.</param>
    /// <returns>The spectral density at each of the specified frequencies.</returns>
    private Tensor<T> CalculateSeasonalSpectralDensity(Tensor<T> freqs, Vector<T> sarParams, Vector<T> smaParams, int s)
    {
        return SpectralDensityRatio(
            freqs, sarParams, smaParams,
            HarmonicMultipliers(sarParams.Length, 1, s), HarmonicMultipliers(smaParams.Length, 1, s));
    }

    /// <summary>
    /// Applies a filter to a signal using convolution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Applying a filter to a signal is like using a special lens on a camera.
    /// The filter (lens) highlights certain features while reducing others. In time series,
    /// we use filters to extract specific patterns like trends or seasonal components.
    /// Convolution is the mathematical operation that combines the signal with the filter.
    /// </para>
    /// </remarks>
    /// <param name="signal">The input signal (time series data).</param>
    /// <param name="filter">The filter to apply.</param>
    /// <returns>The filtered signal.</returns>
    private Vector<T> ApplyFilter(Vector<T> signal, Vector<T> filter)
    {
        int _signalLength = signal.Length;
        int _filterLength = filter.Length;
        if (_signalLength == 0)
        {
            return new Vector<T>(0);
        }

        // Full convolution with the reversed filter and zero padding:
        // result[i] = sum_j signal[i - j] * filter[L - 1 - j] = sum_j signal[i - (L - 1) + j] * filter[j],
        // which is the cross-correlation Conv1D computes with L - 1 zeros of padding on both sides.
        var input = Tensor<T>.FromVector(signal, new[] { 1, 1, _signalLength });
        var kernel = Tensor<T>.FromVector(filter, new[] { 1, 1, _filterLength });
        var full = Engine.Conv1D(input, kernel, 1, _filterLength - 1).Reshape(new[] { _signalLength + _filterLength - 1 });

        // Trim the result to match the original signal length
        int _startIndex = (_filterLength - 1) / 2;
        return Engine.TensorSlice(full, new[] { _startIndex }, new[] { _signalLength }).ToVector();
    }

    /// <summary>
    /// Extracts the trend component from a time series using the SARIMA model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The trend component represents the long-term direction of your data.
    /// Think of it as the general path your data follows when you ignore short-term fluctuations
    /// and seasonal patterns. For example, if your sales are generally increasing over years
    /// despite seasonal ups and downs, that upward direction is the trend.
    /// </para>
    /// </remarks>
    /// <param name="model">The SARIMA model used for extraction.</param>
    /// <returns>A vector representing the trend component.</returns>
    private Vector<T> ExtractTrendComponent(SARIMAModel<T> model)
    {
        // Extract trend component using the non-seasonal part of the SARIMA model
        var _trendOptions = new SARIMAOptions<T>
        {
            P = _sarimaOptions.P,
            D = _sarimaOptions.D,
            Q = _sarimaOptions.Q,
            SeasonalPeriod = 0, // The trend submodel has no seasonal component.
            SeasonalP = 0,
            SeasonalD = 0,
            SeasonalQ = 0
        };

        var _trendModel = new SARIMAModel<T>(_trendOptions);
        _trendModel.Train(new Matrix<T>(TimeSeries.Length, 1), TimeSeries);

        return _trendModel.Predict(new Matrix<T>(TimeSeries.Length, 1));
    }

    /// <summary>
    /// Extracts the seasonal component from a time series using the SARIMA model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The seasonal component represents repeating patterns in your data that occur at
    /// regular intervals (like sales increasing during holidays every year). This method isolates these
    /// recurring patterns from your data, similar to identifying which notes repeat in a musical composition.
    /// </para>
    /// <para>
    /// This method creates a specialized SARIMA model that focuses only on the seasonal aspects of your data
    /// by setting the non-seasonal parameters (P, D, Q) to zero and using only the seasonal parameters.
    /// </para>
    /// </remarks>
    /// <param name="model">The SARIMA model used for extraction.</param>
    /// <returns>A vector representing the seasonal component.</returns>
    private Vector<T> ExtractSeasonalComponent(SARIMAModel<T> model)
    {
        // Extract seasonal component using the seasonal part of the SARIMA model
        var _seasonalOptions = new SARIMAOptions<T>
        {
            P = 0,
            D = 0,
            Q = 0,
            SeasonalP = _sarimaOptions.SeasonalP,
            SeasonalD = _sarimaOptions.SeasonalD,
            SeasonalQ = _sarimaOptions.SeasonalQ,
            SeasonalPeriod = _sarimaOptions.SeasonalPeriod
        };

        var _seasonalModel = new SARIMAModel<T>(_seasonalOptions);
        _seasonalModel.Train(new Matrix<T>(TimeSeries.Length, 1), TimeSeries);

        return _seasonalModel.Predict(new Matrix<T>(TimeSeries.Length, 1));
    }

    /// <summary>
    /// Extracts the irregular component from a time series by removing trend and seasonal components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The irregular component (sometimes called "residual" or "noise") represents the 
    /// random fluctuations in your data that can't be explained by trends or seasonal patterns. 
    /// Think of it like the static on a radio - it's the unpredictable part that remains after 
    /// accounting for all predictable patterns.
    /// </para>
    /// <para>
    /// This method simply subtracts both the trend and seasonal components from the original time series,
    /// leaving only the irregular variations.
    /// </para>
    /// </remarks>
    /// <param name="trend">The trend component to subtract.</param>
    /// <param name="seasonal">The seasonal component to subtract.</param>
    /// <returns>A vector representing the irregular component.</returns>
    private Vector<T> ExtractIrregularComponent(Vector<T> trend, Vector<T> seasonal)
    {
        // Calculate irregular component as the residual
        return TimeSeries.Subtract(trend).Subtract(seasonal);
    }
}
