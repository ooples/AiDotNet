using AiDotNet.Enums;

namespace AiDotNet.Models;

/// <summary>
/// Represents the parameters used for normalizing a single feature or target variable in a machine learning model.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates all the parameters needed to normalize and denormalize a single feature or target variable. 
/// It supports multiple normalization methods, such as min-max scaling, z-score normalization, robust scaling, and 
/// binning, and stores the relevant parameters for each method. These parameters are typically calculated during 
/// training based on the training data and are then used to normalize new data in the same way.
/// </para>
/// <para><b>For Beginners:</b> This class stores the information needed to scale a single feature or target variable.
/// 
/// When normalizing data for machine learning:
/// - Different methods can be used (min-max scaling, z-score normalization, etc.)
/// - Each method requires specific parameters (like minimum/maximum values or mean/standard deviation)
/// - These parameters need to be saved to ensure consistent scaling
/// 
/// This class stores all those parameters for a single feature, including:
/// - Which normalization method is being used
/// - The specific values needed for that method (min/max, mean/stddev, etc.)
/// 
/// For example, if using min-max scaling to normalize house prices from $100,000-$1,500,000 to a 0-1 range,
/// this class would store the minimum ($100,000) and maximum ($1,500,000) values needed for that conversion.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class NormalizationParameters<T>
{
    /// <summary>
    /// The numeric operations provider used for mathematical operations on type T.
    /// </summary>
    /// <remarks>
    /// This field provides access to basic mathematical operations for the generic type T,
    /// allowing the class to perform calculations regardless of the specific numeric type.
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the NormalizationParameters class with default values.
    /// </summary>
    /// <param name="numOps">Optional numeric operations provider. If null, a default provider will be used.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new NormalizationParameters instance with default values. It initializes all numeric 
    /// properties to zero and sets the normalization method to None. The constructor takes an optional numeric operations 
    /// provider, which is used for mathematical operations on the generic type T. If no provider is specified, a default 
    /// one is obtained from the MathHelper class.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new set of normalization parameters with default values.
    /// 
    /// When creating new normalization parameters:
    /// - All numeric values are initialized to zero
    /// - The normalization method is set to "None"
    /// - A numeric operations provider is set up to handle the math
    /// 
    /// The numeric operations provider:
    /// - Allows the class to work with different numeric types (float, double, decimal)
    /// - Provides methods for basic math operations on type T
    /// - Is usually obtained automatically from MathHelper
    /// 
    /// This constructor is typically used when:
    /// - Creating parameters before calculating actual values
    /// - Deserializing parameters from storage
    /// - Creating empty parameters as placeholders
    /// </para>
    /// </remarks>
    public NormalizationParameters(INumericOperations<T>? numOps = null)
    {
        _numOps = numOps ?? MathHelper.GetNumericOperations<T>();
        Method = NormalizationMethod.None;
        Min = Max = Mean = StdDev = Scale = Shift = Median = IQR = P = MaxAbs = _numOps.Zero;
        Bins = [];
        Quantiles = [];
        OutputDistribution = OutputDistribution.Uniform;
    }

    /// <summary>
    /// Gets or sets the normalization method used.
    /// </summary>
    /// <value>A NormalizationMethod enumeration value indicating which normalization technique is used.</value>
    /// <remarks>
    /// <para>
    /// This property indicates which normalization method is used for the feature or target variable. Different methods 
    /// use different parameters and have different characteristics. For example, min-max scaling normalizes values to a 
    /// specific range (typically 0 to 1), z-score normalization centers the data around zero with a standard deviation 
    /// of one, and robust scaling uses the median and interquartile range to be less sensitive to outliers.
    /// </para>
    /// <para><b>For Beginners:</b> This indicates which scaling technique is being used.
    /// 
    /// The normalization method:
    /// - Determines how the data will be scaled
    /// - Affects which parameters are actually used
    /// - Has different properties and use cases
    /// 
    /// Common methods include:
    /// - None: No normalization is applied
    /// - MinMax: Scales data to a range, typically 0-1 (uses Min and Max)
    /// - ZScore: Centers data around 0 with standard deviation of 1 (uses Mean and StdDev)
    /// - Robust: Similar to ZScore but less affected by outliers (uses Median and IQR)
    /// - Custom: Uses custom Scale and Shift values
    /// - Binning: Divides data into discrete bins
    /// 
    /// Each method has advantages for different types of data and models.
    /// </para>
    /// </remarks>
    public NormalizationMethod Method { get; set; }

    /// <summary>
    /// Gets or sets the minimum value observed in the data.
    /// </summary>
    /// <value>The minimum value, used for min-max normalization.</value>
    /// <remarks>
    /// <para>
    /// This property stores the minimum value observed in the data for the feature or target variable. It is primarily 
    /// used for min-max normalization, where values are scaled to a range based on the minimum and maximum values. The 
    /// minimum value is typically calculated during training based on the training data.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the smallest value observed in the data.
    /// 
    /// The minimum value:
    /// - Is used primarily for min-max scaling
    /// - Represents the lower bound of the original data range
    /// - Is typically mapped to 0 or another lower bound in the normalized range
    /// 
    /// For example, if normalizing house prices and the cheapest house is $100,000,
    /// this value would be 100000.
    /// 
    /// This parameter is important because:
    /// - It defines the lower end of the data range
    /// - It's needed to properly scale new data points
    /// - It helps ensure consistent normalization
    /// </para>
    /// </remarks>
    public T Min { get; set; }

    /// <summary>
    /// Gets or sets the maximum value observed in the data.
    /// </summary>
    /// <value>The maximum value, used for min-max normalization.</value>
    /// <remarks>
    /// <para>
    /// This property stores the maximum value observed in the data for the feature or target variable. It is primarily 
    /// used for min-max normalization, where values are scaled to a range based on the minimum and maximum values. The 
    /// maximum value is typically calculated during training based on the training data.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the largest value observed in the data.
    /// 
    /// The maximum value:
    /// - Is used primarily for min-max scaling
    /// - Represents the upper bound of the original data range
    /// - Is typically mapped to 1 or another upper bound in the normalized range
    /// 
    /// For example, if normalizing house prices and the most expensive house is $1,500,000,
    /// this value would be 1500000.
    /// 
    /// This parameter is important because:
    /// - It defines the upper end of the data range
    /// - It's needed to properly scale new data points
    /// - It helps ensure consistent normalization
    /// </para>
    /// </remarks>
    public T Max { get; set; }

    /// <summary>
    /// Gets or sets the mean (average) value of the data.
    /// </summary>
    /// <value>The mean value, used for z-score normalization.</value>
    /// <remarks>
    /// <para>
    /// This property stores the mean (average) value of the data for the feature or target variable. It is primarily 
    /// used for z-score normalization, where values are scaled by subtracting the mean and dividing by the standard 
    /// deviation. This centers the data around zero. The mean is typically calculated during training based on the 
    /// training data.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the average value of the data.
    /// 
    /// The mean value:
    /// - Is used primarily for z-score normalization
    /// - Represents the center point of the data distribution
    /// - Is subtracted from each value during z-score normalization
    /// 
    /// For example, if the average house price in your dataset is $350,000,
    /// this value would be 350000.
    /// 
    /// This parameter is important because:
    /// - It defines the center of the data distribution
    /// - It's needed to properly center new data points
    /// - It helps ensure consistent normalization
    /// </para>
    /// </remarks>
    public T Mean { get; set; }

    /// <summary>
    /// Gets or sets the standard deviation of the data.
    /// </summary>
    /// <value>The standard deviation, used for z-score normalization.</value>
    /// <remarks>
    /// <para>
    /// This property stores the standard deviation of the data for the feature or target variable. It is primarily used 
    /// for z-score normalization, where values are scaled by subtracting the mean and dividing by the standard deviation. 
    /// This scales the data to have a standard deviation of one. The standard deviation is typically calculated during 
    /// training based on the training data.
    /// </para>
    /// <para><b>For Beginners:</b> This stores how spread out the data values are.
    /// 
    /// The standard deviation:
    /// - Is used primarily for z-score normalization
    /// - Measures how dispersed the data is around the mean
    /// - Is used as a divisor during z-score normalization
    /// 
    /// For example, if house prices in your dataset typically vary by about $150,000 from the mean,
    /// this value would be approximately 150000.
    /// 
    /// This parameter is important because:
    /// - It defines the scale of variation in the data
    /// - It's needed to properly scale new data points
    /// - It helps ensure consistent normalization
    /// </para>
    /// </remarks>
    public T StdDev { get; set; }

    /// <summary>
    /// Gets or sets the scale factor for custom normalization.
    /// </summary>
    /// <value>The scale factor, used for custom normalization.</value>
    /// <remarks>
    /// <para>
    /// This property stores a custom scale factor for the feature or target variable. It is used for custom normalization, 
    /// where values are scaled by multiplying by this factor. Custom normalization allows for more flexibility in how the 
    /// data is scaled, but requires the scale factor to be specified explicitly rather than being calculated from the data.
    /// </para>
    /// <para><b>For Beginners:</b> This stores a custom multiplication factor for scaling.
    /// 
    /// The scale factor:
    /// - Is used for custom normalization
    /// - Represents how much to multiply each value by
    /// - Allows for flexible, manual control of scaling
    /// 
    /// For example, if you want to convert dollars to thousands of dollars,
    /// you might use a scale factor of 0.001.
    /// 
    /// This parameter is useful when:
    /// - You want more control over the normalization process
    /// - You have domain knowledge about the appropriate scaling
    /// - Standard methods don't fit your specific needs
    /// </para>
    /// </remarks>
    public T Scale { get; set; }

    /// <summary>
    /// Gets or sets the shift value for custom normalization.
    /// </summary>
    /// <value>The shift value, used for custom normalization.</value>
    /// <remarks>
    /// <para>
    /// This property stores a custom shift value for the feature or target variable. It is used for custom normalization, 
    /// where values are shifted by adding this value (typically after scaling). Custom normalization allows for more 
    /// flexibility in how the data is transformed, but requires the shift value to be specified explicitly rather than 
    /// being calculated from the data.
    /// </para>
    /// <para><b>For Beginners:</b> This stores a custom value to add after scaling.
    /// 
    /// The shift value:
    /// - Is used for custom normalization
    /// - Represents how much to add to each value after scaling
    /// - Allows for flexible, manual control of normalization
    /// 
    /// For example, if you want to shift temperatures from Celsius to Fahrenheit,
    /// you might use a scale of 1.8 and a shift of 32.
    /// 
    /// This parameter is useful when:
    /// - You want more control over the normalization process
    /// - You have domain knowledge about the appropriate transformation
    /// - Standard methods don't fit your specific needs
    /// </para>
    /// </remarks>
    public T Shift { get; set; }

    /// <summary>
    /// Gets or sets the bin boundaries for binning normalization.
    /// </summary>
    /// <value>A list of values representing bin boundaries.</value>
    /// <remarks>
    /// <para>
    /// This property stores the boundaries between bins for binning normalization. Binning is a normalization technique 
    /// that converts continuous values into discrete bins or categories. Each value in the list represents a boundary 
    /// between adjacent bins. Values less than the first boundary go into the first bin, values between the first and 
    /// second boundaries go into the second bin, and so on. The number of bins is one more than the number of boundaries.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the dividing points for converting continuous data into categories.
    /// 
    /// The bins list:
    /// - Is used for binning normalization
    /// - Contains the boundaries between different bins or categories
    /// - Allows continuous data to be converted to discrete categories
    /// 
    /// For example, if binning house prices, you might have bins at:
    /// [200000, 400000, 600000, 800000]
    /// Creating 5 categories: <200K, 200K-400K, 400K-600K, 600K-800K, >800K
    /// 
    /// This approach is useful when:
    /// - You want to convert continuous data to categorical
    /// - The exact values are less important than the range they fall into
    /// - You want to reduce the impact of outliers
    /// </para>
    /// </remarks>
    public List<T> Bins { get; set; }

    /// <summary>
    /// Gets or sets the median value of the data.
    /// </summary>
    /// <value>The median value, used for robust normalization.</value>
    /// <remarks>
    /// <para>
    /// This property stores the median value of the data for the feature or target variable. It is primarily used for 
    /// robust normalization, where values are scaled by subtracting the median and dividing by the interquartile range. 
    /// This approach is less sensitive to outliers than z-score normalization, which uses the mean and standard deviation. 
    /// The median is the middle value when the data is sorted and is typically calculated during training based on the 
    /// training data.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the middle value of the data when sorted.
    /// 
    /// The median value:
    /// - Is used primarily for robust normalization
    /// - Represents the middle point of the sorted data
    /// - Is less affected by outliers than the mean
    /// 
    /// For example, if the middle house price in your sorted dataset is $320,000,
    /// this value would be 320000.
    /// 
    /// This parameter is important because:
    /// - It provides a robust measure of central tendency
    /// - It's less influenced by extreme values than the mean
    /// - It's used in robust scaling to handle data with outliers
    /// </para>
    /// </remarks>
    public T Median { get; set; }

    /// <summary>
    /// Gets or sets the interquartile range (IQR) of the data.
    /// </summary>
    /// <value>The interquartile range, used for robust normalization.</value>
    /// <remarks>
    /// <para>
    /// This property stores the interquartile range (IQR) of the data for the feature or target variable. The IQR is the 
    /// difference between the 75th percentile (Q3) and the 25th percentile (Q1) of the data. It is primarily used for 
    /// robust normalization, where values are scaled by subtracting the median and dividing by the IQR. This approach is 
    /// less sensitive to outliers than z-score normalization, which uses the mean and standard deviation. The IQR is 
    /// typically calculated during training based on the training data.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the range between the 25th and 75th percentiles of the data.
    /// 
    /// The interquartile range (IQR):
    /// - Is used primarily for robust normalization
    /// - Measures the spread of the middle 50% of the data
    /// - Is less affected by outliers than standard deviation
    /// 
    /// For example, if the middle 50% of house prices in your dataset range from $250,000 to $450,000,
    /// the IQR would be 200000.
    /// 
    /// This parameter is important because:
    /// - It provides a robust measure of data spread
    /// - It's less influenced by extreme values than standard deviation
    /// - It's used as a divisor in robust scaling to handle data with outliers
    /// </para>
    /// </remarks>
    public T IQR { get; set; }

    /// <summary>
    /// Gets or sets a power parameter for certain normalization methods.
    /// </summary>
    /// <value>The power parameter, used for power transformations.</value>
    /// <remarks>
    /// <para>
    /// This property stores a power parameter that can be used for certain normalization methods, such as power transformations
    /// like Box-Cox or Yeo-Johnson transformations. These transformations can help make skewed data more normally distributed
    /// by raising values to a certain power. The optimal power parameter is typically determined during training to maximize
    /// the normality of the transformed data.
    /// </para>
    /// <para><b>For Beginners:</b> This stores a power value used for certain advanced normalization techniques.
    ///
    /// The power parameter:
    /// - Is used for power transformations like Box-Cox or Yeo-Johnson
    /// - Helps make skewed data more normally distributed
    /// - Can be optimized to find the best transformation
    ///
    /// For example, a value of 0.5 would correspond to a square root transformation,
    /// which can help normalize right-skewed data.
    ///
    /// This parameter is useful when:
    /// - Your data has a skewed distribution
    /// - You want to make the data more normally distributed
    /// - Standard normalization methods don't work well
    ///
    /// Power transformations are more advanced techniques but can significantly
    /// improve model performance with certain types of data.
    /// </para>
    /// </remarks>
    public T P { get; set; }

    /// <summary>
    /// Gets or sets the maximum absolute value observed in the data.
    /// </summary>
    /// <value>The maximum absolute value, used for MaxAbsScaler normalization.</value>
    /// <remarks>
    /// <para>
    /// This property stores the maximum absolute value observed in the data for the feature or target variable.
    /// It is used for MaxAbsScaler normalization, which scales data to the range [-1, 1] by dividing each value
    /// by the maximum absolute value. This method preserves the sign of values and maintains zeros (which is
    /// important for sparse data). The maximum absolute value is typically calculated during training based on
    /// the training data.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the largest absolute value (ignoring the sign) in your data.
    ///
    /// The maximum absolute value:
    /// - Is used for MaxAbsScaler normalization
    /// - Represents the farthest distance from zero in either direction
    /// - Is used as a divisor to scale values to the range [-1, 1]
    ///
    /// For example, if your data ranges from -75 to 100, the maximum absolute value would be 100,
    /// and all values would be divided by 100 to scale them to [-0.75, 1.0].
    ///
    /// This parameter is important because:
    /// - It preserves the sign of values (positive stays positive, negative stays negative)
    /// - It keeps zero values as zero (important for sparse data)
    /// - It's simpler than min-max scaling but still effective
    /// </para>
    /// </remarks>
    public T MaxAbs { get; set; }

    /// <summary>
    /// Gets or sets the quantile values used for quantile transformation.
    /// </summary>
    /// <value>A list of quantile values representing the empirical distribution.</value>
    /// <remarks>
    /// <para>
    /// This property stores the quantile values calculated from the training data for QuantileTransformer.
    /// These quantiles represent the empirical cumulative distribution function (CDF) of the data and are
    /// used to map values to either a uniform or normal distribution. The number of quantiles determines
    /// the granularity of the transformation.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the distribution pattern learned from your training data.
    ///
    /// The quantiles list:
    /// - Is used for QuantileTransformer normalization
    /// - Contains values that divide your data into equal-sized groups
    /// - Helps map your data to a target distribution (uniform or normal)
    ///
    /// For example, with 100 quantiles:
    /// - The 25th quantile is the value below which 25% of the data falls
    /// - The 50th quantile is the median
    /// - The 75th quantile is the value below which 75% of the data falls
    ///
    /// This approach is powerful because:
    /// - It can handle any input distribution
    /// - It's very robust to outliers
    /// - It can transform data to match a desired distribution shape
    /// </para>
    /// </remarks>
    public List<T> Quantiles { get; set; }

    /// <summary>
    /// Gets or sets the target output distribution for quantile transformation.
    /// </summary>
    /// <value>An OutputDistribution enum indicating either Uniform or Normal distribution.</value>
    /// <remarks>
    /// <para>
    /// This property specifies whether the QuantileTransformer should map data to a uniform distribution
    /// (where all ranges have equal probability) or a normal distribution (bell-shaped curve). This setting
    /// determines how the quantiles are mapped during transformation.
    /// </para>
    /// <para><b>For Beginners:</b> This specifies what shape you want your data to have after transformation.
    ///
    /// The output distribution:
    /// - Can be Uniform (flat distribution) or Normal (bell curve)
    /// - Affects how values are redistributed
    /// - Depends on what your machine learning algorithm expects
    ///
    /// Uniform distribution:
    /// - All value ranges have equal numbers of data points
    /// - Values are spread evenly across the range
    /// - Good for algorithms that don't assume any particular distribution
    ///
    /// Normal distribution:
    /// - Creates a bell-shaped curve
    /// - Most values cluster around the center
    /// - Good for algorithms that work best with normally-distributed data
    /// </para>
    /// </remarks>
    public OutputDistribution OutputDistribution { get; set; }
}
