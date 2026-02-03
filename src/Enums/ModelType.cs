namespace AiDotNet.Enums;

/// <summary>
/// Defines the types of machine learning models available in the AiDotNet library.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This enum lists all the different AI models you can use in this library.
/// Think of these as different tools in your AI toolbox - each one works best for specific
/// types of problems. Some are simple and easy to understand, while others are more powerful
/// but complex. Start with simpler models like SimpleRegression before moving to more advanced ones.
/// </para>
/// </remarks>
public enum ModelType
{
    /// <summary>
    /// Represents no model selection.
    /// </summary>
    None,

    /// <summary>
    /// An automated machine learning model that automatically selects and trains the best model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AutoML (Automated Machine Learning) automates the process of selecting and
    /// configuring the best machine learning model for your data. Instead of manually trying different
    /// models, AutoML experiments with various algorithms, evaluates them, and chooses the one that
    /// performs best. It's like having an AI assistant that tests different approaches and picks the
    /// winner for you, saving you time and expertise.
    /// </para>
    /// </remarks>
    AutoML,

    /// <summary>
    /// A basic model that finds the relationship between a single input variable and an output variable.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Simple Regression is like drawing a straight line through your data points.
    /// It helps you understand how one thing affects another - for example, how temperature affects
    /// ice cream sales. It's the easiest model to understand and a great starting point.
    /// </para>
    /// </remarks>
    SimpleRegression,

    /// <summary>
    /// A model that finds the relationship between multiple input variables and a single output variable.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multiple Regression is like Simple Regression but with more factors.
    /// Instead of just using temperature to predict ice cream sales, you might also consider
    /// day of the week, local events, and season. This gives you a more complete picture.
    /// </para>
    /// </remarks>
    MultipleRegression,

    /// <summary>
    /// A model that predicts multiple output variables based on multiple input variables.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multivariate Regression predicts several things at once. For example,
    /// instead of just predicting ice cream sales, you might predict sales of ice cream,
    /// cold drinks, and sunscreen all from the same set of inputs like temperature and season.
    /// </para>
    /// </remarks>
    MultivariateRegression,

    /// <summary>
    /// A model that captures non-linear relationships using polynomial functions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Polynomial Regression uses curved lines instead of straight ones.
    /// This is useful when the relationship isn't a straight line - for example, plant growth
    /// might increase with water up to a point, then decrease if there's too much water.
    /// A curved line can capture this pattern better than a straight line.
    /// </para>
    /// </remarks>
    PolynomialRegression,

    /// <summary>
    /// A regression model that gives different importance to different data points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Weighted Regression gives some data points more importance than others.
    /// For example, if you're predicting house prices, you might give more weight to recent sales
    /// and less weight to sales from many years ago, since the market changes over time.
    /// </para>
    /// </remarks>
    WeightedRegression,

    /// <summary>
    /// An ensemble model that combines multiple decision trees to improve prediction accuracy.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Random Forest combines many decision trees (simple flowchart-like models)
    /// and lets them vote on the final prediction. It's like asking a large group of people
    /// instead of just one person - the combined wisdom often gives better results. This model
    /// is powerful, handles complex data well, and is less likely to overfit (memorize) your data.
    /// </para>
    /// </remarks>
    RandomForest,

    /// <summary>
    /// A tree-based model that makes decisions by splitting data based on feature values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Decision Tree works like a flowchart of yes/no questions. For example,
    /// to predict if someone will buy ice cream: "Is temperature > 75°F? If yes, is it a weekend?
    /// If no, is there a special event?" and so on. It's easy to understand but can be less
    /// accurate than more complex models.
    /// </para>
    /// </remarks>
    DecisionTree,

    /// <summary>
    /// An ensemble technique that builds models sequentially, with each new model correcting errors from previous ones.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Gradient Boosting builds a series of models, where each new model focuses on
    /// fixing the mistakes of previous models. It's like having a team where each member specializes
    /// in handling the cases that the rest of the team struggles with. This approach often produces
    /// very accurate predictions but requires careful tuning to avoid overfitting.
    /// </para>
    /// </remarks>
    GradientBoosting,

    /// <summary>
    /// A histogram-based gradient boosting model that uses binned features for fast training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Histogram Gradient Boosting is a faster version of gradient boosting
    /// that groups similar feature values into "bins" (like putting numbers into buckets).
    /// This makes training much faster, especially for large datasets, while often achieving
    /// similar accuracy. It's similar to popular libraries like LightGBM and XGBoost.
    /// </para>
    /// </remarks>
    HistGradientBoosting,

    /// <summary>
    /// A boosting algorithm specifically designed for regression problems.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AdaBoost-R2 is a specialized version of boosting for predicting numeric values.
    /// It works by giving more attention to the data points that are hardest to predict correctly.
    /// With each round of training, it adjusts to focus more on the difficult cases, similar to
    /// how a teacher might give extra attention to students who are struggling with certain concepts.
    /// </para>
    /// </remarks>
    AdaBoostR2,

    /// <summary>
    /// A variation of Random Forest that introduces more randomness in how trees are built.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Extremely Randomized Trees is similar to Random Forest but adds even more
    /// randomness when building each tree. This extra randomness can help prevent overfitting
    /// (when a model learns the training data too specifically and performs poorly on new data).
    /// Think of it as deliberately introducing some "noise" to make the model more flexible.
    /// </para>
    /// </remarks>
    ExtremelyRandomizedTrees,

    /// <summary>
    /// An extension of Random Forest that predicts a range of possible values rather than a single value.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Quantile Regression Forests don't just predict a single value (like "this house
    /// will sell for $300,000") but instead predict a range ("this house will sell for between $280,000
    /// and $320,000 with 90% confidence"). This gives you a better sense of the uncertainty in the prediction,
    /// which can be very valuable for decision-making.
    /// </para>
    /// </remarks>
    QuantileRegressionForests,

    /// <summary>
    /// A decision tree that uses statistical tests to make splitting decisions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Conditional Inference Tree is a type of decision tree that uses statistics to
    /// decide how to split the data. This makes it less biased toward features with many possible values.
    /// For example, a regular decision tree might favor using "zip code" over "temperature" just because
    /// there are more possible zip codes, even if temperature is actually more important. Conditional
    /// Inference Trees help avoid this problem.
    /// </para>
    /// </remarks>
    ConditionalInferenceTree,

    /// <summary>
    /// A tree model that uses linear regression at its leaf nodes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> M5 Model Tree combines decision trees with linear regression. Instead of making
    /// a single prediction at each leaf (end point) of the tree, it fits a small linear regression model.
    /// This is like first sorting your data into groups using yes/no questions, then finding the best
    /// straight-line fit for each group separately. This approach often works well for problems with
    /// numeric outputs.
    /// </para>
    /// </remarks>
    M5ModelTree,

    /// <summary>
    /// A model that captures complex, non-linear relationships in data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Non-Linear Regression captures complex relationships that can't be represented
    /// by straight lines. For example, population growth might follow an S-curve (slow at first,
    /// then rapid, then slowing down again). Non-linear models can capture these more complex patterns,
    /// but they can be harder to interpret and require more data to train effectively.
    /// </para>
    /// </remarks>
    NonLinearRegression,

    /// <summary>
    /// A powerful algorithm that finds patterns by mapping data to higher-dimensional spaces.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Support Vector Regression finds patterns by transforming your data into a different
    /// space where the pattern becomes simpler. Imagine trying to separate mixed red and blue marbles on a table.
    /// It might be hard in 2D, but if you could lift some marbles up (adding a 3rd dimension), the separation
    /// might become easier. SVR uses a similar concept mathematically, making it powerful for complex patterns,
    /// though it can be slower and harder to tune than simpler models.
    /// </para>
    /// </remarks>
    SupportVectorRegression,

    /// <summary>
    /// Combines ridge regression with kernel methods to handle non-linear relationships.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Kernel Ridge Regression is like Support Vector Regression but with a different
    /// mathematical approach. It uses "kernels" (special mathematical functions) to transform your data
    /// into a space where complex relationships become simpler. This allows it to capture non-linear patterns
    /// while still maintaining some of the simplicity and efficiency of linear models.
    /// </para>
    /// </remarks>
    KernelRidgeRegression,

    /// <summary>
    /// Ridge Regression (L2 regularized linear regression) that adds a penalty for large coefficients.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Ridge Regression is a safer version of linear regression that prevents overfitting.
    ///
    /// It works by adding a penalty for large coefficient values (L2 regularization), which:
    /// - Shrinks coefficients toward zero (but never exactly to zero)
    /// - Makes the model more stable when features are correlated
    /// - Reduces sensitivity to noise in the data
    ///
    /// Ridge Regression has a closed-form solution, making it fast to train. Use it when:
    /// - You have correlated features
    /// - You want to prevent overfitting
    /// - You expect all features to contribute to the prediction
    /// </para>
    /// </remarks>
    RidgeRegression,

    /// <summary>
    /// Lasso Regression (L1 regularized linear regression) that can eliminate unimportant features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Lasso Regression performs automatic feature selection by shrinking some
    /// coefficients exactly to zero.
    ///
    /// Unlike Ridge Regression which only shrinks coefficients, Lasso can completely eliminate features:
    /// - L1 regularization can set coefficients to exactly zero
    /// - Creates sparse models with fewer non-zero coefficients
    /// - Useful for identifying the most important features
    ///
    /// Lasso uses coordinate descent optimization (iterative). Use it when:
    /// - You have many features and want to identify the most important ones
    /// - You want a simpler, more interpretable model
    /// - You suspect only a subset of features actually matter
    ///
    /// Note: Lasso may arbitrarily select one feature from a group of correlated features.
    /// Consider ElasticNet for better handling of correlated features.
    /// </para>
    /// </remarks>
    LassoRegression,

    /// <summary>
    /// Elastic Net Regression (combined L1 and L2 regularization) for balanced feature selection and stability.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Elastic Net combines the best of Ridge and Lasso regression.
    ///
    /// It uses both L1 (Lasso) and L2 (Ridge) penalties, controlled by the L1Ratio parameter:
    /// - L1Ratio = 1.0: Pure Lasso (maximum feature selection)
    /// - L1Ratio = 0.0: Pure Ridge (maximum stability)
    /// - L1Ratio = 0.5: Balanced mix (default)
    ///
    /// Key benefits over Lasso alone:
    /// - Groups correlated features together instead of picking one arbitrarily
    /// - Can select more than n features when n samples are available
    /// - More stable when features are highly correlated
    ///
    /// Use Elastic Net when:
    /// - You want feature selection AND have correlated features
    /// - Lasso's behavior on correlated features is problematic
    /// - You're not sure whether Ridge or Lasso is better for your problem
    /// </para>
    /// </remarks>
    ElasticNetRegression,

    /// <summary>
    /// A probabilistic model that provides uncertainty estimates along with predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Gaussian Process Regression doesn't just predict a value - it gives you a range
    /// of possible values with their probabilities. It's like a weather forecast that says "80% chance
    /// of rain" instead of just "it will rain." This is very useful when understanding the uncertainty
    /// of predictions is important, such as in scientific research or risk assessment.
    /// </para>
    /// </remarks>
    GaussianProcessRegression,

    /// <summary>
    /// A non-parametric regression technique that preserves the order of data points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Isotonic Regression finds a non-decreasing (always flat or going up) or non-increasing
    /// (always flat or going down) line that best fits your data. It's useful when you know your output
    /// should never decrease as your input increases (or vice versa). For example, the risk of heart disease
    /// generally increases with age, so a model that sometimes shows risk decreasing with age would be illogical.
    /// </para>
    /// </remarks>
    IsotonicRegression,

    /// <summary>
    /// Predicts specific percentiles of the output distribution rather than just the mean.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Quantile Regression predicts different percentiles of your data. Instead of just
    /// predicting the average house price in an area, it might predict the median price (50th percentile),
    /// the luxury price (90th percentile), and the budget price (10th percentile). This gives you a more
    /// complete picture of the range of possible values.
    /// </para>
    /// </remarks>
    QuantileRegression,

    /// <summary>
    /// A model that uses radial basis functions to approximate complex patterns in data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Radial Basis Function Regression uses special functions that respond strongly 
    /// to data points close to their center and weakly to distant points. Imagine dropping a pebble 
    /// in water - the ripples are strongest near where the pebble fell and fade as they move outward. 
    /// By combining many of these "ripple patterns" centered at different points, the model can 
    /// approximate complex relationships in your data.
    /// </para>
    /// </remarks>
    RadialBasisFunctionRegression,

    /// <summary>
    /// A model that gives more weight to nearby data points when making predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Locally Weighted Regression makes predictions by focusing more on data points 
    /// that are similar to what you're trying to predict. It's like asking for restaurant recommendations 
    /// and giving more weight to opinions from people with similar taste to yours. This approach is 
    /// flexible and can capture complex patterns without requiring a specific mathematical formula.
    /// </para>
    /// </remarks>
    LocallyWeightedRegression,

    /// <summary>
    /// A model that uses piecewise polynomial functions to create smooth curves through data points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Spline Regression connects data points with smooth curves rather than straight lines. 
    /// Imagine drawing a smooth curve through points on a graph by hand - you naturally create gentle 
    /// curves rather than sharp angles. Splines work similarly, creating a series of connected curves 
    /// that flow smoothly through your data points, which is useful for capturing complex patterns.
    /// </para>
    /// </remarks>
    SplineRegression,

    /// <summary>
    /// A model that predicts based on the average of the k closest data points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> K-Nearest Neighbors makes predictions by looking at the most similar examples 
    /// in your data. To predict a house price, it might find the 5 most similar houses (in terms of 
    /// size, location, etc.) and average their prices. It's like asking "What happened in similar 
    /// situations?" rather than trying to find a mathematical formula. Simple to understand but can 
    /// be slow with large datasets.
    /// </para>
    /// </remarks>
    KNearestNeighbors,

    /// <summary>
    /// A model that discovers mathematical formulas that best describe relationships in data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Symbolic Regression tries to find an actual mathematical formula that explains
    /// your data. Instead of just fitting parameters to a pre-defined equation, it searches for the
    /// equation itself. For example, it might discover that your data follows "y = x² + 3x - 2" rather 
    /// than just giving you numbers. This provides insights into the underlying relationships and can 
    /// be more interpretable than other complex models.
    /// </para>
    /// </remarks>
    SymbolicRegression,

    /// <summary>
    /// A probabilistic approach that updates predictions as new data becomes available.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bayesian Regression starts with an initial guess about relationships in your data, 
    /// then updates this guess as it sees more data. It's like having a theory about something, then 
    /// gradually refining your theory as you gather more evidence. A key advantage is that it doesn't 
    /// just give predictions but also tells you how confident it is in those predictions.
    /// </para>
    /// </remarks>
    BayesianRegression,

    /// <summary>
    /// A tree-like structure that represents mathematical expressions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Expression Tree represents mathematical formulas as tree structures that can be 
    /// manipulated and evolved. Each branch and leaf represents part of a formula (like addition, 
    /// multiplication, or variables). This approach is often used in genetic programming to "evolve" 
    /// mathematical formulas that fit your data, similar to how natural selection works in nature.
    /// </para>
    /// </remarks>
    ExpressionTree,

    /// <summary>
    /// A mathematical representation of data as points in multi-dimensional space.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Vector models represent your data as points in space with multiple dimensions. 
    /// Each feature becomes a dimension - so if you have height and weight, each data point is plotted 
    /// in 2D space. With more features, you get more dimensions (though these become hard to visualize). 
    /// This representation allows mathematical operations that can reveal patterns and relationships 
    /// in your data.
    /// </para>
    /// </remarks>
    Vector,

    /// <summary>
    /// A model that uses principles inspired by natural evolution to find optimal solutions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Genetic Algorithm Regression mimics evolution to find the best model. It starts 
    /// with many random models, keeps the best ones ("survival of the fittest"), combines them to create 
    /// "offspring" models, and occasionally introduces random changes ("mutations"). Over many generations, 
    /// this process tends to discover increasingly better models, even for complex problems where traditional 
    /// approaches might struggle.
    /// </para>
    /// </remarks>
    GeneticAlgorithmRegression,

    /// <summary>
    /// A regression technique that accounts for errors in both input and output variables.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Orthogonal Regression accounts for measurement errors in both your input and output 
    /// variables. Standard regression assumes your input measurements are perfect and only the outputs 
    /// have errors. But in reality, you might have uncertainty in both. For example, when studying how 
    /// height affects weight, both measurements might have errors. Orthogonal regression handles this 
    /// situation better than standard approaches.
    /// </para>
    /// </remarks>
    OrthogonalRegression,

    /// <summary>
    /// A regression method that is less sensitive to outliers in the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Robust Regression is designed to handle outliers - unusual data points that don't 
    /// follow the general pattern. Standard regression can be heavily influenced by even a single extreme 
    /// value (like one house that sold for 10x the normal price). Robust regression reduces the influence 
    /// of these outliers, giving you a model that better represents the typical patterns in your data.
    /// </para>
    /// </remarks>
    RobustRegression,

    /// <summary>
    /// A model specifically designed for data that changes over time.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Time Series Regression is specialized for data collected over time, like daily 
    /// temperatures, stock prices, or monthly sales. It accounts for special patterns in time data, 
    /// such as seasonal effects (sales increasing during holidays), trends (gradual increase over years), 
    /// and the fact that recent values often influence future values. This makes it much better for 
    /// forecasting future values in time-based data.
    /// </para>
    /// </remarks>
    TimeSeriesRegression,

    /// <summary>
    /// A flexible model that combines multiple simple functions to capture complex patterns.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Generalized Additive Model Regression builds complex relationships by adding together 
    /// simpler ones. Instead of forcing a specific shape (like a straight line or curve), it lets each 
    /// input variable affect the output in its own way. For example, temperature might have a curved 
    /// relationship with ice cream sales, while day of week has a different pattern. GAMs combine these 
    /// separate patterns to create a flexible overall model.
    /// </para>
    /// </remarks>
    GeneralizedAdditiveModelRegression,

    /// <summary>
    /// A technique that handles correlated input variables by projecting them onto new dimensions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Partial Least Squares Regression works well when your input variables are related 
    /// to each other. For example, a person's height and weight are correlated - taller people tend to 
    /// weigh more. This correlation can confuse standard regression. PLS creates new combined variables 
    /// that capture the most important patterns while avoiding the problems caused by these correlations.
    /// </para>
    /// </remarks>
    PartialLeastSquaresRegression,

    /// <summary>
    /// A dimension reduction technique combined with regression to handle many correlated variables.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Principal Component Regression first simplifies your data by identifying the most 
    /// important patterns (principal components), then builds a regression model using these patterns. 
    /// It's like summarizing a 50-page document into 5 key points, then working with those summaries. 
    /// This approach works well when you have many related input variables and helps avoid overfitting 
    /// (when a model learns noise rather than true patterns).
    /// </para>
    /// </remarks>
    PrincipalComponentRegression,

    /// <summary>
    /// A method that automatically selects the most important variables for prediction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Stepwise Regression automatically chooses which input variables to include in your 
    /// model. It starts with either no variables or all variables, then adds or removes them one by one 
    /// based on how much they improve predictions. This helps create simpler models by focusing only on 
    /// the most important factors, making the model easier to interpret and often more accurate on new data.
    /// </para>
    /// </remarks>
    StepwiseRegression,

    /// <summary>
    /// A regression model for count data (non-negative integers).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Poisson Regression is designed for predicting counts - like the number of customer 
    /// calls per hour, website visits per day, or accidents per month. Unlike standard regression, it 
    /// ensures predictions are always non-negative (you can't have -3 customer calls) and works with the 
    /// special statistical properties of count data, where variance often increases with the mean.
    /// </para>
    /// </remarks>
    PoissonRegression,

    /// <summary>
    /// A model for predicting categorical outcomes with more than two possible values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multinomial Logistic Regression predicts which category something belongs to when 
    /// there are multiple possibilities. For example, predicting if a customer will choose small, medium, 
    /// or large size, or if an email is spam, personal, or work-related. It gives you the probability for 
    /// each possible outcome, allowing you to see not just the most likely category but also how confident 
    /// the model is.
    /// </para>
    /// </remarks>
    MultinomialLogisticRegression,

    /// <summary>
    /// A model for predicting binary outcomes (yes/no, true/false, 0/1).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Logistic Regression predicts the probability of something being in a particular category. 
    /// Despite its name, it's used for classification, not regression. For example, it can predict whether 
    /// an email is spam (1) or not spam (0), or the probability a customer will make a purchase. It works by 
    /// transforming a linear model into probability values between 0 and 1 using a special S-shaped curve 
    /// called the logistic function.
    /// </para>
    /// </remarks>
    LogisticRegression,

    /// <summary>
    /// A model for count data with extra variation (overdispersion).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Negative Binomial Regression is designed for counting events (like number of customer 
    /// complaints or product defects) when there's more variability in the data than expected. While Poisson 
    /// regression assumes the average and variance are equal, real-world count data often has higher variance. 
    /// Negative Binomial Regression handles this extra variability, making it more realistic for many 
    /// real-world counting problems.
    /// </para>
    /// </remarks>
    NegativeBinomialRegression,

    /// <summary>
    /// A flexible model inspired by the human brain's structure that can capture complex patterns.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural Network Regression uses interconnected "neurons" organized in layers to learn 
    /// patterns in data. Think of it as a complex web of calculations that can discover and represent 
    /// relationships that simpler models miss. Neural networks can automatically learn features from data 
    /// without being explicitly programmed, making them powerful for complex problems like image recognition, 
    /// language processing, and predicting outcomes with many interacting factors.
    /// </para>
    /// </remarks>
    NeuralNetworkRegression,

    /// <summary>
    /// A specific type of neural network with multiple layers of neurons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multilayer Perceptron Regression is a specific type of neural network with at least 
    /// three layers: an input layer, one or more hidden layers, and an output layer. Each neuron connects 
    /// to all neurons in the next layer, creating a densely connected network. This structure allows the 
    /// model to learn complex patterns by transforming the data through successive layers, with each layer 
    /// learning increasingly abstract features. It's like having a team of specialists who each focus on 
    /// different aspects of a problem before combining their insights.
    /// </para>
    /// </remarks>
    MultilayerPerceptronRegression,

    /// <summary>
    /// A generative stochastic neural network that can learn a probability distribution over its inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Deep Boltzmann Machine is a type of neural network that learns to recognize patterns 
    /// in data by trying to recreate the input data from scratch. Imagine it as an artist trying to paint a 
    /// picture after only glancing at it briefly. By doing this repeatedly, the network learns the important 
    /// features and relationships in the data. This makes it useful for tasks like feature detection, 
    /// dimensionality reduction, and generating new data similar to the training set.
    /// </para>
    /// </remarks>
    DeepBoltzmannMachine,

    /// <summary>
    /// A type of neural network particularly effective for processing grid-like data such as images.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Convolutional Neural Networks are specialized for processing data with a grid-like 
    /// structure, such as images. They use a mathematical operation called convolution to scan over the input, 
    /// detecting features like edges, textures, and shapes. This is similar to how our eyes focus on different 
    /// parts of an image. CNNs are highly effective for tasks like image classification, object detection, 
    /// and even some types of time series analysis.
    /// </para>
    /// </remarks>
    ConvolutionalNeuralNetwork,

    /// <summary>
    /// A neural network architecture designed to work with sequential data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Recurrent Neural Networks are designed to work with sequences of data, where the 
    /// order matters. They have a form of memory, allowing information to persist. This makes them ideal for 
    /// tasks involving time series, language, or any data where context from previous inputs is important. 
    /// Think of it like reading a book - your understanding of each word is influenced by the words that came before it.
    /// </para>
    /// </remarks>
    RecurrentNeuralNetwork,

    /// <summary>
    /// A type of neural network that learns to encode data into a compressed representation and then decode it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Autoencoders are neural networks that try to copy their input to their output. This 
    /// might seem pointless, but there's a catch: they have to go through a narrow "bottleneck" in the middle. 
    /// This forces the network to learn a compressed representation of the data. It's like learning to describe 
    /// a movie in just a few words, then trying to recreate the whole movie from those words. This can be used 
    /// for dimensionality reduction, feature learning, and generating new data.
    /// </para>
    /// </remarks>
    Autoencoder,

    /// <summary>
    /// A neural network architecture that excels at processing sequential data with long-term dependencies.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Long Short-Term Memory networks are a special kind of Recurrent Neural Network 
    /// capable of learning long-term dependencies. They're like a more sophisticated form of memory, able to 
    /// remember information for long periods of time. This makes them particularly good at tasks like language 
    /// translation, speech recognition, or any task where you need to consider context from much earlier in a sequence.
    /// </para>
    /// </remarks>
    LongShortTermMemory,

    /// <summary>
    /// A neural network architecture that uses attention mechanisms to weigh the importance of different parts of the input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Transformer models use a mechanism called "attention" to weigh the importance of 
    /// different parts of the input when producing an output. It's like being able to focus on the most relevant 
    /// words in a sentence to understand its meaning. This allows them to handle long-range dependencies in data 
    /// very effectively. Transformers have revolutionized natural language processing tasks and are also being 
    /// applied to other domains like computer vision.
    /// </para>
    /// </remarks>
    Transformer,

    /// <summary>
    /// A neural network architecture that uses attention mechanisms to focus on relevant parts of the input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Attention Networks are designed to focus on the most important parts of the input 
    /// when making predictions. Imagine reading a long document and highlighting the key phrases - that's 
    /// similar to what an attention network does. It learns which parts of the input are most relevant for 
    /// the task at hand. This makes them particularly effective for tasks where some input elements are more 
    /// important than others, such as in language translation, image captioning, or analyzing time series data 
    /// where certain time steps might be more crucial than others.
    /// </para>
    /// </remarks>
    AttentionNetwork,

    /// <summary>
    /// A probabilistic generative model composed of multiple layers of stochastic latent variables.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Deep Belief Networks are like a tower of pattern recognizers stacked on top of each other. 
    /// Each layer learns to identify patterns in the output of the layer below it. The network starts by learning 
    /// simple patterns in the raw data, then progressively learns more complex and abstract patterns in higher layers. 
    /// </para>
    /// <para>
    /// For example, if analyzing images:
    /// - The bottom layer might learn to detect edges and simple shapes
    /// - Middle layers might recognize more complex features like eyes, noses, or wheels
    /// - Top layers might identify complete objects or scenes
    /// 
    /// This layer-by-layer approach allows Deep Belief Networks to learn meaningful representations of data 
    /// even when you don't have a lot of labeled examples, making them powerful for tasks like feature learning, 
    /// dimensionality reduction, and generating new data similar to the training set.
    /// </para>
    /// </remarks>
    DeepBeliefNetwork,

    ResidualNeuralNetwork,

    VariationalAutoencoder,

    CapsuleNetwork,

    QuantumNeuralNetwork,

    SelfOrganizingMap,

    LiquidStateMachine,

    HopfieldNetwork,

    GraphNeuralNetwork,

    ExtremeLearningMachine,

    DifferentiableNeuralComputer,

    EchoStateNetwork,

    DeepQNetwork,

    /// <summary>
    /// Double Deep Q-Network - addresses overestimation bias in DQN.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Double DQN fixes a problem in standard DQN where Q-values are often
    /// too optimistic. It uses two networks to make more realistic value estimates - one picks
    /// the best action, another evaluates it. This leads to more stable and accurate learning.
    ///
    /// Strengths: More accurate Q-values, better final performance, same complexity as DQN
    /// </para>
    /// </remarks>
    DoubleDQN,

    /// <summary>
    /// Dueling Deep Q-Network - separates value and advantage estimation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dueling DQN splits Q-values into two parts: the value of being in a state
    /// (how good is this situation?) and the advantage of each action (how much better is this action
    /// than average?). This makes learning more efficient, especially when many actions have similar values.
    ///
    /// Strengths: Faster learning, better performance, especially useful when actions don't always matter
    /// </para>
    /// </remarks>
    DuelingDQN,

    /// <summary>
    /// Rainbow DQN - combines six DQN improvements into one powerful algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Rainbow combines Double DQN, Dueling DQN, Prioritized Replay,
    /// Multi-step Learning, Distributional RL, and Noisy Networks. It's like taking the best features
    /// from six different DQN variants and putting them together. Currently the strongest DQN variant.
    ///
    /// Strengths: State-of-the-art performance, combines multiple improvements, excellent sample efficiency
    /// </para>
    /// </remarks>
    RainbowDQN,

    GenerativeAdversarialNetwork,

    /// <summary>
    /// A Deep Convolutional GAN that uses convolutional layers with specific architectural guidelines.
    /// </summary>
    DCGAN,

    /// <summary>
    /// A Wasserstein GAN that uses Wasserstein distance for more stable training.
    /// </summary>
    WassersteinGAN,

    /// <summary>
    /// A Wasserstein GAN with Gradient Penalty for enforcing Lipschitz constraint.
    /// </summary>
    WassersteinGANGP,

    /// <summary>
    /// A Conditional GAN that generates data conditioned on additional information.
    /// </summary>
    ConditionalGAN,

    /// <summary>
    /// An Auxiliary Classifier GAN that includes class information in generation.
    /// </summary>
    AuxiliaryClassifierGAN,

    /// <summary>
    /// An Information Maximizing GAN that learns disentangled representations.
    /// </summary>
    InfoGAN,

    /// <summary>
    /// A StyleGAN that generates high-quality images with style-based generation.
    /// </summary>
    StyleGAN,

    /// <summary>
    /// A Progressive GAN that grows during training for high-resolution image generation.
    /// </summary>
    ProgressiveGAN,

    /// <summary>
    /// A BigGAN that uses large batch sizes for high-fidelity image generation.
    /// </summary>
    BigGAN,

    /// <summary>
    /// A CycleGAN for unpaired image-to-image translation.
    /// </summary>
    CycleGAN,

    /// <summary>
    /// A Pix2Pix GAN for paired image-to-image translation.
    /// </summary>
    Pix2Pix,

    /// <summary>
    /// A Self-Attention GAN that uses attention mechanisms for modeling long-range dependencies.
    /// </summary>
    SAGAN,

    NeuralTuringMachine,

    NEAT,

    MemoryNetwork,

    LSTMNeuralNetwork,

    HTMNetwork,

    NeuralNetwork,

    OccupancyNetwork,

    /// <summary>
    /// A 3D Convolutional Neural Network that processes voxelized volumetric data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> VoxelCNN is like a regular CNN but for 3D data. Instead of looking at
    /// 2D images, it examines 3D grids of "voxels" (volumetric pixels). Think of voxels like
    /// 3D Minecraft blocks - each block is either filled or empty.
    /// 
    /// VoxelCNN is useful for:
    /// - 3D shape classification (e.g., ModelNet40 dataset)
    /// - Medical image analysis (CT scans, MRI)
    /// - Robotics and spatial understanding
    /// - Point cloud classification (after voxelization)
    /// </para>
    /// </remarks>
    VoxelCNN,

    /// <summary>
    /// 3D U-Net architecture for volumetric semantic segmentation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A 3D U-Net is like an intelligent 3D scanner that can identify and label
    /// every single voxel in a 3D volume. The "U" shape comes from its encoder-decoder design:
    /// - Encoder: Progressively zooms out to understand the big picture
    /// - Decoder: Progressively zooms back in to produce detailed predictions
    /// - Skip connections: Preserve fine details by linking encoder to decoder
    ///
    /// 3D U-Net is useful for:
    /// - Medical image segmentation (organs, tumors in CT/MRI)
    /// - 3D point cloud semantic segmentation
    /// - Part segmentation of 3D shapes
    /// </para>
    /// </remarks>
    UNet3D,

    /// <summary>
    /// MeshCNN neural network for processing 3D triangle meshes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MeshCNN processes 3D shapes represented as triangle meshes
    /// directly, without converting to voxels or point clouds. It learns from the
    /// connectivity of triangles through edge-based convolutions.
    /// 
    /// MeshCNN is useful for:
    /// - Shape classification from mesh data
    /// - Mesh segmentation (labeling different parts)
    /// - Learning from CAD models and 3D scans
    /// </para>
    /// </remarks>
    MeshCNN,

    /// <summary>
    /// SpiralNet++ neural network for mesh vertex processing using spiral convolutions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SpiralNet++ processes 3D meshes by applying convolutions
    /// along spiral sequences of vertices. This provides consistent local feature learning
    /// on irregular mesh structures without requiring mesh registration.
    ///
    /// SpiralNet++ is useful for:
    /// - Mesh shape analysis and classification
    /// - Medical mesh analysis (organs, bones)
    /// - General mesh classification and segmentation
    ///
    /// Strengths: Works on irregular meshes, efficient spiral-based operations
    /// </para>
    /// </remarks>
    SpiralNetPlusPlus,
    /// <summary>
    /// SpiralNet neural network for mesh processing using spiral convolutions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SpiralNet is the original spiral convolution architecture
    /// for processing 3D meshes. It defines consistent spiral ordering of vertex neighbors
    /// for convolutional operations on irregular mesh structures.
    /// </para>
    /// </remarks>
    SpiralNet,

    RestrictedBoltzmannMachine,

    SpikingNeuralNetwork,

    FeedForwardNetwork,

    Linear,
    Polynomial,
    Symbolic,
    TreeBased,

    ARIMAXModel,

    ARMAModel,

    ARModel,

    BayesianStructuralTimeSeriesModel,

    DynamicRegressionWithARIMAErrors,

    ExponentialSmoothingModel,

    GARCHModel,

    GRUNeuralNetwork,

    SiameseNetwork,

    Clip,

    /// <summary>
    /// BLIP (Bootstrapped Language-Image Pre-training) for vision-language tasks.
    /// </summary>
    Blip,

    /// <summary>
    /// BLIP-2 (Bootstrapped Language-Image Pre-training 2) with Q-Former architecture.
    /// </summary>
    Blip2,

    /// <summary>
    /// LLaVA (Large Language and Vision Assistant) for visual instruction following.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LLaVA connects a vision encoder with a large language model,
    /// enabling conversational AI about images. Think of it as giving ChatGPT the ability to see!
    /// </para>
    /// </remarks>
    LLaVA,

    /// <summary>
    /// Flamingo model for in-context visual learning with few-shot capabilities.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Flamingo can learn new visual tasks from just a few examples.
    /// Show it a few image-text pairs, and it learns the pattern to apply to new images.
    /// </para>
    /// </remarks>
    Flamingo,

    /// <summary>
    /// VideoCLIP for video-text understanding and retrieval.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> VideoCLIP extends CLIP to understand videos, not just images.
    /// It can match videos with text descriptions, recognize actions, and caption videos.
    /// </para>
    /// </remarks>
    VideoCLIP,

    /// <summary>
    /// ImageBind for binding 6+ modalities (image, text, audio, video, thermal, depth, IMU).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ImageBind connects ALL types of data in a shared space.
    /// Search audio by text, match thermal images to photos, or find videos from motion data!
    /// </para>
    /// </remarks>
    ImageBind,

    InterventionAnalysisModel,

    TBATSModel,

    STLDecomposition,

    VARModel,

    UnobservedComponentsModel,

    TransferFunctionModel,

    StateSpaceModel,

    ARIMAModel,

    MAModel,

    SARIMAModel,

    SpectralAnalysisModel,

    ProphetModel,

    NeuralNetworkARIMA,

    /// <summary>
    /// A neural network architecture that employs multiple specialist networks (experts) with learned routing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mixture-of-Experts is like having a team of specialists rather than one generalist.
    ///
    /// Imagine a hospital with different specialists:
    /// - A cardiologist handles heart problems
    /// - A neurologist handles brain issues
    /// - A pediatrician handles children's health
    /// - A triage system (gating network) directs patients to the right specialist(s)
    ///
    /// In a MoE neural network:
    /// - Multiple "expert" networks specialize in different patterns
    /// - A "gating network" learns to route inputs to the best expert(s)
    /// - Only a few experts process each input (sparse activation), making it efficient
    /// - Final predictions combine outputs from selected experts
    ///
    /// Key advantages:
    /// - Increased model capacity without proportional compute cost
    /// - Different experts specialize in different aspects of the problem
    /// - Scalable to very large models
    /// - Efficient through sparse expert activation
    /// </para>
    /// </remarks>
    MixtureOfExperts,

    /// <summary>
    /// A general reinforcement learning model type.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Reinforcement Learning models learn through trial and error by interacting
    /// with an environment. Unlike supervised learning (which learns from labeled examples), RL agents
    /// learn from rewards and punishments. Think of training a dog - you give treats for good behavior
    /// and corrections for bad behavior, and the dog learns what actions lead to rewards.
    ///
    /// RL has achieved remarkable successes:
    /// - Playing games at superhuman level (AlphaGo, Atari games, Dota 2)
    /// - Robotic control (walking, manipulation, assembly)
    /// - Resource optimization (data center cooling, traffic control)
    /// - Recommendation systems and personalization
    /// </para>
    /// </remarks>
    ReinforcementLearning,

    /// <summary>
    /// Proximal Policy Optimization - a state-of-the-art policy gradient RL algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> PPO is one of the most popular RL algorithms today. It learns a policy
    /// (strategy for choosing actions) by making small, safe updates to avoid catastrophic performance drops.
    /// Think of it like making small course corrections while driving rather than sudden jerky turns.
    ///
    /// Used by: OpenAI's ChatGPT (RLHF), robotics systems, game AI
    /// Strengths: Stable, sample-efficient, works well for continuous control
    /// </para>
    /// </remarks>
    PPOAgent,

    /// <summary>
    /// Soft Actor-Critic - an off-policy algorithm combining maximum entropy RL with actor-critic.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SAC encourages exploration by maximizing both reward and "entropy"
    /// (randomness/exploration). It's like learning to play a game while also maintaining variety
    /// in your strategies. This makes it very robust and sample-efficient for continuous control tasks.
    ///
    /// Used by: Robotic manipulation, autonomous vehicles, industrial control
    /// Strengths: Very stable, excellent for continuous actions, sample-efficient
    /// </para>
    /// </remarks>
    SACAgent,

    /// <summary>
    /// Deep Deterministic Policy Gradient - an actor-critic algorithm for continuous action spaces.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DDPG learns policies for continuous control (like adjusting steering angle
    /// or motor torque) rather than discrete choices (like "left" or "right"). It's the RL equivalent
    /// of precision control versus binary decisions.
    ///
    /// Used by: Robotic control, autonomous vehicles, continuous resource allocation
    /// Strengths: Handles continuous actions well, deterministic policies
    /// </para>
    /// </remarks>
    DDPGAgent,

    /// <summary>
    /// Twin Delayed Deep Deterministic Policy Gradient - improved version of DDPG.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TD3 improves DDPG by addressing overestimation bias (being too optimistic
    /// about action values). It uses twin networks and delayed updates for more stable learning.
    /// Think of it as DDPG with better safety checks and more conservative estimates.
    ///
    /// Used by: Advanced robotic control, simulated physics environments
    /// Strengths: More stable than DDPG, reduced overestimation, better final performance
    /// </para>
    /// </remarks>
    TD3Agent,

    /// <summary>
    /// Advantage Actor-Critic - a foundational policy gradient algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A2C learns both a policy (actor) and a value function (critic).
    /// The critic helps the actor learn more efficiently by providing better feedback.
    /// It's like having a coach (critic) give you targeted advice rather than just "good" or "bad".
    ///
    /// Strengths: Foundation for many modern RL algorithms, good for parallel training
    /// </para>
    /// </remarks>
    A2CAgent,

    /// <summary>
    /// Asynchronous Advantage Actor-Critic - parallel version of A2C.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A3C runs multiple agents in parallel, each learning from different
    /// experiences simultaneously. It's like having multiple students learn the same subject
    /// independently, then sharing their knowledge. This speeds up learning significantly.
    ///
    /// Used by: Early DeepMind research, parallel game playing
    /// Strengths: Efficient parallel training, works on CPU without GPUs
    /// </para>
    /// </remarks>
    A3CAgent,

    /// <summary>
    /// Trust Region Policy Optimization - ensures safe, monotonic policy improvements.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TRPO guarantees that each policy update improves performance (monotonic improvement)
    /// by limiting how much the policy can change. It's like taking safe, guaranteed steps forward rather than
    /// potentially risky big leaps. PPO was developed as a simpler alternative to TRPO.
    ///
    /// Strengths: Guaranteed improvement, very stable, excellent for continuous control
    /// </para>
    /// </remarks>
    TRPOAgent,

    /// <summary>
    /// REINFORCE (Monte Carlo Policy Gradient) - the foundational policy gradient algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> REINFORCE is the simplest policy gradient method. It plays full episodes,
    /// then updates the policy to make good actions more likely. Simple but can be slow and high-variance.
    ///
    /// Strengths: Simple to understand and implement, works for any differentiable policy
    /// </para>
    /// </remarks>
    REINFORCEAgent,

    /// <summary>
    /// Conservative Q-Learning - offline RL algorithm that avoids out-of-distribution actions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> CQL is designed for offline RL (learning from fixed datasets without interaction).
    /// It penalizes Q-values for actions not seen in the dataset, preventing the agent from being overconfident
    /// about unfamiliar actions. Useful for learning from historical data.
    ///
    /// Strengths: Safe offline learning, works with fixed datasets, prevents distributional shift
    /// </para>
    /// </remarks>
    CQLAgent,

    /// <summary>
    /// Implicit Q-Learning - another offline RL approach using expectile regression.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> IQL avoids explicitly computing policy constraints, making it simpler and
    /// more stable than some other offline RL methods. It learns Q-values and policies separately,
    /// which can be more robust.
    ///
    /// Strengths: Simple, stable, good offline RL performance
    /// </para>
    /// </remarks>
    IQLAgent,

    /// <summary>
    /// Decision Transformer - treats RL as a sequence modeling problem using transformers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Decision Transformer uses the transformer architecture (from language models)
    /// to predict actions conditioned on desired returns. Instead of learning values or policies directly,
    /// it learns to generate action sequences that lead to target rewards.
    ///
    /// Strengths: Leverages powerful transformer architecture, good offline performance, can condition on returns
    /// </para>
    /// </remarks>
    DecisionTransformer,

    /// <summary>
    /// Multi-Agent DDPG - extends DDPG to multi-agent cooperative/competitive settings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MADDPG allows multiple agents to learn simultaneously in shared environments.
    /// Each agent has its own policy but can observe others during training. Used for cooperative tasks
    /// (team coordination) or competitive tasks (games, negotiations).
    ///
    /// Strengths: Handles multi-agent scenarios, centralized training with decentralized execution
    /// </para>
    /// </remarks>
    MADDPGAgent,

    /// <summary>
    /// QMIX - value-based multi-agent RL that factorizes joint action-values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> QMIX learns how to coordinate multiple agents by factorizing the joint
    /// Q-function into individual agent Q-functions. It's particularly good for cooperative multi-agent
    /// tasks where agents need to work together.
    ///
    /// Strengths: Efficient multi-agent coordination, monotonic value factorization
    /// </para>
    /// </remarks>
    QMIXAgent,

    /// <summary>
    /// Dreamer - model-based RL that learns a world model and plans in latent space.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dreamer learns a model of the environment (how the world works), then
    /// "dreams" about possible futures to plan actions. This allows learning from imagined experiences,
    /// making it very sample-efficient.
    ///
    /// Strengths: Very sample-efficient, learns world models, can plan ahead
    /// </para>
    /// </remarks>
    DreamerAgent,

    /// <summary>
    /// MuZero - combines tree search with learned models, mastering games without knowing rules.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MuZero (from DeepMind) learns to play games at superhuman levels without
    /// being told the rules. It learns a model of the game dynamics and uses tree search (like AlphaZero)
    /// to plan. Famous for mastering Chess, Go, Shogi, and Atari.
    ///
    /// Strengths: State-of-the-art game playing, model-based planning, no need for known dynamics
    /// </para>
    /// </remarks>
    MuZeroAgent,

    /// <summary>
    /// World Models - learns compressed spatial and temporal representations for model-based RL.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> World Models learns a compact representation of the environment and trains
    /// agents entirely inside this learned "world model". It can train much faster by learning in
    /// simulation rather than real environments.
    ///
    /// Strengths: Fast training in learned models, good for visual environments, interpretable latent space
    /// </para>
    /// </remarks>
    WorldModelsAgent,

    /// <summary>
    /// A model trained through knowledge distillation - compressing a larger teacher model into a smaller student.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Knowledge Distillation is like having a student learn from an expert teacher.
    /// The "teacher" is a large, accurate model, and the "student" is a smaller, faster model that learns
    /// to mimic the teacher's behavior while being much more efficient to deploy.
    ///
    /// Real-world analogy: An expert chef (teacher) trains an apprentice (student). The apprentice learns
    /// not just the recipes (hard labels), but also the chef's intuitions, techniques, and reasoning process
    /// (soft targets). This deeper knowledge transfer helps the apprentice become highly skilled.
    ///
    /// How it works:
    /// - Teacher model provides "soft" predictions (probabilities) that reveal relationships between classes
    /// - Student learns from both soft predictions and true labels
    /// - Result: Student model that's 40-90% smaller but retains 90-97% of teacher's accuracy
    ///
    /// Key benefits:
    /// - **Model Compression**: Deploy on mobile, edge devices, browsers
    /// - **Faster Inference**: 2-10x speedup with minimal accuracy loss
    /// - **Lower Costs**: Reduced compute and memory requirements
    /// - **Better Calibration**: Improved confidence estimates
    ///
    /// Success stories:
    /// - DistilBERT: 40% smaller than BERT, 97% performance, 60% faster
    /// - MobileNet: Distilled from ResNet, runs on smartphones
    /// - TinyBERT: 7.5x smaller, suitable for edge deployment
    ///
    /// Use ConfigureKnowledgeDistillation() on AiModelBuilder to enable this technique.
    /// </para>
    /// </remarks>
    KnowledgeDistillation,

    /// <summary>
    /// PointNet neural network for direct point cloud processing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> PointNet processes 3D point clouds directly without converting
    /// to voxels or meshes. It learns global features from unordered point sets through
    /// symmetric functions (max pooling) that ensure permutation invariance.
    ///
    /// PointNet is useful for:
    /// - 3D object classification from LiDAR or depth sensors
    /// - Point cloud segmentation
    /// - 3D shape recognition
    ///
    /// Strengths: Simple, efficient, handles raw point clouds directly
    /// </para>
    /// </remarks>
    PointNet,

    /// <summary>
    /// PointNet++ neural network with hierarchical point set learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> PointNet++ extends PointNet by learning local features at multiple scales.
    /// It uses farthest point sampling and ball query to create hierarchical point cloud representations,
    /// capturing both fine details and global structure.
    ///
    /// PointNet++ is useful for:
    /// - High-accuracy 3D classification and segmentation
    /// - Scenes with varying point densities
    /// - Applications requiring local geometric feature learning
    ///
    /// Strengths: Better local feature learning, multi-scale processing, state-of-the-art accuracy
    /// </para>
    /// </remarks>
    PointNetPlusPlus,

    /// <summary>
    /// Dynamic Graph CNN for point cloud processing with dynamic edge convolutions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DGCNN constructs graphs dynamically in feature space and applies
    /// edge convolutions. Unlike fixed graph methods, it recomputes nearest neighbors after
    /// each layer, allowing the network to learn better representations.
    ///
    /// DGCNN is useful for:
    /// - Point cloud classification with high accuracy
    /// - Part segmentation
    /// - 3D shape analysis
    ///
    /// Strengths: Dynamic graph construction, captures both local and global features
    /// </para>
    /// </remarks>
    DGCNN,

    /// <summary>
    /// Neural Radiance Fields for novel view synthesis.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> NeRF learns a continuous 3D representation of a scene from 2D images.
    /// It represents scenes as a function mapping 3D coordinates and viewing direction to color
    /// and density, enabling photorealistic novel view synthesis.
    ///
    /// NeRF is useful for:
    /// - Novel view synthesis from photos
    /// - 3D reconstruction
    /// - Virtual reality content creation
    ///
    /// Strengths: High-quality rendering, continuous representation, view-dependent effects
    /// </para>
    /// </remarks>
    NeRF,

    /// <summary>
    /// Instant Neural Graphics Primitives with multiresolution hash encoding.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Instant-NGP dramatically accelerates NeRF training using multiresolution
    /// hash encoding. Instead of slow positional encoding, it uses learned hash tables at multiple
    /// resolutions, enabling real-time training and rendering.
    ///
    /// Instant-NGP is useful for:
    /// - Real-time 3D reconstruction
    /// - Fast NeRF training (seconds instead of hours)
    /// - Interactive 3D scene editing
    ///
    /// Strengths: 1000x faster than vanilla NeRF, compact representation, real-time rendering
    /// </para>
    /// </remarks>
    InstantNGP,

    /// <summary>
    /// 3D Gaussian Splatting for real-time radiance field rendering.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> 3D Gaussian Splatting represents scenes as millions of 3D Gaussians
    /// that can be efficiently rasterized. Unlike NeRF's ray marching, it uses tile-based
    /// rasterization for real-time rendering at high quality.
    ///
    /// Gaussian Splatting is useful for:
    /// - Real-time novel view synthesis
    /// - High-quality 3D reconstruction
    /// - AR/VR applications requiring fast rendering
    ///
    /// Strengths: Real-time rendering (100+ FPS), high quality, explicit 3D representation
    /// </para>
    /// </remarks>
    GaussianSplatting,

    /// <summary>
    /// DiffusionNet for learning on 3D surfaces using diffusion-based message passing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DiffusionNet processes 3D meshes using heat diffusion as a building block.
    /// It learns features by simulating how heat would spread across the surface, which naturally
    /// captures surface geometry and is robust to mesh discretization.
    ///
    /// DiffusionNet is useful for:
    /// - Mesh segmentation
    /// - Shape correspondence
    /// - Surface-based learning tasks
    ///
    /// Strengths: Robust to mesh quality, captures intrinsic geometry, state-of-the-art on surfaces
    /// </para>
    /// </remarks>
    DiffusionNet,

    // ==================== Classification Models ====================

    /// <summary>
    /// Gaussian Naive Bayes classifier for continuous features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Gaussian Naive Bayes assumes features follow a Gaussian (normal) distribution.
    /// It's fast, simple, and works well when features are continuous measurements like height, weight, or temperature.
    /// The "naive" assumption is that all features are independent of each other given the class.
    /// </para>
    /// </remarks>
    GaussianNaiveBayes,

    /// <summary>
    /// Multinomial Naive Bayes classifier for discrete count features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multinomial Naive Bayes is designed for count data, like word frequencies in text documents.
    /// It's the go-to algorithm for text classification tasks like spam detection, sentiment analysis, and document categorization.
    /// Each feature represents how many times something occurs (e.g., how many times each word appears).
    /// </para>
    /// </remarks>
    MultinomialNaiveBayes,

    /// <summary>
    /// Bernoulli Naive Bayes classifier for binary features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bernoulli Naive Bayes works with binary (yes/no) features. Instead of counting how many times
    /// a word appears, it only cares whether the word appears at all. This is useful for text classification when
    /// you want to focus on the presence or absence of features rather than their frequency.
    /// </para>
    /// </remarks>
    BernoulliNaiveBayes,

    /// <summary>
    /// Complement Naive Bayes classifier designed for imbalanced datasets.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Complement Naive Bayes is a variation designed to work better with imbalanced datasets
    /// where some classes have many more examples than others. Instead of modeling each class directly,
    /// it models the "complement" (everything except that class), which can provide more stable estimates.
    /// </para>
    /// </remarks>
    ComplementNaiveBayes,

    /// <summary>
    /// Categorical Naive Bayes classifier for categorical features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Categorical Naive Bayes handles features that have distinct categories (like color: red/blue/green,
    /// or size: small/medium/large). Unlike Multinomial NB which counts occurrences, Categorical NB directly models
    /// the probability of each category appearing.
    /// </para>
    /// </remarks>
    CategoricalNaiveBayes,

    /// <summary>
    /// Support Vector Classifier using the kernel trick for non-linear boundaries.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Support Vector Classifier (SVC) finds the best boundary between classes by maximizing
    /// the margin (distance from the boundary to the nearest points). Using kernels, it can learn complex,
    /// non-linear decision boundaries. It's powerful but can be slow on large datasets.
    /// </para>
    /// </remarks>
    SupportVectorClassifier,

    /// <summary>
    /// Linear Support Vector Classifier optimized for large-scale data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Linear SVC is a faster version of SVC that only uses linear boundaries (straight lines/planes).
    /// While less flexible than kernel SVC, it's much faster and works well when you have many features
    /// or when a linear boundary is sufficient (which is often the case for high-dimensional data).
    /// </para>
    /// </remarks>
    LinearSupportVectorClassifier,

    /// <summary>
    /// Nu-Support Vector Classifier with nu parameter for margin control.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Nu-SVC is similar to regular SVC but uses a different parameter (nu) that has a more
    /// intuitive interpretation. The nu parameter bounds the fraction of support vectors and training errors,
    /// making it easier to tune for desired model complexity.
    /// </para>
    /// </remarks>
    NuSupportVectorClassifier,

    /// <summary>
    /// K-Nearest Neighbors classifier based on similarity to training examples.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> K-Nearest Neighbors (KNN) classifies new samples by looking at the k most similar
    /// training examples and taking a majority vote. It's simple, intuitive, and makes no assumptions
    /// about the data distribution. The downside is that it can be slow for large datasets since it
    /// needs to compare against all training examples.
    /// </para>
    /// </remarks>
    KNeighborsClassifier,

    /// <summary>
    /// Decision Tree classifier using recursive partitioning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Decision Tree Classifier works like a flowchart of yes/no questions to classify data.
    /// It's easy to understand and interpret (you can visualize the tree), handles both numeric and categorical
    /// features, and requires minimal data preprocessing. However, single trees can overfit and are sensitive
    /// to small changes in the data.
    /// </para>
    /// </remarks>
    DecisionTreeClassifier,

    /// <summary>
    /// Random Forest classifier using ensemble of decision trees.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Random Forest combines many decision trees and lets them vote on the final prediction.
    /// By training each tree on a random subset of data and features, it reduces overfitting and improves
    /// accuracy. It's one of the most reliable "out of the box" classifiers and works well on many problems.
    /// </para>
    /// </remarks>
    RandomForestClassifier,

    /// <summary>
    /// Gradient Boosting classifier that builds trees sequentially.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Gradient Boosting builds trees one at a time, with each new tree correcting errors
    /// made by previous trees. This sequential approach often achieves higher accuracy than Random Forest
    /// but requires more careful tuning and can be slower to train.
    /// </para>
    /// </remarks>
    GradientBoostingClassifier,

    /// <summary>
    /// AdaBoost classifier that focuses on hard examples.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> AdaBoost (Adaptive Boosting) trains a sequence of weak classifiers, giving more weight
    /// to examples that previous classifiers got wrong. This forces the ensemble to focus on the hardest cases.
    /// It's effective and less prone to overfitting than some other boosting methods.
    /// </para>
    /// </remarks>
    AdaBoostClassifier,

    /// <summary>
    /// Extra Trees (Extremely Randomized Trees) classifier with added randomness.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Extra Trees is similar to Random Forest but adds even more randomness when building trees.
    /// Instead of finding the best split for each feature, it chooses splits randomly. This makes it faster
    /// to train and can reduce overfitting, though it may need more trees to achieve the same accuracy.
    /// </para>
    /// </remarks>
    ExtraTreesClassifier,

    /// <summary>
    /// Bagging classifier that combines predictions from multiple base estimators.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bagging (Bootstrap Aggregating) trains multiple copies of any base classifier on
    /// random subsets of the data, then combines their predictions. It's a general technique that can
    /// improve stability and accuracy of any classifier that tends to overfit.
    /// </para>
    /// </remarks>
    BaggingClassifier,

    /// <summary>
    /// Stochastic Gradient Descent classifier with various loss functions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SGD Classifier uses stochastic gradient descent to train linear classifiers efficiently.
    /// It can handle very large datasets that don't fit in memory by processing one example at a time.
    /// Different loss functions allow it to mimic logistic regression, SVM, or other linear models.
    /// </para>
    /// </remarks>
    SGDClassifier,

    /// <summary>
    /// Passive Aggressive classifier for large-scale online learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Passive Aggressive classifiers update aggressively when they make a mistake but stay
    /// "passive" (don't change) when they predict correctly. This makes them suitable for online learning
    /// where you receive data one example at a time.
    /// </para>
    /// </remarks>
    PassiveAggressiveClassifier,

    /// <summary>
    /// Perceptron classifier - the simplest neural network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Perceptron is the simplest possible neural network - just one layer that learns
    /// a linear boundary. While limited to linearly separable problems, it's historically important and
    /// forms the building block for more complex neural networks.
    /// </para>
    /// </remarks>
    PerceptronClassifier,

    /// <summary>
    /// Ridge classifier that uses L2 regularization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Ridge Classifier converts the classification problem to a regression problem and uses
    /// L2 regularization (like Ridge Regression). It's fast and works well for multi-class problems,
    /// especially when the number of classes is large.
    /// </para>
    /// </remarks>
    RidgeClassifier,

    /// <summary>
    /// Linear Discriminant Analysis classifier that finds linear combinations of features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LDA finds the directions in the feature space that best separate the classes.
    /// It's both a classifier and a dimensionality reduction technique. It assumes features follow
    /// a Gaussian distribution with equal covariance for all classes, making it fast and effective
    /// when these assumptions hold.
    /// </para>
    /// </remarks>
    LinearDiscriminantAnalysis,

    /// <summary>
    /// Quadratic Discriminant Analysis classifier with class-specific covariance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> QDA is like LDA but allows each class to have its own covariance matrix.
    /// This creates quadratic (curved) decision boundaries instead of linear ones. It's more flexible
    /// than LDA but requires more data to estimate the additional parameters.
    /// </para>
    /// </remarks>
    QuadraticDiscriminantAnalysis,

    /// <summary>
    /// One-vs-Rest classifier for multi-class classification.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> One-vs-Rest (also called One-vs-All) trains one binary classifier per class,
    /// where each classifier distinguishes one class from all others. The class with the highest
    /// confidence wins. It's simple and works with any binary classifier.
    /// </para>
    /// </remarks>
    OneVsRestClassifier,

    /// <summary>
    /// One-vs-One classifier for multi-class classification.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> One-vs-One trains one classifier for each pair of classes. For N classes,
    /// this means N*(N-1)/2 classifiers. Each classifier votes, and the class with the most votes wins.
    /// It can be more accurate than One-vs-Rest but slower with many classes.
    /// </para>
    /// </remarks>
    OneVsOneClassifier,

    /// <summary>
    /// Classifier Chain for multi-label classification.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Classifier Chain handles multi-label problems (where items can have multiple labels)
    /// by training classifiers in sequence, each using predictions from previous classifiers as features.
    /// This captures dependencies between labels, like "if it's labeled 'beach', it's more likely to also be labeled 'summer'".
    /// </para>
    /// </remarks>
    ClassifierChain,

    /// <summary>
    /// Multi-output classifier for predicting multiple targets independently.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multi-Output Classifier trains one classifier per target variable, treating each
    /// target independently. It's useful when you need to predict multiple related but independent outputs
    /// from the same input features.
    /// </para>
    /// </remarks>
    MultiOutputClassifier,

    /// <summary>
    /// Stacking classifier that combines multiple classifiers with a meta-learner.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Stacking trains multiple base classifiers and then trains a "meta-learner" to combine
    /// their predictions. The meta-learner learns which classifiers to trust for different types of inputs.
    /// It often achieves better performance than any single classifier.
    /// </para>
    /// </remarks>
    StackingClassifier,

    /// <summary>
    /// Voting classifier that combines predictions through voting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Voting Classifier combines multiple classifiers by letting them vote. Hard voting
    /// counts class predictions (majority wins), while soft voting averages predicted probabilities.
    /// It's a simple but effective way to combine diverse classifiers.
    /// </para>
    /// </remarks>
    VotingClassifier,

    // ==================== Semi-Supervised Learning ====================

    /// <summary>
    /// Self-Training classifier that iteratively labels unlabeled data with high confidence predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Self-Training is like a student who learns from a small set of labeled examples,
    /// then uses what they learned to make guesses about unlabeled examples. When they're confident about
    /// a guess, they treat it as a real example and continue learning. This is useful when labeling data
    /// is expensive but you have lots of unlabeled data.
    /// </para>
    /// </remarks>
    SelfTrainingClassifier,

    /// <summary>
    /// Label Propagation classifier that spreads labels through a similarity graph.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Label Propagation builds a graph where similar data points are connected.
    /// Labels from the few labeled examples then spread through the graph to nearby unlabeled points,
    /// like colors spreading through connected water droplets. Points that are similar to labeled
    /// examples get the same labels.
    /// </para>
    /// </remarks>
    LabelPropagation,

    /// <summary>
    /// Label Spreading classifier similar to Label Propagation but with normalization for stability.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Label Spreading is like Label Propagation but more careful. It normalizes
    /// the connections in the graph so that labels spread more evenly and the process is more stable.
    /// This helps when your data has clusters of different sizes or densities.
    /// </para>
    /// </remarks>
    LabelSpreading,

    // ==================== Multi-Label Classification ====================

    /// <summary>
    /// Binary Relevance classifier that trains one binary classifier per label independently.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Binary Relevance is the simplest multi-label approach. It trains a separate
    /// binary classifier for each label (e.g., "is this a comedy?" "is this an action movie?"). Each classifier
    /// makes an independent yes/no decision. Simple but ignores label correlations.
    /// </para>
    /// </remarks>
    BinaryRelevanceClassifier,

    /// <summary>
    /// Label Powerset classifier that treats each unique label combination as a separate class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Label Powerset converts multi-label into multi-class by treating each unique
    /// combination of labels as its own class. For example, "action+comedy" becomes one class, "drama+romance"
    /// becomes another. Captures label correlations perfectly but may have many classes with few examples.
    /// </para>
    /// </remarks>
    LabelPowersetClassifier,

    // ==================== Finance Models ====================

    /// <summary>
    /// PatchTST time series forecasting model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> PatchTST splits a time series into small patches and uses
    /// transformer attention to forecast future values.
    /// </para>
    /// </remarks>
    PatchTST,

    /// <summary>
    /// Inverted Transformer (iTransformer) for time series forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> iTransformer treats each variable as a token so the model
    /// learns relationships across features while forecasting.
    /// </para>
    /// </remarks>
    ITransformer,

    /// <summary>
    /// DeepAR probabilistic forecasting model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DeepAR predicts future values by learning autoregressive
    /// patterns and uncertainty from historical data.
    /// </para>
    /// </remarks>
    DeepAR,

    /// <summary>
    /// N-BEATS neural basis expansion model for forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> N-BEATS breaks a time series into trend and seasonal parts
    /// to make accurate forecasts.
    /// </para>
    /// </remarks>
    NBEATS,

    /// <summary>
    /// Temporal Fusion Transformer (TFT) forecasting model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TFT combines attention with gating to forecast multiple
    /// steps ahead and provide interpretable signals.
    /// </para>
    /// </remarks>
    TFT,

    /// <summary>
    /// Neural Value at Risk model for financial risk estimation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> NeuralVaR estimates how much you could lose in a worst-case
    /// scenario given a confidence level.
    /// </para>
    /// </remarks>
    NeuralVaR,

    /// <summary>
    /// TabNet model for structured financial risk prediction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TabNet uses attentive feature selection to focus on the
    /// most important columns in tabular data.
    /// </para>
    /// </remarks>
    TabNet,

    /// <summary>
    /// TabTransformer model for tabular financial data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TabTransformer applies transformer attention to columns
    /// in a table to model feature interactions.
    /// </para>
    /// </remarks>
    TabTransformer,

    // ==================== Clustering Models ====================

    /// <summary>
    /// A clustering algorithm that groups similar data points together.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Clustering algorithms find natural groups in data without being told
    /// what groups to look for. Unlike classification (which learns from labeled examples),
    /// clustering discovers patterns on its own.
    ///
    /// Common clustering algorithms:
    /// - K-Means: Divides data into k spherical clusters
    /// - DBSCAN: Finds clusters of arbitrary shape based on density
    /// - Hierarchical: Creates a tree of nested clusters
    /// - GMM: Models clusters as Gaussian distributions
    ///
    /// Applications:
    /// - Customer segmentation
    /// - Image segmentation
    /// - Anomaly detection
    /// - Document grouping
    /// </para>
    /// </remarks>
    Clustering,

    // ==================== Video AI Models ====================

    /// <summary>
    /// Video Super-Resolution model for enhancing video resolution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Video Super-Resolution models upscale low-resolution videos to higher resolution
    /// while maintaining temporal consistency across frames. Unlike single-image super-resolution, video models
    /// use information from multiple frames to produce sharper and more consistent results.
    ///
    /// Examples: BasicVSR++, VRT, VideoGigaGAN
    /// </para>
    /// </remarks>
    VideoSuperResolution,

    /// <summary>
    /// Frame Interpolation model for generating intermediate frames.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Frame Interpolation models generate new frames between existing ones to increase
    /// video frame rate (e.g., 30fps to 60fps) or create slow-motion effects. They estimate motion between
    /// frames and synthesize intermediate frames.
    ///
    /// Examples: RIFE, FILM, VFIMamba
    /// </para>
    /// </remarks>
    FrameInterpolation,

    /// <summary>
    /// Optical Flow model for estimating motion between frames.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Optical Flow models estimate the apparent motion of pixels between consecutive
    /// video frames. This motion information is essential for many video processing tasks like frame
    /// interpolation, video stabilization, and action recognition.
    ///
    /// Examples: RAFT, SEA-RAFT, GMFlow
    /// </para>
    /// </remarks>
    OpticalFlow,

    /// <summary>
    /// Video Depth Estimation model for predicting depth from video.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Video Depth Estimation models predict how far each pixel is from the camera
    /// using temporal information across frames. This enables 3D scene understanding, autonomous driving,
    /// and AR/VR applications.
    ///
    /// Examples: Depth Anything V2, Video Depth Anything
    /// </para>
    /// </remarks>
    VideoDepthEstimation,

    /// <summary>
    /// Video Object Segmentation model for tracking and segmenting objects.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Video Object Segmentation models identify and track specific objects across
    /// video frames, creating pixel-accurate masks that follow the object through time.
    ///
    /// Examples: SAM2, Cutie, XMem
    /// </para>
    /// </remarks>
    VideoObjectSegmentation,

    /// <summary>
    /// Video Inpainting model for removing and filling content.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Video Inpainting models remove unwanted objects from videos and fill in the
    /// removed areas with realistic content that maintains temporal consistency.
    ///
    /// Examples: ProPainter, E2FGVI
    /// </para>
    /// </remarks>
    VideoInpainting,

    /// <summary>
    /// Video Action Recognition model for understanding activities.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Video Action Recognition models identify what actions or activities are
    /// happening in a video, such as "running", "cooking", or "playing basketball".
    ///
    /// Examples: VideoMAE, InternVideo2
    /// </para>
    /// </remarks>
    VideoActionRecognition,

    /// <summary>
    /// Text-to-Video generation model for creating videos from text.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Text-to-Video models generate video content from text descriptions,
    /// enabling creative content generation with natural language prompts.
    ///
    /// Examples: OpenSora, Stable Video Diffusion
    /// </para>
    /// </remarks>
    TextToVideo,

    /// <summary>
    /// Object Tracking model for tracking objects across video frames.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Object Tracking models follow objects as they move through a video,
    /// maintaining consistent identity even when objects are temporarily occluded.
    ///
    /// Examples: ByteTrack, SORT, DeepSORT
    /// </para>
    /// </remarks>
    ObjectTracking,

    /// <summary>
    /// Video Matting model for extracting foreground with alpha matte.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Video Matting models separate foreground subjects from backgrounds
    /// with soft alpha mattes that preserve fine details like hair and semi-transparent regions.
    ///
    /// Examples: RVM (Robust Video Matting), MODNet
    /// </para>
    /// </remarks>
    VideoMatting,

    /// <summary>
    /// Video Stabilization model for removing camera shake.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Video Stabilization models smooth out shaky video by estimating
    /// and compensating for camera motion, producing steady footage.
    ///
    /// Examples: DIFRINT, DUT
    /// </para>
    /// </remarks>
    VideoStabilization,

    /// <summary>
    /// Monocular Depth Estimation model for predicting depth from single images.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Depth Estimation models predict how far each pixel is from the camera
    /// using only a single image, enabling 3D understanding without stereo cameras.
    ///
    /// Examples: MiDaS, Depth Anything, ZoeDepth
    /// </para>
    /// </remarks>
    DepthEstimation,

    /// <summary>
    /// Video Denoising model for removing noise from video.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Video Denoising models remove noise (graininess) from video while
    /// preserving details and maintaining temporal consistency across frames.
    ///
    /// Examples: FastDVDNet, VRT
    /// </para>
    /// </remarks>
    VideoDenoising,

    // ==================== Imbalanced Learning ====================

    /// <summary>
    /// SMOTE (Synthetic Minority Over-sampling Technique) for handling imbalanced data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SMOTE creates synthetic minority class samples by interpolating
    /// between existing minority samples. This helps balance datasets where one class is
    /// much smaller than others, improving classifier performance on minority classes.
    /// </para>
    /// </remarks>
    SMOTE,

    /// <summary>
    /// ADASYN (Adaptive Synthetic Sampling) for handling imbalanced data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ADASYN is like SMOTE but smarter - it creates more synthetic samples
    /// in regions where minority samples are harder to learn (surrounded by majority samples).
    /// This focuses oversampling effort where it's most needed.
    /// </para>
    /// </remarks>
    ADASYN,

    /// <summary>
    /// Borderline-SMOTE for focused oversampling near decision boundaries.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Borderline-SMOTE only creates synthetic samples from minority samples
    /// that are near the decision boundary (borderline cases). This is where the classifier
    /// needs the most help distinguishing between classes.
    /// </para>
    /// </remarks>
    BorderlineSMOTE,

    /// <summary>
    /// Random Oversampling that duplicates minority samples.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Random Oversampling simply duplicates existing minority samples
    /// until classes are balanced. It's the simplest approach but can lead to overfitting
    /// since it creates exact copies rather than new synthetic samples.
    /// </para>
    /// </remarks>
    RandomOverSampler,

    /// <summary>
    /// Random Undersampling that removes majority samples.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Random Undersampling randomly removes majority class samples
    /// until classes are balanced. It's fast but loses potentially useful information.
    /// Good for very large datasets where data loss is acceptable.
    /// </para>
    /// </remarks>
    RandomUnderSampler,

    /// <summary>
    /// Tomek Links undersampling for cleaning decision boundaries.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Tomek Links removes majority samples that form "Tomek links"
    /// with minority samples - pairs where each is the other's nearest neighbor. This cleans
    /// up ambiguous boundary regions without removing too many samples.
    /// </para>
    /// </remarks>
    TomekLinksUndersampler,

    /// <summary>
    /// Edited Nearest Neighbors (ENN) undersampling for removing noisy samples.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ENN removes majority samples that would be misclassified by
    /// their k nearest neighbors. This removes noisy and borderline samples, creating
    /// a cleaner decision boundary.
    /// </para>
    /// </remarks>
    EditedNearestNeighborsUndersampler,

    /// <summary>
    /// NearMiss undersampling that keeps majority samples near minority samples.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> NearMiss intelligently selects which majority samples to keep
    /// based on their distance to minority samples. It keeps the most informative samples
    /// near the decision boundary rather than removing randomly.
    /// </para>
    /// </remarks>
    NearMissUndersampler,

    /// <summary>
    /// SMOTE + ENN combination (SMOTEENN) for balanced and clean data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SMOTEENN first applies SMOTE to oversample, then applies ENN
    /// to clean up noisy samples from both classes. This combination often produces better
    /// results than either method alone - balanced AND clean data.
    /// </para>
    /// </remarks>
    SMOTEENN,

    /// <summary>
    /// SMOTE + Tomek Links combination (SMOTETomek) for balanced and clean data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SMOTETomek first applies SMOTE to oversample, then removes
    /// Tomek links to clean the boundary. Less aggressive than SMOTEENN but still
    /// improves the decision boundary after oversampling.
    /// </para>
    /// </remarks>
    SMOTETomek,

    // ==================== Survival Analysis ====================

    /// <summary>
    /// Kaplan-Meier estimator for non-parametric survival analysis.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Kaplan-Meier creates a survival curve showing the probability
    /// of surviving beyond each time point. It handles censored data (subjects who haven't
    /// had the event yet) naturally. It's the simplest survival method but doesn't use
    /// patient features - everyone gets the same survival curve.
    /// </para>
    /// </remarks>
    KaplanMeierEstimator,

    /// <summary>
    /// Cox Proportional Hazards model for survival analysis with covariates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cox model is the most widely used survival model. It tells you
    /// how patient features (age, treatment, etc.) affect survival risk. Each feature gets
    /// a "hazard ratio" - a value greater than 1 means higher risk, less than 1 means
    /// lower risk. Unlike Kaplan-Meier, it personalizes survival predictions.
    /// </para>
    /// </remarks>
    CoxProportionalHazards,

    // ===== Causal Inference Models =====

    /// <summary>
    /// Propensity Score Matching for causal effect estimation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> PSM estimates treatment effects by matching treated individuals
    /// to control individuals with similar propensity scores. Think of it as finding "twins"
    /// who were equally likely to receive treatment, then comparing their outcomes.
    /// </para>
    /// </remarks>
    PropensityScoreMatching,

    /// <summary>
    /// Inverse Probability Weighting for causal effect estimation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> IPW weights each observation by the inverse of its probability
    /// of being in its actual treatment group. This creates a "pseudo-population" where
    /// treatment is independent of covariates, like a randomized experiment.
    /// </para>
    /// </remarks>
    InverseProbabilityWeighting,

    /// <summary>
    /// Doubly Robust estimator combining outcome modeling and propensity scores.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The doubly robust estimator combines two approaches: modeling
    /// the outcome and modeling the propensity score. It's "doubly robust" because it gives
    /// correct results if EITHER model is correct, providing protection against misspecification.
    /// </para>
    /// </remarks>
    DoublyRobustEstimator,

    /// <summary>
    /// Causal Forest for heterogeneous treatment effect estimation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Causal Forest is like Random Forest but for treatment effects.
    /// It estimates how treatment effects vary across individuals based on their
    /// characteristics, identifying who benefits most from treatment.
    /// </para>
    /// </remarks>
    CausalForest,

    // ===== Online Learning Models =====

    /// <summary>
    /// Online SGD Classifier for incremental binary classification.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This classifier learns from streaming data one example at a time.
    /// It uses stochastic gradient descent to update a logistic regression model incrementally,
    /// making it suitable for large-scale or real-time learning scenarios.
    /// </para>
    /// </remarks>
    OnlineSGDClassifier,

    /// <summary>
    /// Online SGD Regressor for incremental regression.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Similar to OnlineSGDClassifier but for continuous targets.
    /// Uses SGD to incrementally fit a linear regression model to streaming data.
    /// </para>
    /// </remarks>
    OnlineSGDRegressor,

    /// <summary>
    /// Online Passive-Aggressive classifier for margin-based online learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> PA algorithms aggressively update when mistakes are made but stay
    /// passive when predictions are correct. Good for noisy data and concept drift scenarios.
    /// </para>
    /// </remarks>
    OnlinePassiveAggressiveClassifier
}




