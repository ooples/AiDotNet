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

    /// <summary>
    /// A neural network architecture that uses skip connections to allow training of very deep networks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Residual Neural Networks solve a problem that occurs when neural networks get too deep 
    /// (have too many layers). Normally, adding more layers should make a network smarter, but in practice, very 
    /// deep networks become harder to train. ResNets use "shortcut connections" that let information skip ahead, 
    /// allowing the network to learn which information should take the shortcut and which needs more processing. 
    /// This is like having both an express lane and a local lane on a highway - some information can take the 
    /// fast route while other information needs to stop at every exit.
    /// </para>
    /// </remarks>
    ResidualNeuralNetwork,

    /// <summary>
    /// A type of autoencoder that learns to generate new data similar to the training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Variational Autoencoders learn to compress data (like images) into a compact form and 
    /// then reconstruct it. What makes them special is they learn a smooth, continuous representation where 
    /// similar inputs are close together. This allows you to generate new data by sampling from this space. 
    /// For example, a VAE trained on faces could generate new, realistic faces that don't belong to any real 
    /// person, or you could blend characteristics from different faces by moving through this learned space.
    /// </para>
    /// </remarks>
    VariationalAutoencoder,

    /// <summary>
    /// A neural network architecture that better preserves spatial hierarchies between features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Capsule Networks improve on traditional neural networks by preserving information about 
    /// position and orientation of features. Regular networks might recognize a face if it has eyes, nose, and mouth, 
    /// regardless of their arrangement. Capsule Networks also care about the spatial relationships - eyes above nose, 
    /// nose above mouth. This makes them better at understanding 3D objects from different viewpoints and less likely 
    /// to be fooled by images where the parts are present but arranged incorrectly.
    /// </para>
    /// </remarks>
    CapsuleNetwork,

    /// <summary>
    /// A neural network that leverages quantum computing principles for processing information.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Quantum Neural Networks combine principles from quantum computing with neural networks. 
    /// While traditional computers use bits (0 or 1), quantum computers use qubits that can exist in multiple states 
    /// simultaneously. This allows quantum neural networks to explore many possible solutions at once, potentially 
    /// solving certain complex problems much faster than classical approaches. This is an emerging field at the 
    /// intersection of quantum physics and machine learning.
    /// </para>
    /// </remarks>
    QuantumNeuralNetwork,

    /// <summary>
    /// A type of unsupervised neural network that organizes data based on similarity.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Self-Organizing Maps learn to arrange data so that similar items are placed close together. 
    /// Imagine organizing books on shelves so that similar topics are near each other - that's what SOMs do with data. 
    /// They create a "map" where each location represents a particular pattern, and similar patterns are placed nearby. 
    /// This makes them useful for visualizing high-dimensional data, discovering clusters, and finding relationships 
    /// between different data points.
    /// </para>
    /// </remarks>
    SelfOrganizingMap,

    /// <summary>
    /// A neural network model inspired by how neurons process information in a liquid.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Liquid State Machines are inspired by how signals ripple through a liquid. When you drop 
    /// a stone in water, it creates ripples that spread and interact in complex ways. Similarly, these networks 
    /// process information by letting it "ripple" through a collection of randomly connected neurons. This approach 
    /// is particularly good at processing time-varying inputs like speech or video, where the timing and sequence 
    /// of information matters.
    /// </para>
    /// </remarks>
    LiquidStateMachine,

    /// <summary>
    /// A recurrent neural network that can store and retrieve patterns.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hopfield Networks are designed to store and recall patterns, similar to how human memory 
    /// works. You can train the network to "remember" certain patterns, and then when given a partial or noisy version 
    /// of a pattern, it can recover the complete original pattern. It's like recognizing a song from just hearing a few 
    /// notes, or recognizing a friend's face even when they're partially obscured. These networks are useful for pattern 
    /// completion, noise reduction, and content-addressable memory.
    /// </para>
    /// </remarks>
    HopfieldNetwork,

    /// <summary>
    /// A neural network designed to work with graph-structured data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Graph Neural Networks are specialized for data that has connections between elements, like 
    /// social networks, molecules, or road systems. While traditional neural networks work well with grid-like data 
    /// (images) or sequences (text), GNNs can understand and process the relationships between entities in a graph. 
    /// They learn by passing information along the connections in the graph, allowing each node to gather information 
    /// from its neighbors. This makes them powerful for tasks like predicting interactions between proteins, 
    /// recommending friends on social media, or analyzing traffic patterns.
    /// </para>
    /// </remarks>
    GraphNeuralNetwork,

    /// <summary>
    /// A neural network with a fast, one-pass learning algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Extreme Learning Machines offer a shortcut to training neural networks. Traditional networks 
    /// require many iterations to learn, which can be time-consuming. ELMs randomly assign the connections in the hidden 
    /// layer and only train the output layer, which can be done in one quick step. This is like building a team where you 
    /// randomly assign roles to most members but carefully select the team leaders. While this approach sacrifices some 
    /// accuracy, it can be thousands of times faster to train, making it useful for applications where speed is critical 
    /// or for providing quick initial solutions.
    /// </para>
    /// </remarks>
    ExtremeLearningMachine,

    /// <summary>
    /// A neural network architecture that combines neural networks with external memory systems.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Differentiable Neural Computers combine a neural network with a memory matrix that it can 
    /// read from and write to. This is similar to how your computer has both a processor (CPU) and memory (RAM). The 
    /// neural network learns how to store information in the memory and how to retrieve it when needed. This gives the 
    /// system the ability to remember facts and use them later, making it particularly good at tasks that require 
    /// reasoning with stored information, like answering questions about a story or navigating complex environments.
    /// </para>
    /// </remarks>
    DifferentiableNeuralComputer,

    /// <summary>
    /// A recurrent neural network where the internal connections are fixed and only output connections are trained.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Echo State Networks use a clever shortcut to make recurrent neural networks easier to train. 
    /// They create a large, randomly connected "reservoir" of neurons that transforms inputs in complex ways. Only the 
    /// connections from this reservoir to the output are trained, which is much simpler than training all connections. 
    /// It's like having a complex but fixed system of pipes and valves (the reservoir) that processes water flow in 
    /// intricate ways, and you only need to learn how to read the resulting patterns. This approach works well for 
    /// time series prediction, speech recognition, and other sequential data problems.
    /// </para>
    /// </remarks>
    EchoStateNetwork,

    /// <summary>
    /// A neural network architecture used in reinforcement learning to learn optimal action policies.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Deep Q-Networks learn to make sequences of decisions to maximize a reward. They're used in 
    /// reinforcement learning, where an agent learns by interacting with an environment. The network predicts the 
    /// "value" of taking each possible action in a given situation, allowing the agent to choose the best action. 
    /// It's like learning to play chess by estimating how good each possible move is. DQNs have been used to master 
    /// video games, control robots, and optimize complex systems like data center cooling.
    /// </para>
    /// </remarks>
    DeepQNetwork,

    /// <summary>
    /// A framework consisting of two neural networks that compete with each other to generate realistic data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Generative Adversarial Networks consist of two neural networks competing against each other: 
    /// a generator that creates fake data (like images) and a discriminator that tries to distinguish real data from 
    /// fakes. As they train, the generator gets better at creating convincing fakes, and the discriminator gets better 
    /// at spotting them. It's like a counterfeiter and detective constantly improving to outdo each other. This 
    /// competition drives both to improve until the generator creates data so realistic that it's hard to distinguish 
    /// from real data. GANs have created remarkably realistic images, videos, and even music.
    /// </para>
    /// </remarks>
    GenerativeAdversarialNetwork,

    /// <summary>
    /// A neural network architecture that combines neural networks with external memory access mechanisms.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural Turing Machines combine neural networks with an external memory that they can read 
    /// from and write to in a precise way. This is inspired by how computers use memory and how humans can store and 
    /// retrieve information. The network learns to control where to store information, what to store, and how to retrieve 
    /// it later. This gives it the ability to learn algorithms - step-by-step procedures for solving problems - rather 
    /// than just pattern recognition. NTMs can learn tasks like copying sequences, sorting numbers, or even simple 
    /// programming-like tasks.
    /// </para>
    /// </remarks>
    NeuralTuringMachine,

    /// <summary>
    /// A method for evolving neural networks through genetic algorithms.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> NEAT (NeuroEvolution of Augmenting Topologies) uses principles from evolution to design 
    /// neural networks. Instead of manually designing the network structure, NEAT starts with simple networks and 
    /// gradually makes them more complex through a process similar to natural selection. Networks that perform better 
    /// are more likely to "reproduce" and pass on their characteristics. Over many generations, this can discover 
    /// innovative network designs that human engineers might not think of. NEAT is particularly useful for reinforcement 
    /// learning problems like game playing or robot control.
    /// </para>
    /// </remarks>
    NEAT,

    /// <summary>
    /// A neural network architecture designed to store and retrieve information from an external memory.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Memory Networks are designed to improve how neural networks handle information that needs
    /// to be remembered over time. They combine neural networks with an external memory component that can be read
    /// from and written to. This allows the network to store facts, context, or other information and retrieve it
    /// when needed. Memory Networks are particularly useful for tasks like answering questions about a story, where
    /// the network needs to remember details from earlier parts of the text to answer questions correctly.
    /// </para>
    /// </remarks>
    MemoryNetwork,

    /// <summary>
    /// A specialized recurrent neural network architecture for processing sequential data with long-term dependencies.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LSTM Neural Networks are a special type of recurrent neural network designed to remember
    /// information for long periods of time. Regular neural networks struggle to connect information from far apart
    /// in a sequence. LSTMs solve this with a "cell state" that acts like a conveyor belt of information, allowing
    /// relevant information to flow through many steps while filtering out irrelevant details. This makes them
    /// excellent for tasks like language translation, speech recognition, and time series prediction where context
    /// from much earlier can be important.
    /// </para>
    /// </remarks>
    LSTMNeuralNetwork,

    /// <summary>
    /// A neural network inspired by the structure and function of the neocortex in the human brain.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hierarchical Temporal Memory Networks are inspired by how the human neocortex processes
    /// information. They learn patterns in a hierarchical way, with higher levels learning more abstract patterns
    /// based on the patterns detected at lower levels. HTM Networks are particularly good at finding patterns in
    /// time-based data and can learn continuously without forgetting previous patterns. They're designed to
    /// recognize anomalies, make predictions, and understand sequences in data, similar to how our brains process
    /// sensory information.
    /// </para>
    /// </remarks>
    HTMNetwork,

    /// <summary>
    /// A general-purpose computational model inspired by the structure and function of biological neural networks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural Networks are computing systems inspired by the human brain. They consist of
    /// interconnected "neurons" that process information and learn patterns from data. Each connection has a
    /// "weight" that strengthens or weakens the signal between neurons, and these weights are adjusted during
    /// training to improve performance. Neural networks can learn to recognize patterns, classify data, make
    /// predictions, and solve complex problems without being explicitly programmed with rules. They're the
    /// foundation of many modern AI systems.
    /// </para>
    /// </remarks>
    NeuralNetwork,

    /// <summary>
    /// A neural network architecture for 3D scene understanding and representation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Occupancy Networks represent 3D objects or scenes by learning which points in space
    /// are inside or outside an object. This is like having a detailed 3D model where you can check any point
    /// in space to see if it's part of the object. Unlike traditional 3D representations that use fixed grids
    /// or point clouds, Occupancy Networks can represent shapes at any resolution. They're useful for 3D
    /// reconstruction, shape generation, and scene understanding in applications like robotics, virtual reality,
    /// and computer-aided design.
    /// </para>
    /// </remarks>
    OccupancyNetwork,

    /// <summary>
    /// A stochastic neural network that can learn a probability distribution over its inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Restricted Boltzmann Machines learn patterns in data by trying to recreate the input
    /// from scratch. They consist of two layers of neurons - visible and hidden - with connections between layers
    /// but not within layers (that's the "restricted" part). RBMs can learn to recognize patterns and extract
    /// features without supervision. They're like a simplified version of Deep Boltzmann Machines and are often
    /// used as building blocks for deeper networks. RBMs have been applied to problems like recommendation systems,
    /// feature extraction, and dimensionality reduction.
    /// </para>
    /// </remarks>
    RestrictedBoltzmannMachine,

    /// <summary>
    /// A neural network model that mimics the behavior of biological neurons more closely than traditional artificial neurons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Spiking Neural Networks more closely mimic how real neurons in the brain communicate.
    /// While traditional neural networks use continuous values, spiking networks use discrete "spikes" or pulses
    /// of activity. Neurons accumulate input until they reach a threshold, then fire a spike. This approach is
    /// more biologically realistic and potentially more energy-efficient. SNNs are particularly interesting for
    /// neuromorphic computing (brain-inspired hardware), real-time processing of sensory data, and understanding
    /// how the brain processes information.
    /// </para>
    /// </remarks>
    SpikingNeuralNetwork,

    /// <summary>
    /// A basic neural network architecture where information flows in one direction from input to output.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Feed Forward Networks are the simplest type of neural network, where information
    /// travels in only one direction - from input to output, with no loops or cycles. Data passes through
    /// multiple layers of neurons, with each layer processing the information and passing it to the next.
    /// This straightforward architecture makes them easier to understand and train compared to more complex
    /// networks. They're effective for many tasks like classification, regression, and pattern recognition
    /// where the input is fixed and doesn't depend on previous inputs or states.
    /// </para>
    /// </remarks>
    FeedForwardNetwork,

    /// <summary>
    /// A model that represents relationships as straight lines.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Linear models represent relationships as straight lines (or their higher-dimensional
    /// equivalents). They assume that the output changes proportionally with changes in the input. For example,
    /// if doubling an input doubles the output, that's a linear relationship. These models are simple,
    /// interpretable, and computationally efficient. While they can't capture complex curved relationships on
    /// their own, they form the foundation of many more sophisticated models and can be surprisingly effective
    /// for many real-world problems.
    /// </para>
    /// </remarks>
    Linear,

    /// <summary>
    /// A model that uses polynomial functions to represent non-linear relationships.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Polynomial models use curved lines (like parabolas) instead of straight lines to
    /// represent relationships in data. They can capture more complex patterns where the rate of change varies.
    /// For example, a plant's growth might accelerate then slow down over time, forming an S-curve. Polynomial
    /// models can represent these curved relationships by including squared terms (x²), cubed terms (x³), and
    /// so on. They're more flexible than linear models but need to be used carefully to avoid overfitting.
    /// </para>
    /// </remarks>
    Polynomial,

    /// <summary>
    /// A model that discovers mathematical expressions to describe relationships in data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Symbolic models try to find actual mathematical formulas that explain your data.
    /// Instead of just fitting parameters to a pre-defined equation, they search for the equation itself.
    /// For example, they might discover that your data follows "y = sin(x) + x²" rather than just giving
    /// you numbers. This provides insights into the underlying relationships and can be more interpretable
    /// than black-box models. Symbolic models are particularly useful when you want to understand the
    /// mathematical laws governing a system.
    /// </para>
    /// </remarks>
    Symbolic,

    /// <summary>
    /// A model that uses decision trees or tree-based ensembles for prediction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Tree-Based models make predictions by following a series of yes/no questions,
    /// similar to a flowchart. They split the data based on features, creating a tree-like structure.
    /// For example, to predict house prices: "Is the area > 2000 sq ft? If yes, is it in neighborhood A?
    /// If no, is it older than 20 years?" and so on. Tree-based models include single decision trees and
    /// powerful ensembles like Random Forests and Gradient Boosted Trees. They're flexible, handle different
    /// types of data well, and can capture complex interactions between features.
    /// </para>
    /// </remarks>
    TreeBased,

    /// <summary>
    /// An extension of ARIMA that includes external variables in the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ARIMAX Models extend ARIMA by including external factors that might influence
    /// your time series. For example, when forecasting ice cream sales, an ARIMA model might only look at
    /// past sales patterns, but an ARIMAX model could also consider temperature, holidays, or marketing
    /// campaigns. This often improves predictions by incorporating known influences. It's like forecasting
    /// traffic not just based on historical patterns but also considering if there's a major concert or
    /// sporting event happening.
    /// </para>
    /// </remarks>
    ARIMAXModel,

    /// <summary>
    /// A time series model that combines autoregressive and moving average components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ARMA Models combine two approaches for time series forecasting: they look at
    /// how past values of the series influence future values (the AR part) and how past prediction errors
    /// affect future values (the MA part). This combination often works better than either approach alone.
    /// It's like predicting tomorrow's weather based both on weather patterns from recent days and on how
    /// accurate your previous forecasts were. ARMA models work best for stationary time series, where the
    /// statistical properties don't change over time.
    /// </para>
    /// </remarks>
    ARMAModel,

    /// <summary>
    /// A time series model where future values depend on past values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Autoregressive (AR) Models predict future values based on past values in the
    /// time series. They assume that what happens next depends on what happened before. For example, tomorrow's
    /// temperature is likely related to today's temperature. The "order" of an AR model (like AR(3)) tells you
    /// how many past time periods it considers - AR(3) looks at the three most recent values. These models are
    /// relatively simple but effective for many time series where recent history influences the future.
    /// </para>
    /// </remarks>
    ARModel,


    /// <summary>
    /// A flexible time series model that combines Bayesian methods with structural components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Bayesian Structural Time Series Models combine several components to model time series data. 
    /// They can include trends, seasonality, and the effects of external factors. What makes them special is their 
    /// Bayesian approach, which provides uncertainty estimates and allows you to incorporate prior knowledge. 
    /// These models are particularly useful for forecasting with limited data, understanding which factors 
    /// influence your time series, and measuring the impact of interventions like marketing campaigns.
    /// </para>
    /// </remarks>
    BayesianStructuralTimeSeriesModel,

    /// <summary>
    /// A hybrid model combining regression with ARIMA error correction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dynamic Regression with ARIMA Errors combines two powerful approaches. The regression 
    /// part models relationships between your time series and external factors (like how temperature affects 
    /// ice cream sales). The ARIMA part then models the errors from this regression, capturing patterns that 
    /// the regression missed. It's like having two specialists working together - one focusing on known 
    /// relationships and the other catching any remaining patterns. This hybrid approach often outperforms 
    /// either method used alone.
    /// </para>
    /// </remarks>
    DynamicRegressionWithARIMAErrors,

    /// <summary>
    /// A time series forecasting method that gives more weight to recent observations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Exponential Smoothing Models forecast by averaging past values, but giving more 
    /// weight to recent observations. Imagine predicting tomorrow's temperature by looking at the past week, 
    /// but considering yesterday's temperature more important than last week's. These models can capture 
    /// trends (values consistently increasing or decreasing) and seasonality (regular patterns like higher 
    /// sales during holidays). They're intuitive, computationally simple, and work well for many business 
    /// forecasting problems.
    /// </para>
    /// </remarks>
    ExponentialSmoothingModel,

    /// <summary>
    /// A model designed to capture volatility clustering in financial time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GARCH Models are specialized for financial data where volatility (the amount of 
    /// fluctuation) tends to cluster - periods of high volatility are followed by more high volatility, 
    /// and calm periods tend to persist as well. Think of stock markets having "nervous" periods with 
    /// big swings up and down, followed by "calm" periods with smaller changes. GARCH models capture 
    /// this pattern, making them valuable for risk management, option pricing, and financial forecasting.
    /// </para>
    /// </remarks>
    GARCHModel,

    /// <summary>
    /// A recurrent neural network architecture with gating mechanisms for efficient sequence processing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GRU Neural Networks are a type of recurrent neural network designed for sequential 
    /// data like text or time series. They're similar to LSTM networks but with a simpler structure that 
    /// makes them faster to train while still capturing long-term patterns. GRUs use "gates" that control 
    /// what information to keep or forget, allowing them to learn which parts of the sequence are important. 
    /// They're widely used for language processing, speech recognition, and time series forecasting.
    /// </para>
    /// </remarks>
    GRUNeuralNetwork,

    /// <summary>
    /// A neural network architecture designed to compare and find similarities between inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Siamese Networks are specialized for comparing two inputs to determine how similar 
    /// they are. They process both inputs through identical sub-networks (hence the name "Siamese," like 
    /// Siamese twins) and then measure the distance between the outputs. This is useful for tasks like 
    /// facial recognition (is this the same person?), signature verification, or finding similar products. 
    /// They're particularly valuable when you have many categories with few examples of each.
    /// </para>
    /// </remarks>
    SiameseNetwork,

    /// <summary>
    /// A time series model that assesses the impact of specific events or interventions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Intervention Analysis Models measure how specific events affect a time series. 
    /// For example, how did a new policy, marketing campaign, or competitor's action impact your sales? 
    /// These models separate the normal pattern of your time series from the effects of the intervention, 
    /// allowing you to quantify the impact. It's like measuring how much a medicine improves health by 
    /// comparing patient outcomes before and after treatment, while accounting for other factors.
    /// </para>
    /// </remarks>
    InterventionAnalysisModel,

    /// <summary>
    /// A flexible forecasting model that handles multiple seasonal patterns.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TBATS Models are designed for time series with complex seasonal patterns. While 
    /// simple models might handle yearly patterns (like holiday sales spikes), TBATS can simultaneously 
    /// model multiple seasonal patterns of different lengths. For example, retail data might show weekly 
    /// patterns (busier on weekends), monthly patterns (busier after payday), and yearly patterns (holiday 
    /// seasons). TBATS can capture all these overlapping cycles, making it powerful for complex forecasting 
    /// problems.
    /// </para>
    /// </remarks>
    TBATSModel,

    /// <summary>
    /// A method that breaks down a time series into trend, seasonal, and remainder components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> STL Decomposition breaks a time series into three parts: the overall trend 
    /// (long-term direction), seasonal patterns (regular cycles), and remainder (what's left after 
    /// removing trend and seasonality). It's like separating a music recording into bass, melody, and 
    /// percussion. This decomposition helps you understand what's driving your time series and can 
    /// improve forecasting by modeling each component separately. It's particularly useful for 
    /// visualizing and interpreting time series data.
    /// </para>
    /// </remarks>
    STLDecomposition,

    /// <summary>
    /// A model that captures relationships between multiple related time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Vector Autoregression Models analyze multiple time series that influence each 
    /// other. For example, how prices, advertising, and competitor actions all affect sales. Unlike 
    /// simpler models that look at each series separately, VAR models capture how each series affects 
    /// the others. It's like modeling an ecosystem where changes in one species affect others. This 
    /// makes VAR models powerful for understanding complex systems and forecasting interrelated variables.
    /// </para>
    /// </remarks>
    VARModel,

    /// <summary>
    /// A flexible time series model that represents observed data in terms of unobserved components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Unobserved Components Models break down a time series into underlying components 
    /// that aren't directly observable, such as trend, cycle, seasonal patterns, and irregular fluctuations. 
    /// It's like a detective inferring underlying motives from observed behaviors. These models are 
    /// particularly useful for economic data, where concepts like "potential GDP" or "natural rate of 
    /// unemployment" can't be directly measured but can be estimated from observable data.
    /// </para>
    /// </remarks>
    UnobservedComponentsModel,

    /// <summary>
    /// A model that describes how input changes propagate through a system over time.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Transfer Function Models describe how changes in an input variable affect an 
    /// output variable over time. For example, how a change in advertising budget affects sales in the 
    /// following weeks. These models capture both the magnitude of the effect (how much sales increase) 
    /// and its timing (immediate boost, gradual increase, delayed response, etc.). They're particularly 
    /// useful for understanding cause-and-effect relationships in time series data.
    /// </para>
    /// </remarks>
    TransferFunctionModel,

    /// <summary>
    /// A flexible framework for modeling time series data using hidden states.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> State Space Models represent a time series using hidden "states" that evolve 
    /// over time. Think of it like tracking a car's position when you can only see its speed - the 
    /// position is the hidden state that you infer from observable data. These models are very flexible 
    /// and can represent many time series patterns. They're particularly useful for tracking systems 
    /// that change over time, filtering noisy data, and forecasting complex time series.
    /// </para>
    /// </remarks>
    StateSpaceModel,

    /// <summary>
    /// A comprehensive time series model that handles non-stationary data with trends and seasonality.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ARIMA Models combine three powerful techniques for time series forecasting: 
    /// they use past values (AR), past forecast errors (MA), and differences between consecutive 
    /// observations (I for "Integrated") to handle data with trends. ARIMA is like a Swiss Army knife 
    /// for time series - versatile and effective for many forecasting problems. The model is specified 
    /// as ARIMA(p,d,q), where p, d, and q control how many past values, differences, and errors to use.
    /// </para>
    /// </remarks>
    ARIMAModel,

    /// <summary>
    /// A time series model where future values depend on past forecast errors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Moving Average (MA) Models predict future values based on past forecast errors 
    /// rather than past values themselves. They assume that if recent forecasts were too high or too low, 
    /// that information can help improve future forecasts. It's like adjusting your aim in a game based 
    /// on whether your previous shots were too high or too low. MA models are particularly good at 
    /// handling short-term, irregular fluctuations in time series data.
    /// </para>
    /// </remarks>
    MAModel,

    /// <summary>
    /// An extension of ARIMA that also captures seasonal patterns in data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Seasonal ARIMA Models extend ARIMA to handle regular patterns that repeat at 
    /// fixed intervals, like higher retail sales during holidays or increased ice cream consumption in 
    /// summer. SARIMA adds seasonal components to the standard ARIMA model, allowing it to capture both 
    /// short-term dependencies and longer seasonal patterns. It's specified with additional parameters 
    /// that control the seasonal aspects of the model, making it powerful for data with clear seasonal effects.
    /// </para>
    /// </remarks>
    SARIMAModel,

    /// <summary>
    /// A frequency-domain approach to analyzing time series data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Spectral Analysis Models examine time series data in terms of cycles and frequencies 
    /// rather than time. It's like breaking down a musical chord into individual notes - the model identifies 
    /// which cycles (daily, weekly, monthly, etc.) are present in your data and how strong each one is. This 
    /// approach is particularly useful for finding hidden periodicities in complex data, understanding cyclical 
    /// behavior, and filtering out noise. It's widely used in fields like signal processing, economics, and 
    /// climate science.
    /// </para>
    /// </remarks>
    SpectralAnalysisModel,

    /// <summary>
    /// A decomposable time series forecasting model developed by Facebook.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Prophet is a forecasting model designed by Facebook to handle time series with 
    /// strong seasonal effects and multiple seasons. It's particularly good with data that has holidays, 
    /// seasonal patterns, and trend changes. Prophet automatically decomposes your data into trend, seasonality, 
    /// and holiday effects, making it easier to understand what's driving changes in your time series. It's 
    /// designed to be robust to missing data and outliers, and requires minimal manual tuning, making it 
    /// accessible for non-experts.
    /// </para>
    /// </remarks>
    ProphetModel,

    /// <summary>
    /// A hybrid model combining neural networks with ARIMA for time series forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural Network ARIMA combines the pattern-recognition power of neural networks 
    /// with the time series expertise of ARIMA models. ARIMA is good at capturing linear relationships and 
    /// temporal dependencies, while neural networks excel at learning complex, non-linear patterns. By 
    /// combining them, this hybrid approach can capture both types of patterns in your data. It's like 
    /// having both a statistics expert and a pattern recognition expert working together on your forecasting 
    /// problem, often resulting in more accurate predictions than either approach alone.
    /// </para>
    /// </remarks>
    NeuralNetworkARIMA
}