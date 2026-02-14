namespace AiDotNet.Enums;

/// <summary>
/// Defines different loss functions used to measure how well a model's predictions match the actual values.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Loss functions are like a "scoring system" for AI models. They measure the difference
/// between what the model predicted and what the correct answer actually is. A lower loss means the model
/// is making better predictions. During training, the optimizer adjusts the model's parameters to minimize
/// this loss value, gradually improving the model's accuracy.
/// </para>
/// </remarks>
public enum LossType
{
    /// <summary>
    /// Mean Squared Error - measures the average squared difference between predictions and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MSE squares the errors, so large mistakes are penalized much more than small ones.
    /// It's the most common loss function for regression problems (predicting numbers like prices or temperatures).
    /// </para>
    /// </remarks>
    MeanSquaredError,

    /// <summary>
    /// Mean Absolute Error - measures the average absolute difference between predictions and actual values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> MAE treats all errors equally regardless of size. It's more robust to outliers
    /// than MSE because it doesn't square the errors. Use it when you want a loss that isn't dominated by
    /// a few large mistakes.
    /// </para>
    /// </remarks>
    MeanAbsoluteError,

    /// <summary>
    /// Root Mean Squared Error - the square root of the average squared differences.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> RMSE is like MSE but returns a value in the same units as your data,
    /// making it easier to interpret. If you're predicting house prices in dollars, RMSE gives you
    /// an error in dollars too.
    /// </para>
    /// </remarks>
    RootMeanSquaredError,

    /// <summary>
    /// Huber Loss - combines MSE for small errors and MAE for large errors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Huber loss gives you the best of both worlds: it's smooth like MSE for small
    /// errors but doesn't overreact to outliers like MAE. The delta parameter controls the transition point.
    /// </para>
    /// </remarks>
    Huber,

    /// <summary>
    /// Cross Entropy - measures the difference between two probability distributions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cross entropy is the go-to loss for classification problems. It heavily penalizes
    /// confident wrong predictions and rewards confident correct predictions. If your model says "I'm 99% sure
    /// this is a cat" and it's actually a dog, the loss will be very high.
    /// </para>
    /// </remarks>
    CrossEntropy,

    /// <summary>
    /// Binary Cross Entropy - cross entropy specialized for two-class (yes/no) problems.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when your model predicts one of two categories, like spam/not-spam
    /// or positive/negative sentiment.
    /// </para>
    /// </remarks>
    BinaryCrossEntropy,

    /// <summary>
    /// Categorical Cross Entropy - cross entropy for multi-class classification with one-hot encoded labels.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when your model predicts one of many categories (like classifying
    /// images of animals into cat, dog, bird, etc.) and your labels are one-hot encoded vectors.
    /// </para>
    /// </remarks>
    CategoricalCrossEntropy,

    /// <summary>
    /// Sparse Categorical Cross Entropy - like categorical cross entropy but with integer labels instead of one-hot.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Same as categorical cross entropy, but your labels are simple numbers (0, 1, 2, ...)
    /// instead of one-hot vectors. This is more memory-efficient for problems with many categories.
    /// </para>
    /// </remarks>
    SparseCategoricalCrossEntropy,

    /// <summary>
    /// Focal Loss - gives more weight to hard-to-classify examples, reducing the impact of easy ones.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Focal loss is designed for imbalanced datasets where one class vastly outnumbers
    /// others. It focuses training on the difficult examples that the model struggles with most.
    /// </para>
    /// </remarks>
    Focal,

    /// <summary>
    /// Hinge Loss - used for "maximum-margin" classification, especially with Support Vector Machines.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hinge loss tries to make the model not just classify correctly, but do so with
    /// confidence. It penalizes predictions that are correct but not confident enough.
    /// </para>
    /// </remarks>
    Hinge,

    /// <summary>
    /// Squared Hinge Loss - a smoother variant of hinge loss that squares the penalty.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Like hinge loss but with a smoother gradient, making optimization easier.
    /// The squaring means larger margin violations are penalized more heavily.
    /// </para>
    /// </remarks>
    SquaredHinge,

    /// <summary>
    /// Log-Cosh Loss - the logarithm of the hyperbolic cosine of the prediction error.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Log-cosh is approximately equal to MSE for small errors and MAE for large errors,
    /// similar to Huber loss but smoother. It's twice differentiable everywhere, which helps some optimizers.
    /// </para>
    /// </remarks>
    LogCosh,

    /// <summary>
    /// Quantile Loss - predicts specific percentiles rather than the mean.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Instead of predicting the average outcome, quantile loss lets you predict
    /// specific percentiles. For example, you could predict the 90th percentile to understand worst-case scenarios.
    /// </para>
    /// </remarks>
    Quantile,

    /// <summary>
    /// Poisson Loss - appropriate for count data that follows a Poisson distribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use Poisson loss when you're predicting counts (like number of events, visitors,
    /// or occurrences). It assumes the target values follow a Poisson distribution.
    /// </para>
    /// </remarks>
    Poisson,

    /// <summary>
    /// Kullback-Leibler Divergence - measures how one probability distribution diverges from another.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> KL divergence measures how different two probability distributions are.
    /// It's commonly used in variational autoencoders and knowledge distillation.
    /// </para>
    /// </remarks>
    KullbackLeiblerDivergence,

    /// <summary>
    /// Cosine Similarity Loss - measures the cosine of the angle between prediction and target vectors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This loss focuses on the direction of vectors rather than their magnitude.
    /// It's useful for text similarity, recommendation systems, and embedding learning.
    /// </para>
    /// </remarks>
    CosineSimilarity,

    /// <summary>
    /// Contrastive Loss - learns embeddings by pulling similar pairs together and pushing dissimilar pairs apart.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Contrastive loss is used in siamese networks to learn a similarity metric.
    /// It makes similar items closer and dissimilar items farther apart in the embedding space.
    /// </para>
    /// </remarks>
    Contrastive,

    /// <summary>
    /// Triplet Loss - learns embeddings using anchor, positive, and negative examples.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Triplet loss uses three samples at a time: an anchor, a positive (similar to anchor),
    /// and a negative (different from anchor). It ensures the anchor is closer to the positive than the negative.
    /// </para>
    /// </remarks>
    Triplet,

    /// <summary>
    /// Dice Loss - measures overlap between predicted and actual segmentation masks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Dice loss is popular in image segmentation tasks. It measures how much the predicted
    /// region overlaps with the actual region, handling class imbalance well.
    /// </para>
    /// </remarks>
    Dice,

    /// <summary>
    /// Jaccard Loss - also known as Intersection over Union (IoU) loss for segmentation tasks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Similar to Dice loss, Jaccard loss measures the overlap between predicted and actual
    /// regions. It's commonly used as an evaluation metric in object detection and segmentation.
    /// </para>
    /// </remarks>
    Jaccard,

    /// <summary>
    /// Elastic Net Loss - combines L1 (Lasso) and L2 (Ridge) regularization penalties.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Elastic Net combines two regularization techniques to prevent overfitting.
    /// L1 encourages sparse models (fewer features) while L2 prevents any single feature from dominating.
    /// </para>
    /// </remarks>
    ElasticNet,

    /// <summary>
    /// Exponential Loss - used in boosting algorithms like AdaBoost.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Exponential loss grows very quickly for wrong predictions, making it very
    /// sensitive to misclassified examples. It's the foundation of the AdaBoost algorithm.
    /// </para>
    /// </remarks>
    Exponential,

    /// <summary>
    /// Modified Huber Loss - a smooth approximation combining hinge loss and quadratic loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Modified Huber loss is designed for classification. It provides a smooth
    /// loss function that's robust to outliers while still being differentiable everywhere.
    /// </para>
    /// </remarks>
    ModifiedHuber,

    /// <summary>
    /// Charbonnier Loss - a differentiable variant of L1 loss, popular in image restoration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Charbonnier loss is a smooth approximation of MAE that avoids the
    /// non-differentiability at zero. It's commonly used in super-resolution and denoising tasks.
    /// </para>
    /// </remarks>
    Charbonnier,

    /// <summary>
    /// Mean Bias Error - measures the average bias (systematic over/under-prediction) of the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Unlike MSE which treats over-prediction and under-prediction equally,
    /// MBE tells you if your model consistently predicts too high or too low.
    /// </para>
    /// </remarks>
    MeanBiasError,

    /// <summary>
    /// Wasserstein Loss - based on the Earth Mover's Distance, commonly used in GANs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Wasserstein loss measures the "cost" of transforming one distribution into
    /// another. It's used in Wasserstein GANs (WGANs) to provide more stable training signals.
    /// </para>
    /// </remarks>
    Wasserstein,

    /// <summary>
    /// Margin Loss - used in capsule networks with separate margins for positive and negative classes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Margin loss is designed for capsule networks. It uses two thresholds
    /// (margins) to determine the acceptable range for positive and negative class activations.
    /// </para>
    /// </remarks>
    Margin,

    /// <summary>
    /// CTC Loss - Connectionist Temporal Classification for sequence-to-sequence alignment.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> CTC loss is used for tasks where the input and output sequences have different
    /// lengths, like speech recognition or handwriting recognition. It handles alignment automatically.
    /// </para>
    /// </remarks>
    CTC,

    /// <summary>
    /// Noise Contrastive Estimation Loss - approximates softmax for large vocabulary problems.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> NCE loss is an efficient alternative to full softmax when you have many classes.
    /// Instead of computing probabilities for all classes, it samples a few "noise" classes for comparison.
    /// </para>
    /// </remarks>
    NoiseContrastiveEstimation,

    /// <summary>
    /// Ordinal Regression Loss - preserves the ordering relationship between classes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when your classes have a natural order, like rating scales (1-5 stars)
    /// or severity levels (mild, moderate, severe). It ensures the model respects this ordering.
    /// </para>
    /// </remarks>
    OrdinalRegression,

    /// <summary>
    /// Weighted Cross Entropy Loss - cross entropy with class-specific weights.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is cross entropy but with different importance given to different classes.
    /// Use it when some classes are more important to get right, or when classes are imbalanced.
    /// </para>
    /// </remarks>
    WeightedCrossEntropy,

    /// <summary>
    /// Scale Invariant Depth Loss - measures depth prediction quality regardless of absolute scale.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This loss is designed for monocular depth estimation tasks. It focuses on
    /// getting the relative depth relationships right rather than exact depth values.
    /// </para>
    /// </remarks>
    ScaleInvariantDepth,

    /// <summary>
    /// Perceptual Loss - compares feature representations from a pretrained network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Instead of comparing pixels directly, perceptual loss compares how a neural
    /// network "sees" the images. This produces more visually pleasing results in image generation tasks.
    /// </para>
    /// </remarks>
    Perceptual,

    /// <summary>
    /// Quantum Loss - a loss function inspired by quantum computing principles.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Quantum loss applies concepts from quantum mechanics to measure prediction
    /// errors. It can capture complex relationships that traditional loss functions might miss.
    /// </para>
    /// </remarks>
    Quantum,

    /// <summary>
    /// RealESRGAN Loss - a composite loss for Real-ESRGAN super-resolution models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This combines multiple loss functions (L1, perceptual, GAN) designed specifically
    /// for image super-resolution. It balances pixel accuracy with perceptual quality.
    /// </para>
    /// </remarks>
    RealESRGAN,

    /// <summary>
    /// Rotation Prediction Loss - a self-supervised loss that predicts image rotation angles.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This loss is used in self-supervised learning where the model learns to predict
    /// how much an image has been rotated. This helps the model learn useful visual features without labels.
    /// </para>
    /// </remarks>
    RotationPrediction,
}
