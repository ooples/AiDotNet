namespace AiDotNet.Interpretability
{
    /// <summary>
    /// Enumeration of interpretation methods supported by interpretable models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These are the different explanation techniques you can use to understand
    /// why your model makes certain predictions. Each method has different strengths:
    ///
    /// - <b>Model-Agnostic Methods</b> (work with any model):
    ///   - SHAP, LIME, PartialDependence, Counterfactual, DiCE, Anchor, FeatureImportance, FeatureInteraction, Occlusion, FeatureAblation
    ///
    /// - <b>Neural Network Methods</b> (require gradient access):
    ///   - IntegratedGradients, DeepLIFT, DeepSHAP, GradientSHAP, GradCAM, LayerGradCAM, GuidedBackprop, GuidedGradCAM, NoiseTunnel
    ///
    /// - <b>Tree-Based Methods</b> (for tree models):
    ///   - TreeSHAP
    ///
    /// - <b>Concept-Based Methods</b> (for high-level understanding):
    ///   - TCAV
    ///
    /// - <b>Training Data Attribution</b>:
    ///   - InfluenceFunctions
    ///
    /// Enable methods through <see cref="Models.Options.InterpretabilityOptions"/> when configuring your model.
    /// </para>
    /// </remarks>
    public enum InterpretationMethod
    {
        /// <summary>
        /// SHAP (SHapley Additive exPlanations) values for feature importance.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> SHAP comes from game theory and fairly distributes "credit"
        /// for a prediction among all features. It provides both local (per-prediction) and
        /// global (overall) importance scores.
        ///
        /// <b>Requires:</b> Background data representing baseline distribution.
        /// <b>Best for:</b> Any model when you need theoretically-grounded attributions.
        /// </para>
        /// </remarks>
        SHAP,

        /// <summary>
        /// LIME (Local Interpretable Model-agnostic Explanations) for local explanations.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> LIME explains individual predictions by fitting a simple
        /// linear model around each prediction. Fast and intuitive but can be unstable.
        ///
        /// <b>Best for:</b> Quick local explanations when speed matters.
        /// </para>
        /// </remarks>
        LIME,

        /// <summary>
        /// Partial dependence plots to show feature effects.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> PDP shows how a feature affects predictions on average.
        /// Useful for understanding overall trends but can miss interactions.
        ///
        /// <b>Requires:</b> Background data for marginal effect computation.
        /// <b>Best for:</b> Understanding how individual features affect predictions.
        /// </para>
        /// </remarks>
        PartialDependence,

        /// <summary>
        /// Counterfactual explanations to understand decision boundaries.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Counterfactuals answer "What would need to change to get
        /// a different prediction?" Very intuitive for users ("If your income was $5k higher,
        /// you would qualify").
        ///
        /// <b>Best for:</b> Actionable explanations in high-stakes decisions.
        /// </para>
        /// </remarks>
        Counterfactual,

        /// <summary>
        /// Anchor explanations for rule-based interpretations.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Anchors find sufficient conditions that "anchor" the prediction.
        /// Results in simple rules like "If age > 40 AND income > $50k, then approved (95% precision)".
        ///
        /// <b>Best for:</b> Creating human-readable rules that explain predictions.
        /// </para>
        /// </remarks>
        Anchor,

        /// <summary>
        /// Feature importance analysis using permutation importance.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Measures how important each feature is by shuffling it and
        /// measuring how much worse the model performs. Requires test data with ground truth labels.
        ///
        /// <b>Requires:</b> Data with ground truth labels.
        /// <b>Best for:</b> Global understanding of which features matter most.
        /// </para>
        /// </remarks>
        FeatureImportance,

        /// <summary>
        /// Feature interaction analysis using Friedman's H-statistic.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> The H-statistic measures how much features interact (when
        /// the effect of one feature depends on another). Values range from 0 (no interaction)
        /// to 1 (pure interaction).
        ///
        /// <b>Requires:</b> Background data for PD computation.
        /// <b>Best for:</b> Understanding feature dependencies.
        /// </para>
        /// </remarks>
        FeatureInteraction,

        /// <summary>
        /// Integrated Gradients for neural network attribution.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Integrated Gradients is a theoretically-grounded method
        /// that satisfies important axioms (completeness, sensitivity). It computes attributions
        /// by integrating gradients along a path from baseline to input.
        ///
        /// <b>Requires:</b> Neural network model.
        /// <b>Best for:</b> Rigorous neural network explanations.
        /// </para>
        /// </remarks>
        IntegratedGradients,

        /// <summary>
        /// DeepLIFT (Deep Learning Important FeaTures) for neural network attribution.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> DeepLIFT explains predictions by comparing activations
        /// to a reference baseline. Faster than Integrated Gradients and handles
        /// non-linearities better than vanilla gradients.
        ///
        /// <b>Requires:</b> Neural network model.
        /// <b>Best for:</b> Fast neural network explanations.
        /// </para>
        /// </remarks>
        DeepLIFT,

        /// <summary>
        /// Grad-CAM (Gradient-weighted Class Activation Mapping) for CNN visual explanations.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Grad-CAM creates visual heatmaps showing which parts of
        /// an image were most important for a CNN's prediction. Bright regions indicate
        /// high importance.
        ///
        /// <b>Requires:</b> Convolutional neural network model.
        /// <b>Best for:</b> Visual explanations for image classifiers.
        /// </para>
        /// </remarks>
        GradCAM,

        /// <summary>
        /// TreeSHAP for exact SHAP values on tree-based models.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> TreeSHAP computes exact (not approximate) SHAP values
        /// specifically for tree-based models like Decision Trees, Random Forests, and
        /// Gradient Boosting. It's much faster than Kernel SHAP for these models.
        ///
        /// <b>Requires:</b> Tree-based model (Decision Tree, Random Forest, Gradient Boosting).
        /// <b>Best for:</b> Fast, exact explanations for tree models.
        /// </para>
        /// </remarks>
        TreeSHAP,

        /// <summary>
        /// DeepSHAP combining GradientSHAP with DeepLIFT for efficient neural network explanations.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> DeepSHAP is optimized for deep networks. It uses DeepLIFT's
        /// backpropagation rules combined with SHAP's sampling strategy to produce fast,
        /// approximate SHAP values for neural networks.
        ///
        /// <b>Requires:</b> Deep neural network model.
        /// <b>Best for:</b> Large neural networks where regular SHAP would be too slow.
        /// </para>
        /// </remarks>
        DeepSHAP,

        /// <summary>
        /// GradientSHAP for gradient-based SHAP approximation on neural networks.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> GradientSHAP uses expected gradients to approximate SHAP
        /// values. It samples baselines and interpolates between baseline and input to
        /// get attributions that satisfy SHAP's completeness axiom.
        ///
        /// <b>Requires:</b> Differentiable model (neural network).
        /// <b>Best for:</b> Gradient-based attributions with SHAP theoretical properties.
        /// </para>
        /// </remarks>
        GradientSHAP,

        /// <summary>
        /// TCAV (Testing with Concept Activation Vectors) for concept-level explanations.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> TCAV explains models using human-understandable concepts
        /// (like "striped" or "furry") rather than individual features. It tests how
        /// sensitive model predictions are to the presence of these concepts.
        ///
        /// <b>Requires:</b> Concept examples (positive and negative examples for each concept).
        /// <b>Best for:</b> High-level concept-based understanding of CNN decisions.
        /// </para>
        /// </remarks>
        TCAV,

        /// <summary>
        /// Influence Functions for training data attribution.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Influence Functions identify which training examples were
        /// most responsible for a particular prediction. They answer "Which training
        /// examples would change this prediction if removed?"
        ///
        /// <b>Requires:</b> Access to training data and model gradients.
        /// <b>Best for:</b> Data debugging, identifying mislabeled training examples.
        /// </para>
        /// </remarks>
        InfluenceFunctions,

        /// <summary>
        /// Occlusion-based attribution by systematically hiding parts of input.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Occlusion works by covering parts of the input with a
        /// baseline value (like a gray patch on an image) and seeing how predictions
        /// change. Simple and model-agnostic but computationally expensive.
        ///
        /// <b>Best for:</b> Visual explanations for image models without gradient access.
        /// </para>
        /// </remarks>
        Occlusion,

        /// <summary>
        /// Feature Ablation for attribution by removing/replacing features.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Feature Ablation measures feature importance by replacing
        /// features with baseline values one at a time (or in groups) and measuring
        /// prediction changes. Similar to occlusion but for tabular/general data.
        ///
        /// <b>Best for:</b> Simple, interpretable feature importance for any model.
        /// </para>
        /// </remarks>
        FeatureAblation,

        /// <summary>
        /// DiCE (Diverse Counterfactual Explanations) for generating diverse what-if scenarios.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> DiCE generates multiple different counterfactual
        /// explanations for a single prediction, showing diverse ways to change
        /// the outcome. Uses genetic algorithms to find diverse, sparse, actionable changes.
        ///
        /// <b>Best for:</b> Actionable recourse with multiple alternative paths.
        /// </para>
        /// </remarks>
        DiCE,

        /// <summary>
        /// Guided Backpropagation for cleaner gradient visualizations.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Guided Backpropagation modifies gradient computation
        /// to only propagate positive gradients through ReLU activations, producing
        /// cleaner visualizations that highlight positive feature contributions.
        ///
        /// <b>Requires:</b> Neural network with ReLU activations.
        /// <b>Best for:</b> Clean, high-resolution feature visualizations.
        /// </para>
        /// </remarks>
        GuidedBackprop,

        /// <summary>
        /// LayerGradCAM for class activation mapping at a specific network layer.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> LayerGradCAM applies GradCAM at a chosen layer rather
        /// than just the last convolutional layer. This lets you visualize what the
        /// model sees at different levels of abstraction.
        ///
        /// <b>Requires:</b> Convolutional neural network.
        /// <b>Best for:</b> Layer-by-layer visualization of CNN focus.
        /// </para>
        /// </remarks>
        LayerGradCAM,

        /// <summary>
        /// GuidedGradCAM combining GuidedBackprop with GradCAM for high-resolution class-specific visualizations.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> GuidedGradCAM multiplies GuidedBackprop (fine-grained)
        /// with GradCAM (class-discriminative) to get the best of both: high-resolution
        /// visualizations that are also class-specific.
        ///
        /// <b>Requires:</b> Convolutional neural network.
        /// <b>Best for:</b> High-quality visual explanations combining detail and class-specificity.
        /// </para>
        /// </remarks>
        GuidedGradCAM,

        /// <summary>
        /// NoiseTunnel (SmoothGrad) for noise-averaged gradient smoothing.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> NoiseTunnel adds noise to inputs and averages the
        /// resulting attributions. This smooths out noisy gradients and produces
        /// more stable, visually pleasing explanations.
        ///
        /// <b>Variants:</b> SmoothGrad (average), SmoothGrad² (squared average), VarGrad (variance).
        /// <b>Best for:</b> Reducing noise in any gradient-based explanation.
        /// </para>
        /// </remarks>
        NoiseTunnel,

        /// <summary>
        /// Input × Gradient attribution - multiplies input values by their gradients.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> A simple and fast baseline method that multiplies each
        /// input feature by its gradient. Captures both feature value and sensitivity.
        ///
        /// <b>Formula:</b> attribution[i] = input[i] × gradient[i]
        /// <b>Best for:</b> Quick baseline explanations, sanity checks.
        /// </para>
        /// </remarks>
        InputXGradient,

        /// <summary>
        /// Neuron-level attribution for understanding individual neuron contributions.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> While most methods explain INPUT features, neuron attribution
        /// explains which NEURONS in a hidden layer contribute to the output.
        ///
        /// <b>Variants:</b> NeuronGradient, NeuronIntegratedGradients, NeuronConductance, NeuronDeepLIFT.
        /// <b>Best for:</b> Understanding hidden representations, feature discovery, pruning analysis.
        /// </para>
        /// </remarks>
        NeuronAttribution,

        /// <summary>
        /// Layer-level attribution for computing attributions at intermediate layers.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> Explains which neurons in a specific layer are important
        /// for the output. Useful for understanding what the model "thinks" at each stage.
        ///
        /// <b>Variants:</b> LayerGradient, LayerIntegratedGradients, LayerDeepLIFT, LayerConductance.
        /// <b>Best for:</b> Understanding CNN feature maps, debugging layer-wise information flow.
        /// </para>
        /// </remarks>
        LayerAttribution,

        /// <summary>
        /// TracIn (Tracing Influence) for efficient training data attribution.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>For Beginners:</b> TracIn is a simpler alternative to Influence Functions.
        /// It sums gradient dot products from training checkpoints to identify influential
        /// training examples.
        ///
        /// <b>Requires:</b> Gradient checkpoints saved during training.
        /// <b>Best for:</b> Fast training data attribution when checkpoints are available.
        /// </para>
        /// </remarks>
        TracIn
    }
}
