# Hello World — Iris classifier

The classic starting-line for any ML library: a 3-class classifier on
Fisher's 1936 Iris dataset (150 flowers × 4 measurements × 3 species).
End-to-end in ~50 lines including the inlined dataset:

```bash
dotnet run
```

```text
=== AiDotNet Hello World: Iris classifier ===

Test accuracy: 29/30 = 96.7%
Final loss:    0.0421
```

## What this sample shows

1. **Data → tensors.** Iris is small enough to inline (150 rows × 5 cols).
   We pack features into a `[N, 4]` `Tensor<double>` and labels into a
   `[N, 3]` one-hot tensor for the 3-class softmax target.
2. **80/20 train/test split.** Seeded shuffle for reproducibility.
3. **Build + train via the fluent `AiModelBuilder`.** Three lines:

   ```csharp
   var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
       .ConfigureModel(new NeuralNetwork<double>(architecture))
       .BuildAsync(trainX, trainY);
   ```

4. **Evaluate.** Argmax the prediction logits, compare against the held-out
   labels, report accuracy.

The neural network uses the `NetworkComplexity.Simple` default
architecture, which is a sensible 1-hidden-layer feedforward classifier.
For something more elaborate, see [BasicClassification](../BasicClassification/)
which walks through layer-by-layer architecture configuration.

## Why Iris and not XOR

XOR is the textbook example for "non-linearly separable" but it doesn't
exercise much of the library. Iris exercises:

- Real-world feature scales (sepal length ~5cm vs petal width ~0.2cm)
- 3-way classification (XOR is binary)
- Train/test split discipline
- The realistic 90-95%+ accuracy range that production classifiers hit

The prior XOR sample is preserved in version control for reference;
see `git log` if you want to compare.

## Next steps

- [BasicClassification](../BasicClassification/) — Layer-by-layer architecture configuration
- [BasicRegression](../BasicRegression/) — Continuous-valued outputs
- [aidotnet.dev/docs](https://aidotnet.dev/docs) — Full documentation
