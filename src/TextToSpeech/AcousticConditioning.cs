namespace AiDotNet.TextToSpeech;

/// <summary>
/// The phoneme hidden sequence after a variance-adaptor model adds its acoustic conditions, with what its decoder is
/// conditioned on and any auxiliary loss the conditioning contributes in training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <param name="Hidden">The conditioned phoneme hidden sequence the variance adaptor reads.</param>
/// <param name="DecoderCondition">The vector conditional layer normalizations in the decoder read (AdaSpeech's speaker
/// embedding); null for an unconditioned decoder.</param>
/// <param name="AuxiliaryLoss">A scalar loss term added to the objective (AdaSpeech's phoneme-level predictor loss);
/// null for none.</param>
public sealed record AcousticConditioning<T>(Tensor<T> Hidden, Tensor<T>? DecoderCondition, Tensor<T>? AuxiliaryLoss);
