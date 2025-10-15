using System;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Models
{
    /// <summary>
    /// Represents the output from a tokenizer encoding operation.
    /// Contains all necessary tensors for model input.
    /// </summary>
    public class TokenizerOutput
    {
        /// <summary>
        /// Matrix of token IDs (batch_size x sequence_length)
        /// </summary>
        public Matrix<int> InputIds { get; set; } = Matrix<int>.Empty();

        /// <summary>
        /// Attention mask matrix indicating real tokens (1) vs padding (0)
        /// Shape: (batch_size x sequence_length)
        /// </summary>
        public Matrix<int> AttentionMask { get; set; } = Matrix<int>.Empty();

        /// <summary>
        /// Token type IDs matrix for models that distinguish between sequences
        /// Shape: (batch_size x sequence_length)
        /// Used by models like BERT for sentence pair tasks
        /// </summary>
        public Matrix<int>? TokenTypeIds { get; set; }

        /// <summary>
        /// Position IDs for models with learnable position embeddings
        /// Shape: (batch_size x sequence_length)
        /// </summary>
        public Matrix<int>? PositionIds { get; set; }

        /// <summary>
        /// Original sequence lengths before padding
        /// </summary>
        public Vector<int> SequenceLengths { get; set; } = Vector<int>.Empty();

        /// <summary>
        /// Gets the batch size
        /// </summary>
        public int BatchSize => InputIds.Rows;

        /// <summary>
        /// Gets the sequence length
        /// </summary>
        public int SequenceLength => InputIds.Columns;

        /// <summary>
        /// Indicates if this output contains valid data
        /// </summary>
        public bool IsValid => BatchSize > 0 && SequenceLength > 0;

        /// <summary>
        /// Creates a TokenizerOutput for a single sequence
        /// </summary>
        /// <param name="inputIds">Token IDs</param>
        /// <param name="attentionMask">Attention mask</param>
        /// <param name="tokenTypeIds">Optional token type IDs</param>
        /// <returns>TokenizerOutput instance</returns>
        public static TokenizerOutput FromSingleSequence(
            Vector<int> inputIds, 
            Vector<int> attentionMask,
            Vector<int>? tokenTypeIds = null)
        {
            if (inputIds.Length != attentionMask.Length)
            {
                throw new ArgumentException("Input IDs and attention mask must have the same length");
            }

            var output = new TokenizerOutput
            {
                InputIds = new Matrix<int>(1, inputIds.Length),
                AttentionMask = new Matrix<int>(1, attentionMask.Length),
                SequenceLengths = new Vector<int>(1)
            };

            // Copy data to matrices
            for (int i = 0; i < inputIds.Length; i++)
            {
                output.InputIds[0, i] = inputIds[i];
                output.AttentionMask[0, i] = attentionMask[i];
            }

            output.SequenceLengths[0] = inputIds.Length;

            if (tokenTypeIds != null)
            {
                output.TokenTypeIds = new Matrix<int>(1, tokenTypeIds.Length);
                for (int i = 0; i < tokenTypeIds.Length; i++)
                {
                    output.TokenTypeIds[0, i] = tokenTypeIds[i];
                }
            }

            return output;
        }

        /// <summary>
        /// Validates the consistency of the tokenizer output
        /// </summary>
        /// <exception cref="InvalidOperationException">Thrown when output is inconsistent</exception>
        public void Validate()
        {
            if (!IsValid)
            {
                throw new InvalidOperationException("TokenizerOutput contains no data");
            }

            if (InputIds.Rows != AttentionMask.Rows || InputIds.Columns != AttentionMask.Columns)
            {
                throw new InvalidOperationException("InputIds and AttentionMask dimensions must match");
            }

            if (TokenTypeIds != null && 
                (TokenTypeIds.Rows != InputIds.Rows || TokenTypeIds.Columns != InputIds.Columns))
            {
                throw new InvalidOperationException("TokenTypeIds dimensions must match InputIds");
            }

            if (PositionIds != null && 
                (PositionIds.Rows != InputIds.Rows || PositionIds.Columns != InputIds.Columns))
            {
                throw new InvalidOperationException("PositionIds dimensions must match InputIds");
            }

            if (SequenceLengths.Length != BatchSize)
            {
                throw new InvalidOperationException("SequenceLengths must have one entry per batch");
            }
        }

        /// <summary>
        /// Creates a deep copy of this TokenizerOutput
        /// </summary>
        /// <returns>New TokenizerOutput instance with copied data</returns>
        public TokenizerOutput DeepCopy()
        {
            return new TokenizerOutput
            {
                InputIds = new Matrix<int>(InputIds),
                AttentionMask = new Matrix<int>(AttentionMask),
                TokenTypeIds = TokenTypeIds != null ? new Matrix<int>(TokenTypeIds) : null,
                PositionIds = PositionIds != null ? new Matrix<int>(PositionIds) : null,
                SequenceLengths = new Vector<int>(SequenceLengths)
            };
        }
    }
}