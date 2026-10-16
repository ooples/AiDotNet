using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.VisionLanguage.Grounding;

internal sealed partial class GroundingDinoNetwork<T>
{
    /// <summary>
    /// Loads a Hugging Face <c>GroundingDinoForObjectDetection</c> state dict (as converted from the official checkpoints),
    /// given a reader of a named tensor of a given shape.
    /// </summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        int d = c.Dim, half = c.EncoderFfn / 2;
        void Linear(DenseLayer<T> layer, string name, int inputs, int outputs)
            => TorchParameters.Linear(_engine, layer, inputs, outputs, read($"{name}.weight", new[] { outputs, inputs }), read($"{name}.bias", new[] { outputs }));
        void Norm(LayerNormalizationLayer<T> norm, string name, int dim)
            => TorchParameters.LayerNorm(_engine, norm, read($"{name}.weight", new[] { dim }), read($"{name}.bias", new[] { dim }));
        void Attention(Attention a, string name)
        {
            Linear(a.Query, $"{name}.query", d, d);
            Linear(a.Key, $"{name}.key", d, d);
            Linear(a.Value, $"{name}.value", d, d);
            Linear(a.Output, $"{name}.out_proj", d, d);
        }
        void Deform(Deformable a, string name)
        {
            Linear(a.Offsets, $"{name}.sampling_offsets", d, a.Heads * c.Levels * a.Points * 2);
            Linear(a.Weights, $"{name}.attention_weights", d, a.Heads * c.Levels * a.Points);
            Linear(a.Value, $"{name}.value_proj", d, d);
            Linear(a.Output, $"{name}.output_proj", d, d);
        }
        void Mlp(DenseLayer<T>[] layers, string name, int inputs)
        {
            for (int i = 0; i < layers.Length; i++)
                Linear(layers[i], $"{name}.layers.{i}", i == 0 ? inputs : d, i == layers.Length - 1 && layers.Length == 3 ? 4 : d);
        }

        Backbone.LoadTorchWeights(read, "model.backbone.conv_encoder.model.");
        Text.LoadTorchWeights(read, "model.text_backbone.");
        Linear(TextProjection, "model.text_projection", c.Text.Hidden, d);
        var outChannels = Backbone.OutChannels;
        for (int level = 0; level < c.Levels; level++)
        {
            int k = InputKernel[level];
            int inputs = level < outChannels.Length ? outChannels[level] : level == outChannels.Length ? outChannels[^1] : d;
            TorchParameters.Linear(_engine, InputProjections[level], inputs * k * k, d,
                read($"model.input_proj_vision.{level}.0.weight", new[] { d, inputs, k, k }), read($"model.input_proj_vision.{level}.0.bias", new[] { d }));
            var gamma = InputNorms[level].GetGammaTensor();
            var beta = InputNorms[level].GetBetaTensor();
            var g = read($"model.input_proj_vision.{level}.1.weight", new[] { d });
            var bb = read($"model.input_proj_vision.{level}.1.bias", new[] { d });
            for (int i = 0; i < d; i++) { gamma[i] = NumOps.FromDouble(g[i]); beta[i] = NumOps.FromDouble(bb[i]); }
            _engine.InvalidatePersistentTensor(gamma);
            _engine.InvalidatePersistentTensor(beta);
        }
        TorchParameters.Linear(LevelEmbedding, 1, c.Levels * d, read("model.level_embed", new[] { c.Levels, d }));
        TorchParameters.Linear(QueryEmbedding, 1, c.Queries * d, read("model.query_position_embeddings.weight", new[] { c.Queries, d }));
        for (int i = 0; i < EncoderLayers.Count; i++)
        {
            var e = EncoderLayers[i];
            string n = $"model.encoder.layers.{i}";
            var f = e.Fusion;
            Norm(f.VisionNorm, $"{n}.fusion_layer.layer_norm_vision", d);
            Norm(f.TextNorm, $"{n}.fusion_layer.layer_norm_text", d);
            Linear(f.VisionProj, $"{n}.fusion_layer.attn.vision_proj", d, half);
            Linear(f.TextProj, $"{n}.fusion_layer.attn.text_proj", d, half);
            Linear(f.VisionValue, $"{n}.fusion_layer.attn.values_vision_proj", d, half);
            Linear(f.TextValue, $"{n}.fusion_layer.attn.values_text_proj", d, half);
            Linear(f.VisionOut, $"{n}.fusion_layer.attn.out_vision_proj", half, d);
            Linear(f.TextOut, $"{n}.fusion_layer.attn.out_text_proj", half, d);
            TorchParameters.Linear(f.VisionGamma, 1, d, read($"{n}.fusion_layer.vision_param", new[] { d }));
            TorchParameters.Linear(f.TextGamma, 1, d, read($"{n}.fusion_layer.text_param", new[] { d }));
            Attention(e.TextAttention, $"{n}.text_enhancer_layer.self_attn");
            Linear(e.TextFc1, $"{n}.text_enhancer_layer.fc1", d, half);
            Linear(e.TextFc2, $"{n}.text_enhancer_layer.fc2", half, d);
            Norm(e.TextNormBefore, $"{n}.text_enhancer_layer.layer_norm_before", d);
            Norm(e.TextNormAfter, $"{n}.text_enhancer_layer.layer_norm_after", d);
            Deform(e.ImageAttention, $"{n}.deformable_layer.self_attn");
            Norm(e.ImageNorm, $"{n}.deformable_layer.self_attn_layer_norm", d);
            Linear(e.ImageFc1, $"{n}.deformable_layer.fc1", d, c.EncoderFfn);
            Linear(e.ImageFc2, $"{n}.deformable_layer.fc2", c.EncoderFfn, d);
            Norm(e.ImageFinalNorm, $"{n}.deformable_layer.final_layer_norm", d);
        }
        Linear(EncoderOutput, "model.enc_output", d, d);
        Norm(EncoderOutputNorm, "model.enc_output_norm", d);
        Mlp(EncoderBoxHead, "model.encoder_output_bbox_embed", d);
        for (int i = 0; i < DecoderLayers.Count; i++)
        {
            var layer = DecoderLayers[i];
            string n = $"model.decoder.layers.{i}";
            Attention(layer.SelfAttention, $"{n}.self_attn");
            Norm(layer.SelfNorm, $"{n}.self_attn_layer_norm", d);
            Attention(layer.TextAttention, $"{n}.encoder_attn_text");
            Norm(layer.TextNorm, $"{n}.encoder_attn_text_layer_norm", d);
            Deform(layer.ImageAttention, $"{n}.encoder_attn");
            Norm(layer.ImageNorm, $"{n}.encoder_attn_layer_norm", d);
            Linear(layer.Fc1, $"{n}.fc1", d, c.DecoderFfn);
            Linear(layer.Fc2, $"{n}.fc2", c.DecoderFfn, d);
            Norm(layer.FinalNorm, $"{n}.final_layer_norm", d);
        }
        Norm(DecoderNorm, "model.decoder.layer_norm", d);
        Mlp(ReferenceHead, "model.decoder.reference_points_head", 2 * d);
        Mlp(BoxHead, "bbox_embed.0", d);
    }

    /// <summary>
    /// The reference initialization (Hugging Face <c>GroundingDinoPreTrainedModel._init_weights</c> and the Swin/BERT
    /// ones): linear layers N(0, 0.02²) with zero bias; LayerNorm and GroupNorm unit; deformable offsets zero with the
    /// per-head ray bias, attention weights zero, value/output projections Xavier-uniform; the fusion projections
    /// Xavier-uniform and its gates 1e-4; every box head's last layer zero; the level embedding N(0, 1); embeddings
    /// N(0, 0.02²); the input projections' convolutions Xavier-uniform with zero bias (the official DINO init).
    /// </summary>
    public void InitializeLikeReference(Random random)
    {
        var c = Configuration;
        double Normal() => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        double[] Xavier(int[] shape)
        {
            int count = shape.Aggregate(1, (a, s) => a * s), receptive = shape.Skip(2).Aggregate(1, (a, s) => a * s);
            double bound = Math.Sqrt(6.0 / ((double)shape[1] * receptive + (double)shape[0] * receptive));
            return Enumerable.Range(0, count).Select(_ => (2 * random.NextDouble() - 1) * bound).ToArray();
        }
        LoadTorchWeights((name, shape) =>
        {
            int count = shape.Aggregate(1, (a, s) => a * s);
            var values = new double[count];
            bool weight = name.EndsWith(".weight", StringComparison.Ordinal);
            if (name.Contains("norm") || name.Contains("LayerNorm") || Regex(name, "input_proj_vision.", ".1."))
            {
                if (weight) for (int i = 0; i < count; i++) values[i] = 1.0;
                return values;
            }
            if (name.EndsWith("vision_param", StringComparison.Ordinal) || name.EndsWith("text_param", StringComparison.Ordinal))
                return Enumerable.Repeat(1e-4, count).ToArray();
            if (name == "model.level_embed") return Enumerable.Range(0, count).Select(_ => Normal()).ToArray();
            if (name.Contains("sampling_offsets"))
                return weight ? values : RayBias(shape[0]);
            if (name.Contains(".attention_weights")) return values;
            if (name.Contains("value_proj") || name.Contains("output_proj") || name.Contains("fusion_layer.attn.") || Regex(name, "input_proj_vision.", ".0."))
                return weight ? Xavier(shape) : values;
            bool lastBoxLayer = (name.Contains("bbox_embed") && name.Contains(".layers.2."));
            if (lastBoxLayer) return values;
            if (weight || name.Contains("embeddings") || name.Contains("relative_position_bias_table"))
                for (int i = 0; i < count; i++) values[i] = 0.02 * Normal();
            return values;
        });

        double[] RayBias(int length)
        {
            int perHead = length / 1;
            var bias = new double[length];
            // sampling_offsets bias [heads, levels, points, 2]: head m on the ray at angle 2πm/heads, normalized to the
            // unit square's edge, point p at (p + 1) times that.
            int heads = c.EncoderHeads, points = c.EncoderPoints;
            if (length != heads * c.Levels * points * 2) { heads = c.DecoderHeads; points = c.DecoderPoints; }
            for (int m = 0; m < heads; m++)
            {
                double theta = m * (2.0 * Math.PI / heads), cx = Math.Cos(theta), cy = Math.Sin(theta);
                double scale = Math.Max(Math.Abs(cx), Math.Abs(cy));
                for (int l = 0; l < c.Levels; l++)
                    for (int p = 0; p < points; p++)
                    {
                        int at = ((m * c.Levels + l) * points + p) * 2;
                        bias[at] = cx / scale * (p + 1);
                        bias[at + 1] = cy / scale * (p + 1);
                    }
            }
            return bias;
        }

        static bool Regex(string name, string prefix, string part) => name.Contains(prefix) && name.Substring(name.IndexOf(prefix, StringComparison.Ordinal) + prefix.Length).Contains(part);
    }
}
