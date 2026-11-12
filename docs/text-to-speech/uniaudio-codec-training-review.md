# UniAudio universal-codec training review

The scope is the codec pretraining stage in [UniAudio Appendix E](https://arxiv.org/html/2310.00704v3#A5), separate from the frozen codec used to train the language model. This is component evidence, not signoff on all UniAudio tasks or all 18 PR models.

The paper requires six mel discriminators, with hidden widths 64/128/256/512/512/512 and hops 32/64/128/256/512/1024. The native implementation now includes these critics and the author's linear/log mel inputs, seven weight-normalized convolution layers, least-squares adversarial objectives, feature matching, full-band and six-band PQMF spectral reconstruction, commitment loss, alternating generator/discriminator training, bandwidth sampling, and the released AdamW/epoch scheduler recipe.

Primary implementation source: `yangdongchao/UniAudio` at `c1146a8cecf84f1c2e877170d84998393b7e1e78`. Released configuration and weights: `Dongchao/UniAudio` at `835f07dcf1bdf586b0f427be59e4d48d0320f00e`, `universal_model/config.yaml` and `model.pth` inside `universal_model.zip`.

## Independent numerical checks

`tools/reference-data/uniaudio_codec_training_reference.py` executes the author's actual critic, spectral frontend, generator/discriminator losses and PQMF modules. It records outputs, every critic parameter gradient, input gradients, and the author's optimizer schedule. Compatibility adaptations preserve the old real/imaginary STFT layout and moved SciPy Kaiser-window symbol. A separately labeled double-operator context isolates explicit float32 casts while retaining the original neural methods.

`tools/reference-data/uniaudio_rvq_training_reference.py` executes the original LSX quantizer's training forward and autograd. It checks all three codebooks' embedding, EMA sums and counts after a step, along with codes, commitment loss and input gradients. The dead-code fixture contains one vector, making random replacement values deterministic without substituting a sampler.

These checks exposed and corrected three substantive errors:

- Feature matching observes the preceding feature after the author's in-place LeakyReLU. Saving the unactivated convolution output gives the wrong loss and gradients.
- Subband spectral convergence uses one Frobenius norm across the subband batch. Averaging independently normalized band losses gives a different objective.
- The LSX quantizer expires dead codes after EMA normalization. Replacing them before normalization overwrites the replacements. UniAudio now explicitly selects the author's ordering; other models retain their existing ordering.

Eleven focused tests pass: four author critic/frontend/objective comparisons, two complete quantizer training comparisons, and five integrated training lifecycle/configuration checks. These cover parameter ownership, freezing the LM during codec updates, freezing codec state during LM updates, generator/discriminator updates, epoch rates, evaluation without state mutation, and retaining the critic graph in copied options after returning to LM training.

`tools/reference-data/uniaudio_critic_pretrained_reference.py` additionally loads all 126 learned tensors of the six released critics into the unchanged author networks. It retains original float32 output separately from a double-operator reference with regenerated nonlearned Hann/mel buffers. `tools/TtsCheckpointAudit` mode `uniaudio-critics` checks the public checkpoint loader, all six real/fake score tensors, generator/discriminator objectives and waveform gradient. Production verification passes against Build38: all 126 original critic tensors load, scores differ by at most 1.90e-11, generator/discriminator objectives by 9.94e-15/8.15e-14, and waveform gradients by 6.22e-12. Later Build41 passes both framework targets and the integrated codec-stage regressions.

## Limits still requiring review

The existing complete LM/codec checkpoint audit validates released inference mathematics. Neither those checks nor the codec tests establish identical stochastic training trajectories, distributed multi-GPU behavior, corpus reproduction, or all eleven non-TTS task pipelines. Production framework builds and regression tests must also pass after these new codec-training changes.
