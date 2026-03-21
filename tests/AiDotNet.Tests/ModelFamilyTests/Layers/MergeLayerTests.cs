// AddLayer and ConcatenateLayer require multiple inputs (multi-branch architecture).
// The single-input LayerTestBase pattern doesn't apply. These layers need
// specialized multi-input tests, not the standard invariant test suite.
// TODO: Create MultiInputLayerTestBase for merge layers.
