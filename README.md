# Image Tracing

A basic Python script doing the following:

1. Training an auto-encoder with the [minst](https://keras.io/api/datasets/mnist/) dataset.
2. Saving the trained auto-encoder (.json/.h5 in models directory) for future use.
3. Making predictions (testing or aka decoding with auto-encoders).
4. Writing the decoded images in decoded directory.
5. Converting the png images (decoded) into svg using [potrace](http://potrace.sourceforge.net/) then writing them in decoded_to_svg directory.
