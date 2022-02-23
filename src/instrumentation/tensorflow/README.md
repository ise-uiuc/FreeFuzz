# Instrumentation 

This folder contains code to perform instrumentation on TensorFlow in order to collect dynamic execution information.

We hook the invocation of `1900` TensorFlow APIs, and the API names are listed in `api_list.txt`.

The key function is `def hijack(output_dir)` in `hijack.py`, where `output_dir` represents the path that all the traced information will be saved to.

## Usage:

(1) Copy the folder `instrumentation` to the root directory where TensorFlow is installed. For example, if TensorFlow is installed with `virtualenv`, then just copy it to `site-packages/tensorflow/instrumentation/`.

(2) Append these lines to the `site-packages/tensorflow/__init__.py`.

```
from tensorflow.instrumentation.hijack import hijack
hijack()
```

Then we execute the code collected in the first stage to trace various dynamic execution information for each API invocation. The outputs will be stored in the directory `signature_db`.
