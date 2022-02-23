import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import inspect
from tensorflow.instrumentation.decorators import dump_signature_of_class, dump_signature_of_function
def hijack(output_dir="signature_db"):
    hijack_all(output_dir)


def hijack_api(obj, func_name_str, output_dir):
    """
    Function to hijack an API.

    Args:
        obj: the base module. This function is currently specific to TensorFlow.
            So obj should be tensorflow.
        func_name_str: A string. The full name of the api (except 'tf.'). For example, the name of
            `tf.keras.losses.MeanSquaredError` should be 'keras.losses.MeanSquaredError'.

    Returns:
        A boolean, indicating if hijacking is successful.


    The targeted API can be either a function or a class (the type will be detected by this function).
    This function would replace the original api with the new decorated api we created. This is achieved
    in a fairly simple and straight-forward way. For the example above, we just set the attribute by calling
    `setattr(tf.keras.losses, 'MeanSquaredError', wrapped_func)`.
    """
    func_name_list = func_name_str.split('.')
    func_name = func_name_list[-1]

    # Get the module object and the api object.
    module_obj = obj
    if len(func_name_list) > 1:
        for module_name in func_name_list[:-1]:
            module_obj = getattr(module_obj, module_name)
    orig_func = getattr(module_obj, func_name)

    # Utilities.
    def is_class(x):
        return inspect.isclass(x)
    def is_callable(x):
        return callable(x)
    def is_built_in_or_extension_type(x):
        if is_class(x) and hasattr(x, '__dict__') and not '__module__' in x.__dict__:
            return True
        else:
            return False
    # Handle special cases of types.
    if is_built_in_or_extension_type(orig_func):
      return False
    if is_class(orig_func):
        if hasattr(orig_func, '__slots__'):
            return False
        wrapped_func = dump_signature_of_class(orig_func, func_name_str, output_dir=output_dir)
        setattr(module_obj, func_name, wrapped_func)
        return True
    else:
      if is_callable(orig_func):
        wrapped_func = dump_signature_of_function(orig_func, func_name_str, output_dir=output_dir)
        setattr(module_obj, func_name, wrapped_func)
        return True
      else:
        return False

def should_skip(api):

    skip_list = [
        'tf.keras.layers.Layer',
        'tf.compat.v1.keras.layers.Layer',
        'tf.Module',
        'tf.compat.v1.Module',
        'tf.compat.v1.flags.FLAGS',
        'tf.compat.v1.app.flags.EnumClassListSerializer',
        'tf.compat.v1.app.flags.EnumClassSerializer',
        'tf.compat.v1.flags.EnumClassListSerializer',
        'tf.compat.v1.flags.EnumClassSerializer',
        'tf.init_scope',
        'tf.TensorShape',
        'tf.Variable',
        'tf.compat.v1.Variable',
        'tf.ResourceVariable',
        'tf.Tensor',
        'tf.compat.v1.Tensor',
        'tf.compat.v1.flags.tf_decorator.make_decorator',
        'tf.compat.v1.flags.tf_decorator.tf_stack.extract_stack',
        'tf.compat.v1.flags.tf_decorator.unwrap',
        'tf.compat.v1.flags.tf_decorator.rewrap',
        'tf.compat.v1.app.flags.tf_decorator.make_decorator',
        'tf.compat.v1.app.flags.tf_decorator.rewrap',
        'tf.compat.v1.app.flags.tf_decorator.tf_stack',
        'tf.compat.v1.app.flags.tf_decorator.tf_stack.CurrentModuleFilter',
        'tf.compat.v1.app.flags.tf_decorator.tf_stack.FrameSummary',
        'tf.compat.v1.app.flags.tf_decorator.tf_stack.StackSummary',
        'tf.compat.v1.app.flags.tf_decorator.tf_stack.StackTraceFilter',
        'tf.compat.v1.app.flags.tf_decorator.tf_stack.StackTraceMapper',
        'tf.compat.v1.app.flags.tf_decorator.tf_stack.StackTraceTransform',
        'tf.compat.v1.app.flags.tf_decorator.tf_stack.extract_stack',
        'tf.compat.v1.app.flags.tf_decorator.unwrap',

    ]
    skip_key_word = [
        'tf.compat.v1',
        'tf.debugging',
        'tf.distribute',
        'tf.errors',
        'tf.profiler',
        'tf.test',
        'tf.tpu',
        'tf.summary',
        'tpu',
        'TPU',
        # 'tf.quantization', 
        # 'tf.experimental.numpy',

    ]
    
    if api.find('tf.') != 0:
        return True
    # Skip the current api if it's in the skip list.
    if api in skip_list:
        return True
    # Skip the current api if it has some keywords.
    for kw in skip_key_word:
        if kw in api:
            return True

def hijack_all(output_dir, verbose=False):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    success_list = []
    failed_list = []
    skip_list = []
    import os
    api_file = __file__.replace("hijack.py", "api_list.txt")
    with open(api_file, 'r') as fr:
        apis = fr.readlines()
    print('Number of total apis: ', len(apis)) 
    skip_apis = False
    cnt = 0
    for i, api in enumerate(apis):
        api = api.strip()
        if skip_apis:
            if should_skip(api):
                skip_list.append(api + "\n")
                continue

        hijack_api(tf, api[3:], output_dir)
