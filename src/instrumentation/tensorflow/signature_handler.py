import tensorflow as tf
import numpy as np

import json
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

def is_iterable(v):
    return isinstance(v, list) or isinstance(v, tuple)
def get_var_class_full_name(v):
    return v.__class__.__module__ + '.' + v.__class__.__name__

def json_serialize_value(v):
    """Return the json serializable value of v. """
    try:
        return json.dumps(v)
    except Exception as e:
        return str(v)

def json_deserialize_value(v):
    """Return the json serializable value of v. """
    try:
        return json.loads(v)
    except Exception as e:
        return v

class SignatureHandler:
    python_built_in_types = [str,
                             int, float, complex,
                             list, tuple, range,
                             dict,
                             set, frozenset,
                             bool,
                             bytes, bytearray, memoryview]
    def get_var_signature(self, v):
        if self.check_var_tensor(v):
            return self.get_tensor_signature(v)
        if self.check_var_variable(v):
            return self.get_variable_signature(v)
        if self.check_var_nparray(v):
            return self.get_nparray_signature(v)
        if self.check_var_tf_object(v):
            return self.get_tf_object_signature(v)

        if self.check_var_list(v):
            return self.get_list_signature(v)
        
        if self.check_var_raw(v):
            return self.get_raw_signature(v)
            
        return self.get_other_signature(v)

    def check_var_raw(self, v):
        """ Check if a variable is a python built-in object. """
        if type(v) in self.python_built_in_types:
            return True
        else:
            return False

    def get_raw_signature(self, v):
        s = dict()
        s['Label'] = 'raw'
        s['value'] = json_serialize_value(v)
        return s


    def check_var_list(self, v):
        """ Check if a variable is a list. """
        return isinstance(v, list) or isinstance(v, tuple)

    def get_list_signature(self, v):
        s = dict()
        s['Label'] = 'list'
        s['value'] = [self.get_var_signature(e) for e in v]
        return s

    def check_var_tuple(self, v):
        """ Check if a variable is a list. """
        return isinstance(v, list)

    def get_tuple_signature(self, v):
        s = dict()
        s['Label'] = 'tuple'
        s['value'] = (self.get_var_signature(e) for e in v)
        return s

    def check_var_tensor(self, v):
        """ Check if a variable is a TensorFlow tensor """
        if isinstance(v, tf.Tensor) or isinstance(v, KerasTensor):
            return True
        else:
            return False
    
    def check_var_variable(self, v):
        """ Check if a variable is a TensorFlow variable """
        if isinstance(v, tf.Variable):
            return True
        else:
            return False

    def get_tensor_shape(self, v):
        assert isinstance(v.shape, tf.TensorShape)
        try:
            return v.shape.as_list()
        except ValueError:
            # s has unknown shape (unknown rank): TensorShape(None)
            return None

    def get_tensor_signature(self, v):
        """ v is a Tensor."""
        s = dict()
        s['Label'] = 'tensor'
        if isinstance(v, KerasTensor):
            s['Label'] = 'KerasTensor'
        assert isinstance(v.dtype, tf.dtypes.DType)
        s['dtype'] = v.dtype.name
        s['shape'] = self.get_tensor_shape(v)
        return s

    def get_variable_signature(self, v):
        """ v is a variable"""
        s = dict()
        s['Label'] = 'variable'
        assert isinstance(v.dtype, tf.dtypes.DType)
        s['dtype'] = v.dtype.name
        s['shape'] = self.get_tensor_shape(v)
        return s

    def check_var_nparray(self, v):
        return isinstance(v, np.ndarray)

    def get_nparray_signature(self, v):
        s = dict()
        s['Label'] = 'nparray'
        s['shape'] = v.shape
        s['dtype'] = v.dtype.name
        return s

    def check_var_tf_object(self, v):
        return 'tensorflow' in get_var_class_full_name(v)

    def get_tf_object_signature(self, v):
        s = dict()
        s['Label'] = 'tf_object'
        s['class_name'] = get_var_class_full_name(v)
        if s["class_name"] == "tensorflow.python.framework.dtypes.DType":
            s['to_str'] = str(v)
        if hasattr(v, 'shape'):
            s['shape'] = self.get_tensor_shape(v)
        if hasattr(v, 'dtype'):
            if isinstance(v.dtype, tf.dtypes.DType):
                s['dtype'] = v.dtype.name
            else:
                s['type'] = str(v.dtype)

        return s

    def get_other_signature(self, v):
        s = dict()
        s['Label'] = 'other'
        s['type'] = str(type(v))
        return s

