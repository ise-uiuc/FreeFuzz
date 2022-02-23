from functools import wraps
import json
import os
from write_tools import write_fn


def decorate_function(func, hint):
    @wraps(func)
    def wrapper(*args, **kwargs):
        def json_serialize(v):
            """Return a json serializable object. """
            try:
                json.dumps(v)
                return v  # v is a int, float, list, ...
            except Exception as e:
                if hasattr(v, 'shape'):  # v numpy array
                    return get_var_signature(
                        v)  # A dict of signature {'shape':..., 'type':...}
                if hasattr(v, '__name__'):  #  v is a function
                    return v.__name__
                elif hasattr(v, '__class__'):  # v is a class
                    res = []
                    if isinstance(v, tuple) or isinstance(v, list):
                        for vi in v:
                            if hasattr(vi, 'shape'):
                                res.append(get_var_signature(vi))
                            elif isinstance(vi, tuple) or isinstance(vi, list):
                                res2 = []
                                for vii in vi:
                                    if (hasattr(vii, 'shape')):
                                        res2.append(get_var_signature(vii))
                                res.append(res2)
                        return res
                    else:
                        return v.__class__.__module__ + v.__class__.__name__  # v.name
                else:
                    raise Exception('Error [json serialize ] %s' % v)

        def build_param_dict(*args, **kwargs):
            param_dict = dict()
            for ind, arg in enumerate(args):
                param_dict['parameter:%d' % ind] = json_serialize(arg)
            for key, value in kwargs.items():
                param_dict[key] = json_serialize(value)
            return param_dict

        def get_var_shape(var):
            if hasattr(var, 'shape'):  # var is numpy.ndarray or tensor
                s = var.shape
                if isinstance(s, list):
                    return s
                elif isinstance(s, tuple):
                    return list(s)
                else:
                    try:
                        return list(s)  # convert torch.Size to list
                    except Exception as e:
                        print(e.message)

        def get_var_dtype(var):
            if hasattr(var, 'dtype'):
                return str(var.dtype)  # string
            if isinstance(var, list):
                res = '['
                for varx in var:
                    res += type(varx).__name__ + ","
                return res[:-1] + "]"  # remove the ending ","
            elif isinstance(var, tuple):
                res = '['
                for varx in var:
                    res += type(varx).__name__ + ","
                return res[:-1] + "]"
            else:
                try:
                    return type(var).__name__
                except Exception as e:
                    print(e.message)

        def get_shape_for_tensors(t):
            if isinstance(t, list):
                input_shape = [get_var_shape(i) for i in t]
            else:
                input_shape = get_var_shape(t)
            return input_shape

        def get_var_signature(var):
            s = dict()
            s['shape'] = get_var_shape(var)
            s['dtype'] = get_var_dtype(var)
            return s

        def get_signature_for_tensors(t):
            if isinstance(t, list):
                signatures = [get_var_signature(i) for i in t]
            else:
                signatures = get_var_signature(t)
            return signatures

        outputs = func(*args, **kwargs)
        output_signature = get_signature_for_tensors(outputs)
        param_dict = build_param_dict(*args, **kwargs)
        write_fn(hint, param_dict, None, output_signature)
        return outputs

    if not callable(func):
        return func

    return wrapper
