import json
from write_tools import write_fn

def decorate_class(klass, hint):
    if not hasattr(klass, '__call__'):
        return klass
    old_init = klass.__init__
    old_call = klass.__call__
    init_params = dict()

    def json_serialize(v):
        try:
            json.dumps(v)
            return v
        except Exception as e:
            if hasattr(v, '__name__'):
                return v.__name__
            elif hasattr(v, '__class__'):
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
                    return v.__class__.__module__ + v.__class__.__name__
            return str(type(v))

    def build_param_dict(*args, **kwargs):
        param_dict = dict()
        for ind, arg in enumerate(args):
            param_dict['parameter:%d' % ind] = json_serialize(arg)
        for key, value in kwargs.items():
            param_dict[key] = json_serialize(value)
        return dict(param_dict)

    def get_var_shape(var):
        if hasattr(var, 'shape'):
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
        if isinstance(t, list) or isinstance(t, tuple):
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
        if isinstance(t, list) or isinstance(t, tuple):
            signatures = [get_var_signature(i) for i in t]
        else:
            signatures = get_var_signature(t)
        return signatures

    def new_init(self, *args, **kwargs):
        nonlocal init_params
        init_params = build_param_dict(*args, **kwargs)
        old_init(self, *args, **kwargs)

    def new_call(self, *inputs, **kwargs):
        nonlocal init_params
        input_signature = get_signature_for_tensors(inputs)
        outputs = old_call(self, *inputs, **kwargs)
        output_signature = get_signature_for_tensors(outputs)
        write_fn(hint, dict(init_params), input_signature, output_signature)
        return outputs

    klass.__init__ = new_init
    klass.__call__ = new_call
    return klass
