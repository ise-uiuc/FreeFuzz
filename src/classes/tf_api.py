from functools import WRAPPER_UPDATES
import inspect
import json
import random
from typing import List, Dict
import numpy as np
import tensorflow as tf
from numpy.random import choice, randint

from constants.keys import *
from classes.argument import ArgType, Argument
from classes.api import API
from termcolor import colored

from classes.api import API
from classes.database import TFDatabase

from classes.argument import OracleType
from utils.probability import do_type_mutation, do_select_from_db

class TFArgument(Argument):
    _str_values = ["", "1", "sum", "same", "valid", "zeros"]
    _float_values = [0.0, 1.0, -1.0, 63.0, -63.0]
    _tensor_arg_dtypes = [ArgType.TF_TENSOR, ArgType.KERAS_TENSOR, ArgType.TF_VARIABLE]
    _dtypes = [
        tf.bfloat16, tf.bool, tf.complex128, tf.complex64, tf.double,
        tf.float16, tf.float32, tf.float64, tf.half,
        tf.int16, tf.int32, tf.int64, tf.int8,
        tf.uint8, tf.uint16, tf.uint32, tf.uint64,
    ]
    _support_types = [
        ArgType.TF_TENSOR, ArgType.TF_VARIABLE, ArgType.KERAS_TENSOR,
        ArgType.TF_DTYPE, ArgType.TF_OBJECT
    ]

    def __init__(self, value, type: ArgType, minv=0, maxv=0, shape=None, dtype=None) -> None:
        if isinstance(dtype, str):
            dtype = self.str_to_dtype(dtype)
        shape = self.shape_to_list(shape)

        super().__init__(value, type)
        self.minv = minv
        self.maxv = maxv
        self.shape = shape
        self.dtype = dtype

    @staticmethod
    def str_to_dtype(dt: str):
        dt = dt.strip().replace("_ref", "")
        if not dt.startswith("tf."):
            dt = "tf." + dt
        try:
            return eval(dt)
        except:
            return tf.float32

    @staticmethod
    def shape_to_list(shape): 
        if shape is None: return None   
        if not isinstance(shape, list):
            try:
                shape = shape.as_list()
            except:
                shape = list(shape)
            else:
                shape = list(shape)
        shape = [1 if x is None else x for x in shape]
        return shape

    @staticmethod
    def get_type(x):
        res = Argument.get_type(x)
        if res != None:
            return res
        if tf.is_tensor(x):
            if tf.keras.backend.is_keras_tensor(x):
                return ArgType.KERAS_TENSOR
            return ArgType.TF_TENSOR
        elif isinstance(x, tf.DType):
            return ArgType.TF_DTYPE
        
    def mutate_value_random(self) -> None:
        """ Apply random value mutation. """
        if self.type == ArgType.INT:
            self.value = self.mutate_int_value(self.value)
        elif self.type == ArgType.STR:
            self.value = self.mutate_str_value(self.value)
        elif self.type == ArgType.FLOAT:
            self.value = self.mutate_float_value(self.value)
        elif self.type == ArgType.BOOL:
            self.value = self.mutate_bool_value(self.value)
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            for arg in self.value:
                arg.mutate_value_random()
        elif self.type in self._tensor_arg_dtypes:
            self.minv, self.maxv = self.random_tensor_value_range(self.dtype)
        elif self.type == ArgType.TF_DTYPE:
            self.value = TFArgument.mutate_dtype()
        elif self.type == ArgType.TF_OBJECT:
            pass
        elif self.type == ArgType.NULL:
            pass
        else:
            raise ValueError(self.type)
            assert (0)

    def if_mutate_shape(self):
        return random.random() < 0.3

    def if_mutate_shape_value(self):
        return random.random() < 0.3

    def if_expand_dim(self):
        return random.random() < 0.3

    def if_squeeze(self):
        return random.random() < 0.3

    def mutate_shape(self, old_shape):
        new_shape = old_shape

        # Change rank
        if self.if_expand_dim():
            new_shape.append(1)
        elif len(new_shape) > 0 and self.if_squeeze():
            new_shape.pop()
        # Change value
        for i in range(len(new_shape)):
            if self.if_mutate_shape_value():
                new_shape[i] = self.mutate_int_value(new_shape[i], minv=0)
               
        return new_shape

    def generate_value_random(self) -> None:

        if self.type == ArgType.INT:
            self.value = self.mutate_int_value(0)
        elif self.type == ArgType.STR:
            self.value = self.mutate_str_value("")
        elif self.type == ArgType.FLOAT:
            self.value = self.mutate_float_value(0.)
        elif self.type == ArgType.BOOL:
            self.value = self.mutate_bool_value(True)
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            self.value = [TFArgument(1, ArgType.INT), TFArgument(1, ArgType.INT)]
        elif self.type in self._tensor_arg_dtypes:
            shape = [randint(1, 3), randint(1, 3)]
            dtype = choice([tf.int32, tf.float32, tf.float64])
            self.shape, self.dtype = shape, dtype
            self.value, self.minv, self.maxv = None, 0, 1
        elif self.type == ArgType.TF_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.TF_OBJECT:
            self.value = None
            pass
        elif self.type == ArgType.NULL:
            self.value = None
            pass
        else:
            assert (0)

    def mutate_type(self) -> None:
        def if_mutate_primitive():
            return random.random() < 0.1

        def if_mutate_null():
            return random.random() < 0.1

        if self.type in [ArgType.INT, ArgType.FLOAT, ArgType.STR, ArgType.BOOL]:
            if not if_mutate_primitive(): return False
            # change the type
            types = [ArgType.INT, ArgType.FLOAT, ArgType.STR, ArgType.BOOL]
            types.remove(self.type)
            self.type = choice(types)
            # change the value
            if self.type == ArgType.INT:
                self.value = self.mutate_int_value(0)
            elif self.type == ArgType.FLOAT:
                self.value = self.mutate_float_value(0.0)
            elif self.type == ArgType.STR:
                self.value = self.mutate_str_value("")
            elif self.type == ArgType.BOOL:
                self.value = choice([True, False])
        elif self.type in [ArgType.LIST, ArgType.TUPLE]:
            if random.random() < 0.01: 
                self.value = [] # with a probability return an empty list
            for arg in self.value:
                arg.mutate_type()
        elif self.type == ArgType.TF_TENSOR:
            dtype = choice(self._dtypes)
            shape = self.shape
            if self.if_mutate_shape():
                shape = self.mutate_shape(shape)
            self.shape, self.dtype = shape, dtype
        elif self.type == ArgType.TF_OBJECT:
            pass
        elif self.type == ArgType.NULL:
            if not if_mutate_null():
                return False
            new_type = choice(self._support_types + super()._support_types)
            if new_type == ArgType.LIST or new_type == ArgType.TUPLE:
                self.value = [
                    TFArgument(2, ArgType.INT),
                    TFArgument(3, ArgType.INT)
                ]
            elif new_type == ArgType.TF_TENSOR:
                self.shape = [2, 2]
                self.dtype = tf.float32

            if new_type != ArgType.NULL:
                try:
                    self.type = new_type
                    self.generate_value_random()
                except:
                    pass
        elif self.type == ArgType.TF_DTYPE:
            self.value = choice(TFArgument._dtypes)
        return True

    @staticmethod
    def if_mutate_int_random():
        return random.random() < 0.2

    @staticmethod
    def if_mutate_str_random():
        return random.random() < 0.1

    @staticmethod
    def if_mutate_float_random():
        return random.random() < 0.2

    
    def mutate_bool_value(self, value) -> bool:
        return choice([True, False])

    def mutate_int_value(self, value, minv=None, maxv=None) -> int:
        if TFArgument.if_mutate_int_random():
            value = choice(self._int_values)
        else:
            value += randint(-2, 2)
        if minv is not None:
            value = max(minv, value)
        if maxv is not None:
            value = min(maxv, value)
        return value
    
    def mutate_str_value(self, value) -> str:
        if TFArgument.if_mutate_str_random():
            return choice(self._str_values)
        return value

    @staticmethod
    def mutate_dtype() -> tf.dtypes.DType:
        return choice(TFArgument._dtypes)

    @staticmethod
    def low_precision_dtype(dtype):
        if dtype in [tf.int16, tf.int32, tf.int64]:
            return tf.int8
        elif dtype in [tf.float32, tf.float64]:
            return tf.float16
        elif dtype in [ tf.complex128]:
            return tf.complex64
        return dtype

    @staticmethod
    def random_tensor_value_range(dtype):
        assert isinstance(dtype, tf.dtypes.DType)
        minv = 0
        maxv = 1
        if dtype.is_floating or dtype.is_complex or dtype == tf.string or dtype == tf.bool:
            pass
        elif "int64" in dtype.name or "int32" in dtype.name or "int16" in dtype.name:
            minv = 0 if "uint" in dtype.name else - (1 << 8)
            maxv = (1 << 8)
        else:
            try:
                minv = dtype.min
                maxv = dtype.max
            except Exception as e:
                minv, maxv = 0, 1
        return minv, maxv

    def to_code_tensor(self, var_name, low_precision=False):
        dtype = self.dtype
        if low_precision:
            dtype = self.low_precision_dtype(dtype)
        shape = self.shape
        if dtype is None:
            assert (0)
        code = ""
        var_tensor_name = f"{var_name}_tensor"
        if dtype.is_floating:
            code += "%s = tf.random.uniform(%s, dtype=tf.%s)\n" % (var_tensor_name, shape, dtype.name)
        elif dtype.is_complex:
            ftype = "float64" if dtype == tf.complex128 else "float32"
            code += "%s = tf.complex(tf.random.uniform(%s, dtype=tf.%s)," \
                    "tf.random.uniform(%s, dtype=tf.%s))\n" % (var_tensor_name, shape, ftype, shape, ftype)
        elif dtype == tf.bool:
            code += "%s = tf.cast(tf.random.uniform(" \
                   "%s, minval=0, maxval=2, dtype=tf.int32), dtype=tf.bool)\n" % (var_tensor_name, shape)
        elif dtype == tf.string:
            code += "%s = tf.convert_to_tensor(np.ones(%s, dtype=str))\n" % (var_tensor_name, shape)
        elif dtype in [tf.int32, tf.int64]:
            code += "%s = tf.random.uniform(%s, minval=%d, maxval=%d, dtype=tf.%s)\n" \
                % (var_tensor_name, shape, self.minv, self.maxv + 1, dtype.name)
        else:
            code += "%s = tf.saturate_cast(" \
                "tf.random.uniform(%s, minval=%d, maxval=%d, dtype=tf.int64), " \
                "dtype=tf.%s)\n" % (var_tensor_name, shape, self.minv, self.maxv + 1, dtype.name)
        code += f"{var_name} = tf.identity({var_tensor_name})\n"
        return code

    def to_code_keras_tensor(self, var_name, low_precision=False):
        return self.to_code_tensor(var_name, low_precision=low_precision)

    def to_code(self, var_name, low_precision=False) -> str:
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_code(f"{var_name}_{i}", low_precision)
                arg_name_list += f"{var_name}_{i},"

            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
            return code
        elif self.type == ArgType.TF_OBJECT:
            return "%s = None\n" % (var_name)
        elif self.type == ArgType.TF_DTYPE:
            return "%s = tf.%s\n" % (var_name, self.value.name)
        elif self.type in self._tensor_arg_dtypes:
            # Did not consider cloning for in-place operation here.
            code = ""
            if self.type == ArgType.TF_TENSOR:
                code = self.to_code_tensor(var_name, low_precision=low_precision)
            elif self.type == ArgType.TF_VARIABLE:
                code = self.to_code_tensor(var_name, low_precision=low_precision)
                code += "%s = tf.Variable(%s)\n" % (var_name, var_name)
            elif self.type == ArgType.KERAS_TENSOR:
                code = self.to_code_keras_tensor(var_name, low_precision=low_precision)
            return code
        return super().to_code(var_name)


    def to_diff_code(self, var_name, low_precision=False) -> str:
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_diff_code(f"{var_name}_{i}", low_precision)
                arg_name_list += f"{var_name}_{i},"
            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
            return code
        elif self.type == ArgType.TF_OBJECT:
            return "%s = None\n" % (var_name)
        elif self.type == ArgType.TF_DTYPE:
            return "%s = tf.%s\n" % (var_name, self.value.name)
        elif self.type in self._tensor_arg_dtypes:
            code = f"{var_name} = tf.identity({var_name}_tensor)\n"
            if not low_precision:
                code += f"{var_name} = tf.cast({var_name}, tf.{self.dtype.name})\n"
            if self.type == ArgType.TF_VARIABLE:
                code += "%s = tf.Variable(%s)\n" % (var_name, var_name)
            return code
        return ""

    def mutate_value(self):
        self.mutate_value_random()

    @staticmethod
    def generate_arg_from_signature(signature):
        if isinstance(signature, bool):
            return TFArgument(signature, ArgType.BOOL)
        if isinstance(signature, int):
            return TFArgument(signature, ArgType.INT)
        if isinstance(signature, float):
            return TFArgument(signature, ArgType.FLOAT)
        if isinstance(signature, str):
            return TFArgument(signature, ArgType.STR)
        if isinstance(signature, list):
            value = []
            for elem in signature:
                value.append(TFArgument.generate_arg_from_signature(elem))
            return TFArgument(value, ArgType.LIST)
        if isinstance(signature, tuple):
            value = []
            for elem in signature:
                value.append(TFArgument.generate_arg_from_signature(elem))
            return TFArgument(value, ArgType.TUPLE)

        if (not isinstance(signature, dict)):
            return TFArgument(None, ArgType.NULL)

        if "type" not in signature and "Label" not in signature:
            return TFArgument(None, ArgType.NULL)

        label = signature["type"] if "type" in signature else signature["Label"]

        if label == "tf_object":
            if "class_name" not in signature:
                return TFArgument(None, ArgType.TF_OBJECT)

            if signature["class_name"] == "tensorflow.python.keras.engine.keras_tensor.KerasTensor" or \
                signature["class_name"] == "tensorflow.python.ops.variables.RefVariable":
                dtype = signature["dtype"]
                shape = signature["shape"]
                dtype = TFArgument.str_to_dtype(dtype)
                minv, maxv = TFArgument.random_tensor_value_range(dtype)
                return TFArgument(None, ArgType.TF_TENSOR, minv, maxv, shape, dtype)
            if signature["class_name"] == "tensorflow.python.framework.dtypes.DType":
                name = signature["to_str"].replace("<dtype: '", "").replace("'>", "")
                value = eval("tf." + name)
                return TFArgument(value, ArgType.TF_DTYPE)
            try:
                value = eval(signature.class_name)
            except:
                value = None
            return TFArgument(value, ArgType.TF_OBJECT)
        if label == "raw":
            try:
                value = json.loads(signature['value'])
            except:
                value = signature['value']
                pass
            if isinstance(value, int):
                return TFArgument(value, ArgType.INT)
            if isinstance(value, str):
                return TFArgument(value, ArgType.STR)
            if isinstance(value, float):
                return TFArgument(value, ArgType.FLOAT)
            if isinstance(value, tuple):
                tuple_value = []
                for elem in value:
                    tuple_value.append(TFArgument.generate_arg_from_signature(elem))
                return TFArgument(tuple_value, ArgType.TUPLE)
            if isinstance(value, list):
                list_value = []
                for elem in value:
                    list_value.append(TFArgument.generate_arg_from_signature(elem))
                return TFArgument(list_value, ArgType.LIST)

        if label == "tuple":
            try:
                value = json.loads(signature['value'])
                tuple_value = []
                for elem in value:
                    tuple_value.append(TFArgument.generate_arg_from_signature(elem))
                return TFArgument(tuple_value, ArgType.TUPLE)
            except:
                raise ValueError("Wrong signature " + str(signature))
        if label == "list":
            try:
                try:
                    value = json.loads(signature['value'])
                except:
                    value = signature['value']
                list_value = []
                for elem in value:
                    list_value.append(TFArgument.generate_arg_from_signature(elem))
                return TFArgument(list_value, ArgType.LIST)
            except:
                raise ValueError("Wrong signature " + str(signature))
        if label in ["tensor", "KerasTensor", "variable", "nparray"]:
            if not ('shape' in signature.keys()
                    and 'dtype' in signature.keys()):
                raise Exception('Wrong signature {0}'.format(signature))
            shape = signature['shape']
            dtype = signature["dtype"]
            dtype = TFArgument.str_to_dtype(dtype)

            if isinstance(shape, (list, tuple)):
                minv, maxv = TFArgument.random_tensor_value_range(dtype)
                return TFArgument(None, ArgType.TF_TENSOR, minv, maxv, shape, dtype)
            else:
                minv, maxv = 0, 1
                shape = [1, ]  
                return TFArgument(None, ArgType.TF_TENSOR, minv, maxv, shape, dtype)

        return TFArgument(None, ArgType.NULL)

class TFAPI(API):
    def __init__(self, api_name, record=None) -> None:
        super().__init__(api_name)
        self.record = TFDatabase.get_rand_record(api_name) if record is None else record
        self.args = TFAPI.generate_args_from_record(self.record)
        self.is_class = inspect.isclass(eval(self.api))

    def mutate(self, enable_value=True, enable_type=True, enable_db=True):
        num_arg = len(self.args)
        if num_arg == 0:
            return
        num_Mutation = randint(1, num_arg + 1)
        for _ in range(num_Mutation):
            arg_name = choice(list(self.args.keys()))
            arg = self.args[arg_name]

            if enable_type and do_type_mutation():
                arg.mutate_type()
            do_value_mutation = True
            if enable_db and do_select_from_db():
                new_arg, success = TFDatabase.select_rand_over_db(self.api, arg_name)
                if success:
                    new_arg = TFArgument.generate_arg_from_signature(new_arg)
                    self.args[arg_name] = new_arg
                    do_value_mutation = False
            if enable_value and do_value_mutation:
                arg.mutate_value()

    def to_code_oracle(self,
                prefix="arg", oracle=OracleType.CRASH) -> str:
        
        if oracle == OracleType.CRASH:
            code = self.to_code(prefix=prefix, res_name=RESULT_KEY)
            return self.wrap_try(code, ERROR_KEY)
        elif oracle == OracleType.CUDA:
            cpu_code = self.to_code(prefix=prefix, res_name=RES_CPU_KEY, 
                use_try=True, err_name=ERR_CPU_KEY, wrap_device=True, device_name="CPU")
            gpu_code = self.to_diff_code(prefix=prefix, res_name=RES_GPU_KEY,
                use_try=True, err_name=ERR_GPU_KEY, wrap_device=True, device_name="GPU:0")
            
            code = cpu_code + gpu_code
            return self.wrap_try(code, ERROR_KEY)
        elif oracle == OracleType.PRECISION:
            low_code = self.to_code(prefix=prefix, res_name=RES_LOW_KEY, low_precision=True,
                use_try=True, err_name=ERR_LOW_KEY, time_it=True, time_var=TIME_LOW_KEY)
            high_code = self.to_diff_code(prefix=prefix, res_name=RES_HIGH_KEY,
                use_try=True, err_name=ERR_HIGH_KEY, time_it=True, time_var=TIME_HIGH_KEY)
            code = low_code + high_code
            return self.wrap_try(code, ERROR_KEY)
        return ''

    @staticmethod
    def generate_args_from_record(record: dict):

        def generate_args_from_signatures(signatures):
            if isinstance(signatures, dict):
                if signatures['Label'] == 'list':
                    s = signatures['value']
                    if isinstance(s, list):
                        signatures = s
            args = []
            if signatures == None:
                return args
            for signature in signatures:
                x = TFArgument.generate_arg_from_signature(signature)
                args.append(x)
            return args

        args = {}
        for key in record.keys():
            if key == "input_signature":
                value = generate_args_from_signatures(record[key])
                args[key] = TFArgument(value, ArgType.LIST)
            elif key != "output_signature":
                args[key] = TFArgument.generate_arg_from_signature(record[key])
        return args

    def _to_arg_code(self, prefix="arg", low_precision=False):
        args = []
        kwargs = {}
        for key in self.args.keys():
            if "parameter:" in key:
                args.append(self.args[key])
            elif key != "output_signature" and key != "input_signature":
                kwargs[key] = self.args[key]

        arg_code = ""
        arg_str = ""
        index = 0
        for arg in args:
            arg_code += arg.to_code(f"{prefix}_{index}", low_precision=low_precision)
            arg_str += f"{prefix}_{index},"
            index += 1
        for key, arg in kwargs.items():
            arg_code += arg.to_code(key, low_precision=low_precision)
            arg_str += "%s=%s," % (key, key)
        return arg_code, arg_str
        
    def _to_diff_arg_code(self, prefix="arg", low_precision=False):
        args = []
        kwargs = {}
        for key in self.args.keys():
            if "parameter:" in key:
                args.append(self.args[key])
            elif key != "output_signature" and key != "input_signature":
                kwargs[key] = self.args[key]

        arg_code = ""
        arg_str = ""
        index = 0
        for arg in args:
            arg_code += arg.to_diff_code(f"{prefix}_{index}", low_precision=low_precision)
            arg_str += f"{prefix}_{index},"
            index += 1
        for key, arg in kwargs.items():
            arg_code += arg.to_diff_code(key, low_precision=low_precision)
            arg_str += "%s=%s," % (key, key)
        return arg_code, arg_str

    def to_code(self, prefix="arg", res_name="", low_precision=False, **kwargs) -> str:
        
        inputs = None
        input_name = ''
        if "input_signature" in self.args:
            inputs = self.args["input_signature"]
        if inputs:
            input_name = f"{prefix}_input"

        arg_code, arg_str = self._to_arg_code(prefix=prefix, low_precision=low_precision)
        res_code = ""
        if self.is_class:
            cls_name = f"{prefix}_class"
            arg_code += f"{cls_name} = {self.api}({arg_str})\n"
            if inputs:
                arg_code += inputs.to_code(input_name, low_precision=low_precision)
                res_code += f"{RES_KEY}[\"{res_name}\"] = {cls_name}(*{input_name})\n"
        else:
            res_code = f"{RES_KEY}[\"{res_name}\"] = {self.api}({arg_str})\n"
        invocation = self._to_invocation_code(arg_code, res_code, **kwargs)
        return invocation

    def to_diff_code(self, prefix="arg", res_name="", low_precision=False, **kwargs) -> str:
        
        inputs = None
        input_name = ''
        if "input_signature" in self.args:
            inputs = self.args["input_signature"]
        if inputs:
            input_name = f"{prefix}_input"

        arg_code, arg_str = self._to_diff_arg_code(prefix=prefix, low_precision=low_precision)
        res_code = ""
        if self.is_class:
            cls_name = f"{prefix}_class"
            res_code = f""
            if inputs:
                arg_code += inputs.to_diff_code(input_name, low_precision=low_precision)
                res_code += f"{RES_KEY}[\"{res_name}\"] = {cls_name}(*{input_name})\n"
        else:
            res_code = f"{RES_KEY}[\"{res_name}\"] = {self.api}({arg_str})\n"
        
        invocation = self._to_invocation_code(arg_code, res_code, **kwargs)
        return invocation

    def _to_res_code(self, res_name, arg_str, input_name=None, prefix="arg"):
        res_code = ""
        if self.is_class:
            cls_name = f"{prefix}_class"
            if input_name:
                res_code += f"{RES_KEY}[\"{res_name}\"] = {cls_name}(*{input_name})\n"
        else:
            res_code = f"{RES_KEY}[\"{res_name}\"] = {self.api}({arg_str})\n"
        return res_code

    def _to_invocation_code(self, arg_code, res_code, use_try=False, err_name="", 
        wrap_device=False, device_name="", time_it=False, time_var="", **kwargs) -> str:
        if time_it:
            res_code = res_code + self.wrap_time(res_code, time_var)
        code = arg_code + res_code
        inv_code = code
        if wrap_device:
            inv_code = self.wrap_device(inv_code, device=device_name)
        if use_try:
            inv_code = self.wrap_try(inv_code, error_var=err_name)
        return inv_code

    @staticmethod
    def wrap_try(code:str, error_var) -> str:
        wrapped_code = "try:\n"
        if code.strip() == "":
            code = "pass"
        wrapped_code += API.indent_code(code)
        wrapped_code += f"except Exception as e:\n  {RES_KEY}[\"{error_var}\"] = \"Error:\"+str(e)\n"
        return wrapped_code

    @staticmethod
    def wrap_device(code:str, device) -> str:
        device_code = f"with tf.device('/{device}'):\n" + API.indent_code(code)
        return device_code

    @staticmethod
    def wrap_time(code:str, time_var) -> str:
        wrapped_code = "t_start = time.time()\n"
        wrapped_code += code
        wrapped_code += "t_end = time.time()\n"
        wrapped_code += f"{RES_KEY}[\"{time_var}\"] = t_end - t_start\n"
        return wrapped_code


        
def test_tf_arg():
    arg = TFArgument(None, ArgType.TF_TENSOR, shape=[2, 2], dtype=tf.int64)
    arg.mutate_value()
    print(arg.to_code("var"))
    print(arg.to_code("var", True))

def test_tf_api():
    api_name = "tf.keras.layers.Conv2D"
    record = TFDatabase.get_rand_record(api_name)
    api = TFAPI(api_name, record)
    api.mutate()
    print(api.to_code_oracle(oracle=OracleType.CRASH))
    print(api.to_code_oracle(oracle=OracleType.CUDA))
    print(api.to_code_oracle(oracle=OracleType.PRECISION))

if __name__ == '__main__':
    # test_tf_arg()
    test_tf_api()
