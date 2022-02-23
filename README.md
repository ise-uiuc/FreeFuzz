# FreeFuzz

This is the artifact of the research paper, "Free Lunch for Testing: Fuzzing Deep-Learning Libraries from Open Source", at ICSE 2022.

## About

FreeFuzz is the first approach to fuzzing DL libraries via mining from open source. It collects code/models from three different sources: 1) code snippets from the library documentation, 2) library developer tests, and 3) DL models in the wild. Then, FreeFuzz automatically runs all the collected code/models with instrumentation to collect the dynamic information for each covered API. Lastly, FreeFuzz will leverage the traced dynamic information to perform fuzz testing for each covered API.

This is the FreeFuzz's implementation for testing PyTorch and TensorFlow.

## Getting Started

### 1. Requirements

1. Our testing framework leverages [MongoDB](https://www.mongodb.com/) so you should [install and run MongoDB](https://docs.mongodb.com/manual/installation/) first.
	- Run the command `ulimit -n 64000` to adjust the limit that the system resources a process may use. You can see this [document](https://docs.mongodb.com/manual/reference/ulimit/) for more details.
2. You should check our dependent python libraries in `requirements.txt` and run `pip install -r requirements.txt` to install them
3. Python version >= 3.8.0 (It must support f-string.)

### 2. Setting Up with Dataset

#### Using Our Dataset

Run the following command to load the database.

```shell
mongorestore dump/
```

#### Collecting Data by Yourself

1. Go to `src/instrumentation/{torch, tensorflow}` to see how to intrument the dynamic information and add them into the database
2. After adding invocation data, you should run the following command to preprocess the data for PyTorch

```shell
cd src && python preprocess/process_data.py torch
```

or for TensorFlow
```shell
cd src && python preprocess/process_data.py tf
```

### 3. Configuration

There are some hyper-parameters in FreeFuzz and they could be easily configured as follows.

In `src/config/demo.conf`:

1. MongoDB database configuration.

```conf
[mongodb]
# your-mongodb-server
host = 127.0.0.1
# mongodb port
port = 27017 
# name of pytorch database
torch_database = freefuzz-torch
# name of tensorflow database
tf_database = freefuzz-tf
```

2. Output directory configuration.

```conf
[output]
# output directory for pytorch
torch_output = torch-output
# output directory for tensorflow
tf_output = tf-output
```

3. Oracle configuration.

```conf
[oracle]
# enable crash oracle
enable_crash = true
# enable cuda oracle
enable_cuda = true
# enable precision oracle
enable_precision = true
# float difference bound: if |a-b| > bound, a is different than b
float_difference_bound = 1e-5
# max time bound: if time(low_precision) > bound * time(high_precision),
# it will be considered as a potential bug
max_time_bound = 10
# only consider the call with time(call) > time_thresold
time_thresold = 1e-3
```

4. Mutation stratgy configuration.

```conf
[mutation]
enable_value_mutation = true
enable_type_mutation = true
enable_db_mutation = true
# the number of times each api is executed
each_api_run_times = 1000
```

### 4. Start

After finishing above steps, run the following command to start FreeFuzz to test PyTorch

```shell
cd src && python FreeFuzz.py --conf demo_torch.conf
```

Or run this command to test TensorFlow

```shell
cd src && python FreeFuzz.py --conf demo_tf.conf
```

To run the full experiment, run the following command
```shell
cd src && python FreeFuzz.py --conf expr.conf
```
If you want to use another configuration file, you can put it in `src/config`.

Note that you should specify the configuration file you want to use.

## Notes

1. Some APIs will be skipped since they may crash the program. You can set what you want to skip in the file `src/config/skip_torch.txt` or `src/config/skip_tf`.
2. For the details of three mutation strategies, please refer to our paper.
