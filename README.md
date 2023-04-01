# A baseline for the Leaf Nothing Behind competition

This repository shows a valid baseline code for the Leaf Nothing Behind competition. We
encourage participants to use it :
- as a starting point for their investigations,
- as a scalable structure for AI projects adapted with a pytorch Dataset for the competition,
- as an example of a submission with a valid format.

### Requirements

This code requires the following libraries :
- numpy
- scikit-image
- pytorch
- YAECS

Requirements can be installed using :
```
$ pip install -r requirements.txt
```

The code was tested, including the requirements' installation, and works with python 3.8 or on
google colab. There are supposedly issues with python 3.11 when using the versions provided in
the requirements.txt, but installing the libraries one by one with compatible versions was found
to solve these issues.

This code runs with pytorch as framework and YAECS as config system
(https://github.com/Antzyx/yaecs). Participants may however switch to any other tool they wish,
as long as their final code satisfies the submission format.

### Usage

To train a network, first configure the training as you wish in a config file or use the one in
`configs/train_config.yaml`. Also adapt the path to the dataset's CSV file in
`configs/paths_on_your_machine.yaml`. Then, simply run :
```
$ python main.py --config <path_to_your_config>
```
You may also modify any parameter from the command line :
```
$ python main.py --config <path_to_your_config> --<param_name> <new_value>
```
To further develop this code, do not forget that any new parameter must first be added to the
default config `configs/default.yaml` for it to be available in your code. Specifying parameter
types in that default config is optional.

### Submission format

When uploading a submission for the competition, please **MAKE SURE YOU ABIDE BY THE FORMAT
BELOW**. In this competition, you will submit your code which we will use on our private test set.
For safety reasons, your code **WILL BE READ** before we execute it on our machine.
Any suspicion of malevolent code will be punished by an immediate expulsion from the competition.

Additionally, we must enforce certain time constraints for the inference. At the absolute minimum,
the submitted solution must be able to process 2000 test samples in one hour on our test hardware,
which has the following specifications :
- GPU: Tesla P100-PCIE-16GB,
- CPUs: 4x Intel(R) Xeon(R) CPU @ 2.00GHz,
- RAM: 24 GB

The code will be tested on a google colab instance. Therefore you are encouraged to test your
submission on google colab to make sure it runs without issue. In particular, using conda with
colab might be difficult.

Your final submission should be a zip file that can be unzipped using the ubuntu 20.04 "unzip"
command. It should contain all the files in the root directory of your project, in such a way that
your project can be run using the content of the zip alone. The total size of the zip file should
be no more than 1 GB. If your weights file makes your project exceed 1 GB, you may have your code
download those weights from a self-hosted source at the start of your inference, but be aware that
the download time will count towards your inference time. We cannot guarantee that, in such cases,
the download will be as fast as you hope it should be.

In your final submission, you are free to use any library or method you prefer. However,
since we need to use your method on our test data to rank it, it should follow our standard
interface. Your final submission **MUST** be in one of the following two cases :
1) have a pip-installable `requirements.txt` and a `main.py` in the root folder
2) have a `infer.sh` script in the root folder

When your code is called (please see below how we will call it), it should accept the
following 3 arguments :
- `mode` : this argument will be given the value 'infer' ;
- `csv_path` : we will use this argument to give your code the path to the CSV file for the
  test data. Those CSV files will obey the same format as the train dataset, and the folder
  structure of the dataset will also be the same ;
- `save_infers_under` : we will use this argument to give your code the path to the folder
  where to save the results.

When called (please see below how we will call it), your code should read the provided CSV and
load, for each test sample, the 2 sentinel-2 images with corresponding masks, and the 3
sentinel-1 images. Then, for each test sample, it should predict the third sentinel-2 image in
the series. These result images should then be concatenated in numpy array format, resulting in
an array named for instance `image_results` of shape `(number_of_test_samples, 256, 256, 1)`.
The array should have the dtype float32.

Similarly, store the names of the predicted samples (ie. the third element in each line of the
CSV) in a list of strings, ordered in the same order as the first dimension of `image_results`.

Finally, store `image_results` and the name list in a python dictionary with the following
keys : `"paths"` and `"outputs"`. Save this dictionary in the provided folder (see
`save_infers_under` argument) as a pickle called `results.pickle`.

We provide below a shortened example that works with the data sample provided in the
baseline code (see `data` folder).

```python
import os, pickle
config = ...  # configuration where the parameters (including save_infers_under) are stored
image_results = ...  # concatenated result images. In the case of the provided dummy test data, shape should be (1, 256, 256, 1)
names_list = ...  # list of sample names. In the case of the provided dummy test data, should be ['LATVIA_LITHUANIA_2019-05-29_2019-06-10-2-0-25-29.tiff']
results = {
  "outputs": image_results,
  "paths": names_list,
}
with open(os.path.join(config.save_infers_under, "results.pickle"), 'wb') as file:
    pickle.dump(results, file)
```

Finally, in what follows, we explain in each possible case how we will call your code. Please
choose the option that satisfies the needs of your project and make sure you make it
compatible with our arguments and commands. For reference, the provided baseline code uses
version 1.

#### 1) Pip and main.py

We will use this option if and only if the following 2 requirements are
fulfilled :
- the root folder contains at least 2 files named `requirements.txt` and `main.py`
- the root folder does not contain any file named `infer.sh`

In this case, we will run the following commands from the project root :
```
$ pip install -r requirements.txt
$ python main.py \
   --mode infer \
   --csv_path /content/leaf_nothing_behind_baseline/data/test_data.csv \
   --save_infers_under /content/leaf_nothing_behind_baseline/data/inference_results
```

#### 2) infer.sh

We will use this option if and only if the following requirement is
fulfilled :
- the root folder contains a file named `infer.sh`

In this case, we will run the following commands from the project root :
```
$ sudo ./infer.sh \
   --mode infer \
   --csv_path /content/leaf_nothing_behind_baseline/data/test_data.csv \
   --save_infers_under /content/leaf_nothing_behind_baseline/data/inference_results
```
We will check the content of your `infer.sh` file before running it.