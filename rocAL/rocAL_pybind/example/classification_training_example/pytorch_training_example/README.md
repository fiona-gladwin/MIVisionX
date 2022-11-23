## Running Classification_training_Example with flower dataset 

## Prerequisites
* MIVisionX
* RPP
* pytorch
* PIP3 - `sudo apt install python3-pip`

### Building the required pytorch Rocm docker
Use the instructions in the [docker section](../../../../../docker) to build the required [pytorch docker](../../../../../docker/pytorch)

### Dataset preprocessing

* For first run, to setup dataset, run the download_and_preprocess_dataset.py file 
```
python download_and_preprocess_dataset.py
```
### Running the training

To run this example for the first run or subsequent runs, just execute:
```
python3.9 Classification_training_flowerdataset.py
```
