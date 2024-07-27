# Sentiment Analysis of Textual Customer Reviews with PyTorch

Brief description of your project.

## Installation

### Requirements

- Python 3.8
- Other dependencies listed in `requirements.txt`

To install the required dependencies, run:  
```pip install -r requirements.txt```


## Usage

### Training

To train the model, navigate to the `src` folder and run:
```python train.py```


This script does the following:
- Loads the default configuration
- Prepares the data
- Splits the data into training and validation sets
- Sets up data loaders
- Initializes the model
- Trains the model using PyTorch Lightning

Checkpoints are saved in `src/output/<version>/` folder.
Logs are saved in `lightning_logs/<version>/` folder.

### Visualizing Training Logs

To visualize the training logs using TensorBoard, run the following command from the `src` folder:

```tensorboard --logdir=lightning_logs```

Then open a web browser and go to `http://localhost:6006` to view the TensorBoard dashboard.

### Testing

A Jupyter notebook `test.ipynb` is provided in the `src` folder for testing the model. The current accuracy achieved is 90%.

To run the test notebook:

1. Start Jupyter Notebook or JupyterLab
2. Navigate to the `src` folder
3. Open `test.ipynb`
4. Run the cells in the notebook

## License

Specify your project's license here.