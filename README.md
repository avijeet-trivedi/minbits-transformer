# MINBITS TRANSFORMER: MINIMUM PRECISION FOR STOCK PRICE PREDICTION

This repository implements a Transformer model that supports both full-precision floating-point and N-bit fixed-point arithmetic for activations and weights. This flexibility allows experimentation with reduced precision arithmetic, enabling trade-offs between computational efficiency and model accuracy. The model is specifically applied to the task of stock price prediction.

![image2](https://github.com/user-attachments/assets/a6cb8b8b-509c-4cc2-a2a6-459f4e869952)
* Generated using DALLE


---

## Overview

This repository contains a Jupyter Notebook implementation of a quantized Transformer model for stock price prediction. The notebook is organized into clearly defined sections to facilitate reproducibility and ease of understanding. Each section corresponds to a specific stage in the workflow, from data preparation to model analysis. Below is a summary of the key sections:

1. **Setup**: Importing all necessary libraries and packages required for the implementation.  
2. **Data Fetch and Pre-processing**: Fetching stock price data and preparing it for training and evaluation.  
3. **Quantized Transformer Model**: Defining the architecture of the Transformer model with support for quantized computations.  
4. **Training and Inference Function**: Functions for model training and generating predictions using the trained model.  
5. **Visualization Function**: Functions to visualize stock price predictions and evaluate the model's performance.  
6. **Model Analysis Function**: Tools to analyze and compare model performance across different quantization levels.  
7. **Execution**: Bringing all components together to train, evaluate, and analyze the model.  
8. **FP16 Quantization Using Torch.amp Library**: Demonstrating model quantization to FP16 precision using PyTorch’s `torch.amp` library.


## Setup

Before running the notebook, make sure you have the following Python packages installed:  

- `os`  
- `torch`  
- `pickle`  
- `numpy`  
- `pandas`  
- `torch.nn`  
- `yfinance`  
- `matplotlib`  
- `torch.nn.functional`  
- `ta`  
- `sklearn`  
- `torch.utils.data`  
- `torch.amp`  

You can install these packages using pip as follows:  
```bash
pip install torch numpy pandas yfinance matplotlib ta scikit-learn
```

This code was developed and tested on a MacBook Pro using the MPS (Metal Performance Shaders) backend for GPU acceleration. If you are running the notebook on Google Colab or another system, you may need to select the appropriate device for your setup.

To do this, locate the following line in the code:

```python
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

Replace 'mps' with the device available on your system, such as 'cuda' for NVIDIA GPUs or 'cpu' for standard CPU execution. If no compatible device is available, the code will default to running on the CPU.


## Data Fetch and Pre-processing

This section focuses on fetching historical stock market data, enriching it with technical indicators, and preparing it for training and evaluation.

### Functions:

1. **`store_stock_data(ticker, start_date, end_date, file_path)`**  
   This function fetches historical stock market data for a given ticker symbol and time period, using the `yfinance` library. It saves the data in CSV format at the specified file path.  
   - Columns included: `Date`, `High`, `Low`, `Open`, `Adj Close`, `Close`, `Volume`.

2. **`load_data_with_indicators(file_path, column='Close', time_step=100, train_fraction=0.8)`**  
   This function loads the stored stock data and performs the following:  
   - Ensures the `Date` column is in the correct format and adds `DayOfWeek` and `Month` as additional features.  
   - Adds technical indicators such as RSI, MACD, and Bollinger Bands using the `ta` library.  
   - Includes lagged `Close` values as features for improved model performance.  
   - Normalizes features and target using `MinMaxScaler`.  
   - Splits the data into training and testing sets based on the `train_fraction`.  
   - Creates time-series sequences of a specified length (`time_step`).  
   - Converts the processed data into PyTorch tensors.

### Output:
The function returns the following:  
- `train_X`, `train_Y`: Training features and labels as PyTorch tensors.  
- `test_X`, `test_Y`: Testing features and labels as PyTorch tensors.  
- `feature_scaler`, `target_scaler`: Scalers used for normalizing features and targets.

### Prerequisites:
Ensure the `yfinance` and `ta` libraries are installed. You can install them using pip:  
```bash
pip install yfinance ta scikit-learn
```

Notes:
* The data fetching and processing code assumes the input dataset contains a Date column. Ensure your dataset meets this requirement.
* Missing or invalid dates in the dataset will be dropped during processing.
* This implementation normalizes data to the range [0, 1] and splits it into train and test sets, making it ready for input to the Transformer model.


## Quantized Transformer Model

This section defines a custom transformer model with quantization applied to both the activations and weights of each layer. The model includes custom multi-head attention and feed-forward layers that support fixed-point quantization.

### Functions

- **`quantize_tensor(tensor, bits)`**:
    Quantizes the input tensor to a fixed-point representation with a specified number of bits. The range is determined by the minimum and maximum values of the tensor.

- **`quantize_tensor_weights(tensor, bits)`**:
    Quantizes the weight tensor to a fixed-point representation with a specified number of bits. Special handling is applied when `bits` is 1 (binarization).

- **`quantize_layer_weights(layer, bits)`**:
    Quantizes the weights of a given layer. This function iterates over all parameters in the layer and applies `quantize_tensor_weights` to the weights.

- **`QuantizedCustomMultiheadAttention`**:
    A custom multi-head attention layer that supports quantization for both activations and weights. The layer includes methods to quantize weights and activations during the forward pass.

- **`QuantizedTransformerStockPredictor`**:
    A transformer model designed for stock prediction that supports quantization for both activations and weights. It includes positional encoding, multiple self-attention layers, feed-forward layers and a fully connected layer at the end. Quantization is applied at each layer during both training and inference.

### Key Features

- **Quantization**: Both activations and weights are quantized to fixed-point representations, with the number of bits specified by the user. This reduces model size and computation during inference.
- **Custom Multi-Head Attention**: Implements a quantized version of multi-head attention that supports the same operations as the standard transformer, but with reduced precision for both weights and activations.
- **Flexible Configuration**: Users can adjust the number of bits for both activations and weights, allowing a trade-off between accuracy and performance.

The `QuantizedTransformerStockPredictor` class includes an implementation of the forward pass where quantization is applied to the inputs, self-attention layers, and feed-forward layers, followed by the output layer. Quantization is applied on each layer's activations and weights to enable more efficient inference while retaining the functionality of the original transformer architecture.


## Training and Inference Functions

### Training Function

The `training` function trains the model using Mean Squared Error (MSE) loss and the Adam optimizer. The training process includes both training and validation steps. The function accepts the following parameters:

- `model`: The neural network model to train.
- `train_X`, `train_Y`: The training features and labels.
- `test_X`, `test_Y`: The validation features and labels.
- `epochs`: The number of epochs to train the model (default is 50).
- `batch_size`: The batch size used for training (default is 32).
- `lr`: The learning rate for the optimizer (default is 0.001).
- `device`: The device (CPU or GPU) to run the model on (default is CPU).

#### Steps involved in training:

1. **Model and Data Initialization**:
   - The model and data are moved to the specified device (either CPU or GPU).
   
2. **Loss Calculation (Epoch 0)**:
   - The model is set to evaluation mode (`model.eval()`) to calculate the initial training and validation loss without updating the model parameters.
   - A batch-wise calculation of the training loss is done using the MSE loss function.
   - Similarly, validation loss is computed on the validation dataset.

3. **Training Loop**:
   - The model is set to training mode (`model.train()`).
   - A random permutation of the training dataset is generated, and batches are processed.
   - For each batch:
     - The optimizer gradients are cleared using `optimizer.zero_grad()`.
     - The model is forward-passed to obtain predictions.
     - The loss is computed, and backpropagation (`loss.backward()`) is performed.
     - Gradients are clipped to prevent exploding gradients using `torch.nn.utils.clip_grad_norm_()`.
     - The optimizer updates the model weights with `optimizer.step()`.
   - The training loss is calculated and averaged for the entire epoch.
   
4. **Validation Loop**:
   - The model is evaluated on the validation dataset after each training epoch.
   - No gradients are computed during validation (`with torch.no_grad()`).
   - The validation loss is computed similarly to the training loss.
   
5. **Learning Rate Scheduling**:
   - A learning rate scheduler (`StepLR`) is used to reduce the learning rate by a factor of 0.1 every 10 epochs.

6. **Return**:
   - After all epochs, the model is moved back to CPU, and a dictionary containing the training and validation losses for each epoch is returned.

### Inference Function

The `inference` function is used to make predictions on a dataset using the trained model. It accepts the following parameters:

- `model`: The trained PyTorch model.
- `data_loader`: A DataLoader containing the dataset (could be for training, validation, or test data).

#### Steps involved in inference:

1. **Model Evaluation**:
   - The model is set to evaluation mode (`model.eval()`) to disable dropout and batch normalization, ensuring consistent predictions.
   
2. **Batch-wise Prediction**:
   - For each batch of data in the `data_loader`:
     - The model performs a forward pass to obtain predictions.
     - Predictions and true labels (ground truth) are stored in `predictions` and `true_values` lists, respectively.
   
3. **Return**:
   - The function returns the predictions and true values as numpy arrays.

This function allows the model to generate predictions for a given dataset and is typically used after training to evaluate the model's performance on a test set or to make predictions on new data.


## Visualization Functions

### `plot_stock_prediction`

The `plot_stock_prediction` function visualizes the stock price predictions made by the trained model, along with the actual prices for both the training and test datasets. The function performs the following steps:

#### Parameters:
- `model`: The trained model used for making predictions.
- `train_X`, `train_Y`: The feature and target data for the training set.
- `test_X`, `test_Y`: The feature and target data for the test set.
- `batch_size`: The batch size used for data loading.
- `target_scaler`: The scaler used to inverse the scaling of the target variable (e.g., stock prices).
- `ticker`: The stock ticker symbol (used in the plot title).

#### Steps involved in plotting stock predictions:

1. **Data Preparation**:
   - `DataLoader` instances are created for both the training and test datasets, using the provided `train_X`, `train_Y`, `test_X`, and `test_Y` tensors. The batches are not shuffled as time-series data needs to maintain its order.

2. **Prediction**:
   - The `inference` function is called on both the training and test DataLoaders to obtain predictions (`train_preds`, `test_preds`) and actual values (`train_actuals`, `test_actuals`).

3. **Inverse Scaling**:
   - The predicted and actual values are inverse-transformed using the `target_scaler` to restore the original scale of stock prices. The predictions and actuals are reshaped and flattened for the plotting process.

4. **Data Combination**:
   - The training actual values (`train_actuals_unscaled`) and the test actual values (`test_actuals_unscaled`) are combined into a single array (`full_actuals`).
   - The predictions for the training set are combined with `NaN` values for the test set (`full_preds`).
   - Similarly, the test predictions are combined with `NaN` values for the training set (`test_preds_combined`).

5. **Plotting**:
   - The plot is generated with the following lines:
     - Blue line: Represents the actual stock prices (`full_actuals`).
     - Green dashed line: Represents the predicted prices for the training set (`full_preds`).
     - Red dashed line: Represents the predicted prices for the test set (`test_preds_combined`).
   - The plot is customized with a title, axis labels, and a grid for better readability.
   
6. **Displaying the Plot**:
   - The plot is displayed using `plt.show()` to visualize the results.

### `plot_loss`

The `plot_loss` function visualizes the training and validation loss over the course of the model's training. The function performs the following steps:

#### Parameters:
- `training_loss`: A list or array containing the training loss at each epoch.
- `validation_loss`: A list or array containing the validation loss at each epoch.
- `epochs`: The total number of epochs the model was trained for.
- `model_name`: The name of the model, which will be included in the plot title.

#### Steps involved in plotting loss curves:

1. **Plotting the Losses**:
   - The function creates a plot with the following lines:
     - The training loss (`training_loss`) is plotted over the epochs.
     - The validation loss (`validation_loss`) is plotted over the epochs.

2. **Customization**:
   - The plot includes:
     - A title indicating the model name (`{model_name} - Training vs Validation Loss`).
     - X-axis labeled as "Epochs".
     - Y-axis labeled as "Loss".
     - A legend to differentiate between training and validation losses.
     - A grid for better readability.

3. **Displaying the Plot**:
   - The plot is displayed using `plt.show()` to visualize the loss trends during training and validation.

These two functions help visualize the performance of the model during training, both in terms of loss reduction and prediction accuracy on the stock price prediction task.


## Model Analysis Function

### `model_analysis`

The `model_analysis` function trains or loads a pretrained model, performs stock price prediction analysis, and visualizes the results. It also tracks the model's loss convergence during training. The function executes the following steps:

#### Parameters:
- `activation_bits`: The number of bits used for activation precision (e.g., 32, 16, 8, 4). Default is `None`, meaning full precision (FP32).
- `weight_bits`: The number of bits used for weight precision (e.g., 32, 16, 8, 4). Default is `None`, meaning full precision (FP32).
- `load_pretrained_model`: A boolean flag indicating whether to load a pretrained model (`True`) or train a new one (`False`).
- `epochs`: The number of epochs for training. Default is 50.
- `batch_size`: The batch size for training. Default is 64.
- `lr`: The learning rate for training. Default is 0.001.

#### Steps involved in model analysis:

1. **Precision Selection**:
   - Based on the provided `activation_bits` and `weight_bits`, the precision for the model is determined:
     - If both `activation_bits` and `weight_bits` are `None`, the precision is set to 'FP32' (floating point 32-bit).
     - If both `activation_bits` and `weight_bits` are the same (e.g., 32, 16, 8, 4), the corresponding fixed-point precision is used (e.g., 'FX32', 'FX16').
     - If only `weight_bits` is set to 1, the model will use binary weights (i.e., 'BinarizeWeights').

2. **Model and Loss Tracker Paths**:
   - Paths for saving and loading the model and loss tracker are generated based on the ticker symbol, precision, and number of epochs.
   - These paths are used to save and load the model's state and the training loss history.

3. **Loading and Preprocessing Data**:
   - The function loads the data using the `load_data_with_indicators` function. The data is scaled using `feature_scaler` and `target_scaler`, and the data is split into training and test datasets based on the provided `train_fraction`.

4. **Model Creation**:
   - A `QuantizedTransformerStockPredictor` model is instantiated with the specified hyperparameters such as `input_dim`, `d_model`, `nhead`, `num_encoder_layers`, `dim_feedforward`, `drop_out`, and `activation_bits`/`weight_bits`.

5. **Training the Model**:
   - If `load_pretrained_model` is set to `False`, the model is trained on the loaded data using the `training` function, which tracks the loss over the epochs. The model is trained on the available device (`mps` or `cpu`), and the model's state is saved to `model_path`, while the loss tracker is saved to `loss_tracker_path`.

6. **Loading a Pretrained Model**:
   - If `load_pretrained_model` is set to `True`, the model loads its state from the `model_path`, and the loss tracker is loaded from the `loss_tracker_path`. This allows the user to resume from a previously saved model.

7. **Stock Price Prediction Visualization**:
   - The `plot_stock_prediction` function is called to visualize the actual vs predicted stock prices for both training and testing datasets. This helps in evaluating the model's prediction performance.

8. **Loss Convergence Visualization**:
   - The `plot_loss` function is called to visualize the training and validation loss curves over the epochs. This helps in evaluating the model's convergence and whether it is overfitting or underfitting.

In summary, this function provides a comprehensive approach to training, evaluating, and analyzing the performance of a stock price prediction model. It handles both training from scratch and loading pretrained models, while also providing insightful visualizations of the model's performance and loss convergence.

## Execution

### Overview

In this section, the code configures the parameters for the data, model, and training, and then initiates the training process for different quantization configurations. The goal is to train a stock price prediction model using various bit-widths for activations and weights, including full precision (FP32) and various fixed-point precisions (FX32, FX16, FX8, FX4), as well as binarized weights.

### Steps:

1. **Data Configuration**:
   - The `ticker` variable specifies the stock symbol to be used for data collection (e.g., 'AAPL' for Apple).
   - The `start_date` and `end_date` define the time period for which stock data is retrieved (from January 1, 2016, to January 1, 2024).
   - The `time_step` is set to 5, which determines the sequence length for the model.
   - The `train_fraction` is set to 0.8, meaning 80% of the data will be used for training, and the remaining 20% will be used for testing.
   - The `data_path` defines the location of the stock data file (`stocks_data/AAPL.csv`).

   The `store_stock_data` function is called with the above parameters to download and save the stock data to the specified `data_path`.

2. **Model Configuration**:
   - The model's configuration is set with the following parameters:
     - `drop_out`: Dropout rate of 0.3 is used to prevent overfitting.
     - `dim_feedforward`: The size of the feedforward layer in the model is set to 128.
     - `num_encoder_layers`: The model will use 4 encoder layers.
     - `nhead`: The number of attention heads is set to 8.
     - `dmodel`: The model dimension is set to 64.

3. **Training Configuration**:
   - The `load_pretrained_model` flag is set to `False`, indicating that the model will be trained from scratch.
   - `epochs` is set to 50, meaning the model will be trained for 50 epochs.
   - `learning_rate`: The learning rate for training is set to 0.001.
   - `batch_size`: The batch size for training is set to 64.

4. **Launching Training for Various Quantization Configurations**:
   - A loop is used to train the model with different quantization configurations. The following combinations of activation and weight bit-widths are tested:
     - **FP32**: Full precision (32-bit floating-point) for both activations and weights.
     - **FX32**: 32-bit fixed-point precision for both activations and weights.
     - **FX16**: 16-bit fixed-point precision for both activations and weights.
     - **FX8**: 8-bit fixed-point precision for both activations and weights.
     - **FX4**: 4-bit fixed-point precision for both activations and weights.
     - **BinarizeWeights**: Binary weights with full precision activations (1-bit for weights).

   For each combination of `activation_bits` and `weight_bits`, the `model_analysis` function is called, which trains the model or loads a pretrained model, and visualizes the results of the stock price predictions and loss convergence.

### Conclusion

This section automates the training process for different quantization schemes and provides a comprehensive analysis of how varying precision levels for activations and weights affect the performance of the stock price prediction model.


## FP16 Quantization using torch.amp library

### Model

The model used in this section is the same as the one described earlier, but without any quantization configuration capabilities. The model is implemented to support training with FP16 precision using the `torch.amp` library, which enables mixed-precision training.

### Training and Inference Function

In this section, the training and inference functions are modified to support FP16 mixed-precision. The key differences from the earlier training and inference functions are as follows:

#### Training Function (`training_fp16`)
1. **Mixed-Precision Training**: The `torch.amp` library is used for automatic mixed-precision (AMP) training, where `autocast` is used for performing operations in FP16 precision, and `GradScaler` is used to scale the gradients during backpropagation to avoid underflow.
2. **Loss Scaling**: The loss is scaled using `scaler.scale(loss)` to ensure stable gradient updates during training, which is particularly important for FP16 precision.

#### Inference Function (`inference_fp16`)
1. **Mixed-Precision Inference**: Similar to the training function, `autocast` is used during the inference phase to use FP16 precision for predictions.

### Execution

The execution in this section follows a similar approach to what was done previously. The data and model configurations remain the same, and the training is launched for FP16 precision using the modified `training_fp16` function. The model is then evaluated using the `inference_fp16` function for prediction and performance evaluation in FP16 precision.
