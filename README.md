# High density EEG and deep learning outcome prediction on the first day of coma after cardiac arrest

This repository contains the implementation of the model and pretrained weights (`.pt` files) for predicting outcome of comatose patients on the first and second day after cardiac arrest.  

---

## Usage

### 1. Import and initialize the model

```
from model import DeepFcCNN
import torch  # Initialize model
model = DeepFcCNN(in_chans=62, in_length=500)
```

### 2. Load pretrained weights

```
state_dict = torch.load("trained_model.pt", map_location="cpu") model.load_state_dict(state_dict)
model.eval()
```

### 3. Run inference

```
# Example input: batch of EEG signals
# Shape = (batch_size, n_channels, signal_length)
x = torch.randn(8, 62, 500)  # Forward pass
y = model(x)
print(y.shape)  # torch.Size([8])
print(y)        # Values between 0 and 1
```

---

## Input / Output

- **Input:** EEG tensor of shape `(batch_size, n_channels, 500)`
    
- **Output:** Tensor of shape `(batch_size,)`, values between 0 and 1 (class probability)
    

---

## Files

| File                | Description                             |
| ------------------- | --------------------------------------- |
| `deep_fc.py`        | Contains the model class description    |
| `models/<model>.pt` | Pretrained weights (PyTorch state_dict) |

## Citation

If you use this model or code in your work, please cite:

```
@article{Pelentritou2025.01.14.25320516,
	title={High density EEG and deep learning improves outcome prediction on the first day of coma after cardiac arrest},
	author={Pelentritou, Andria and Gruaz, Lucas and Iten, Manuela and Haenggi, Matthias and Zubler, Frederic and Rossetti, Andrea O and De Lucia, Marzia},
	year={2025},
	doi = {10.1101/2025.01.14.25320516},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2025/01/15/2025.01.14.25320516},
	journal = {medRxiv}
	}
```

