graph_segmentation_project/
│── data_loader.py          # Loads MRI data & converts to graph
│── models/
│   │── gcn.py              # GCN implementation
│   │── gat.py              # GAT implementation
│   │── graph_unet.py       # Deep Graph U-Net implementation
│── train.py                # Trains and evaluates models
│── results.py              # Saves results (loss, accuracy, Dice score)
│── visualize.py            # Plots MRI slices, graph structure, segmentation results