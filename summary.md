/mri_graph_segmentation
├── /data_loader.py        # Loads MRI and mask data, converts to graph
├── /models
│   ├── /gcn.py            # GCN model
│   ├── /gat.py            # GAT model
│   └── /g_unet.py         # Graph U-Net model
├── /train
│   ├── /train_gcn.py      # Train GCN
│   ├── /train_gat.py      # Train GAT
│   └── /train_gunet.py    # Train Graph U-Net
├── /results
│   └── /compare_results.py # Compare model performance
└── /visualize.py          # Visualize predictions