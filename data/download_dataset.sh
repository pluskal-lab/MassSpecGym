#!/bin/bash

# TODO: Download the datasets from the Hugging Face Hub
# wget ...

# TODO (Anton): It may better not to have this scipt but rather have as a part of the MassSpecDataset constructor.
# Similar to how its done in PyG: https://arc.net/l/quote/idfpqgij
# Probably we can pass the dataset name as an argument to the constructor and it will download it if it's not present.
# Or alternatively pass url instead of mgf_pth and download it if it's not present.
