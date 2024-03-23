import pyarrow as pa
import torch
import torch.utils.data as data

n_legs = pa.array([2, 4, 5, 100])
animals = pa.array(["Flamingo", "Horse", "Brittle stars", "Centipede"])
names = ["n_legs", "animals"]

batch = pa.record_batch([n_legs, animals], names=names)
table = pa.Table.from_batches([batch])
# Create a PyArrow table
#table = pa.Table.from_arrays([pa.array([1, 2, 3]), pa.array([4, 5, 6])])

# Create a PyTorch dataset
dataset = data.ArrowDataset(table)

# Create a PyTorch dataloader
dataloader = data.DataLoader(dataset, batch_size=3, shuffle=True)

# Iterate over the dataloader
for batch in dataloader:
    # Do something with the batch
    print(batch)

