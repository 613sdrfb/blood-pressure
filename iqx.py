import h5py

file_path = r"D:\leida\data_iq_origin\data01.h5"

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name} | Shape: {obj.shape}")

with h5py.File(file_path, 'r') as f:
    print("正在探测文件结构...")
    f.visititems(print_structure)