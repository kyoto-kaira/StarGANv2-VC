import os

train_list = []
val_list = []

for i, dir in enumerate(sorted(os.listdir("data"))):
    for j, path in enumerate(sorted(os.listdir(f"data/{dir}/parallel100/wav24kHz16bit"))):
        full_path = f"./Data/data/{dir}/parallel100/wav24kHz16bit/{path}"
        if j < 13:
            val_list.append([full_path, i])
        else:
            train_list.append([full_path, i])

    for path in sorted(os.listdir(f"data/{dir}/nonpara30/wav24kHz16bit")):
        full_path = f"./Data/data/{dir}/nonpara30/wav24kHz16bit/{path}"
        train_list.append([full_path, i])

with open("train_list.txt", "w") as f:
    for path, i in train_list:
        f.write(f"{path}|{i}\n")

with open("val_list.txt", "w") as f:
    for path, i in val_list:
        f.write(f"{path}|{i}\n")


