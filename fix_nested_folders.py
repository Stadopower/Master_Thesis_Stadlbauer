import os
import shutil

# Path to the main directory containing all subfolders
# Get the directory where the current .py file is located


# Change the working directory to that location
main_dir = r"D:\RUG\Master_Thesis\Master_Thesis_Stadlbauer\data"

for sub in os.listdir(main_dir):
    sub_path = os.path.join(main_dir, sub)
    nested_path = os.path.join(sub_path, sub)

    # Check if the nested folder exists
    if os.path.isdir(nested_path):
        print(f"Fixing: {sub_path}")

        # Move all contents from nested folder to parent
        for item in os.listdir(nested_path):
            src = os.path.join(nested_path, item)
            dst = os.path.join(sub_path, item)
            shutil.move(src, dst)

        # Remove the now-empty nested folder
        os.rmdir(nested_path)

# Path to the main directory containing your sub-Pxxx folders

for folder in os.listdir(main_dir):
    old_path = os.path.join(main_dir, folder)

    # Only rename if it's a folder and starts with "sub-P"
    if os.path.isdir(old_path) and folder.startswith("sub-P"):
        new_name = folder.replace("sub-", "", 1)  # remove only the first "sub-"
        new_path = os.path.join(main_dir, new_name)

        print(f"Renaming: {old_path} → {new_path}")
        os.rename(old_path, new_path)