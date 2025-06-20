import os
import numpy as np
from PIL import Image
from tqdm import tqdm  # Progress bar

def convert_exe_to_image(exe_path, save_path, width=256):
    """
    Convert a .exe file into a grayscale .png image.
    """
    try:
        with open(exe_path, 'rb') as file:
            byte_data = file.read()

        byte_array = np.frombuffer(byte_data, dtype=np.uint8)
        height = int(np.ceil(len(byte_array) / width))
        padded_length = width * height
        byte_array = np.pad(byte_array, (0, padded_length - len(byte_array)), 'constant', constant_values=0)
        byte_array = byte_array.reshape((height, width))

        image = Image.fromarray(byte_array)
        image.save(save_path)
        return True  # Successfully converted
    except Exception as e:
        raise Exception(f"Error processing {exe_path}: {e}")  # Propagate the error

def batch_convert(exe_folder, output_folder, width=256):
    """
    Convert all .exe files inside a folder to .png images, preserving the folder structure.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    failed_files = []  # List to store files that couldn't be converted

    # Walk through the folder recursively
    for root, dirs, files in os.walk(exe_folder):
        for filename in tqdm(files, desc=f"Converting EXE to PNG in {root}"):
            if filename.endswith('.exe'):
                exe_path = os.path.join(root, filename)
                try:
                    # Preserve folder structure by creating corresponding directories
                    relative_path = os.path.relpath(root, exe_folder)
                    save_dir = os.path.join(output_folder, relative_path)
                    os.makedirs(save_dir, exist_ok=True)

                    img_name = filename.replace('.exe', '.png')
                    save_path = os.path.join(save_dir, img_name)

                    # Try converting the file
                    if not convert_exe_to_image(exe_path, save_path, width=width):
                        failed_files.append((exe_path, "Conversion failed for an unknown reason"))
                except Exception as e:
                    # Log failed file and reason
                    failed_files.append((exe_path, str(e)))

    # Print the summary of failures
    if failed_files:
        print("⚠️ Some files failed to convert:")
        for file, reason in failed_files:
            print(f"File: {file}\nReason: {reason}\n")
    else:
        print("✅ All files were converted successfully!")

    print(f"✅ Finished converting all files! Images saved to {output_folder}")
    print(f"Total files processed: {len(files)}")
    print(f"Total successful conversions: {len(files) - len(failed_files)}")
    print(f"Total failed conversions: {len(failed_files)}")

if __name__ == "__main__":
    # Set your paths
    exe_folder = "/home/subash/Desktop/phase2/Dataset/Virus/Virus train/Winwebsec"       # Your malware exe files location
    output_folder = "/home/subash/Desktop/phase2/Dataset_image/Virus_img/Virus_img_train/Winwebsec_img"   # Where to save .png images

    batch_convert(exe_folder, output_folder, width=256)
