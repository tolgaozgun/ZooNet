import os

import pandas as pd

def hyperparam_randomizer():
    lr = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    epoch = [50, 100]
    batch_size = [16, 32, 64, 128]

    lr_i = randint(0, len(lr) - 1)
    epoch_i = randint(0, len(epoch) - 1)
    batch_size_i = randint(0, len(batch_size) - 1)

    return lr[lr_i], batch_size[batch_size_i], epoch[epoch_i]


def create_run_dir(output_dir):
    # If output_dir is None, set it to the current directory
    if output_dir is None:
        output_dir = os.getcwd()

    # Get the list of directories in the folder that has the syntax "run_***" where *** is a number
    # Get the latest run number
    # Calculate the current run number by incrementing the latest run number by 1
    # Create a directory with the name "run_***" where *** is the current run number
    # Return the current run number
    print("Calculating run number...")
    run_number = 0
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for file in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, file)):
            if file.startswith("run_"):
                # Compare the current run number with the run number in the file name
                # If the run number in the file name is greater than the current run number, set the current run number to the run number in the file name
                if int(file[4:]) > run_number:
                    run_number = int(file[4:])

    run_number += 1

    # Set the directory name to "run_***" where *** is the current run number in 3 digits
    main_dir = os.path.join(output_dir, f"run_{run_number:04d}")
    os.mkdir(main_dir)
    
    # If subfolder is not None, create a subfolder in the run directory
    print(f"Run number: {run_number:04d}")
    return run_number, main_dir


def training_history_to_excel(training_history: dict, out_path: str):
    df = pd.DataFrame.from_dict(training_history)
    df.to_excel(out_path)
