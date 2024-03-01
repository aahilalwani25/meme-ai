import pandas as pd
import os

class DatasetController:

    def load_dataset_csv(self, dataset_file_name: str):
        
        csv_files_list= os.listdir('Datasets/csv_files')
        
        csv_file = [file for file in csv_files_list if file==dataset_file_name]
        if csv_file:
        # Load the first CSV file found into a pandas DataFrame
            csv_file_path = os.path.join('Datasets/csv_files', csv_file[0])
            df = pd.read_csv(csv_file_path)
            return (df)
        else:
            return("\nNo CSV files found in the directory.")

#print(load_dataset_csv('english_roman.csv'))