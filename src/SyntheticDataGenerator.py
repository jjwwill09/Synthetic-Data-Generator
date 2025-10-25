import pandas as pd
from pathlib import Path as pt
import os
import csv


from faker import Faker
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer


import tkinter as tk
from tkinter import filedialog as fd


import numpy as np
import tensorflow as tf


import time


# The name of the new file
file_name = 'synthetic_data.xlsx'


# Upload a file and it will call the main function after
def upload_file_path():
   window.withdraw()


   file_path = fd.askopenfilename(
       title="Select a file",
       initialdir="/"
   )


   if file_path:
       global directory_path
       directory_path = file_path
       print(f"Selected file path: {file_path}")
       window.deiconify()
       main()
   else:
       print("No file selected.")
       window.deiconify()


def download_file():
   # Downloads a new spreadsheet based on the synthetic data table and prints the file name
   synthetic_data.to_excel(file_name, index=False)
   print(f"file downloaded as {file_name}")


# Sets up the main window GUI
window = tk.Tk()
window.title("Synthetic Data Generator (SDG)")
window.geometry("320x250")


label = tk.Label(window, text="upload a table file and get back synthetic data")
label.pack(pady=10)


# Add user input for number of synthetic rows
row_label = tk.Label(window, text="Enter number of synthetic rows:")
row_label.pack()
row_entry = tk.Entry(window)
row_entry.insert(0, "1000")
row_entry.pack(pady=5)


# TensorFlow Toggle
use_tensor = False


def toggle_tensor():
   global use_tensor
   use_tensor = not use_tensor
   print(f"TensorFlow is {'enabled' if use_tensor else 'disabled'}")


tensor_toggle = tk.Checkbutton(window, text="Toggle TensorFlow - (More Time)", command=toggle_tensor)
tensor_toggle.pack(pady=5)


# Upload and Download buttons
upload_button = tk.Button(window, text="upload file", command=upload_file_path)
download_button = tk.Button(window, text=f"download new file as {file_name}", command=download_file)
upload_button.pack(pady=10)
download_button.pack(pady=10)


# Convert .text files into .csv files
def txt_to_csv(txt_input, csv_output, delimiter=' '):
   print("----- Writing the CSV file -----")
   try:
       # Try encoding with standard utf-8
       with open(txt_input, 'r', encoding='utf-8') as infile, open(csv_output, 'w', newline='') as outfile:
           reader = csv.reader(infile, delimiter=delimiter)
           writer = csv.writer(outfile)
           for row in reader:
               writer.writerow(row)
   except:
       # If utf-8 fails, try with latin-1 (which can read all 8-bit characters)
       with open(txt_input, 'r', encoding='latin-1') as infile, open(csv_output, 'w', newline='', encoding='utf-8') as outfile:
           reader = csv.reader(infile, delimiter=delimiter)
           writer = csv.writer(outfile)
           for row in reader:
               writer.writerow(row)
   return csv_output


# Read file script made to take the type of file and then read it based on the file type
def readFile():
   try:
       fileType = file_path.suffix
       print("----------------------------")
       ss = None


       if fileType == ".xlsx":
           ss = pd.read_excel(file_path)
       if fileType == ".csv":
           ss = pd.read_csv(file_path)
       if fileType == ".json":
           ss = pd.read_json(file_path)
       if fileType == ".xml":
           ss = pd.read_xml(file_path)
       if fileType == ".txt":
           global csv_path
           csv_path = txt_to_csv(file_path, 'output.csv')
           print(f"New file path: {pt(csv_path)}")
           ss = pd.read_csv(csv_path)


       if ss is None:
           raise ValueError("Failed to read file")


       print(f"File extension: {fileType}")
       return ss
  
   except Exception as e:
       print(f"Error reading file: {str(e)}")
       raise


# TensorFlow Synthetic Name Generator 
def generate_synthetic_names(original_names, num_to_generate=1000, epochs=19900):
   print("Training TensorFlow model to generate synthetic names...")
   text = "\n".join(original_names.astype(str).str.lower())


   # Build vocabulary
   vocab = sorted(set(text))
   char2idx = {u: i for i, u in enumerate(vocab)}
   idx2char = np.array(vocab)


   # Convert to integer representation
   text_as_int = np.array([char2idx[c] for c in text])
   seq_length = 20
   char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
   sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)


   def split_input_target(chunk):
       return chunk[:-1], chunk[1:]


   dataset = sequences.map(split_input_target).shuffle(10000).batch(64, drop_remainder=True)


   # Model parameters
   vocab_size = len(vocab)
   embedding_dim = 64
   rnn_units = 128


   # Build functional model (stateless)
   inputs = tf.keras.Input(shape=(None,))
   x = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
   x = tf.keras.layers.GRU(rnn_units, return_sequences=True)(x)
   outputs = tf.keras.layers.Dense(vocab_size)(x)
   model = tf.keras.Model(inputs, outputs)


   model.compile(
       optimizer='adam',
       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
   )


   # Train model silently
   model.fit(dataset, epochs=epochs, verbose=0)


   # Text generation function (stateless)
   def generate_name(start_string="a"):
       input_eval = [char2idx.get(s, 0) for s in start_string]
       input_eval = tf.expand_dims(input_eval, 0)
       name_generated = []


       for _ in range(20):
           predictions = model(input_eval)
           predictions = tf.squeeze(predictions, 0)
           predicted_id = tf.random.categorical(predictions[-1:], num_samples=1)[0, 0].numpy()
           input_eval = tf.expand_dims([predicted_id], 0)
           name_generated.append(idx2char[predicted_id])
           if idx2char[predicted_id] == '\n':
               break


       return ''.join(name_generated).strip().title()


   # Generate synthetic names
   synthetic_names = [
       generate_name(np.random.choice(list("abcdefghijklmnopqrstuvwxyz")))
       for _ in range(num_to_generate)
   ]


   print(f"Generated {len(synthetic_names)} synthetic names.")
   return synthetic_names


# Main function
def main():
   try:
       global file_path
       file_path = pt(directory_path)


       print(f"File path to uploaded file: {file_path}")


       # Variable containing the file pathway information
       ss = readFile()
       faker = Faker()


       # Get number of rows from user input
       try:
           num_rows = int(row_entry.get())
       except:
           num_rows = 1000
       print(f"Generating {num_rows} synthetic rows...")


      
       # Exclude the name column from SDV synthesis
       has_name = 'name' in ss.columns
       if has_name:
           print("Excluding 'name' column from SDV synthesis")
           ss_no_name = ss.drop(columns=['name'])
       else:
           ss_no_name = ss


       # Starts Debugging Timer
       start_time = time.time()


       # Detect metadata AFTER removing the name column
       metadata = SingleTableMetadata()
       metadata.detect_from_dataframe(data=ss_no_name)


       # Creates synthesized data based on the file for however many rows you want
       synthesizer = GaussianCopulaSynthesizer(metadata)
       synthesizer.fit(data=ss_no_name)


       global synthetic_data
       synthetic_data = synthesizer.sample(num_rows=num_rows)


       # If the file contains a name column and TensorFlow is toggled, regenerate names using TensorFlow
       if has_name and use_tensor:
           print("Now using TensorFlow")
           synthetic_names = generate_synthetic_names(ss['name'], num_to_generate=len(synthetic_data))


           # Reinsert the synthetic names in the same column position
           cols = list(ss.columns)
           if 'name' in cols:
               name_index = cols.index('name')
               synthetic_data.insert(name_index, 'name', synthetic_names)
           else:
               synthetic_data['name'] = synthetic_names
       # If TensorFlow isn't toggled use Faker instead
       elif has_name:
           print("now using Faker")
           cols = list(ss.columns)
           if 'name' in cols:
               name_index = cols.index('name')
               fake_names = [faker.first_name() for _ in range(len(synthetic_data))]
               synthetic_data.insert(name_index, 'name', fake_names)


       # Ends time
       enlapsed_time = time.time() - start_time
       print(f"generating done in {enlapsed_time} seconds.")
       # Prints a summary of the new table to the terminal
       print("\nSynthetic data generation complete. Preview:")
       print(synthetic_data.head())


       if 'csv_path' in globals() and os.path.exists(csv_path):
           os.remove(csv_path)


       # Save synthetic dataset
       synthetic_data.to_excel(file_name, index=False)
       print(f"\nSynthetic data (with names) saved to {file_name}")


   # If main function fails print an error
   except Exception as e:
       print(f"Error in main function: {str(e)}")


# Main GUI loop
window.mainloop()