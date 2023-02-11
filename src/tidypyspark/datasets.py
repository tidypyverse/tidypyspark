from importlib import resources

def get_penguins_path():
  '''
  get path to penguins csv in the package directory
  '''
  with resources.path("tidypyspark.data", "pen.csv") as f:
      data_file_path = f
  return str(data_file_path)
