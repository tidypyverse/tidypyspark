from importlib import resources

def get_penguins_path():
  '''
  get path to penguins csv in the package directory
  '''
  with resources.path("tidypandas.data", "pen.csv") as f:
      data_file_path = f
  return data_file_path
