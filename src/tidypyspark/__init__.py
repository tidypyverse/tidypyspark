# read version from installed package
import tidypyspark.tidypyspark_class as ts
from importlib.metadata import version
__version__ = version("tidypyspark")
