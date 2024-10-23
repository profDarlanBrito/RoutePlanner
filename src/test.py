import Config
from Reconstruction import ConvexHull

settings = Config.Settings.get()

ConvexHull.load_variables()
ConvexHull.test()
