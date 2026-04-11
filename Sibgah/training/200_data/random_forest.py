# Random Forest training entry-point (convenience wrapper).
# The actual implementation lives in random_forest/train.py
import runpy, os
runpy.run_path(os.path.join(os.path.dirname(__file__), "random_forest", "train.py"), run_name="__main__")
