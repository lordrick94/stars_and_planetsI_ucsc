from dustmaps.bayestar import BayestarQuery
import dustmaps.bayestar
from dustmaps.config import config

# Set the location for downloading dustmaps
config['data_dir'] = '~/.dustmaps'

# Download the full Bayestar map
dustmaps.bayestar.fetch()
