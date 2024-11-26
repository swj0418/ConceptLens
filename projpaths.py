import os
import sys

# HIERARCHY 1
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESOURCES_ROOT = os.path.join(PROJECT_ROOT, 'resources')
SERVER_ROOT = os.path.join(PROJECT_ROOT, 'server')
SERVED_DATA_ROOT = os.path.join(SERVER_ROOT, 'served_data')
WALKING_ROOT = os.path.join(PROJECT_ROOT, 'data_walking')

WALKED_DATA = os.path.join(WALKING_ROOT, 'output')


ALL_PATHS = [
    PROJECT_ROOT,
    RESOURCES_ROOT,
    SERVER_ROOT,
    SERVED_DATA_ROOT,
    WALKED_DATA
]

# Create folders if they don't exist
[os.makedirs(p, exist_ok=True) for p in ALL_PATHS]
