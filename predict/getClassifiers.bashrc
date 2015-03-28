#!/bin/bash

python classifyMutation.py L70P -c 2 -mk True
python classifyMutation.py L70P -c 22 -mk True
python classifyMutation.py L70P -c 3 -mk True
python classifyMutation.py L70P -c 4 -mk True

python classifyMutation.py L70P -c 2 -mk True -pr True
python classifyMutation.py L70P -c 22 -mk True -pr True
python classifyMutation.py L70P -c 3 -mk True -pr True
python classifyMutation.py L70P -c 4 -mk True -pr True