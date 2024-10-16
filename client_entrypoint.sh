#!/bin/bash

REPLICA_NUMBER=$(hostname | grep -o '[0-9]*$')

export FEDN_DATA_PATH="data/clients/${REPLICA_NUMBER}/"

# Hacky puke but it works
exec "$@"
