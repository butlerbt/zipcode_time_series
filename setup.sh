#!/bin/bash

# install necessary packages
# note: storing everything printed to the Terminal to the log/log.txt file
sh src/requirements/install.sh | tee -a log/01_install_log.txt


# download necessary data and place it in a PostgreSQL database
# note: storing everything printed to the Terminal to the log/log.txt file
sh src/data/get_data.sh | tee -a log/02_get_data_log.txt
