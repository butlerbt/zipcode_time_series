#!/bin/bash
# /*
#     Author:     Brent Butler, coached by Cristian Nuno
#     Purpose:    Import CSV file into a PSQL table
#     Date:       December 2, 2019
# */

echo "Start downloading data and documentation at $(date)"

# bash function used to retrieve the absolute file path of a file as a string
# note: thank you to peterh's answer on SO 
#       https://stackoverflow.com/a/21188136
get_str_abs_filename() {
  # $1 : relative filename
  echo "'$(cd "$(dirname "$1")" && pwd)/$(basename "$1")'"
}


# store the absolute file path of the .csv file that stores the Residential Building data
export RESBLDG_PATH=$(get_str_abs_filename "data/raw/EXTR_ResBldg.csv")

# store the absolute file path of the .csv file that stores the Resedential Property Sales data
export RPSALE_PATH=$(get_str_abs_filename "data/raw/EXTR_RPSale.csv")

# store the absolute file path of the .csv file that stores the Parcel data
export PARCEL_PATH=$(get_str_abs_filename "data/raw/EXTR_Parcel.csv")

# create a PostgreSQL database
createdb kc_housing

# import the csv files into the kc_housing database
# note: great tutorial on bash & psql found here
#       https://www.manniwood.com/postgresql_and_bash_stuff/index.html
psql \
    --dbname=kc_housing \
    --file=src/data/import_csv.sql \
    --set RESBLDG_PATH=$RESBLDG_PATH \
    --set RPSALE_PATH=$RPSALE_PATH \
    --set PARCEL_PATH=$PARCEL_PATH \
    --echo-all

echo "Finished downloading data and documentation at $(date)"

