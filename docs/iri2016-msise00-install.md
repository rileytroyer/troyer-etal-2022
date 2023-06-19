# semeter-inversion

This project takes an electron density profile as measured by the Poker Flat Incoherent Scatter Radar (PFISR) and estimates an incoming differential energy flux.

Based on: https://doi.org/10.1029/2004RS003042

Be aware, this is not intended to be production quality code.

Python packages iri2016 and msise00 are required. You can see how to install them below

msise00: https://github.com/space-physics/msise00
iri2016: https://github.com/space-physics/iri2016

Note: for iri2016 you should check that the indices include the dates that you are using.

Updated indices are here: http://irimodel.org/indices/

On my anaconda installation indices are stored for the code at:
envs/semeter-inversion/lib/python3.9/site-packages/iri2016/data/index/{apf107,ig_rz}.dat

For these to run properly, on Linux, you will need to run 
`sudo apt install cmake` and `sudo apt install gfortran`

## Issues
As of 2023-05-25, if using the newest version of Pandas, the misise00 will fail. This is due to a dependency on the geomagindices package: https://github.com/space-physics/geomagindices

This package uses a depreciated version of pandas.index.get_loc(). If you get an error that says datetime.datetime is not indexable you will need to edit the source file. In my installation this is located here: envs/semeter-inversion/lib/python3.9/site-packages/geomagindices/base.py

In this file change the line `i = [dat.index.get_loc(t, method="nearest") for t in dtime]` to `i = [dat.index.get_indexer([t], method="nearest")[0] for t in dtime]`

This is based on the solution here: https://stackoverflow.com/questions/71027193/datetimeindex-get-loc-is-deprecated

There is an issue reported in the geomagindices module, but the fix hasn't been implemented yet. 

