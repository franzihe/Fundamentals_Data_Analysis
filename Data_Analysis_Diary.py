# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Intercomparison between CMIP6 models and ERA5 reanalysis of cloud phase and snowfall
# <br>
# <font size=4>
#     <p><b> Franziska Hellmuth</b></p>
#     <p><i> Fundamentals of Ocean/Atmosphere Data Analysis</i></p>
#     </font>
#
# In this course I want to begin on my work for the global comparison of cloud phase and snowfall. I'll start with CMIP6 data since I don't have all the ERA5 reanalysis yet.
# Variables which will be useful:
# - specific cloud liquid water content
# - specific cloud ice water content
# - temperature
# - surface snowfall

# +
# import packages
from imports import (xr, fct)

# reload imports
# %load_ext autoreload
# %autoreload 2

# plotting cosmetics
fct.plot_style()
# -

# Load 

fn = xr.open_dataset('/home/franzihe/nird_franzihe/NorESM2-LM/r1i1p1f1/clw_Amon_NorESM2-LM_historical_r1i1p1f1_gn_199001-199912.nc')

fn

fn.p0

p = fn.a*fn.p0 + fn.b*fn.ps

p.isel(time = 0)/100

fn.close()


