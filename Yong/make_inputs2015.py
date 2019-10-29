#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 17:12:03 2018

@author: smalldave
"""

from netCDF4 import Dataset
import numpy as np
import pandas as pd

lat0 =  -19
lat1 = 45
lon0 = 60
lon1 = 188
start_year = 1985 #>=
end_year = 2015 #<=
start_month = 7 #>=
end_month = 9 #<=
fname = '../Data/total_forecast_precipitation_mean_spread_corrected.nc'
infile = Dataset(fname)
lat=infile.variables['lat'][:]
lon=infile.variables['lon'][:]

y0=np.where(lat == lat0 )
y1 = np.where(lat==lat1)
lat_=infile.variables['lat'][y0[0][0]:y1[0][0]]

x0=np.where(lon == lon0)
x1 = np.where(lon==lon1)
lon_=infile.variables['lon'][x0[0][0]:x1[0][0]]

time=infile.variables['time'][:]
time=time/100.
time=pd.DataFrame(time)
time=time.astype(int)
time=time.astype(str)
time=time.reset_index()
time.columns=['precip_index','time']
time['year'] = time['time'].apply(lambda x: int(x[0:4]))
time['month'] = time['time'].apply(lambda x: int(x[4:6]))
time['day'] = time['time'].apply(lambda x: int(x[6:]))
a1=time.year<=end_year
a2=time.year>=start_year
a3=time.month<=end_month
a4=time.day>=start_month
time_=time.loc[a1 & a2 & a3 & a4]
precip = infile.variables['precipitation'][time_.index,y0[0][0]:y1[0][0],x0[0][0]:x1[0][0],:,:]

dirname='../Data/'
rootgrp = Dataset(''.join([dirname,'total_forecast_precipitation_mean_spread_input_2015.nc']), "w", format="NETCDF4")  

time = rootgrp.createDimension("time", len(time_))
lat = rootgrp.createDimension("lat", len(lat_))
lon = rootgrp.createDimension("lon", len(lon_))    
fhour = rootgrp.createDimension("fhour",15)
variable = rootgrp.createDimension("variable",2)

times = rootgrp.createVariable("time","f8",("time",))
fhour = rootgrp.createVariable("fhour","i4",("fhour",))
latitudes = rootgrp.createVariable("lat","f4",("lat",))
longitudes = rootgrp.createVariable("lon","f4",("lon",))
variable = rootgrp.createVariable("variable","int",("variable",))
temp = rootgrp.createVariable("precipitation","f8",("time","lat","lon","fhour","variable"))
#temp = rootgrp.createVariable("precipitation","f8",("time","fhour","lat","lon",))

latitudes[:] = lat_
longitudes[:] = lon_
times[:] = np.array(time_.time.astype(str))
temp[:,:,:,:] = precip
variable[:] = np.array([0,1])        
fhour = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
rootgrp.close() 
prec = np.array(precip)
np.save('../Data/total_forecast_precipitation_mean_spread_input_2015.npy',prec)
#time['forecast_period'] = 0

#for yr in time.year:
    #start = np.where( (time.year==yr) & (time.month=='05') & (time.day=='17') )[0][0]
    #stop =  np.where( (time.year == yr) & (time.month=='09') & (time.day == '30'))[0][0]
    #time.loc[start:stop,'forecast_period'] = 1

print 'Streamflow'
ganges = pd.read_csv('../Data/Ganges.csv')
dates = (ganges.Year >= start_year) & (ganges.Year<=end_year)
ganges = ganges[dates]
ganges = ganges.reset_index()
for lag in np.arange(-20,16):
    new_index = ganges.index + lag
    x = ganges.loc[new_index, 'Q (m3/s)' ]
    x = pd.DataFrame(x)
    x.columns = [''.join(['Q_',str(lag)])]   
    x.index = ganges.index
    ganges = pd.concat([ganges,x],axis=1)
ganges = ganges[ (ganges.Month >= start_month) & (ganges.Month<=end_month)]

columns = ['Date',]
for lag in np.arange(1,16):
    columns.append(''.join(['Q_',str(lag)]))
    
Y_Ganges = ganges[columns]

columns = ['Date']
for lag in np.arange(-20,1):
    columns.append(''.join(['Q_',str(lag)]))    
X_Ganges = ganges[columns]

X_Ganges.to_csv(''.join([dirname,'X_Ganges_2015.csv']))
Y_Ganges.to_csv(''.join([dirname,'Y_Ganges_2015.csv']))
'''########################################################################################################'''
input_ = '../Data/PERSIANN_1_X_1_withlatlong.nc'
infile = Dataset(input_,'r')

date = pd.DataFrame(infile.variables['time'][:])
date.columns=['date']  
date['Month'] = date['date'].apply(lambda x: x.split('-')[1])
date['Year'] = date['date'].apply(lambda x: x.split('-')[0])
date.Month=date.Month.astype(float)
date.Year = date.Year.astype(float)



lat_=list(infile.variables['lat'][:])
####lat_=[int(x) for x in lat_]
lat_0 = lat_.index(lat0)
lat_f = lat_.index(lat1)
lat_ = lat_[lat_0:lat_f]

lon_= list(infile.variables['lon'][:])
lon_0 = lon_.index(lon0)
lon_f = lon_.index(lon1)
lon_ = lon_[lon_0:lon_f]

# not_good = (date.Year>2016) | (date.Year<1985) 
# date=date.loc[~not_good]
good = (date.Month >= start_month) & (date.Month<= end_month) & (date.Year>= start_year) & (date.Year<= end_year)
date=date.loc[good]

Oprecip = infile.variables['precipitation'][:,lat_0:lat_f,lon_0:lon_f]
#date=date[date.Month<10]

ntime,nlat,nlon=Oprecip.shape
times = len(date)
Nl = -20
DATA=np.zeros((times,nlat,nlon,Nl))
date0=[]
for day in np.arange(Nl,1):
    index=np.array(date.index+day)
    print day,date.iloc[0+day].date
    DATA[:,:,:,day] = Oprecip[index,:,:]
    date0.append([day,date.loc[index[0]].date])    

fhour=np.array(range(Nl))


outfile = '../Data/persiann_look_1_x_1_20_2015.nc'
rootgrp = Dataset(outfile, "w", format="NETCDF4")  
latitude = infile.variables['lat'][:]
longitude = infile.variables['lon'][:]

time = rootgrp.createDimension("time", len(np.array(date)))
lat = rootgrp.createDimension("lat", len(lat_))
lon = rootgrp.createDimension("lon", len(lon_))    
fdy= rootgrp.createDimension("fdy",len(fhour))
times = rootgrp.createVariable("time","S10",("time",))

latitudes = rootgrp.createVariable("lat","f4",("lat",))
longitudes = rootgrp.createVariable("lon","f4",("lon",))  
temp = rootgrp.createVariable('precipitation',"f8",("time","fdy","lat","lon"))
fday = rootgrp.createVariable("fdy","f4",("fdy",))

latitudes[:] = lat_
longitudes[:] = lon_
times[:] = np.array(date.date)
fday[:] = fhour
temp[:,:,:,:] = DATA

prec1 = np.array(DATA)
np.save('../Data/persiann_look_1_x_1_20_2015.npy',prec1)
rootgrp.close()       


