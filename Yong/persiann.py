import numpy as np
import pandas as pd

class sample_construction:
    def __init__(self,ext_len=5,lead=0,lookforward=15,start_year=1983,end_year=2017,lat_s=17,lat_e=40,lon_s=62,lon_e=109,data_path=''):
        #data: includes all the variables F, the last feature should be the target feature
        self.ext = ext_len
        self.lead = lead
        self.look = lookforward
        self.path=data_path
        self.start=start_year
        self.end=end_year
        self.lat_s=lat_s
        self.lat_e=lat_e
        self.lon_s=lon_s
        self.lon_e=lon_e

    def create_samples(self,target_river = 'G'): #G,B,M
        persiann = np.load(self.path+'persiann.npy')
        print ('persiann shape: ',persiann.shape)
        timestamps = np.load(self.path+'time.npy')    
        timestamps= pd.to_datetime(timestamps,format='%Y-%m-%d') 
        dates=pd.DataFrame(timestamps, index=timestamps,dtype='datetime64[ns]')

        lat = np.load(self.path+'lat.npy')
        lon = np.load(self.path+'lon.npy')
        # get flow data 1983 - 2017
        if target_river == 'G':
            target = pd.read_csv(self.path+'Ganges.csv',index_col=3,header=0,parse_dates=True)
        elif target_river == 'B':
            target = pd.read_csv(self.path+'Brahmaputra.csv',index_col=3,header=0,parse_dates=True)
        else:
            target = pd.read_csv(self.path+'Meghna.csv',index_col=3,header=0,parse_dates=True)   
        idx1  = (dates.index.year>(self.start-1))&(dates.index.year<(self.end+1)) 
        lat_idx = np.where(np.logical_and(lat>=self.lat_s, lat<=self.lat_e))
        lat_idx = np.array(lat_idx).flatten()
        lon_idx = np.where(np.logical_and(lon>=self.lon_s, lon<=self.lon_e))
        lon_idx = np.array(lon_idx).flatten()
        persiann = persiann[idx1,:,:][:,lat_idx,:][:,:,lon_idx]
        idx2  = (target.index.year>(self.start-1))&(target.index.year<(self.end+1))
        target= target.loc[target.index[idx2],'Q (m3/s)']
        target.rename(columns={"Q (m3/s)": "Q"})
        no_days = min(dates.shape[0],target.shape[0])
        persiann=persiann[:no_days]
        target=target.iloc[:no_days]
        print ('persiann shape: ',persiann.shape, 'target shape', target.shape)
        #print (dates.shape)
        #print (target.shape)
        #no_days = min(dates.shape[0],target.shape[0])
        #print (no_days)
        Y_dates=[]
        E = np.zeros((no_days-self.ext-self.lead-self.look,self.ext), float)
        X = np.zeros((no_days-self.ext-self.lead-self.look,self.look,len(lat_idx),len(lon_idx)), float)
        Y = np.zeros((no_days-self.ext-self.lead-self.look,1), float)
        #E = np.zeros((no_days-self.ext-self.lead-self.look,self.ext), float)
        i=0
        for index, val in target.iteritems():
            #print(i, val)
            if i+self.ext+self.lead+self.look >no_days-1:
                print('break.')
                break
            #use ith - i+self.look th days precipitation data to predict the flow of the day i+self.look+self.lead .
            E[i,:] = target.iloc[i:i+self.ext] 
            X[i,:self.look,:,:]= persiann[i+self.ext:i+self.ext+self.look,:,:]
            Y[i] = target.iloc[i+self.ext+self.lead+self.look] 
            Y_dates.append(target.index[i+self.ext+self.lead+self.look])
            i+=1
            #Y_dates.append(val)
        Y_df=pd.DataFrame(Y, index=Y_dates)
        #print (Y_ts.shape)
        # predict the flow of Jun, Jul, Aug, and Sep 
        idx  = (Y_df.index.month > 5)&(Y_df.index.month<10)
        E = E[idx,:]
        X = X[idx,:,:,:]
        Y_df = Y_df.iloc[idx,:]    
        np.save('E',E)    
        np.save('X',X)
        #np.save('Y',Y)
        Y_df.to_pickle('Y_df.pkl')