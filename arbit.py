zlowest = timeSlice.DEPTH.min() 
number_bins = np.abs(zlowest) // zbin
zbin_exact = np.abs(zlowest) / float(number_bins)
depth_bins = np.linspace(zlowest-1, 0, number_bins)

DSW = (timeSlice.gamman > 28.27) & (timeSlice.PSAL_ADJUSTED > 34.5) & (timeSlice.CTEMP <= -1.8) & (timeSlice.CTEMP >= -1.9)
lssw = (timeSlice.PSAL_ADJUSTED >= 34.3) & (timeSlice.PSAL_ADJUSTED <= 34.4) & (timeSlice.CTEMP <= -1.5) & (timeSlice.CTEMP > -1.9)
ISW = (timeSlice.CTEMP < -1.9)
CDW = (timeSlice.CTEMP >= 0) & (timeSlice.PSAL_ADJUSTED >= 34.5)
mCDW = (timeSlice.CTEMP < 0) & (timeSlice.CTEMP > -1.8) & (timeSlice.gamman > 28) & (timeSlice.gamman < 28.27)

totalcount = timeSlice.groupby(pd.cut(timeSlice.DEPTH, depth_bins) ).CTEMP.count().values
DSWcount = timeSlice.loc[DSW].groupby(pd.cut(timeSlice.loc[DSW].DEPTH, depth_bins) ).CTEMP.count().values
LSSWcount = timeSlice.loc[lssw].groupby(pd.cut(timeSlice.loc[lssw].DEPTH, depth_bins) ).CTEMP.count().values
ISWcount = timeSlice.loc[ISW].groupby(pd.cut(timeSlice.loc[ISW].DEPTH, depth_bins) ).CTEMP.count().values
mCDWcount = timeSlice.loc[mCDW].groupby(pd.cut(timeSlice.loc[mCDW].DEPTH, depth_bins) ).CTEMP.count().values
CDWcount = timeSlice.loc[CDW].groupby(pd.cut(timeSlice.loc[CDW].DEPTH, depth_bins) ).CTEMP.count().values
