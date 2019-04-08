def create_merged_df(arr_data):
    for i in range((len(arr_data))):
        
        nlev = len(arr_data[i]['PRES_ADJUSTED'][0])
        nprof = len(arr_data[i]['PRES_ADJUSTED'])

        ind = np.arange(nprof*nlev)
        lat = np.array([[arr_data[i]["LATITUDE"][j].values]* nlev for j in range(nprof) ])
        lon = np.array([[arr_data[i]["LONGITUDE"][j].values]* nlev for j in range(nprof) ])
        posqc = np.array([[arr_data[i]["POSITION_QC"][j].values]* nlev for j in range(nprof) ])
        juld = np.array([[arr_data[i]["JULD"][j].values]* nlev for j in range(nprof) ])
        
        if(i == 0):    
            df = {'PLATFORM_NUMBER': pd.Series([str(arr_data[i]["PLATFORM_NUMBER"][0].values)]*len(ind), index=ind), 
                  'TEMP_ADJUSTED': pd.Series(arr_data[i]["TEMP_ADJUSTED"].values.flatten(), index=ind), 
                  'PSAL_ADJUSTED': pd.Series(arr_data[i]["PSAL_ADJUSTED"].values.flatten(), index=ind), 
                  'PRES_ADJUSTED': pd.Series(arr_data[i]["PRES_ADJUSTED"].values.flatten(), index=ind),  
                  'TEMP_ADJUSTED_QC': pd.Series(arr_data[i]["TEMP_ADJUSTED_QC"].values.flatten(), index=ind), 
                  'PRES_ADJUSTED_QC': pd.Series(arr_data[i]["PRES_ADJUSTED_QC"].values.flatten(), index=ind), 
                  'PSAL_ADJUSTED_QC': pd.Series(arr_data[i]["PSAL_ADJUSTED_QC"].values.flatten(), index=ind),  
                  'JULD': pd.Series(juld.flatten(), index=ind),  
                  'LATITUDE': pd.Series(lat.flatten(), index=ind),  
                  'LONGITUDE': pd.Series(lon.flatten(), index=ind),  
                  'POSITION_QC': pd.Series(posqc.flatten(), index=ind) }
            df = pd.DataFrame(df)
        else:
            df_i = {'PLATFORM_NUMBER': pd.Series([str(arr_data[i]["PLATFORM_NUMBER"][0].values)]*len(ind), index=ind), 
                  'TEMP_ADJUSTED': pd.Series(arr_data[i]["TEMP_ADJUSTED"].values.flatten(), index=ind), 
                  'PSAL_ADJUSTED': pd.Series(arr_data[i]["PSAL_ADJUSTED"].values.flatten(), index=ind), 
                  'PRES_ADJUSTED': pd.Series(arr_data[i]["PRES_ADJUSTED"].values.flatten(), index=ind),  
                  'TEMP_ADJUSTED_QC': pd.Series(arr_data[i]["TEMP_ADJUSTED_QC"].values.flatten(), index=ind), 
                  'PRES_ADJUSTED_QC': pd.Series(arr_data[i]["PRES_ADJUSTED_QC"].values.flatten(), index=ind), 
                  'PSAL_ADJUSTED_QC': pd.Series(arr_data[i]["PSAL_ADJUSTED_QC"].values.flatten(), index=ind),  
                  'JULD': pd.Series(juld.flatten(), index=ind),  
                  'LATITUDE': pd.Series(lat.flatten(), index=ind),  
                  'LONGITUDE': pd.Series(lon.flatten(), index=ind),  
                  'POSITION_QC': pd.Series(posqc.flatten(), index=ind) }
            df_i = pd.DataFrame(df_i)
            
            pdb.set_trace()
            df.append(df_i, ignore_index=True)
            
    return df
